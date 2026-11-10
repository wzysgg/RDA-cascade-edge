#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EdgeServing-Stability (arXiv 2605.05527) as a live two-GPU T016 dispatcher.

This module is a direct port of the paper's Stability score as resolved against
the raw paper HTML by ``literature_baselines/EDGE_SERVING_METHOD_AUDIT.md``.
The paper-to-code map, every ADAPTED mechanism and the three items the paper
does not publish (numeric clip, tie-break, service costs) are documented in
``literature_baselines/EDGE_SERVING_ADAPTATION.md``.

Decision rule (B_max = 1, two cards, at most one component per call)
--------------------------------------------------------------------
1.  Queue heads.  Every executable component kind is one of the paper's
    fragment queues, represented by its FIFO head only: Fast components in the
    order the frozen kernel orders them (arrival, task id), Accurate components
    by (enqueue, task id), a pinned Suffix by (enqueue, task id) inside its own
    card list.  The candidate set is therefore
    ``{head(FAST_FULL), head(FAST_PREFIX), head(ACCURATE), head(pinned g)}``.
2.  Placement.  A ready candidate goes to the eligible free card that minimises
    its predicted completion ``R_g + L(m, e, B*)``; ties go to the smallest
    card index.  A pinned Suffix is eligible only on its own card (structural).
3.  Duration.  ``L`` is the frozen measured mean service time of the component
    plus the card's IPC leg.  A FAST_PREFIX commits its FAST_SUFFIX on the same
    card, so its candidate duration is the composite
    ``409954 + 1182290 = 1592244`` ns of service (plus IPC) - exactly the card
    occupancy the frozen kernel charges when it dispatches an inline prefix.
4.  Score. Candidate a advances the unserved queues by its own duration L_a
    (paper V-C), never by the maximum duration of all candidates::

        S_a = sum_{i in P, i != a} f(w_i + R_a + L_a)

    Batch remains 1 because only batch=1 service is measured. The single-GPU
    projection is applied locally at each free-card decision; this two-card
    extension does not reproduce the paper's batching or exit-selection knob.
5.  Selection.  ``argmin_a S_a``.  Score ties are broken by the frozen kernel's
    own public order - (predicted service work, arrival for a Fast component /
    enqueue for an Accurate one, card index) - and, only if all of those are
    equal, by the kernel job id, so the choice never depends on Python's
    iteration order.  The number of score ties broken and which element decided
    them are counted in :meth:`ledger`.
6.  Feasibility (paper Eq. (6), residual aware for two cards).  A candidate is
    admissible only while ``w_max + R_g + L <= tau``, where ``w_max`` is the
    candidate queue's own maximum waiting time (the paper's ``w_{m,max}``).
    When no candidate is admissible the arm either serves the score's argmin
    anyway and counts the fallback, or leaves the cards idle
    (``eq6_fallback_serve`` / ``eq6_deadline_filter``).

The candidate-specific duration remains in the score at batch=1. This is
an adapted stability mechanism experiment, not a full paper reproduction.

Only the :class:`lit_api.PublicView` handed to :meth:`choose_action` is read:
no future arrivals, no private state, no label / probability / phase field, and
the view is never mutated.  Deterministic: no randomness and no wall clock.
"""

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lit_api import LiteraturePolicy, register  # noqa: E402

#: Paper Eq. (3) numeric clip.  The paper never states C: it only illustrates
#: the point ``w > tau * (1 + ln 10) ~ 3.3 * tau`` where the *unclipped* weight
#: passes 10 (``_paper_cache/edgeserving_full.txt:219``; the normalization
#: ``w / tau - 1`` is inside the exponent, line 218).  The arm uses the
#: Lead-locked ``C = 10 * e = 27.18281828459045`` - that illustration read as
#: the saturation condition - and implements the paper's ``min()`` exactly, so
#: the clipped branch engages at ``w >= tau * (1 + ln C) = 4.3026 * tau``.
#: Unreachable at runtime (a queued component always has ``w <= tau``), but
#: implemented and unit-tested arithmetically.
CLIP_C_DEFAULT = 10.0 * math.e               # 27.18281828459045
CLIP_LOG_C_DEFAULT = 1.0 + math.log(10.0)    # 3.302585092994046, i.e. ln C

#: Roles the shared executor accepts.
ROLES = ('Fast', 'Accurate')

#: Frozen mean service calibration of every executable component (ns).  Used
#: only as a documented fallback for a view without a service table.
DEFAULT_SERVICE_NS = {
    'FAST_FULL': 1422843,
    'FAST_PREFIX': 409954,
    'FAST_SUFFIX': 1182290,
    'ACCURATE': 2735549,
}


class EdgeServingAdapter(LiteraturePolicy):
    """The EdgeServing-Stability dispatcher (arm name ``edge_serving``)."""

    name = 'edge_serving'
    version = 'stability-2gpu-adapted-2'
    audit_document = 'literature_baselines/EDGE_SERVING_ADAPTATION.md'
    bills_reference_inference = False

    def __init__(self, config=None):
        LiteraturePolicy.__init__(self, config)
        self._load_config()

    # ------------------------------------------------------------------ setup
    def _load_config(self):
        cfg = self.config or {}
        self.tau_fallback_ns = int(cfg.get('deadline_ns', 50000000))
        self.clip_c = float(cfg.get('clip_C', CLIP_C_DEFAULT))
        self.clip_log_c = float(cfg.get('clip_log_C', math.log(self.clip_c)))
        if self.clip_c <= 0.0 or self.clip_log_c <= 0.0:
            self.clip_c = CLIP_C_DEFAULT
            self.clip_log_c = CLIP_LOG_C_DEFAULT
        self.b_max = int(cfg.get('batch_max', 1))
        if self.b_max < 1:
            self.b_max = 1
        self.gpu_cards_hint = int(cfg.get('gpu_cards', 2))
        self.eq6_deadline_filter = bool(cfg.get('eq6_deadline_filter', True))
        self.eq6_fallback_serve = bool(cfg.get('eq6_fallback_serve', True))
        self.network_upload_ns = int(cfg.get('network_upload_ns', 0))
        self.network_return_ns = int(cfg.get('network_return_ns', 0))
        self.service_cfg = dict(cfg.get('gpu_service_ns') or {})
        self.ipc_cfg = int(cfg.get('gpu_ipc_budget_ns', 0))
        self._reset_ledger()

    def _reset_ledger(self):
        self._epochs_seen = []
        self._f_min = None
        self._f_max = None
        self._horizon_min = None
        self._horizon_max = None
        self._first_round = None
        self._last_round = None
        self._counters = {
            'choose_action_calls': 0,
            'rounds': 0,
            'candidate_rounds': 0,
            'dispatch_returns': 0,
            'idle_returns': 0,
            'no_pending': 0,
            'no_free_card': 0,
            'no_candidate': 0,
            'candidates_total': 0,
            'candidates_max': 0,
            'ties_broken': 0,
            'tie_break_none': 0,
            'tie_break_work': 0,
            'tie_break_fifo': 0,
            'tie_break_card': 0,
            'tie_break_job_id': 0,
            'clip_activated': 0,
            'tau_fallback_uses': 0,
            'eq6_checked': 0,
            'eq6_infeasible': 0,
            'eq6_all_infeasible': 0,
            'eq6_fallback_used': 0,
            'control_ticks': 0,
            'feedback_rows': 0,
            'feedback_on_time': 0,
            'feedback_late': 0,
            'feedback_model': {},
            'served_kind': {},
            'served_role': {},
        }

    def reset(self, builder, config):
        """Runner lifecycle hook: (re)load the frozen arm configuration."""
        if isinstance(config, dict) and config:
            self.config = dict(config)
        self._load_config()

    # -------------------------------------------------------------- decision
    def choose_action(self, now_ns, view):
        """Return ``(card, job, role)`` for one component, or ``None``."""
        c = self._counters
        c['choose_action_calls'] += 1
        if view is None:
            c['no_pending'] += 1
            return None
        now = int(now_ns if now_ns is not None else view.now_ns)
        c['rounds'] += 1
        epoch = getattr(view, 'adapter_epoch', None)
        if epoch not in self._epochs_seen:
            self._epochs_seen.append(epoch)

        pending = self._pending_jobs(view)
        if not pending:
            c['no_pending'] += 1
            return None
        free = view.free_cards()
        if not free:
            c['no_free_card'] += 1
            return None

        cands = self._candidates(now, view, pending, free)
        if not cands:
            c['no_candidate'] += 1
            return None
        c['candidate_rounds'] += 1
        c['candidates_total'] += len(cands)
        if len(cands) > c['candidates_max']:
            c['candidates_max'] = len(cands)

        horizon = 0
        for cand in cands:
            if cand['delta_ns'] > horizon:
                horizon = cand['delta_ns']
        if self._horizon_min is None or horizon < self._horizon_min:
            self._horizon_min = horizon
        if self._horizon_max is None or horizon > self._horizon_max:
            self._horizon_max = horizon

        # Eq. (4): candidate-specific projected state, served component deleted.
        f_lo, f_hi = None, None
        for cand in cands:
            # Paper V-C: each action advances OTHER queues by its own L_a.
            # A shared maximum horizon erases the service-cost tradeoff.
            total, lo, hi = self._projection_total(now, pending, cand['delta_ns'])
            cand['removed_term'] = self._f(cand['w_ns'] + cand['delta_ns'],
                                           cand['tau_ns'])
            cand['score'] = total - cand['removed_term']
            if lo is not None:
                f_lo = lo if f_lo is None else min(f_lo, lo)
                f_hi = hi if f_hi is None else max(f_hi, hi)
        if f_lo is not None:
            if self._f_min is None or f_lo < self._f_min:
                self._f_min = f_lo
            if self._f_max is None or f_hi > self._f_max:
                self._f_max = f_hi

        compared = cands
        if self.eq6_deadline_filter:
            c['eq6_checked'] += len(cands)
            feasible = [x for x in cands if x['slo_ok']]
            c['eq6_infeasible'] += len(cands) - len(feasible)
            if not feasible:
                c['eq6_all_infeasible'] += 1
                if not self.eq6_fallback_serve:
                    c['idle_returns'] += 1
                    self._record_round(now, len(pending), free, horizon, cands,
                                       None, 0, 'eq6-idle', True)
                    return None
                c['eq6_fallback_used'] += 1
            else:
                compared = feasible

        ranked = sorted(compared, key=self._rank_key)
        best = ranked[0]
        min_score = best['score']
        tied = [x for x in compared if x['score'] == min_score]
        tie_element = 'none'
        if len(tied) > 1:
            c['ties_broken'] += 1
            tie_element = self._tie_element(tied)
            c['tie_break_' + tie_element] += 1

        job = best['job']
        c['dispatch_returns'] += 1
        c['served_kind'][job.kind] = c['served_kind'].get(job.kind, 0) + 1
        c['served_role'][job.role] = c['served_role'].get(job.role, 0) + 1
        self._record_round(now, len(pending), free, horizon, cands, best,
                           len(tied), tie_element, False)
        return (int(best['card_id']), job, job.role)

    # --------------------------------------------------------- candidate set
    def _pending_jobs(self, view):
        """Every pending component of both roles and both card lists."""
        pending = []
        for role in ROLES:
            pending.extend(view.ready_jobs(role))
        for card in (view.cards or ()):
            pending.extend(view.pinned_jobs(int(card.card)))
        return pending

    def _queue_heads(self, now, pending):
        """``(head, w_max)`` per fragment queue: FIFO head and the queue's
        maximum waiting time ``w_{m,max}`` of paper Eq. (6)."""
        groups = {}
        for job in pending:
            affinity = getattr(job, 'card_affinity', None)
            if affinity is not None:
                key = ('card', int(affinity))
            else:
                key = ('ready', str(job.kind))
            waited = now - int(job.arrival_ns)
            if waited < 0:
                waited = 0
            row = groups.get(key)
            if row is None:
                groups[key] = [job, waited]
            else:
                if self._queue_order(job) < self._queue_order(row[0]):
                    row[0] = job
                if waited > row[1]:
                    row[1] = waited
        heads = []
        for key in sorted(groups, key=lambda k: (k[0], k[1])):
            row = groups[key]
            heads.append((row[0], row[1]))
        return heads

    def _queue_order(self, job):
        """FIFO order inside one fragment queue (the frozen kernel's own)."""
        if getattr(job, 'card_affinity', None) is not None:
            return (int(job.enqueue_ns), int(job.task_id), int(job.job_id))
        if job.role == 'Accurate':
            return (int(job.enqueue_ns), int(job.task_id), int(job.job_id))
        return (int(job.arrival_ns), int(job.task_id), int(job.job_id))

    def _candidates(self, now, view, pending, free):
        out = []
        for job, w_max in self._queue_heads(now, pending):
            card = self._card_for(view, job, free)
            if card is None:
                continue
            length = self._duration_ns(view, card, job)
            residual = self._residual_ns(card)
            delta = residual + length
            waited, tau = self._w_tau(job, now)
            out.append({
                'job': job,
                'job_id': int(job.job_id),
                'kind': str(job.kind),
                'role': str(job.role),
                'card': card,
                'card_id': int(card.card),
                'L_ns': int(length),
                'R_ns': int(residual),
                'delta_ns': int(delta),
                'w_ns': int(waited),
                'tau_ns': int(tau),
                'w_max_ns': int(w_max),
                'fifo_ns': self._fifo_ns(job),
                'slo_ok': bool(w_max + delta <= tau),
                'removed_term': 0.0,
                'score': 0.0,
            })
        return out

    def _card_for(self, view, job, free):
        affinity = getattr(job, 'card_affinity', None)
        if affinity is not None:
            for card in free:
                if int(card.card) == int(affinity):
                    return card
            return None
        best = None
        best_key = None
        for card in free:
            key = (self._duration_ns(view, card, job), int(card.card))
            if best_key is None or key < best_key:
                best_key = key
                best = card
        return best

    def _duration_ns(self, view, card, job):
        """Paper ``L(m, e, B*)``: measured service plus the card's IPC leg,
        plus the committed Suffix of an inline Fast Prefix."""
        index = int(card.card)
        try:
            length = int(view.predicted_service_ns(index, job))
        except (AttributeError, KeyError, TypeError):
            base = self.service_cfg.get(str(job.kind),
                                        DEFAULT_SERVICE_NS.get(str(job.kind), 0))
            length = int(base) + self.ipc_cfg
        if job.kind == 'FAST_PREFIX':
            try:
                length += int(view.suffix_commitment_ns(index))
            except (AttributeError, KeyError, TypeError):
                base = self.service_cfg.get('FAST_SUFFIX',
                                            DEFAULT_SERVICE_NS['FAST_SUFFIX'])
                length += int(base)
        return length

    @staticmethod
    def _residual_ns(card):
        if not card.busy:
            return 0
        remaining = card.busy_remaining_ns
        if remaining is None:
            return 0
        return max(0, int(remaining))

    def _w_tau(self, job, now):
        arrival = int(job.arrival_ns)
        waited = now - arrival
        if waited < 0:
            waited = 0
        tau = int(job.deadline_abs_ns) - arrival
        if tau <= 0:
            tau = self.tau_fallback_ns
        return waited, tau

    @staticmethod
    def _fifo_ns(job):
        return int(job.enqueue_ns) if job.role == 'Accurate' else int(job.arrival_ns)

    # ------------------------------------------------------------ score math
    def _f(self, w_ns, tau_ns):
        """Paper Eq. (3) weight ``min(exp(w / tau - 1), C)`` without overflow."""
        x = (float(w_ns) / float(tau_ns)) - 1.0
        if x >= self.clip_log_c:
            self._counters['clip_activated'] += 1
            return self.clip_c
        return math.exp(x)

    def _projection_total(self, now, pending, horizon):
        """``T(H) = sum_i f(w_i + H)`` over every pending component."""
        counters = self._counters
        log_c = self.clip_log_c
        clip = self.clip_c
        exp = math.exp
        tau_fallback = self.tau_fallback_ns
        total = 0.0
        f_lo = None
        f_hi = None
        fallbacks = 0
        for job in pending:
            arrival = int(job.arrival_ns)
            tau = int(job.deadline_abs_ns) - arrival
            if tau <= 0:
                tau = tau_fallback
                fallbacks += 1
            w = now - arrival + horizon
            if w < 0:
                w = 0
            x = (w / float(tau)) - 1.0
            if x >= log_c:
                value = clip
                counters['clip_activated'] += 1
            else:
                value = exp(x)
            total += value
            if f_lo is None or value < f_lo:
                f_lo = value
            if f_hi is None or value > f_hi:
                f_hi = value
        counters['tau_fallback_uses'] += fallbacks
        return total, f_lo, f_hi

    # ------------------------------------------------------------- tie-break
    @staticmethod
    def _rank_key(cand):
        """Paper Eq. (7) argmin, then the frozen kernel's own public order.

        ``(S_m, predicted service work, arrival | enqueue, card index)`` and,
        only when all of those are equal, the kernel job id.
        """
        return (cand['score'], cand['L_ns'], cand['fifo_ns'],
                cand['card_id'], cand['job_id'])

    @staticmethod
    def _tie_element(tied):
        """Which element of the frozen order decided a score tie."""
        for label, field in (('work', 'L_ns'), ('fifo', 'fifo_ns'),
                             ('card', 'card_id'), ('job_id', 'job_id')):
            if len(set(x[field] for x in tied)) > 1:
                return label
        return 'none'

    # ------------------------------------------------------------- lifecycle
    def on_control_tick(self, now_ns, view):
        self._counters['control_ticks'] += 1

    def on_completion(self, record):
        c = self._counters
        if not isinstance(record, dict):
            return
        c['feedback_rows'] += 1
        if record.get('on_time'):
            c['feedback_on_time'] += 1
        else:
            c['feedback_late'] += 1
        model = record.get('model')
        if model is not None:
            key = str(model)
            c['feedback_model'][key] = c['feedback_model'].get(key, 0) + 1

    # ---------------------------------------------------------------- audit
    def _record_round(self, now, n_pending, free, horizon, cands, best,
                      n_tied, tie_element, eq6_idle):
        record = {
            'now_ns': int(now),
            'pending': int(n_pending),
            'free_cards': [int(card.card) for card in free],
            'horizon_ns': int(horizon),
            'n_candidates': len(cands),
            'n_score_ties': int(n_tied),
            'tie_break': tie_element,
            'eq6_idle': bool(eq6_idle),
            'candidates': [{
                'kind': x['kind'], 'role': x['role'], 'job_id': x['job_id'],
                'card': x['card_id'], 'wait_ns': x['w_ns'], 'tau_ns': x['tau_ns'],
                'w_max_ns': x['w_max_ns'], 'R_ns': x['R_ns'], 'L_ns': x['L_ns'],
                'delta_ns': x['delta_ns'], 'score': round(x['score'], 6),
                'slo_ok': x['slo_ok']} for x in cands],
            'chosen': None if best is None else {
                'kind': best['kind'], 'job_id': best['job_id'],
                'card': best['card_id'], 'wait_ns': best['w_ns'],
                'score': round(best['score'], 6)},
        }
        self._last_round = record
        if self._first_round is None:
            self._first_round = record

    def ledger(self):
        c = self._counters
        return {
            'adapter': self.name,
            'version': self.version,
            'audit_document': self.audit_document,
            'method': 'EdgeServing-Stability (arXiv 2605.05527)',
            'batch_max': self.b_max,
            'gpu_cards_config': self.gpu_cards_hint,
            'clip_C': self.clip_c,
            'clip_log_C': self.clip_log_c,
            'clip_activated_terms': c['clip_activated'],
            'tie_break_order': ['S_m', 'predicted_service_work_ns',
                                'arrival_ns|enqueue_ns', 'card_index', 'job_id'],
            'ties_broken': c['ties_broken'],
            'tie_break_used': {
                'work_ns': c['tie_break_work'],
                'fifo_key': c['tie_break_fifo'],
                'card_index': c['tie_break_card'],
                'job_id': c['tie_break_job_id'],
                'none': c['tie_break_none']},
            'horizon_mode': 'per_candidate_service_increment',
            'horizon_ns': {'min': self._horizon_min, 'max': self._horizon_max},
            'f_range_projected': {'min': self._f_min, 'max': self._f_max},
            'clip_range_note': (
                'every queued component satisfies w <= tau while it sits in the '
                'frozen ready queue (the kernel expires a component as soon as '
                'now > deadline_abs_ns), so its unprojected weight is at most '
                '1; each candidate adds its own service increment on top of w, far '
                'below the saturation point tau*(1+ln C) = 4.3026*tau, so the '
                'clip cannot activate in this simulator; the clipped branch is '
                'still implemented and unit-tested arithmetically'),
            'eq6': {'deadline_filter': self.eq6_deadline_filter,
                    'fallback_serve': self.eq6_fallback_serve,
                    'checked': c['eq6_checked'],
                    'infeasible': c['eq6_infeasible'],
                    'rounds_all_infeasible': c['eq6_all_infeasible'],
                    'fallback_used': c['eq6_fallback_used'],
                    'network_legs_excluded': True,
                    'network_upload_ns': self.network_upload_ns,
                    'network_return_ns': self.network_return_ns},
            'counters': {
                'choose_action_calls': c['choose_action_calls'],
                'rounds': c['rounds'],
                'candidate_rounds': c['candidate_rounds'],
                'dispatch_returns': c['dispatch_returns'],
                'idle_returns': c['idle_returns'],
                'no_pending': c['no_pending'],
                'no_free_card': c['no_free_card'],
                'no_candidate': c['no_candidate'],
                'candidates_total': c['candidates_total'],
                'candidates_max': c['candidates_max'],
                'tau_fallback_uses': c['tau_fallback_uses'],
                'control_ticks': c['control_ticks'],
                'feedback_rows': c['feedback_rows'],
                'feedback_on_time': c['feedback_on_time'],
                'feedback_late': c['feedback_late']},
            'served_kind': dict(c['served_kind']),
            'served_role': dict(c['served_role']),
            'feedback_model': dict(c['feedback_model']),
            'epochs_seen': list(self._epochs_seen),
            'first_round': self._first_round,
            'last_round': self._last_round,
            'derived_note': (
                'S_a=sum over unserved i of f(w_i+L_a); candidate-specific '
                'service time remains in the decision even at batch=1. '
                'The maximum horizon in a round is diagnostic only.'),
            'feedback_note': ('completion feedback is counted for the audit '
                              'ledger only; the dispatch rule does not consume '
                              'it'),
        }


register('edge_serving', lambda config=None: EdgeServingAdapter(config))
