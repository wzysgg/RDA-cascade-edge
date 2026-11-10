"""T012 bounded candidate deadline replay worker (WS-D).

Authority
---------
* ``TASK.md`` 5.1 (bounded 2 s profile / overflow / one inflight + one latest
  replacement / two passes / cycle-phase mapping / receipt budget),
  5.2 (real ``select_job``+``dispatch_choice``, Prefix scenarios, Yhat/Vhat,
  per-class table, sensitivity), 5.3 (``epsilon_select``, bounded comparable
  history, no full-history scan).
* ``docs/T012_INTERFACE.md`` 6.5 (frozen API and return schema).
* ``REFERENCE.json`` (``algorithm``, ``performance``, ``CPU_expected``,
  ``burst_kernel_fixtures``).

Frozen public API (verbatim, interface 6.5)::

    REPLAY_SCHEMA, CANDIDATE_GRID, PROFILE_MAX_ROOTS, PROFILE_SPAN_NS, REPEATS,
    T_PROFILE_S, EPSILON_SELECT_FLOOR
    class ProfileRing: add(root); snapshot(now_ns)
    def replay_candidates(request, costs, gates, ipc, head_ns, select_job, dispatch_choice)
    class ReplayWorker: __init__(ctx=None, timeout_s=2.0); request(payload, now_ns);
                        poll(now_ns); stop(); stats()

Conventions that are easy to get wrong
--------------------------------------
* ``costs``/``gates`` are per card (index 0/1 = the two candidate card shares)
  and per kind, exactly like ``quota_forecast_v8.dispatch_choice``.  ``costs`` is
  the *mean* service used as the actor duration and the predicted charge;
  ``gates`` is the start budget used by the real kernel's quota test.
  ``gates=None`` falls back to ``costs`` (the frozen scalar-fixture behaviour).
* Start latency and return (IPC) latency extend card occupancy / output
  availability only.  The GPU charge stays the mean service on
  ``[actor_start, actor_end]`` and is split on 50 ms quota boundaries; a start
  or return delay never masquerades as GPU charge.
* On-time is ``terminal_available_ns <= deadline_abs_ns`` (inclusive).  The
  REFERENCE burst oracle pins this boundary: 25 directF roots of 2 ms finish at
  exactly the 50 ms deadline and the fixture counts 25 (total 29).  Admission
  still refuses when ``deadline - now <= 0`` (TASK 3.1).
* The counted pass is the last repeat (``repeat_index == repeats - 1``); the
  earlier passes only carry boundary commitments (busy cards, owed Prefix
  suffixes, Head backlog) into it.  Virtual ids append the repeat index so the
  two passes never collide, while the SHA256 scenario quantile uses the original
  ``root_id`` so both passes follow the same route.
* ``ProfileRing`` never silently drops to the easiest task: over
  ``PROFILE_MAX_ROOTS`` it sets ``overflow=True`` and the replay refuses to
  issue (``valid=False``, ``invalid_reason='profile_overflow'``).
* Expired source roots stay in the denominator (``by_class[...]['input']``) and
  are never executed: a root marked ``expired`` by the producer, a root whose
  mapped release is already past its deadline, and a root whose publish-time
  remaining budget is below every eligible card's start gate are all counted and
  not replayed (TASK 5.1 "cannot be rescued by a candidate").
* ``replay_candidates`` reads at most a bounded tail of the comparable-profile
  history (``history_index`` or ``history[-64:]``); it never iterates the full
  closed history (REFERENCE.performance ``history_scans_expected=0``).
"""

import hashlib
import heapq
import math
import multiprocessing
import os
import pickle
import sys
import time
import traceback
from resource_axis import CANDIDATE_GRID, CARD_COUNT, CARD_IDS, card_shares

# ---------------------------------------------------------------------------
# frozen constants (interface 6.5 / REFERENCE.algorithm)
# ---------------------------------------------------------------------------
REPLAY_SCHEMA = 'QF4R_replay_v3'
# Bounded capacity covers a 2 s window at 3x the 33-node source rate,
# including its 1.4x rate bins. Never treat a truncated window as complete.
PROFILE_MAX_ROOTS = 16384
PROFILE_SPAN_NS = 2_000_000_000
REPEATS = 2
T_PROFILE_S = 2.0
EPSILON_SELECT_FLOOR = 0.5      # 1/T_profile

PERIOD_NS = 50_000_000
CANDIDATE_NOISE_MAX_PROFILES = 16
CANDIDATE_NOISE_MIN_PROFILES = 8


def age_ns(sent_ns, now_ns):
    """Message age on the request/reply pair's OWN two clocks.

    The sender stamps ``sent_ns`` and the receiver stamps ``now_ns``; the age
    is their difference and nothing else.  No wall-clock reconciliation and no
    clamping: an unreadable pair is the caller's problem, not a silently
    repaired zero.
    """
    return int(now_ns) - int(sent_ns)

SCENARIOS = ('point', 'unknown_F', 'unknown_A', 'unknown_Abstain')
CLASSES = ('DF', 'DA', 'B')
# Optional verification switch: when True every dispatch also runs the caller's
# ``dispatch_choice`` and any disagreement with the per-card ``select_job`` scan
# is counted in ``dispatch_mismatch`` (0 == the real core agrees).  Off by
# default so the production path pays for one selection, not two.
CROSSCHECK_DISPATCH = False
_ENTRY_CLASS = {'directF': 'DF', 'directA': 'DA', 'Prefix-origin': 'B',
                'DF': 'DF', 'DA': 'DA', 'B': 'B'}

_KIND_FULL = 'FAST_FULL'
_KIND_PREFIX = 'FAST_PREFIX'
_KIND_SUFFIX = 'FAST_SUFFIX'
_KIND_ACCURATE = 'ACCURATE'
_KINDS = (_KIND_FULL, _KIND_PREFIX, _KIND_SUFFIX, _KIND_ACCURATE)
_KIND_CODE = {_KIND_FULL: 0, _KIND_ACCURATE: 1, _KIND_PREFIX: 2}
_KIND_NAME = (0, 1, 2)

# status codes (lists, not dicts: the kernel walks them per event)
_S_PENDING = 0
_S_RUNNING = 1
_S_HEAD = 2
_S_DONE = 3
_S_REJECTED = 4
_S_ABSTAIN = 5

# remaining-budget CDF grid, ms (uniform 0..50 ms, candidate deadline window)
CDF_GRID_MS = tuple(i * 2.5 for i in range(21))
NS_PER_S_CONST = 1_000_000_000

_REQUEST_KEYS = ('profile', 'grid', 'cycle_origin_ns', 'receipt_budget_ns',
                 'receipt_budget_source', 'delivery_budget_ns',
                 'delivery_budget_source', 'ipc_return_ns',
                 'scenario', 'price_version',
                 'snapshot_ns')

_REASON_KEYS = ('source_budget', 'estimated_budget_shortfall',
                'expired_at_release', 'expired_waiting',
                'expired_head', 'producer_expired')


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------
def _percentile(values, q):
    """REFERENCE.algorithm ``percentile_method``: sorted linear interpolation."""
    if not values:
        return None
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 1:
        return vals[0]
    pos = (n - 1) * float(q)
    lo = int(math.floor(pos))
    hi = lo + 1
    if hi > n - 1:
        hi = n - 1
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _sha_quantile(root_id, scenario_id):
    """SHA256(root_id|scenario_id) first 64 bits / 2**64 (TASK 5.2)."""
    raw = ('%s|%s' % (root_id, scenario_id)).encode('utf-8')
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], 'big') / float(1 << 64)


def _as_int(value, default=0):
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _card_share(gF):
    """TASK 5.2: card shares ``[min(gF,1), max(gF-1,0)]``, gA = 2 - gF."""
    gF = float(gF)
    return card_shares(gF)


def _normalise_per_card(value, default=None):
    """Return two per-card dicts keyed by kind.

    Accepts a flat ``{kind: ns}`` mapping (applied to both cards), a per-card
    list/tuple, or ``{0: {...}, 1: {...}}``.
    """
    if value is None:
        value = default
    if value is None:
        return [{} for _ in CARD_IDS]
    if isinstance(value, dict):
        if any(k in value for k in _KINDS):
            base = dict((k, value[k]) for k in _KINDS if k in value)
            return [dict(base) for _ in CARD_IDS]
        out = []
        for g in CARD_IDS:
            item = value.get(g)
            if item is None:
                item = value.get(str(g))
            out.append(dict(item) if isinstance(item, dict) else {})
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for g in CARD_IDS:
            if g < len(value) and isinstance(value[g], dict):
                out.append(dict(value[g]))
            elif value and isinstance(value[-1], dict):
                out.append(dict(value[-1]))
            else:
                out.append({})
        return out
    return [{} for _ in CARD_IDS]


def _fill_kinds(d0, d1):
    """Fill missing kinds from the sibling card / the card's own FAST_FULL."""
    for k in _KINDS:
        if k not in d0:
            d0[k] = d1.get(k, d0.get(_KIND_FULL, 1_000_000))
        if k not in d1:
            d1[k] = d0.get(k, d1.get(_KIND_FULL, 1_000_000))
    for d in (d0, d1):
        for k in _KINDS:
            try:
                d[k] = int(round(float(d[k])))
            except (TypeError, ValueError):
                d[k] = 1_000_000


def _normalise_pair(value, default=0):
    if value is None:
        return [int(default)] * CARD_COUNT
    if isinstance(value, (list, tuple)):
        if not value:
            return [int(default)] * CARD_COUNT
        return [_as_int(value[min(g, len(value)-1)], default) for g in CARD_IDS]
    v = _as_int(value, default)
    return [v] * CARD_COUNT


def _normalise_head(head_ns):
    """Return ``(service_ns, message_ns)`` for the single arm Head."""
    if head_ns is None:
        return (0, 0)
    if isinstance(head_ns, dict):
        svc = head_ns.get('service_ns')
        if svc is None:
            svc = head_ns.get('compute_ns')
        if svc is None:
            svc = head_ns.get('head_ns')
        if svc is None:
            svc = head_ns.get('ns')
        msg = head_ns.get('message_ns')
        if msg is None:
            msg = head_ns.get('msg_ns')
        return (max(0, _as_int(svc, 0)), max(0, _as_int(msg, 0)))
    if isinstance(head_ns, (list, tuple)):
        vals = [float(v) for v in head_ns if v is not None]
        if not vals:
            return (0, 0)
        return (int(_percentile(vals, 0.95) or 0), 0)
    return (max(0, _as_int(head_ns, 0)), 0)


# ---------------------------------------------------------------------------
# 1. ProfileRing  (TASK 5.1)
# ---------------------------------------------------------------------------
class ProfileRing(object):
    """Recent ``PROFILE_SPAN_NS`` of unique GPU source roots, at most
    ``PROFILE_MAX_ROOTS`` entries.

    ``add`` de-duplicates by ``root_id`` (a re-offered root updates its record),
    evicts records older than the span measured from the newest publish and
    flags ``overflow`` when the unique population exceeds the limit.  It never
    trims the sample towards the easiest tasks: the over-limit population is
    reported and the caller must disable issuing for the round.

    ``snapshot`` keeps expired roots in the denominator: they are returned with
    ``expired=True`` (producer flag, or a deadline that already passed at source
    publish, or a publish-time budget of zero) so the replay can count them
    without executing them.
    """

    def __init__(self, max_roots=PROFILE_MAX_ROOTS, span_ns=PROFILE_SPAN_NS):
        self.max_roots = int(max_roots)
        self.span_ns = int(span_ns)
        self._roots = {}
        self._order = []
        self._newest_publish_ns = None
        self.overflow = False
        self.overflow_dropped = 0
        self._last_dropped_publish_ns = None
        self.pruned = 0
        self.received = 0
        self.deduped = 0

    # -- ingest ------------------------------------------------------------
    def add(self, root):
        if not isinstance(root, dict):
            return
        rid = root.get('root_id')
        if rid is None:
            rid = root.get('source_seq')
        if rid is None:
            rid = root.get('task_id')
        if rid is None:
            return
        publish = _as_int(root.get('source_publish_ns'), 0)
        record = dict(root)
        record['root_id'] = rid
        # Normalize once at ingestion, rather than at every candidate request.
        record['source_seq'] = _as_int(root.get('source_seq'), 0)
        record['source_publish_ns'] = publish
        record['deadline_ns'] = _as_int(root.get('deadline_ns'), 0)
        if root.get('budget_ns') is not None:
            record['budget_ns'] = _as_int(root['budget_ns'], 0)
        record['expired'] = bool(root.get('expired', False) or
                                 (record['deadline_ns'] and record['deadline_ns'] <= publish) or
                                 (record.get('budget_ns') is not None and record['budget_ns'] <= 0))
        if rid in self._roots:
            self._roots[rid] = record
            self.deduped += 1
        else:
            self._roots[rid] = record
            self._order.append(rid)
            self.received += 1
        if self._newest_publish_ns is None or publish > self._newest_publish_ns:
            self._newest_publish_ns = publish
        self._evict()
        if len(self._roots) > self.max_roots:
            self._trim_oldest()

    def _evict(self):
        if self._newest_publish_ns is None:
            return
        floor = self._newest_publish_ns - self.span_ns
        if (self._last_dropped_publish_ns is not None
                and self._last_dropped_publish_ns < floor):
            self.overflow = False
        while self._order:
            rid = self._order[0]
            rec = self._roots.get(rid)
            if rec is None:
                self._order.pop(0)
                continue
            if _as_int(rec.get('source_publish_ns'), 0) < floor:
                self._order.pop(0)
                del self._roots[rid]
                self.pruned += 1
            else:
                break

    def _trim_oldest(self):
        self.overflow = True
        drop = []
        while len(self._roots) - len(drop) > self.max_roots and self._order:
            rid = self._order.pop(0)
            if rid in self._roots:
                drop.append(rid)
        for rid in drop:
            publish = self._roots[rid]['source_publish_ns']
            self._last_dropped_publish_ns = max(
                publish, self._last_dropped_publish_ns
                if self._last_dropped_publish_ns is not None else publish)
            del self._roots[rid]
        self.overflow_dropped += len(drop)

    # -- read --------------------------------------------------------------
    def snapshot(self, now_ns, include_diagnostics=True):
        """The latest COMPLETE fixed ``span_ns`` public source window.

        T014 s3.2.3: the span is fixed (2 s) and explicitly bounded as
        ``[window_start, window_end)``; the empty time between two roots and each
        release phase are preserved instead of squeezing a sparse profile to the
        first-to-last-publish distance.  ``coverage_s`` is the nominal window
        length and ``n_dropped_by_window`` counts ring records that fall outside
        it.
        """
        now_ns = _as_int(now_ns, 0)
        end = None if self._newest_publish_ns is None else int(self._newest_publish_ns) + 1
        start = None if end is None else end - int(self.span_ns)
        # Missing roots invalidate only windows which still contain them.
        self.overflow = bool(start is not None
            and self._last_dropped_publish_ns is not None
            and self._last_dropped_publish_ns >= start)
        records = sorted(self._roots.values(),
                         key=lambda r: (r['source_publish_ns'], r['source_seq']))
        # add() has normalized the immutable replay fields. Online requests
        # need just the window filter; age/deadline diagnostics stay opt-in.
        out = [r for r in records if start is None or r['source_publish_ns'] >= start]
        dropped = len(records) - len(out)
        if include_diagnostics:
            records = out
            out = []
        for rec in (records if include_diagnostics else ()):
            publish = rec['source_publish_ns']
            if start is not None and publish < start:
                continue
            deadline = rec['deadline_ns']
            budget = rec.get('budget_ns')
            expired = bool(rec.get('expired', False))
            if deadline and deadline <= publish:
                expired = True
            if budget is not None and budget <= 0:
                expired = True
            # add() replaces a record rather than mutating it.  The worker
            # request can therefore share these immutable dictionaries while
            # preserving all inputs used by the scheduling replay.  Per-root
            # age/deadline-expired are diagnostics, not replay inputs.
            item = rec
            if include_diagnostics:
                item = dict(rec)
                item['expired'] = expired
                item['deadline_expired'] = bool(deadline and deadline <= now_ns)
                item['age_ns'] = max(0, now_ns - publish)
            out.append(item)
        if start is None:
            start = 0
            end = 0
        return {'roots': out,
                'overflow': bool(self.overflow),
                'n': len(out),
                'span_ns': int(self.span_ns),
                'window_start_ns': int(start),
                'window_end_ns': int(end),
                'coverage_s': int(self.span_ns) / float(NS_PER_S_CONST),
                'first_publish_ns': int(start),
                'last_publish_ns': (int(out[-1]['source_publish_ns']) if out else None),
                'now_ns': int(now_ns),
                'pruned': int(self.pruned),
                'overflow_dropped': int(self.overflow_dropped),
                'last_dropped_publish_ns': self._last_dropped_publish_ns,
                'n_dropped_by_window': int(dropped),
                'n_received': int(self.received),
                'n_deduped': int(self.deduped)}

    # -- introspection (tests / diagnostics; not part of the frozen API) ---
    def __len__(self):
        return len(self._roots)

    def reset_overflow(self):
        # Diagnostics may be reset, but cannot certify missing active roots.
        end = self._newest_publish_ns
        self.overflow = bool(end is not None
            and self._last_dropped_publish_ns is not None
            and self._last_dropped_publish_ns >= end + 1 - self.span_ns)
        self.overflow_dropped = 0


# ---------------------------------------------------------------------------
# 2. route assignment per scenario (TASK 5.2)
# ---------------------------------------------------------------------------
def _revealed_mix(roots, override=None):
    """Public revealed F/A/Abstain proportions of Prefix-origin roots."""
    if override is not None:
        try:
            fast = float(override.get('Fast', override.get('F', 0.0)) or 0.0)
            acc = float(override.get('Accurate', override.get('A', 0.0)) or 0.0)
            abst = float(override.get('Abstain', override.get('Ab', 0.0)) or 0.0)
        except AttributeError:
            return None
        total = fast + acc + abst
        if total <= 0:
            return None
        return {'Fast': fast / total, 'Accurate': acc / total,
                'Abstain': abst / total, 'n': int(total)}
    counts = {'Fast': 0, 'Accurate': 0, 'Abstain': 0}
    total = 0
    for root in roots:
        if root['cls'] != 'B':
            continue
        role = root.get('revealed_role')
        if role in counts:
            counts[role] += 1
            total += 1
    if total <= 0:
        return None
    return {'Fast': counts['Fast'] / float(total),
            'Accurate': counts['Accurate'] / float(total),
            'Abstain': counts['Abstain'] / float(total),
            'n': total}


def _quality_means(roots):
    """Conditional public-quality mean per (entry class, revealed route)."""
    bucket = {}
    for root in roots:
        role = root.get('revealed_role')
        q = root.get('public_quality')
        if role is None or q is None:
            continue
        key = (root['cls'], role)
        acc = bucket.setdefault(key, [0.0, 0])
        acc[0] += float(q)
        acc[1] += 1
    return dict((k, v[0] / v[1]) for k, v in bucket.items() if v[1] > 0)


def _assign_scenario(roots, scenario, mix, quality_means, mix_override=None):
    """Return ``(routes, qualities, info)`` for one scenario.

    ``routes[i]``/``qualities[i]`` are None only when the scenario cannot be
    built (``point`` without a single revealed Prefix sample).
    """
    routes = [None] * len(roots)
    qualities = [None] * len(roots)
    unknown = 0
    revealed = 0
    if scenario == 'point' and mix is None:
        # The point scenario needs the revealed F/A/Abstain mix only to route
        # roots whose route is *unknown*.  A profile without a single
        # Prefix-origin root is therefore fully determined (every route is
        # already revealed) and must still be replayed.
        needs_mix = False
        for root in roots:
            if root.get('revealed_role') is None:
                needs_mix = True
                break
        if needs_mix:
            mix = _revealed_mix(roots, override=mix_override)
            if mix is None:
                return None, None, {'valid': False,
                                    'reason': 'point_no_revealed_samples',
                                    'unknown': 0, 'revealed': 0, 'mix': None}
    for i, root in enumerate(roots):
        role = root.get('revealed_role')
        if role is not None:
            routes[i] = role
            qualities[i] = root.get('public_quality')
            revealed += 1
            continue
        unknown += 1
        if scenario == 'point':
            u = _sha_quantile(root['root_id'], 'point')
            if u < mix['Fast']:
                role = 'Fast'
            elif u < mix['Fast'] + mix['Accurate']:
                role = 'Accurate'
            else:
                role = 'Abstain'
        elif scenario == 'unknown_F':
            role = 'Fast'
        elif scenario == 'unknown_A':
            role = 'Accurate'
        else:
            role = 'Abstain'
        routes[i] = role
        if root['cls'] != 'B':
            # direct roots are never unknown; defensive fallback only
            qualities[i] = root.get('public_quality')
        else:
            qualities[i] = quality_means.get((root['cls'], role))
    info = {'valid': True, 'reason': None, 'unknown': unknown,
            'revealed': revealed, 'mix': mix}
    return routes, qualities, info


# ---------------------------------------------------------------------------
# 3. replay kernel (real select_job / dispatch_choice)
# ---------------------------------------------------------------------------
def _run_kernel(select_job, dispatch_choice, tasks, n, release, deadline, kind_code,
                route, cls, quality,
                disabled, budget, repeat, targets, costs, gates, ipc, start_ns,
                head_svc, head_msg, counted_repeat, window_start, cdf_grid,
                resource_accounting=None, service_debt_balance=None,
                allow_idle_endpoint_loans=False, terminal_return_ns=0, waiting_capacity=None):
    """One pass-pair replay of the counted profile for one candidate.

    Returns the per-candidate counters; the caller maps them into the frozen
    schema.  The card state dicts are exactly the ones
    ``quota_forecast_v8.select_job``/``dispatch_choice`` consume.
    """
    cards = []
    for g in CARD_IDS:
        share = float(targets[g])
        cards.append({'busy': False, 'job': None, 'pinned': [], 'pending': set(),
                      'allow_idle_endpoint_loans': bool(allow_idle_endpoint_loans),
                      'target': share, 'effective': share, 'transition': False,
                      'cycle_charges': {}, 'spent': {'Fast': 0, 'Accurate': 0},
                      'spent_cyc': -1})
    service_debt = resource_accounting == 'service_debt'
    if service_debt:
        from quota_forecast_v8 import enable_service_debt, debt_dispatch, debt_complete
        for g, s in enumerate(cards):
            balances = service_debt_balance or []
            enable_service_debt(s, balances[g] if len(balances) > g else None)

    status = [_S_PENDING] * n
    job_of = [None] * n
    card_of = [-1] * n
    arrival = [int(tasks[i]['arrival_ns']) for i in range(n)]
    fast = []
    acc = []
    heap = []
    serial = [0]
    head_q = []
    # Expiry ends the root, but a queued/running CPU decision still consumes
    # its FIFO service and retains the Prefix activation until its reply.
    head_deferred = set()
    head_ptr = 0
    head_free_at = -1
    # Optional experimental policy lifecycle; absent for the production kernel.
    owner = getattr(select_job, '__self__', None)
    hooks = owner if getattr(owner, 'replay_hooks', False) else None
    collect_metrics = hooks is not None or service_debt
    latency_ms = []
    ontime_latency_ms = []
    wait_ms = []
    service_ns = [{ 'Fast': 0, 'Accurate': 0 } for _ in cards]
    simulated_duration = {}

    # Publish-time remaining budget (T014 s3.2.2): the DEADLINE path needs
    # mean service + dispatch->actor-start + the necessary follow-ups.  The gate
    # is a START/quota condition and must never be used as the execution
    # duration, which used to delete feasible roots (F mean 2 ms, gate 8 ms,
    # 6 ms remaining).  A root below this estimate is labelled
    # ``estimated_budget_shortfall`` and is never "rescued" by a candidate.
    min_needed = [None, None, None]
    for g in CARD_IDS:
        share = float(targets[g])
        if share > 0 or (service_debt and allow_idle_endpoint_loans):
            need = costs[g][_KIND_FULL] + start_ns[g] + ipc[g]
            if min_needed[0] is None or need < min_needed[0]:
                min_needed[0] = need
            need = (costs[g][_KIND_PREFIX] + costs[g][_KIND_SUFFIX]
                    + start_ns[g] + ipc[g])
            if min_needed[2] is None or need < min_needed[2]:
                min_needed[2] = need
        if share < 1 or (service_debt and allow_idle_endpoint_loans):
            need = costs[g][_KIND_ACCURATE] + start_ns[g] + ipc[g]
            if min_needed[1] is None or need < min_needed[1]:
                min_needed[1] = need
    min_needed = [v + terminal_return_ns if v is not None else None for v in min_needed]

    by_class = {}
    for name in CLASSES:
        by_class[name] = {'input': 0, 'completed': 0, 'late': 0,
                          'rejected': 0, 'abstain': 0, 'ontime': 0}
    # per-tid views of the class counter dict and of the counted-pass flag: the
    # hot paths do one list index instead of a closure call plus two dict looks
    bc_of = [by_class[cls[i]] for i in range(n)]
    is_counted = [(repeat[i] == counted_repeat) for i in range(n)]
    # monotone version of every input ``select_job`` reads from a card and from
    # the shared ready queues; a card whose version is unchanged since it last
    # answered "nothing eligible" cannot have become eligible (see dispatch()).
    reasons = dict((k, 0) for k in _REASON_KEYS)
    reasons['queue_capacity'] = 0
    ontime_role = {'Fast': 0, 'Accurate': 0}
    role_outcomes = {r: {'completed': 0, 'ontime': 0, 'quality': 0.,
                        'quality_earliness': 0., 'all_quality': 0.,
                        'quality_ok': True, 'all_quality_ok': True}
                     for r in ('Fast', 'Accurate')}
    hist = [0] * (len(cdf_grid) + 1)
    state = {'ontime': 0, 'quality': 0.0, 'quality_earliness': 0.0,
             'quality_ok': True, 'gate_starts': 0,
             'started': 0, 'n_cdf': 0, 'executed': 0, 'dispatch_mismatch': 0}
    from resource_objective import summarize_role, timely_utility
    resource_inputs = {'Fast': 0, 'Accurate': 0}
    resource_utility = {'Fast': 0.0, 'Accurate': 0.0}
    for tid in range(n):
        if is_counted[tid]:
            # A Prefix retains its public fixed route in each scenario.
            # A known abstention has no final output under either allocation;
            # keep it as a zero-utility Fast-origin input, never drop failures.
            assigned = 'Accurate' if kind_code[tid] == 1 or (
                kind_code[tid] == 2 and route[tid] == 'Accurate') else 'Fast'
            resource_inputs[assigned] += 1

    # counted-pass denominator (kept even when a root never executes)
    for i in range(n):
        if is_counted[i]:
            bc_of[i]['input'] += 1

    def count(tid, key):
        if is_counted[tid]:
            bc_of[tid][key] += 1

    def pend_release(tid):
        for s in cards:
            s['pending'].discard(tid)
            if hooks is not None:
                hooks.release_future(s, tid)

    def continuation_enqueue(tid, job, queue, at):
        if waiting_capacity is not None:
            waiting = {j['task_id'] for j in fast+acc if deadline[j['task_id']] > at}
            waiting.update(j['task_id'] for s in cards for j in s['pinned'] if deadline[j['task_id']] > at)
            if tid not in waiting and len(waiting) >= waiting_capacity:
                pend_release(tid)
                status[tid] = _S_REJECTED
                count(tid, 'rejected')
                if is_counted[tid]:
                    reasons['queue_capacity'] += 1
                return False
        queue.append(job)
        status[tid] = _S_PENDING
        return True

    def finish(tid, actor_end, role):
        g = card_of[tid]
        avail = actor_end + ipc[g] + terminal_return_ns
        status[tid] = _S_DONE
        if is_counted[tid]:
            cnt = bc_of[tid]
            cnt['completed'] += 1
            role_result = role_outcomes[role]
            role_result['completed'] += 1
            q = quality[tid]
            if q is None:
                role_result['all_quality_ok'] = False
            else:
                role_result['all_quality'] += float(q)
            if collect_metrics:
                latency_ms.append((avail - arrival[tid]) / 1e6)
            if avail <= deadline[tid]:
                resource_utility[role] += timely_utility(arrival[tid], deadline[tid], avail)
                if collect_metrics:
                    ontime_latency_ms.append((avail - arrival[tid]) / 1e6)
                cnt['ontime'] += 1
                # only the counted pass feeds Yhat/Vhat; pass 0 exists to carry
                # the boundary promises of the previous window (REPEATS=2)
                state['ontime'] += 1
                ontime_role[role] += 1
                role_result['ontime'] += 1
                q = quality[tid]
                if q is None:
                    state['quality_ok'] = False
                    role_result['quality_ok'] = False
                else:
                    state['quality'] += float(q)
                    role_result['quality'] += float(q)
                    # Area under expected-correct delivery count up to the
                    # original deadline. Failed/late inputs contribute zero;
                    # removing slow inputs cannot improve this total score.
                    fraction = max(0.0, min(1.0, (deadline[tid]-avail) /
                        float(max(1, deadline[tid]-arrival[tid]))))
                    state['quality_earliness'] += float(q) * fraction
                    role_result['quality_earliness'] += float(q) * fraction
            else:
                cnt['late'] += 1

    def start_heads(t):
        nonlocal head_free_at, head_ptr
        while head_q and head_free_at <= t:
            tid = -1
            while head_ptr < len(head_q):
                cand = head_q[head_ptr]
                head_ptr += 1
                if status[cand] == _S_HEAD or cand in head_deferred:
                    tid = cand
                    break
            if tid < 0:
                break
            finish_at = t + head_svc
            head_free_at = finish_at
            serial[0] += 1
            heapq.heappush(heap, (finish_at, serial[0], 2, tid))

    def dispatch(t):
        """Start jobs on idle cards with the real ``dispatch_choice`` rule.

        ``work = costs[g][kind] + ipc[g]`` (a ``FAST_PREFIX`` also owes its
        ``FAST_SUFFIX``) and the cross-card argmin on
        ``(work, tasks[tid]['arrival_ns'], g)`` reproduce
        ``quota_forecast_v8.dispatch_choice`` exactly, driven by the *caller's*
        ``select_job`` -- the quota / FIFO / pinned / owed-suffix rules live in
        ``select_job`` and are never re-derived here.  ``CROSSCHECK_DISPATCH``
        additionally runs the caller's real ``dispatch_choice`` every step and
        records any disagreement (and follows it).
        """
        if all(s['busy'] for s in cards):
            return
        if not fast and not acc and not any(s['pinned'] for s in cards):
            return
        cyc = t // PERIOD_NS
        for g in CARD_IDS:
            s = cards[g]
            if not s['busy'] and s['spent_cyc'] != cyc:
                cc = s['cycle_charges'].get(cyc)
                if cc is None:
                    cc = {'Fast': 0, 'Accurate': 0}
                    s['cycle_charges'][cyc] = cc
                s['spent'] = cc
                s['spent_cyc'] = cyc
        for _ in range(8192):
            if hooks is not None:
                hooks.before_dispatch(cards, fast, acc, tasks, costs, t)
            best = None
            best_key = None
            best_g = 0
            best_role = None
            for g in CARD_IDS:
                s = cards[g]
                if s['busy']:
                    continue
                if service_debt and hooks is None:
                    j, role = select_job(s, fast, acc, tasks, costs[g], gates[g], now=t,
                                        ipc=ipc[g])
                else:
                    j, role = select_job(s, fast, acc, tasks, costs[g], gates[g])
                if j is None:
                    continue
                work = costs[g][j['kind']] + ipc[g]
                if j['kind'] == _KIND_PREFIX:
                    work += costs[g][_KIND_SUFFIX] + ipc[g]
                arr = arrival[j['task_id']]
                if best_key is None:
                    best = j
                    best_key = (work, arr, g)
                    best_g = g
                    best_role = role
                else:
                    bk = best_key
                    if work < bk[0] or (work == bk[0] and (arr < bk[1]
                                                           or (arr == bk[1]
                                                               and g < bk[2]))):
                        best = j
                        best_key = (work, arr, g)
                        best_g = g
                        best_role = role
            if CROSSCHECK_DISPATCH:
                if service_debt and hooks is None:
                    g2, j2, role2 = dispatch_choice(cards, fast, acc, tasks,
                                                    costs, ipc, gates, now=t)
                else:
                    g2, j2, role2 = dispatch_choice(cards, fast, acc, tasks,
                                                    costs, ipc, gates)
                if (best is None) != (j2 is None) or (
                        best is not None and (g2 != best_g or j2 is not best
                                              or role2 != best_role)):
                    state['dispatch_mismatch'] += 1
                    best, best_g, best_role = j2, g2, role2
            if best is None:
                return
            g = best_g
            j = best
            role = best_role
            s = cards[g]
            if hooks is not None:
                hooks.on_dispatch(s, j, role, costs[g], fast, acc)
            if service_debt:
                debt_dispatch(s, j, role, costs[g], fast, acc)
            if j['kind'] == _KIND_SUFFIX:
                try:
                    s['pinned'].remove(j)
                except ValueError:
                    pass
            elif role == 'Fast':
                try:
                    fast.remove(j)
                except ValueError:
                    pass
            else:
                try:
                    acc.remove(j)
                except ValueError:
                    pass
            tid = j['task_id']
            card_of[tid] = g
            s['busy'] = True
            s['job'] = j
            if j['kind'] == _KIND_PREFIX:
                s['pending'].add(tid)
            actor_start = t + start_ns[g]
            dur = int(costs[g][j['kind']])
            if hooks is not None:
                dur = hooks.service_duration(g, j, dur, t)
            if dur <= 0:
                dur = 1
            if collect_metrics:
                simulated_duration[(g, tid, j['kind'])] = dur
            actor_end = actor_start + dur
            a = actor_start
            while a < actor_end:
                c = a // PERIOD_NS
                e = (c + 1) * PERIOD_NS
                if e > actor_end:
                    e = actor_end
                cc = s['cycle_charges'].get(c)
                if cc is None:
                    cc = {'Fast': 0, 'Accurate': 0}
                    s['cycle_charges'][c] = cc
                cc[role] += e - a
                a = e
            status[tid] = _S_RUNNING
            if repeat[tid] == counted_repeat:
                if collect_metrics:
                    service_ns[g][role] += dur
                    wait_ms.append((t - j['enqueue_ns']) / 1e6)
                state['gate_starts'] += 1
                state['started'] += 1
                rem_ms = (deadline[tid] - actor_start) / 1e6
                idx = 0
                while idx < len(cdf_grid) and cdf_grid[idx] < rem_ms:
                    idx += 1
                hist[idx] += 1
                state['n_cdf'] += 1
            serial[0] += 1
            # The parent cannot release this card or publish a Head until the
            # actor reply arrives. IPC occupies the dispatch calendar only;
            # simulated_duration still charges the GPU's actual service once.
            heapq.heappush(heap, (actor_end + ipc[g], serial[0], 0, (g, j)))

    rel_order = sorted(range(n), key=lambda i: release[i])
    # deadline drive is a lazy heap: a root is pushed when it is accepted at
    # release, and terminal roots are dropped from the top without ever being
    # allowed to drive the clock (a spent deadline must not cost an event).
    dl_heap = []
    horizon = 0
    for i in range(n):
        if deadline[i] > horizon:
            horizon = deadline[i]
        if release[i] > horizon:
            horizon = release[i]
    horizon += head_svc + head_msg + PERIOD_NS

    t = window_start
    rp = 0
    nrel = len(rel_order)
    dirty = True
    cur_cyc = t // PERIOD_NS
    next_b = (cur_cyc + 1) * PERIOD_NS
    for _step in range(4_000_000):
        while heap and heap[0][0] <= t:
            dirty = True
            at, _seq, typ, payload = heapq.heappop(heap)
            if typ == 0:
                g, j = payload
                s = cards[g]
                s['job'] = None
                s['busy'] = False
                tid = j['task_id']
                kind = j['kind']
                if collect_metrics:
                    duration = simulated_duration.pop((g, tid, kind))
                if hooks is not None:
                    hooks.on_complete(s, j, duration)
                if service_debt:
                    debt_complete(s, j, duration)
                if kind == _KIND_PREFIX:
                    status[tid] = _S_HEAD
                    head_q.append(tid)
                else:
                    if kind == _KIND_SUFFIX:
                        s['pending'].discard(tid)
                        finish(tid, at - ipc[g], 'Fast')
                    elif kind == _KIND_ACCURATE:
                        finish(tid, at - ipc[g], 'Accurate')
                    else:
                        finish(tid, at - ipc[g], 'Fast')
            elif typ == 2:
                # The Head CPU can start its next queued job while this
                # decision's message travels to the parent.
                serial[0] += 1
                heapq.heappush(heap, (at + head_msg, serial[0], 3, payload))
            else:
                tid = payload
                if status[tid] != _S_HEAD:
                    pend_release(tid)
                    head_deferred.discard(tid)
                    continue
                if deadline[tid] <= at:
                    pend_release(tid)
                    status[tid] = _S_ABSTAIN
                    count(tid, 'abstain')
                    if repeat[tid] == counted_repeat:
                        reasons['expired_head'] += 1
                else:
                    role = route[tid]
                    enq = at
                    g = card_of[tid]
                    if role == 'Fast':
                        s = cards[g]
                        s['pending'].discard(tid)
                        j = {'task_id': tid, 'kind': _KIND_SUFFIX,
                             'enqueue_ns': enq, 'affinity': g,
                             'deadline_ns': deadline[tid]}
                        job_of[tid] = j
                        continuation_enqueue(tid, j, s['pinned'], at)
                    elif role == 'Accurate':
                        pend_release(tid)
                        j = {'task_id': tid, 'kind': _KIND_ACCURATE,
                             'enqueue_ns': enq, 'affinity': None,
                             'deadline_ns': deadline[tid]}
                        job_of[tid] = j
                        continuation_enqueue(tid, j, acc, at)
                    else:
                        pend_release(tid)
                        status[tid] = _S_ABSTAIN
                        count(tid, 'abstain')

        while rp < nrel and release[rel_order[rp]] <= t:
            tid = rel_order[rp]
            rp += 1
            dirty = True
            counted = is_counted[tid]
            if disabled[tid]:
                status[tid] = _S_REJECTED
                count(tid, 'rejected')
                if counted:
                    reasons['producer_expired'] += 1
                continue
            if deadline[tid] <= release[tid]:
                status[tid] = _S_REJECTED
                count(tid, 'rejected')
                if counted:
                    reasons['expired_at_release'] += 1
                continue
            kc = kind_code[tid]
            if waiting_capacity is not None:
                waiting = {j['task_id'] for j in fast+acc if deadline[j['task_id']] > t}
                waiting.update(j['task_id'] for s in cards for j in s['pinned'] if deadline[j['task_id']] > t)
                if len(waiting) >= waiting_capacity:
                    status[tid] = _S_REJECTED
                    count(tid, 'rejected')
                    if counted:
                        reasons['queue_capacity'] += 1
                    continue
            need = min_needed[kc]
            if budget[tid] is not None and need is not None \
                    and budget[tid] < need:
                status[tid] = _S_REJECTED
                count(tid, 'rejected')
                if counted:
                    reasons['estimated_budget_shortfall'] += 1
                    reasons['source_budget'] += 1
                continue
            if kc == 0:
                jkind = _KIND_FULL
                queue = fast
                aff = None
            elif kc == 1:
                jkind = _KIND_ACCURATE
                queue = acc
                aff = None
            else:
                jkind = _KIND_PREFIX
                queue = fast
                aff = None
            j = {'task_id': tid, 'kind': jkind, 'enqueue_ns': release[tid],
                 'affinity': aff, 'deadline_ns': deadline[tid]}
            job_of[tid] = j
            queue.append(j)
            status[tid] = _S_PENDING
            serial[0] += 1
            heapq.heappush(dl_heap, (deadline[tid], serial[0], tid))

        while dl_heap and dl_heap[0][0] <= t:
            _dl_at, _dl_seq, tid = heapq.heappop(dl_heap)
            st = status[tid]
            if st == _S_DONE or st == _S_REJECTED or st == _S_ABSTAIN:
                continue
            dirty = True
            if st == _S_PENDING:
                j = job_of[tid]
                if j is not None:
                    if j['kind'] == _KIND_SUFFIX:
                        try:
                            cards[j['affinity']]['pinned'].remove(j)
                        except (ValueError, KeyError, TypeError):
                            pass
                    else:
                        try:
                            fast.remove(j)
                        except ValueError:
                            try:
                                acc.remove(j)
                            except ValueError:
                                pass
                pend_release(tid)
                status[tid] = _S_REJECTED
                count(tid, 'rejected')
                if is_counted[tid]:
                    reasons['expired_waiting'] += 1
            elif st == _S_HEAD:
                head_deferred.add(tid)
                status[tid] = _S_ABSTAIN
                count(tid, 'abstain')
                if is_counted[tid]:
                    reasons['expired_head'] += 1

        if head_ptr < len(head_q) and head_free_at <= t:
            start_heads(t)
        if dirty:
            dispatch(t)
            dirty = False

        # a root that already reached a terminal state must not drive the clock
        while dl_heap:
            _dl_tid = dl_heap[0][2]
            if status[_dl_tid] == _S_DONE or status[_dl_tid] == _S_REJECTED \
                    or status[_dl_tid] == _S_ABSTAIN:
                heapq.heappop(dl_heap)
                continue
            break

        nxt = heap[0][0] if heap else horizon + 1
        if rp < nrel:
            nxt = min(nxt, release[rel_order[rp]])
        if dl_heap:
            nxt = min(nxt, dl_heap[0][0])
        if next_b < nxt:
            nxt = next_b
        if nxt <= t:
            nxt = t + 1
        if nxt > horizon:
            break
        t = nxt
        if t >= next_b:
            cur_cyc = t // PERIOD_NS
            next_b = (cur_cyc + 1) * PERIOD_NS
            dirty = True

    cdf = []
    denom = state['n_cdf']
    running = 0
    cum = []
    for v in hist:
        running += v
        cum.append(running)
    for i in range(len(cdf_grid)):
        cdf.append((cum[i] / float(denom)) if denom else 0.0)
    state['executed'] = state['started']
    result = {'ontime': state['ontime'],
            'resource_outcomes': {r: summarize_role(resource_inputs[r],
                ontime_role[r], resource_utility[r]) for r in resource_inputs},
            'quality': state['quality'],
            'quality_earliness': state['quality_earliness'],
            'quality_ok': state['quality_ok'],
            'by_class': by_class,
            'reasons': reasons,
            'ontime_role': ontime_role,
            'role_outcomes': role_outcomes,
            'gate_starts': state['gate_starts'],
            'started': state['started'],
            'dispatch_mismatch': state['dispatch_mismatch'],
            'remaining_budget_cdf': cdf,
            'remaining_budget_cdf_n': denom}
    if collect_metrics:
        policy_metrics = hooks.diagnostics(cards) if hooks is not None else []
        if service_debt and not policy_metrics:
            policy_metrics = []
            for s in cards:
                d = s['_service_debt']
                policy_metrics.append({
                    'balance': dict(d['balance']), 'earned': dict(d['earned']),
                    'used': dict(d['used']), 'idle_used': dict(d['idle_used']),
                    'inflight_left': len(d['inflight']),
                    'reservations_left': len(s['pending']),
                    'conservation_error_ns': {
                        r: d['origin'][r]+d['earned'][r]-d['used'][r]-d['balance'][r]
                        for r in ('Fast', 'Accurate')}})
        result['validation_metrics'] = {
                'completed_latency_mean_ms': (sum(latency_ms) / len(latency_ms)) if latency_ms else None,
                'completed_latency_p95_ms': _percentile(latency_ms, .95),
                'ontime_latency_mean_ms': (sum(ontime_latency_ms) / len(ontime_latency_ms)) if ontime_latency_ms else None,
                'ontime_latency_p95_ms': _percentile(ontime_latency_ms, .95),
                'stage_wait_mean_ms': (sum(wait_ms) / len(wait_ms)) if wait_ms else None,
                'stage_wait_p95_ms': _percentile(wait_ms, .95),
                'service_ms_by_card': [{r: v / 1e6 for r, v in row.items()} for row in service_ns],
                'policy': policy_metrics}
    return result


# ---------------------------------------------------------------------------
# 4. epsilon_select / selection (TASK 5.3, REFERENCE.algorithm)
# ---------------------------------------------------------------------------
def _history_records(request):
    """At most ``CANDIDATE_NOISE_MAX_PROFILES`` comparable records.

    Reads an index (``history_index``) or a bounded tail of the ring
    (``history``); the full closed history is never iterated.
    """
    epoch = request.get('epoch', request.get('effective_epoch'))
    scenario = request.get('scenario')
    index = request.get('history_index')
    pool = None
    if isinstance(index, dict) and epoch is not None:
        for key in (epoch, str(epoch)):
            if key in index:
                pool = index[key]
                break
    if pool is None:
        bounded = request.get('history_tail')
        if bounded is None:
            bounded = request.get('history')
        if bounded is None:
            return []
        try:
            size = len(bounded)
        except TypeError:
            return []
        start = size - (CANDIDATE_NOISE_MAX_PROFILES * 4)
        if start < 0:
            start = 0
        try:
            pool = bounded[start:size]
        except TypeError:
            return []
    out = []
    if pool is None:
        return out
    for rec in pool:
        if not isinstance(rec, dict):
            continue
        if epoch is not None and rec.get('epoch') != epoch:
            continue
        if scenario is not None and rec.get('scenario') not in (None, scenario):
            continue
        out.append(rec)
    if len(out) > CANDIDATE_NOISE_MAX_PROFILES:
        out = out[-CANDIDATE_NOISE_MAX_PROFILES:]
    return out


def _series(record, key, grid):
    value = record.get(key)
    if isinstance(value, dict):
        series = []
        for g in grid:
            item = value.get(g)
            if item is None:
                item = value.get(repr(float(g)))
            if item is None:
                for k, v in value.items():
                    try:
                        if abs(float(k) - float(g)) < 1e-9:
                            item = v
                            break
                    except (TypeError, ValueError):
                        continue
            series.append(item)
        return series
    if isinstance(value, (list, tuple)) and len(value) == len(grid):
        return list(value)
    return None


def _epsilon(grid, yhat, vhat, ref_idx, records):
    """Per-candidate ``(epsilon_Y, epsilon_V, n_comparable, low_confidence)``."""
    n = len(grid)
    eps_y = [EPSILON_SELECT_FLOOR] * n
    eps_v = [EPSILON_SELECT_FLOOR] * n
    if ref_idx is None:
        return eps_y, eps_v, 0, True
    base_y = yhat[ref_idx]
    base_v = vhat[ref_idx] if ref_idx is not None else None
    if base_y is None:
        return eps_y, eps_v, 0, True
    used = 0
    drift_y = [[] for _ in range(n)]
    drift_v = [[] for _ in range(n)]
    for rec in records:
        ry = _series(rec, 'Yhat', grid)
        if ry is None:
            continue
        ref_rec = rec.get('reference_gF')
        ridx = None
        if ref_rec is not None:
            for i, g in enumerate(grid):
                if abs(float(g) - float(ref_rec)) < 1e-9:
                    ridx = i
                    break
        if ridx is None or ry[ridx] is None:
            continue
        rv = _series(rec, 'Vhat', grid)
        used += 1
        for i in range(n):
            if ry[i] is None or yhat[i] is None:
                continue
            gain_cur = yhat[i] - base_y
            gain_hist = ry[i] - ry[ridx]
            drift_y[i].append(abs(gain_cur - gain_hist))
            if rv is not None and vhat[i] is not None and base_v is not None \
                    and rv[i] is not None and rv[ridx] is not None:
                drift_v[i].append(abs((vhat[i] - base_v) - (rv[i] - rv[ridx])))
    for i in range(n):
        py = _percentile(drift_y[i], 0.95) if drift_y[i] else None
        pv = _percentile(drift_v[i], 0.95) if drift_v[i] else None
        eps_y[i] = max(EPSILON_SELECT_FLOOR, float(py)) if py is not None else EPSILON_SELECT_FLOOR
        eps_v[i] = max(EPSILON_SELECT_FLOOR, float(pv)) if pv is not None else EPSILON_SELECT_FLOOR
    low_confidence = used < CANDIDATE_NOISE_MIN_PROFILES
    return eps_y, eps_v, used, low_confidence


def quality_gain_signature(roots):
    """Public source mix and deadline opportunity; no labels or predictions."""
    classes = {}
    budgets = [0] * 11
    for r in roots:
        cls = str(r.get('entry_class') or r.get('kind'))
        classes[cls] = classes.get(cls, 0) + 1
        b = max(0, _as_int(r.get('budget_ns'), 0))
        budgets[min(10, b // 5_000_000)] += 1
    return {'n': len(roots), 'classes': classes, 'budgets': budgets}


def quality_gain_epsilon(grid, gains, floor, history, epoch, current,
                         snapshot_ns, signature):
    """Bounded same-config gain drift, kept separate from timing resolution.

    Compare only public workload shapes within the existing 20% rate / 0.1 TV
    limits. A regime change or ratio change cannot carry old risk evidence
    into the new demand. This is an empirical guard, not a confidence proof.
    """
    eps, counts = [float(floor)] * len(grid), [0] * len(grid)
    drift = [[] for _ in grid]
    def tv(a, b):
        sa, sb = sum(a.values()), sum(b.values())
        if not sa or not sb:return float('inf')
        return .5 * sum(abs(a.get(k, 0)/sa-b.get(k, 0)/sb) for k in set(a)|set(b))
    for rec in list(history or [])[-CANDIDATE_NOISE_MAX_PROFILES:]:
        if rec.get('epoch') != epoch or abs(float(rec.get('reference_gF', -99))-current)>1e-9:
            continue
        stamp = rec.get('snapshot_ns')
        if stamp is None or not 0 < snapshot_ns-int(stamp) <= 8_000_000_000:
            continue
        prior = rec.get('signature') or {}
        na, nb = signature.get('n', 0), prior.get('n', 0)
        if not na or not nb or abs(na-nb)/max(na, nb)>.2:
            continue
        if tv(signature.get('classes', {}), prior.get('classes', {}))>.1:
            continue
        if tv(dict(enumerate(signature.get('budgets', []))),
              dict(enumerate(prior.get('budgets', []))))>.1:
            continue
        old = _series(rec, 'H_gain', grid)
        if old is None:continue
        for i, gain in enumerate(gains):
            if gain is not None and old[i] is not None:
                drift[i].append(abs(float(gain)-float(old[i])))
    for i, values in enumerate(drift):
        counts[i] = len(values)
        if values:eps[i] = max(eps[i], _percentile(values, .95))
    return eps, counts


def _select(grid, yhat, eps_y, eps_v, vhat, ref_idx, current, direction):
    """Highest Yhat, ties within epsY by Vhat, then smallest move, then gF asc."""
    best = None
    best_move = None
    base_y = yhat[ref_idx] if ref_idx is not None else 0.0
    base_v = vhat[ref_idx] if ref_idx is not None else None
    for i, g in enumerate(grid):
        if ref_idx is not None and i == ref_idx:
            continue
        if yhat[i] is None:
            continue
        if direction == 'Fast' and not (g > current + 1e-9):
            continue
        if direction == 'Accurate' and not (g < current - 1e-9):
            continue
        gy = yhat[i] - base_y
        gy_eps = eps_y[i]
        if gy < -gy_eps:
            continue
        v_ok = None
        if vhat[i] is not None and base_v is not None:
            v_ok = vhat[i] - base_v
        y_branch = gy >= gy_eps and (v_ok is None or v_ok >= -eps_v[i])
        v_branch = (v_ok is not None and v_ok >= eps_v[i] and gy >= -gy_eps)
        if not (y_branch or v_branch):
            continue
        if gy <= 0 and (v_ok is None or v_ok <= 0):
            continue
        move = abs(g - current)
        if best is None:
            best, best_move = i, move
            continue
        best_y = yhat[best] - base_y
        tie_eps = max(gy_eps, eps_y[best])
        if gy > best_y + tie_eps:
            best, best_move = i, move
            continue
        if abs(yhat[i] - yhat[best]) <= tie_eps:
            vi = vhat[i]
            vb = vhat[best]
            if vi is not None and (vb is None or vi > vb):
                best, best_move = i, move
            elif vi == vb:
                if move < best_move:
                    best, best_move = i, move
                elif abs(move - best_move) < 1e-12 and g < grid[best]:
                    best, best_move = i, move
    return best


# ---------------------------------------------------------------------------
# 5. replay_candidates  (frozen signature, interface 6.5)
# ---------------------------------------------------------------------------
def _profile_sha(roots, profile):
    """Stable public-profile fingerprint (root id + time + class + route + q)."""
    import hashlib
    h = hashlib.sha256()
    for r in roots:
        h.update(('|'.join(str(x) for x in (
            r.get('root_id'), r.get('source_publish_ns'), r.get('deadline_ns'),
            r.get('cls'), r.get('revealed_role'), r.get('public_quality'))))
            .encode('utf8'))
        h.update(b'\n')
    h.update(('span=%s' % (profile or {}).get('span_ns')).encode('utf8'))
    h.update(('start=%s' % (profile or {}).get('window_start_ns')).encode('utf8'))
    return h.hexdigest()


def replay_candidates(request, costs, gates, ipc, head_ns, select_job, dispatch_choice):
    """Synchronous replay kernel; the worker and the CPU checks both call it.

    ``request`` keys (interface 6.5):
    ``profile`` (``ProfileRing.snapshot``), ``grid``, ``cycle_origin_ns``,
    ``receipt_budget_ns``, ``receipt_budget_source``, ``scenario``,
    ``price_version``, ``snapshot_ns``.  Additional optional keys used here:
    ``repeats`` (default ``REPEATS``), ``current_gF`` (default 1.0, the step the
    net gain is measured from), ``direction`` ('Fast'/'Accurate' to restrict the
    step axis), ``epoch``/``effective_epoch`` + ``history``/``history_index``
    (bounded comparable profiles for ``epsilon_select``), ``revealed_mix``
    (public revealed F/A/Abstain proportions override), ``start_ns``
    (dispatch->actor-start budget, ns; extends occupancy only) and
    ``costs``/``gates``/``ipc``/``head_ns`` (only when the payload is passed to
    :class:`ReplayWorker`).

    Returns the frozen schema (see module docstring / interface 6.5).
    """
    request = dict(request or {})
    snapshot_ns = _as_int(request.get('snapshot_ns'), 0)
    cycle_origin_ns = _as_int(request.get('cycle_origin_ns'),
                              (snapshot_ns // PERIOD_NS) * PERIOD_NS)
    receipt_budget_ns = _as_int(request.get('receipt_budget_ns'), 0)
    receipt_budget_source = request.get('receipt_budget_source') or 'unspecified'
    scenario = request.get('scenario') or 'point'
    if scenario not in SCENARIOS:
        scenario = 'point'
    price_version = request.get('price_version')
    repeats = _as_int(request.get('repeats'), REPEATS)
    if repeats < 1:
        repeats = 1
    counted_repeat = repeats - 1
    profile = request.get('profile') or {}
    roots_in = list(profile.get('roots') or [])
    base = {'schema': REPLAY_SCHEMA, 'scenario': scenario, 'candidates': [],
            'scenarios': {}, 'n_roots': len(roots_in), 'repeats': repeats,
            'overflow': bool(profile.get('overflow')), 'valid': False,
            'invalid_reason': None, 'receipt_budget_source': receipt_budget_source,
            'price_version': price_version, 'snapshot_ns': snapshot_ns,
            'cycle_origin_ns': cycle_origin_ns,
            'receipt_budget_ns': receipt_budget_ns}
    if (request.get('resource_accounting') == 'service_debt'
            or request.get('delivery_objective') is not None):
        objective = request.get('delivery_objective', 'quality_weighted_earliness')
        if objective not in ('quality_weighted_earliness', 'correct_ontime', 'latency_timeout'):
            base['invalid_reason'] = 'invalid_delivery_objective'
            return base
        base['selection_objective'] = objective
    if request.get('resource_accounting') == 'service_debt':
        base['demand_reference'] = request.get('demand_reference')

    if base['overflow']:
        base['invalid_reason'] = 'profile_overflow'
        return base
    if not roots_in:
        base['invalid_reason'] = 'empty_profile'
        return base

    raw_grid = request.get('grid') or CANDIDATE_GRID
    try:
        grid = tuple(float(g) for g in raw_grid)
        valid_grid = all(not isinstance(g, bool) and math.isfinite(v)
                         and 0 <= v <= CARD_COUNT and abs(v*10-round(v*10)) < 1e-8
                         for g, v in zip(raw_grid, grid))
    except (TypeError, ValueError):
        valid_grid = False
        grid = ()
    if not valid_grid:
        base['invalid_reason'] = 'invalid_grid'
        return base
    if not grid:
        base['invalid_reason'] = 'empty_grid'
        return base

    # ---- costs / gates / ipc / head -------------------------------------
    per_cost = _normalise_per_card(costs)
    per_gate = _normalise_per_card(gates, default=costs)
    if not any(per_cost):
        base['invalid_reason'] = 'missing_costs'
        return base
    if not any(per_gate):
        per_gate = [dict(row) for row in per_cost]
    for g in CARD_IDS:
        _fill_kinds(per_cost[g], per_cost[(g+1) % CARD_COUNT])
        _fill_kinds(per_gate[g], per_gate[(g+1) % CARD_COUNT])
    per_ipc = _normalise_pair(ipc, 0)
    timing_ns = max(1., _as_int(request.get('delivery_budget_ns'), 0), max(per_ipc))
    original_budgets = []
    for r in roots_in:
        d,a = _as_int(r.get('deadline_ns'), None), _as_int(r.get('original_arrival_ns'), None)
        if d is not None and a is not None and d>a:
            original_budgets.append(d-a)
    original_budgets.sort()
    original_budget = original_budgets[len(original_budgets)//2] if original_budgets else 50_000_000
    epsilon_h = EPSILON_SELECT_FLOOR*min(1., timing_ns/original_budget)
    per_start = _normalise_pair(request.get('start_ns', request.get('start_latency_ns')), 0)
    head_svc, head_msg = _normalise_head(head_ns)

    # ---- profile geometry -------------------------------------------------
    origin = profile.get('first_publish_ns')
    if not origin:
        pubs = [_as_int(r.get('source_publish_ns'), 0) for r in roots_in]
        origin = min(pubs) if pubs else 0
    origin = _as_int(origin, 0)
    # cycle-phase mapping (TASK 5.1): the profile start is placed on the same
    # 50 ms quota phase it had, at or after the current quota-cycle origin; each
    # root keeps its exact relative offset (no per-root bucket snapping).
    phase = (origin - cycle_origin_ns) % PERIOD_NS
    window_start = cycle_origin_ns + phase
    span_ns = _as_int(profile.get('span_ns'), PROFILE_SPAN_NS) or PROFILE_SPAN_NS

    roots = []
    for idx, r in enumerate(roots_in):
        entry = r.get('entry_class')
        cls = _ENTRY_CLASS.get(entry, 'DF')
        if entry is None:
            cls = 'B' if r.get('kind') == _KIND_PREFIX else 'DF'
        publish = _as_int(r.get('source_publish_ns'),
                          _as_int(r.get('original_arrival_ns'), 0))
        arrival = _as_int(r.get('original_arrival_ns'), publish)
        deadline = _as_int(r.get('deadline_ns'), publish + 50_000_000)
        budget = r.get('budget_ns')
        role = r.get('revealed_role')
        if role is None:
            role = r.get('revealed')
            if role is True:
                role = r.get('role')
            elif role is False:
                role = None
        if role not in ('Fast', 'Accurate', 'Abstain'):
            role = None
        # The point replay repeats the observed input timeline. A tail
        # percentile is a warning, not the delay of every root in that timeline.
        observed_receipt = r.get('input_receipt_ns')
        if observed_receipt is not None:
            if isinstance(observed_receipt, bool) or int(observed_receipt) < 0:
                raise ValueError('input_receipt_ns must be a nonnegative duration')
            root_receipt_ns = int(observed_receipt)
        else:
            root_receipt_ns = receipt_budget_ns  # explicit legacy-input fallback
        rel_release = (publish - origin) + root_receipt_ns
        rel_arrival = arrival - origin
        rel_deadline = deadline - origin
        if cls == 'DF':
            kind_code = 0
            forced = 'Fast'
        elif cls == 'DA':
            kind_code = 1
            forced = 'Accurate'
        else:
            kind_code = 2
            forced = None
        roots.append({'index': idx, 'root_id': r.get('root_id', idx),
                      'cls': cls, 'kind_code': kind_code,
                      'revealed_role': role if forced is None else forced,
                      'public_quality': r.get('public_quality'),
                      'rel_release': rel_release, 'rel_arrival': rel_arrival,
                      'rel_deadline': rel_deadline,
                      'budget_ns': budget,
                      'expired': bool(r.get('expired', False)),
                      'source_publish_ns': publish})
    n_roots = len(roots)
    base['n_roots'] = n_roots

    quality_means = _quality_means(roots)
    mix_override = request.get('revealed_mix')

    # ---- static per-tid arrays (shared by every candidate) ---------------
    n_tid = n_roots * repeats
    release = [0] * n_tid
    deadline = [0] * n_tid
    kind_code = [0] * n_tid
    cls = ['DF'] * n_tid
    disabled = [False] * n_tid
    budget = [None] * n_tid
    repeat = [0] * n_tid
    tasks = [None] * n_tid
    for i, root in enumerate(roots):
        for rep in range(repeats):
            tid = i * repeats + rep
            shift = rep * span_ns
            rel = window_start + root['rel_release'] + shift
            dl = window_start + root['rel_deadline'] + shift
            release[tid] = rel
            deadline[tid] = dl
            kind_code[tid] = root['kind_code']
            cls[tid] = root['cls']
            disabled[tid] = root['expired']
            b = root['budget_ns']
            budget[tid] = None if b is None else _as_int(b, 0)
            repeat[tid] = rep
            tasks[tid] = {'arrival_ns': window_start + root['rel_arrival'] + shift,
                          'deadline_ns': dl}

    base['window_start_ns'] = window_start
    base['profile_origin_ns'] = origin
    base['counted_repeat'] = counted_repeat
    # T014 s3.3.4: the worker table carries the input / profile / epoch / time
    # fingerprints so a consumer can reject a stale or mismatched table.
    base['input_n'] = int(n_roots)
    base['profile_sha'] = _profile_sha(roots, profile)
    base['epoch'] = request.get('epoch', request.get('effective_epoch'))
    base['start_ns'] = list(per_start)
    if request.get('resource_accounting') == 'service_debt':
        base['quality_gain_signature'] = quality_gain_signature(roots_in)
        base['quality_history_eligible'] = bool(request.get('quality_history_eligible'))

    # ---- per-scenario runs ------------------------------------------------
    mix = _revealed_mix(roots, override=mix_override)
    cache = {}
    per_scenario = {}
    for scen in SCENARIOS:
        routes, qualities, info = _assign_scenario(roots, scen, mix,
                                                   quality_means, mix_override)
        if routes is None:
            per_scenario[scen] = {'valid': False,
                                  'invalid_reason': info['reason'],
                                  'info': info, 'results': None}
            continue
        key = tuple(routes)
        runs = cache.get(key)
        if runs is None:
            # per-*tid* views (one entry per root per repeat; routes/qualities
            # are per-root and the repeat shifts only the time axis)
            route_tid = [routes[i] for i in range(n_roots)
                         for _rep in range(repeats)]
            qual_tid = [qualities[i] for i in range(n_roots)
                        for _rep in range(repeats)]
            runs = []
            for gF in grid:
                targets = _card_share(gF)
                runs.append(_run_kernel(select_job, dispatch_choice, tasks, n_tid,
                                        release, deadline,
                                        kind_code, route_tid, cls, qual_tid,
                                        disabled, budget, repeat, targets,
                                        per_cost, per_gate, per_ipc,
                                        per_start, head_svc, head_msg,
                                        counted_repeat, window_start,
                                        CDF_GRID_MS, request.get('resource_accounting'),
                                        request.get('service_debt_balance'),
                                        request.get('allow_idle_endpoint_loans', False),
                                        int(request.get('terminal_return_ns', 0)),
                                        request.get('waiting_capacity')))
            cache[key] = runs
        per_scenario[scen] = {'valid': True, 'invalid_reason': None,
                              'info': info, 'results': runs,
                              'cached': len(cache) < len(SCENARIOS)}

    # ---- epsilon / selection / assembly ----------------------------------
    current = float(request.get('current_gF', 1.0))
    if current not in grid:
        nearest = min(grid, key=lambda g: (abs(g - current), g))
        current = nearest
    ref_idx = grid.index(current)
    base['reference_gF'] = current
    direction = request.get('direction')
    records = _history_records(request)
    if isinstance(records, dict):
        records = []

    scenario_summary = {}
    ordered = {}
    for scen in SCENARIOS:
        entry = per_scenario.get(scen)
        if entry is None or not entry['valid']:
            scenario_summary[scen] = {
                'scenario': scen, 'valid': False,
                'invalid_reason': (entry or {}).get('invalid_reason', 'scenario_unavailable'),
                'epsilon_Y': EPSILON_SELECT_FLOOR,
                'epsilon_V': EPSILON_SELECT_FLOOR, 'low_confidence': True,
                'n_comparable': 0, 'n_unknown': (entry or {}).get('info', {}).get('unknown'),
                'n_revealed': (entry or {}).get('info', {}).get('revealed'),
                'selected_gF': None, 'best_Yhat': None, 'best_Vhat': None,
                'reference_gF': current}
            continue
        yhat = []
        vhat = []
        for res in entry['results']:
            yhat.append(res['ontime'] / T_PROFILE_S)
            vhat.append((res['quality'] / T_PROFILE_S) if res['quality_ok'] else None)
        eps_y, eps_v, used, low_conf = _epsilon(grid, yhat, vhat, ref_idx, records)
        best = _select(grid, yhat, eps_y, eps_v, vhat, ref_idx, current, direction)
        scenario_summary[scen] = {
            'scenario': scen, 'valid': True, 'invalid_reason': None,
            'epsilon_Y': eps_y[ref_idx], 'epsilon_V': eps_v[ref_idx],
            'low_confidence': bool(low_conf), 'n_comparable': used,
            'n_unknown': entry['info']['unknown'],
            'n_revealed': entry['info']['revealed'],
            'revealed_mix': entry['info']['mix'],
            'selected_gF': (grid[best] if best is not None else None),
            'best_Yhat': (yhat[best] if best is not None else None),
            'best_Vhat': (vhat[best] if best is not None else None),
            'reference_gF': current}
        ordered[scen] = {'yhat': yhat, 'vhat': vhat, 'eps_y': eps_y,
                         'eps_v': eps_v, 'results': entry['results'],
                         'info': entry['info'], 'low_conf': low_conf,
                         'n_comparable': used}
    base['scenarios'] = scenario_summary

    # point validity is the gate for the whole comparison (TASK 5.3)
    point_entry = per_scenario.get('point')
    point_valid = bool(point_entry and point_entry['valid'])
    if not point_valid:
        base['invalid_reason'] = 'point_no_revealed_samples'

    target = ordered.get(scenario)
    if target is None:
        base['valid'] = False
        if base['invalid_reason'] is None:
            base['invalid_reason'] = 'scenario_unavailable'
        return base

    point_y = ordered['point']['yhat'] if 'point' in ordered else None
    point_v = ordered['point']['vhat'] if 'point' in ordered else None
    h_eps, h_n = [epsilon_h] * len(grid), [0] * len(grid)
    if request.get('resource_accounting') == 'service_debt':
        reference = target['results'][ref_idx]
        gains = [(r['quality_earliness']-reference['quality_earliness'])/T_PROFILE_S
                 if r['quality_ok'] and reference['quality_ok'] else None
                 for r in target['results']]
        h_eps, h_n = quality_gain_epsilon(grid, gains, epsilon_h,
            request.get('quality_gain_history'), base['epoch'], current,
            snapshot_ns, base['quality_gain_signature'])
    candidates = []
    for i, gF in enumerate(grid):
        res = target['results'][i]
        by_class = {}
        class_ontime = {}
        for name in CLASSES:
            src = res['by_class'][name]
            by_class[name] = {'input': src['input'], 'completed': src['completed'],
                              'late': src['late'], 'rejected': src['rejected'],
                              'abstain': src['abstain'], 'ontime': src['ontime']}
            class_ontime[name] = src['ontime']
        yhat = target['yhat'][i]
        vhat = target['vhat'][i]
        gain_y = yhat - target['yhat'][ref_idx]
        gain_v = None
        if vhat is not None and target['vhat'][ref_idx] is not None:
            gain_v = vhat - target['vhat'][ref_idx]
        y_ok = gain_y >= target['eps_y'][i] and (gain_v is None or gain_v >= -target['eps_v'][i])
        v_ok = (gain_v is not None and gain_v >= target['eps_v'][i]
                and gain_y >= -target['eps_y'][i])
        positive = gain_y > 0 or (gain_v is not None and gain_v > 0)
        direction_ok = True
        if direction == 'Fast':
            direction_ok = gF > current + 1e-9
        elif direction == 'Accurate':
            direction_ok = gF < current - 1e-9
        effective = bool(point_valid and direction_ok and i != ref_idx
                         and (y_ok or v_ok) and positive)
        if request.get('resource_accounting') == 'service_debt':
            current_result = target['results'][ref_idx]
            gain_h = ((res['quality_earliness']-current_result['quality_earliness'])/T_PROFILE_S
                      if res['quality_ok'] and current_result['quality_ok'] else None)
            # H is continuous. Its resolution is one output advanced by the
            # observed delivery timing budget, not one additional whole output.
            effective = bool(point_valid and direction_ok and i != ref_idx
                and gain_y >= 0 and gain_v is not None and gain_v >= 0
                and gain_h is not None and gain_h > 0 and gain_h >= h_eps[i])
        if base.get('selection_objective') == 'correct_ontime':
            effective = bool(point_valid and direction_ok and i != ref_idx
                and gain_y >= 0 and (gain_v > 0 if gain_v is not None else gain_y > 0))
        sens = {}
        for other in ('unknown_F', 'unknown_A', 'unknown_Abstain'):
            src = ordered.get(other)
            if src is None:
                sens[other] = {'Yhat': None, 'Vhat': None,
                               'dYhat': None, 'dVhat': None}
                continue
            oy = src['yhat'][i]
            ov = src['vhat'][i]
            sens[other] = {
                'Yhat': oy, 'Vhat': ov,
                'dYhat': (oy - src['yhat'][ref_idx]) if oy is not None else None,
                'dVhat': (ov - src['vhat'][ref_idx]) if (ov is not None
                                                and src['vhat'][ref_idx] is not None) else None}
            if request.get('resource_accounting') == 'service_debt':
                other_result, other_current = src['results'][i], src['results'][ref_idx]
                sens[other]['dQualityEarliness'] = (
                    (other_result['quality_earliness']-other_current['quality_earliness']) / T_PROFILE_S
                    if other_result['quality_ok'] and other_current['quality_ok'] else None)
        donor_role = 'Accurate' if gF >= current else 'Fast'
        receiver_role = 'Fast' if donor_role == 'Accurate' else 'Accurate'
        candidates.append({
            'gF': float(gF),
            'numeric_target_gF': float(gF),
            'Yhat': yhat,
            'Vhat': vhat,
            'V_available': bool(res['quality_ok']),
            'ontime_roots': res['ontime'],
            'quality_sum': res['quality'] if res['quality_ok'] else None,
            'by_class': by_class,
            'donor_final': res['ontime_role'].get(donor_role, 0),
            'receiver_final': res['ontime_role'].get(receiver_role, 0),
            'gate_starts': res['gate_starts'],
            'input_n': int(sum(by_class[c]['input'] for c in CLASSES)),
            'remaining_budget_cdf': list(res['remaining_budget_cdf']),
            # T014 s3.3.3: the boolean effective-step MARKER and the numeric
            # target stay in separate fields; ``effective_step`` is kept only
            # so an older offline log can still be read (never float()-ed).
            'is_effective_step': bool(effective),
            'effective_step': bool(effective),
            'point_valid': bool(point_valid),
            'sensitivity': sens,
            'card_share': list(_card_share(gF)),
            'epsilon_Y': target['eps_y'][i],
            'epsilon_V': target['eps_v'][i],
            'low_confidence': bool(target['low_conf']),
            'n_comparable': int(target['n_comparable']),
            'net_gain_Yhat': gain_y,
            'net_gain_Vhat': gain_v,
            'class_ontime': class_ontime,
            'reasons': dict(res['reasons']),
            'remaining_budget_cdf_n': res['remaining_budget_cdf_n'],
            'dispatch_mismatch': res['dispatch_mismatch'],
            'started': res['started']
        })
        if base.get('selection_objective') == 'correct_ontime':
            candidates[-1]['selection_objective'] = base['selection_objective']
        if request.get('resource_accounting') == 'service_debt':
            current_result = target['results'][ref_idx]
            candidates[-1]['quality_earliness_hat'] = (
                res['quality_earliness'] / T_PROFILE_S if res['quality_ok'] else None)
            candidates[-1]['net_gain_quality_earliness'] = (
                (res['quality_earliness']-current_result['quality_earliness']) / T_PROFILE_S
                if res['quality_ok'] and current_result['quality_ok'] else None)
            candidates[-1]['epsilon_quality_earliness'] = h_eps[i]
            candidates[-1]['quality_earliness_resolution_ns'] = timing_ns
            candidates[-1]['quality_speed_quantum'] = epsilon_h / EPSILON_SELECT_FLOOR
            candidates[-1]['quality_gain_history_n'] = h_n[i]
            if request.get('protect_role_outcomes'):
                def role_deltas(result, reference):
                    delta = {}
                    for role in ('Fast', 'Accurate'):
                        a, b = result['role_outcomes'][role], reference['role_outcomes'][role]
                        delta[role] = {k: (a[k]-b[k])/T_PROFILE_S
                                       for k in ('completed', 'ontime')}
                        for k, available in (('quality', 'quality_ok'),
                                             ('quality_earliness', 'quality_ok'),
                                             ('all_quality', 'all_quality_ok')):
                            delta[role][k] = ((a[k]-b[k])/T_PROFILE_S
                                             if a[available] and b[available] else None)
                    return delta
                candidates[-1]['role_protection_required'] = True
                candidates[-1]['role_outcome_delta'] = role_deltas(res, current_result)
                for scen in ('unknown_F', 'unknown_A', 'unknown_Abstain'):
                    src = ordered.get(scen)
                    sens[scen]['role_outcome_delta'] = (
                        role_deltas(src['results'][i], src['results'][ref_idx]) if src else None)
        if 'validation_metrics' in res:
            candidates[-1]['validation_metrics'] = res['validation_metrics']
        if base.get('selection_objective') == 'latency_timeout':
            from resource_objective import paired_effect
            effect = paired_effect(res['resource_outcomes'],
                target['results'][ref_idx]['resource_outcomes'], T_PROFILE_S)
            row = candidates[-1]
            row['selection_objective'] = 'latency_timeout'
            row['resource_outcomes'] = res['resource_outcomes']
            row['resource_effect'] = effect
            row['resource_span_s'] = T_PROFILE_S
            row['is_effective_step'] = row['effective_step'] = bool(
                point_valid and direction_ok and i != ref_idx and effect.get('permitted'))
            for scen in ('unknown_F', 'unknown_A', 'unknown_Abstain'):
                src = ordered.get(scen)
                sens[scen]['resource_outcomes'] = src['results'][i]['resource_outcomes'] if src else None
                sens[scen]['resource_effect'] = paired_effect(
                    src['results'][i]['resource_outcomes'],
                    src['results'][ref_idx]['resource_outcomes'], T_PROFILE_S) if src else {
                        'available': False}
    if base.get('selection_objective') == 'latency_timeout':
        # The historical Y/V-selected scenario summary is diagnostic only in
        # old modes. Make the active summary use the actual resource objective.
        from resource_objective import paired_effect
        for scen, src in ordered.items():
            allowed = [(i, paired_effect(r['resource_outcomes'],
                src['results'][ref_idx]['resource_outcomes'], T_PROFILE_S))
                for i, r in enumerate(src['results']) if i != ref_idx]
            allowed = [(i, e) for i, e in allowed if e.get('permitted')]
            chosen = max(allowed, key=lambda x: (x[1]['gain'],
                -abs(grid[x[0]]-current))) if allowed else None
            base['scenarios'][scen]['selected_gF'] = grid[chosen[0]] if chosen else None
            base['scenarios'][scen]['selection_objective'] = 'latency_timeout'
    base['candidates'] = candidates
    base['valid'] = bool(point_valid)
    return base


# ---------------------------------------------------------------------------
# 6. ReplayWorker  (TASK 5.1 / interface 6.5)
# ---------------------------------------------------------------------------
def _worker_main(input_q, output_q, kernel_module, sys_path, cpu=None, ready=None,
                 immutable_payload_reply=False):
    """Child process body: one request at a time, no queue accumulation."""
    # Spawn inherits the arm main's single-CPU mask. Apply the separately
    # reserved replay CPU here, then acknowledge the actual child mask before
    # the arm announces readiness. Exporting an environment variable alone
    # does not isolate this CPU-bound worker from the live dispatcher.
    try:
        if cpu is not None:
            if not hasattr(os, 'sched_setaffinity'):
                raise RuntimeError('replay CPU affinity unavailable')
            os.sched_setaffinity(0, {int(cpu)})
        actual = (sorted(os.sched_getaffinity(0))
                  if hasattr(os, 'sched_getaffinity') else None)
        if cpu is not None and actual != [int(cpu)]:
            raise RuntimeError('replay CPU affinity mismatch: %r' % (actual,))
        if ready is not None:
            ready.send({'requested': cpu, 'actual': actual})
    except Exception as exc:
        if ready is not None:
            ready.send({'requested': cpu, 'error': repr(exc)})
        return
    finally:
        if ready is not None:
            ready.close()
    if sys_path:
        for path in sys_path:
            if path and path not in sys.path:
                sys.path.insert(0, path)
    select_job = None
    dispatch_choice = None
    try:
        module = __import__(kernel_module)
        select_job = getattr(module, 'select_job')
        dispatch_choice = getattr(module, 'dispatch_choice')
    except Exception:
        select_job = None
        dispatch_choice = None
    while True:
        try:
            message = input_q.get()
        except (EOFError, OSError):
            return
        if message is None:
            return
        seq, payload = message
        if isinstance(payload, bytes):
            payload = pickle.loads(payload)
        started = time.perf_counter()
        try:
            req, req_costs, req_gates, req_ipc, req_head = _split_payload(payload)
            if select_job is None or dispatch_choice is None:
                raise RuntimeError('kernel import failed for %r' % (kernel_module,))
            result = replay_candidates(req, req_costs, req_gates, req_ipc,
                                      req_head, select_job, dispatch_choice)
            # Carry the exact public request with its result. The controller
            # can archive a recomputable table without reconstructing inputs
            # from a later window or changing its evaluation semantics.
            if immutable_payload_reply:
                result['_input_payload_seq'] = seq
            else:
                result['input_payload'] = payload
            result['worker_ms'] = (time.perf_counter() - started) * 1000.0
        except BaseException:
            result = {'schema': REPLAY_SCHEMA, 'scenario': None, 'candidates': [],
                      'scenarios': {}, 'n_roots': 0, 'repeats': REPEATS,
                      'overflow': False, 'valid': False,
                      'invalid_reason': 'worker_error',
                      'error': traceback.format_exc()[-2000:],
                      'worker_ms': (time.perf_counter() - started) * 1000.0}
        try:
            output_q.put((seq, result))
        except Exception:
            return


def _split_payload(payload):
    """Tolerate both ``{'request': ..., 'costs': ...}`` and a flat payload."""
    payload = payload or {}
    if isinstance(payload.get('request'), dict):
        req = dict(payload['request'])
        return (req, payload.get('costs'), payload.get('gates'),
                payload.get('ipc'), payload.get('head_ns'))
    req = {}
    for key in _REQUEST_KEYS + ('resource_accounting', 'service_debt_balance',
                              'allow_idle_endpoint_loans',
                              'quality_gain_history', 'quality_history_eligible',
                              'demand_reference', 'protect_role_outcomes', 'delivery_objective',
                              'terminal_return_ns', 'waiting_capacity'):
        if key in payload:
            req[key] = payload[key]
    for key in ('repeats', 'current_gF', 'direction', 'epoch', 'effective_epoch',
                'history', 'history_index', 'history_tail', 'revealed_mix',
                'start_ns', 'start_latency_ns'):
        if key in payload:
            req[key] = payload[key]
    return (req, payload.get('costs'), payload.get('gates'),
            payload.get('ipc'), payload.get('head_ns'))


class ReplayWorker(object):
    """One inflight request plus one latest replacement; never a queue.

    A dead or errored child returns an explicit
    ``{'valid': False, 'invalid_reason': 'prediction_unknown'}``; ``stop()``
    always joins (and kills on refusal) so no orphan worker survives.
    """

    def __init__(self, ctx=None, timeout_s=2.0, cpu=None, immutable_payload_reply=False):
        # Opt-in callers must retain an immutable public snapshot until reply.
        # Default callers still receive the child's serialized snapshot.
        self.immutable_payload_reply = bool(immutable_payload_reply)
        self.ctx = ctx if ctx is not None else multiprocessing.get_context('spawn')
        self.timeout_ns = int(max(0.001, float(timeout_s)) * 1e9)
        self.kernel_module = os.environ.get('T014_REPLAY_KERNEL',
                                           'quota_forecast_v8')
        self._in_q = self.ctx.Queue()
        self._out_q = self.ctx.Queue()
        ready_recv, ready_send = self.ctx.Pipe(duplex=False)
        self.proc = self.ctx.Process(
            target=_worker_main,
            args=(self._in_q, self._out_q, self.kernel_module, list(sys.path),
                  None if cpu is None else int(cpu), ready_send,
                  self.immutable_payload_reply),
            name='T014_candidate_replay')
        self.proc.daemon = True
        self.proc.start()
        ready_send.close()
        try:
            if not ready_recv.poll(30.0):
                raise RuntimeError('replay worker startup acknowledgement timeout')
            self.cpu_affinity = ready_recv.recv()
            if self.cpu_affinity.get('error'):
                raise RuntimeError('replay worker CPU setup failed: %s' %
                                   self.cpu_affinity['error'])
        except Exception:
            if self.proc.is_alive():
                self.proc.terminate()
            self.proc.join(2.0)
            for queue in (self._in_q, self._out_q):
                queue.cancel_join_thread()
                queue.close()
            raise
        finally:
            ready_recv.close()
        self.inflight = None
        self.latest = None
        self.seq = 0
        self._stopped = False
        self._stats = {'requests': 0, 'results': 0, 'replaced': 0,
                       'timeouts': 0, 'errors': 0, 'stale': 0, 'compact_replies': 0}
        self._worker_ms = None
        self._worker_ms_max = None
        self._last_invalid_reason = None

    # -- lifecycle ---------------------------------------------------------
    @property
    def alive(self):
        try:
            return bool(not self._stopped and self.proc.is_alive())
        except Exception:
            return False

    def _send(self, payload, now_ns=None, encoded=None):
        # T014 s3.1.8 / CPU_expected.worker_clock: the send stamp and the poll
        # stamp must be the SAME clock of the same request/reply pair, so the
        # caller's runtime clock is used when it supplies one.  Stamping
        # ``time.perf_counter_ns()`` here would mix an absolute process clock
        # with the runtime's rebased clock and make every age (and therefore
        # the 750 ms staleness gate) meaningless.
        self.seq += 1
        # Serialize once on the caller, then transfer bytes.  request() has
        # already checked this payload; queueing the dictionary would make the
        # feeder traverse/serialize the same large profile a second time.
        if encoded is None:
            encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        self._in_q.put((self.seq, encoded))
        self.inflight = {'seq': self.seq,
                         'sent_ns': (int(now_ns) if now_ns is not None
                                     else time.perf_counter_ns()),
                         'payload': payload}

    def _maybe_send_latest(self, now_ns):
        if self.latest is None:
            return
        payload, encoded = self.latest
        self.latest = None
        try:
            self._send(payload, now_ns, encoded)
        except Exception:
            self._stats['errors'] += 1
            self._last_invalid_reason = 'worker_error'

    def request(self, payload, now_ns):
        """Issue a request; an existing inflight keeps only the newest one."""
        if self._stopped:
            return
        try:
            encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            self._stats['errors'] += 1
            self._last_invalid_reason = 'unpicklable_payload'
            return
        if self.inflight is not None:
            self.latest = (payload, encoded)
            self._stats['replaced'] += 1
        else:
            try:
                self._send(payload, now_ns, encoded)
            except Exception:
                self._stats['errors'] += 1
                self._last_invalid_reason = 'worker_error'
        self._stats['requests'] += 1

    def poll(self, now_ns):
        """Non-blocking; returns a result dict or None when nothing is ready."""
        if self._stopped:
            return None
        now_ns = int(now_ns)
        result = None
        drain = 0
        while drain < 8:
            try:
                seq, payload = self._out_q.get_nowait()
            except Exception:
                break
            drain += 1
            if self.inflight is not None and seq == self.inflight['seq']:
                result = payload
            else:
                self._stats['stale'] += 1
        if result is not None:
            sent_ns = self.inflight['sent_ns']
            if '_input_payload_seq' in result:
                payload_seq = result.pop('_input_payload_seq')
                if (not self.immutable_payload_reply or
                        payload_seq != self.inflight['seq']):
                    result = {'schema': REPLAY_SCHEMA, 'valid': False,
                              'candidates': [], 'invalid_reason': 'prediction_unknown',
                              'cause': 'input_payload_sequence_mismatch'}
                    self._stats['errors'] += 1
                else:
                    result['input_payload'] = self.inflight['payload']
                    self._stats['compact_replies'] += 1
            self.inflight = None
            self._stats['results'] += 1
            worker_ms = result.get('worker_ms')
            if worker_ms is not None:
                self._worker_ms = float(worker_ms)
                if self._worker_ms_max is None or self._worker_ms > self._worker_ms_max:
                    self._worker_ms_max = self._worker_ms
            result['age_ns'] = age_ns(sent_ns, now_ns)
            if not result.get('valid'):
                self._last_invalid_reason = result.get('invalid_reason')
            self._maybe_send_latest(now_ns)
            return result
        if self.inflight is not None:
            if not self.alive:
                self._stats['errors'] += 1
                self._last_invalid_reason = 'prediction_unknown'
                sent_ns = self.inflight['sent_ns']
                self.inflight = None
                self._maybe_send_latest(now_ns)
                return {'schema': REPLAY_SCHEMA, 'scenario': None,
                        'candidates': [], 'scenarios': {}, 'n_roots': 0,
                        'repeats': REPEATS, 'overflow': False, 'valid': False,
                        'invalid_reason': 'prediction_unknown',
                        'cause': 'worker_dead', 'worker_ms': None,
                        'age_ns': age_ns(sent_ns, now_ns)}
            if self.timeout_ns and (now_ns - self.inflight['sent_ns']) > self.timeout_ns:
                self._stats['timeouts'] += 1
                self._last_invalid_reason = 'prediction_unknown'
                sent_ns = self.inflight['sent_ns']
                self.inflight = None
                self._maybe_send_latest(now_ns)
                return {'schema': REPLAY_SCHEMA, 'scenario': None,
                        'candidates': [], 'scenarios': {}, 'n_roots': 0,
                        'repeats': REPEATS, 'overflow': False, 'valid': False,
                        'invalid_reason': 'prediction_unknown',
                        'cause': 'timeout', 'worker_ms': None,
                        'age_ns': age_ns(sent_ns, now_ns)}
        return None

    def stop(self):
        """Always join; never leave an orphan worker behind."""
        if self._stopped:
            return
        self._stopped = True
        try:
            self._in_q.put(None)
        except Exception:
            pass
        try:
            self.proc.join(2.0)
        except Exception:
            pass
        if self.proc.is_alive():
            try:
                self.proc.terminate()
                self.proc.join(1.0)
            except Exception:
                pass
        if self.proc.is_alive():
            try:
                self.proc.kill()
                self.proc.join(1.0)
            except Exception:
                pass
        for queue in (self._in_q, self._out_q):
            try:
                queue.cancel_join_thread()
            except Exception:
                pass
            try:
                queue.close()
            except Exception:
                pass
        self.inflight = None
        self.latest = None

    def stats(self):
        return {'requests': self._stats['requests'],
                'cpu_affinity': dict(self.cpu_affinity),
                'results': self._stats['results'],
                'compact_replies': self._stats['compact_replies'],
                'replaced': self._stats['replaced'],
                'timeouts': self._stats['timeouts'],
                'errors': self._stats['errors'],
                'alive': bool(self.alive),
                'worker_ms': self._worker_ms,
                'stale': self._stats['stale'],
                'worker_ms_max': self._worker_ms_max,
                'last_invalid_reason': self._last_invalid_reason}
