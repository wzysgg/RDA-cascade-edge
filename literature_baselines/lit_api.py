"""Frozen public-observation contract for literature baseline policies.

This module is the ONLY channel through which a literature adapter may observe
the simulation.  An adapter never receives the T016 runtime object, the task
table, the frozen frame stream, the label bank, or any phase label: it receives
a :class:`PublicView` built exclusively from **historical public events**:

* roots that have already arrived (and only those),
* the model intent and published quality score the common Local router already
  published for a root,
* the current GPU waiting queues,
* task generation time and absolute deadline,
* the component each GPU is executing and its predicted remaining service,
* the locked/measured GPU service table,
* legitimate feedback of already completed tasks,
* rejections / completions / timeouts that have already happened.

The sanitized public contract excludes: EARLY results of a
Prefix that has not completed, the future frame stream, future phase labels,
future arrival rates, the true classification labels of the frozen test tasks,
future predictions of a model that was never executed, and the offline optimal
GPU configuration.

Adapters implement :class:`LiteraturePolicy`.  The runner calls
``choose_action`` every time the shared discrete-event kernel wants to hand one
component to a card; the returned action is executed by the frozen kernel, so
the adapter pays the real service cost of every step it decides.
"""

import importlib
import copy
import json
import os
import sys
from collections.abc import Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

#: Roles a literature action may request from the shared executor.
ACTION_ROLES = ('Fast', 'Accurate')

#: Job kinds the shared kernel can actually execute on a card.
EXECUTABLE_KINDS = ('FAST_FULL', 'FAST_PREFIX', 'FAST_SUFFIX', 'ACCURATE')

#: Attributes an adapter is forbidden to touch, checked by
#: :func:`audit_adapter_source` and reported in ONLINE_INFORMATION_LEAKAGE_AUDIT.
#: Patterns are attribute-access / private-artifact shaped so that ordinary
#: English prose in a docstring does not produce a false hit.
FORBIDDEN_SOURCE_PATTERNS = (
    'rt.tasks', 'runtime.tasks', 'self.tasks', '.tasks[', '.ready[',
    '.frames', 'source_frames', 'arrival_plan', 'private_selection',
    'bank_manifest', 'ground_truth', 'y_true', 'true_label', 'labels[',
    '.labels', 'experiment_phases', 'phase_label', 'oracle',
    'probabilities', 'p_correct', 'p_selected_final', 'p_selected',
    'quality_need', 'reference_document', 'cost_book.records',
    'rt.ended', 'self.ended', 'self.jobs', '.components[',
)


class FeedbackHistory(Sequence):
    """Read-only, O(1) history handle; returned records are copies."""
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(dict(r) for r in self._rows[index])
        return dict(self._rows[index])


class JobView(object):
    """One already-published component waiting for (or owing) a card.

    ``deadline_abs_ns`` is the ORIGINAL absolute deadline (generation + 50 ms);
    no adapter may move it.  ``card_affinity`` is non-None only for a Fast
    Suffix, whose same-card contiguous constraint is structural: the job exists
    only inside that card's pinned list.
    """

    __slots__ = ('job_id', 'task_id', 'kind', 'role', 'card_affinity',
                 'arrival_ns', 'enqueue_ns', 'deadline_abs_ns', 'component_id',
                 'queue_pos', 'pinned', 'raw', 'stream_id')

    def __init__(self, job_id, task_id, kind, role, card_affinity, arrival_ns,
                 enqueue_ns, deadline_abs_ns, component_id, queue_pos, pinned,
                 raw=None, stream_id=None):
        self.raw = raw
        #: the publishing Local terminal (the paper's "video feed" / terminal
        #: stream).  It is public at arrival: it is the arrival event's own
        #: origin, not a phase label, a future arrival or a task's true class.
        self.stream_id = stream_id
        self.job_id = job_id
        self.task_id = task_id
        self.kind = kind
        self.role = role
        self.card_affinity = card_affinity
        self.arrival_ns = arrival_ns
        self.enqueue_ns = enqueue_ns
        self.deadline_abs_ns = deadline_abs_ns
        self.component_id = component_id
        self.queue_pos = queue_pos
        self.pinned = pinned

    def waiting_ns(self, now_ns):
        """Elapsed time since arrival (the paper's ``w``), never since enqueue."""
        return int(now_ns) - int(self.arrival_ns)

    def slack_ns(self, now_ns):
        return int(self.deadline_abs_ns) - int(now_ns)

    def __repr__(self):
        return 'JobView(t=%d,%s,role=%s,w=%d)' % (
            self.task_id, self.kind, self.role, self.arrival_ns)


class CardView(object):
    __slots__ = ('card', 'busy', 'busy_job_kind', 'busy_job_task_id',
                 'busy_job_role', 'busy_remaining_ns', 'busy_started_ns',
                 'target', 'effective', 'transition', 'n_pending_suffix',
                 'pinned', 'service_used_ns', 'idle_ns')

    def __init__(self, **kw):
        for key in self.__slots__:
            setattr(self, key, kw.get(key))

    @property
    def free(self):
        return not self.busy

    def __repr__(self):
        return 'CardView(%d,busy=%s,rem=%s,target=%.3f)' % (
            self.card, self.busy, self.busy_remaining_ns, self.target)


class PublicView(object):
    """Everything a literature policy may legally know at ``now_ns``."""

    __slots__ = ('now_ns', 'deadline_ns', 'duration_ns', 'cards', 'ready',
                 'pinned', 'service_ns', 'ipc_ns', 'feedback', 'counters',
                 'adapter_epoch')

    def __init__(self, **kw):
        for key in self.__slots__:
            setattr(self, key, kw.get(key))

    # -- convenience accessors (all pure reads of already-public state) ----
    def free_cards(self):
        return [c for c in self.cards if not c.busy]

    def ready_jobs(self, role):
        return list(self.ready.get(role) or ())

    def all_ready(self):
        return list(self.ready.get('Fast') or ()) + \
            list(self.ready.get('Accurate') or ())

    def pinned_jobs(self, card):
        return list(self.pinned.get(card) or ())

    def service_ns_for(self, card, kind):
        return int(self.service_ns[card][kind])

    def predicted_service_ns(self, card, job):
        """The one locked measured cost of executing ``job`` on ``card``.

        A Fast Prefix additionally commits its Suffix on the same card; that
        commitment is returned separately by :meth:`suffix_commitment_ns` and is
        NEVER silently folded into a classification-accuracy decision.
        """
        base = int(self.service_ns[card][job.kind]) + int(self.ipc_ns[card])
        return base

    def suffix_commitment_ns(self, card):
        return int(self.service_ns[card]['FAST_SUFFIX'])

    def to_json(self):
        return {'now_ns': self.now_ns,
                'cards': [{'card': c.card, 'busy': c.busy,
                           'kind': c.busy_job_kind,
                           'remaining_ns': c.busy_remaining_ns,
                           'target': c.target, 'effective': c.effective}
                          for c in self.cards],
                'ready_Fast': len(self.ready.get('Fast') or ()),
                'ready_Accurate': len(self.ready.get('Accurate') or ()),
                'pinned': {str(g): len(v) for g, v in (self.pinned or {}).items()}}


class LiteraturePolicy(object):
    """Base class every literature adapter implements.

    Lifecycle (all calls are made by the runner, never by the adapter):

    ``reset(builder, config)``
        called once after the runtime exists; ``config`` carries the adapter's
        frozen parameters read from ``literature_baselines/lit_config.json``.
    ``choose_action(now_ns, view)``
        called whenever at least one card is free.  Return ``None`` (or
        ``(None, None, None)``) to leave every card idle; otherwise return
        ``(card, job, role)`` where ``job`` is a :class:`JobView` taken from
        ``view.ready[role]`` or ``view.pinned[card]``.  Illegal actions are
        refused by the runner and recorded as a violation, so an adapter cannot
        cheat by naming a job that is not publicly queued.
    ``on_control_tick(now_ns, view)``
        optional hook, called on the frozen 100 ms control tick.  A pure
        bookkeeping opportunity; it may NOT enqueue work.
    ``on_completion(record)``
        optional hook called with a legitimate completed-task feedback record.
    ``ledger()``
        returns a JSON-serialisable dict of decision counters / state used by
        the implementation audit.
    """

    name = 'abstract'
    version = 'v1'
    #: set by the adapter to the audit file that documents its fidelity
    audit_document = ''
    #: True when the adapter needs the billed reference inference of the paper
    bills_reference_inference = False

    def __init__(self, config=None):
        #: frozen adapter parameters (merged ``lit_config.json`` + per-arm file)
        self.config = dict(config or {})

    def reset(self, builder, config):  # pragma: no cover - interface
        pass

    def choose_action(self, now_ns, view):  # pragma: no cover - interface
        raise NotImplementedError

    def on_control_tick(self, now_ns, view):  # pragma: no cover - interface
        pass

    def on_completion(self, record):  # pragma: no cover - interface
        pass

    def forecast_clone(self, view):
        """Independent policy snapshot. No forecast may mutate the live policy."""
        return copy.deepcopy(self)

    def ledger(self):  # pragma: no cover - interface
        return {}


class PublicViewBuilder(object):
    """Builds :class:`PublicView` objects from the frozen kernel.

    The builder is owned by the runner.  Adapters get views, never the runtime.
    """

    def __init__(self, rt, feedback_ring, counters_fn):
        self.rt = rt
        self.feedback_ring = feedback_ring
        self.counters_fn = counters_fn

    def build(self, now_ns):
        rt = self.rt
        tasks = rt.tasks
        ready = {'Fast': [], 'Accurate': []}
        for role in ACTION_ROLES:
            lst = rt.ready[role]
            ordered = sorted(
                lst,
                key=lambda j: ((tasks.get(j['task_id']) or {}).get('arrival_ns', 0)
                               if role == 'Fast' else j.get('enqueue_ns', 0),
                               j['task_id']))
            for pos, j in enumerate(ordered):
                task = tasks.get(j['task_id']) or {}
                ready[role].append(JobView(
                    job_id=j.get('job_id'), task_id=j['task_id'], kind=j['kind'],
                    role=role, card_affinity=None,
                    arrival_ns=task.get('arrival_ns', 0),
                    enqueue_ns=j.get('enqueue_ns', 0),
                    deadline_abs_ns=task.get('deadline_abs_ns',
                                             j.get('deadline_ns', 0)),
                    component_id=j.get('component_id'), queue_pos=pos,
                    pinned=False, raw=None, stream_id=task.get('origin_node')))
        pinned = {}
        for g in range(len(rt.gpus)):
            rows = []
            for j in rt.gpus[g]['pinned']:
                task = tasks.get(j['task_id']) or {}
                rows.append(JobView(
                    job_id=j.get('job_id'), task_id=j['task_id'], kind=j['kind'],
                    role='Fast', card_affinity=g,
                    arrival_ns=task.get('arrival_ns', 0),
                    enqueue_ns=j.get('enqueue_ns', 0),
                    deadline_abs_ns=task.get('deadline_abs_ns',
                                             j.get('deadline_ns', 0)),
                    component_id=j.get('component_id'), queue_pos=0,
                    pinned=True, raw=None, stream_id=task.get('origin_node')))
            pinned[g] = rows
        cards = []
        service_ns = {}
        ipc_ns = {}
        for g, s in enumerate(rt.gpus):
            job = s.get('job')
            remaining = None
            if s.get('busy') and job is not None:
                remaining = max(0, int(s.get('budget_finish_ns', 0)) - int(now_ns))
            cards.append(CardView(
                card=g, busy=bool(s.get('busy')),
                busy_job_kind=(job or {}).get('kind'),
                busy_job_task_id=(job or {}).get('task_id'),
                busy_job_role=rt.job_role(job) if job else None,
                busy_remaining_ns=remaining,
                busy_started_ns=s.get('busy_since'),
                target=float(s.get('target', 0.0)),
                effective=float(s.get('effective', 0.0)),
                transition=bool(s.get('transition')),
                n_pending_suffix=len(s.get('pending') or ()),
                pinned=len(s.get('pinned') or ()),
                service_used_ns={r: int(s.get('spent', {}).get(r, 0))
                                 for r in ACTION_ROLES},
                idle_ns=float(s.get('idle_ns', 0.0))))
            service_ns[g] = {k: int(rt.mean_ns[g][k]) for k in EXECUTABLE_KINDS}
            ipc_ns[g] = int(rt.gpu_ipc_budget[g])
        return PublicView(
            now_ns=int(now_ns), deadline_ns=int(rt.deadline_ns),
            duration_ns=int(getattr(rt, 'duration_ns', 0)),
            cards=cards, ready=ready, pinned=pinned,
            service_ns=service_ns, ipc_ns=ipc_ns,
            # A bounded-cost read-only history view, not a mutable shared list.
            feedback=FeedbackHistory(self.feedback_ring), counters=self.counters_fn(),
            adapter_epoch=int(getattr(rt, 'lit_adapter_epoch', 0)))


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
_ADAPTERS = {}


def register(name, factory):
    _ADAPTERS[name] = factory


def load_adapter(name, config=None):
    """Instantiate a registered adapter; the runner owns construction/lifecycle.

    The adapter module is imported on first use so that an arm only pays for the
    code it actually needs.
    """
    if name not in _ADAPTERS:
        for module in (name + '_adapter', name):
            try:
                importlib.import_module(module)
                break
            except ImportError:
                continue
    if name not in _ADAPTERS:
        raise KeyError('literature adapter %r is not registered (have %r)'
                       % (name, sorted(_ADAPTERS)))
    return _ADAPTERS[name](config or {})


def registered():
    return sorted(_ADAPTERS)


def audit_adapter_source(path):
    """Static leakage audit: report forbidden attribute access in an adapter.

    This is an *additional* guard.  The API exposes sanitized public
    views; this scan is not a Python sandbox or a complete information-flow proof.
    """
    hits = []
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            for lineno, line in enumerate(fh, 1):
                code = line.split('#', 1)[0]
                for pat in FORBIDDEN_SOURCE_PATTERNS:
                    if pat in code:
                        hits.append({'file': os.path.basename(path),
                                     'line': lineno, 'pattern': pat,
                                     'text': line.rstrip('\n')})
    except OSError as exc:
        hits.append({'file': os.path.basename(path), 'line': 0,
                     'pattern': '<unreadable>', 'text': repr(exc)})
    return hits


def dump_json(path, payload):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write('\n')
