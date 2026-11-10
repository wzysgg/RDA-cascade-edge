"""T011 shared Local front end.

ONE online Local service (20 logical nodes / 20 CPU actors) shared by two
concurrent disjoint GPU pairs.  The shared source is the sole plan injector:
one arrival plan, one monotonic ``t0``, one Local front end clock.  Both
consumers read the same immutable frames, independently and non-blockingly.

qf4 additions over shared_local_t010 (TASK.md section 5): every published
frame carries ``frame_id`` and ``source_publish_ns``, and the source keeps a
SINGLE buffered ``shared_frames.jsonl`` archive on the server with the frame
content; consumers correlate their receipt with these ids.  The original
public admission/routing algorithms are byte-for-byte the N10 rules: GPU
continuation intrinsics use the common FROZEN initial cost
(``source_prior_legacy``), never one group's live share/queue/quota; a slow
consumer can never block the source or the other consumer; ``bank_index`` /
image condition / change instant never reach a controller.
"""

import hashlib
import json
import math
import os
import time
from collections import deque, Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

FRAME_SCHEMA = 'T011_shared_frame_v1'

OUTCOME_ARRIVAL = 'SHARED_ARRIVAL'
OUTCOME_LOCAL_TERMINAL = 'COMMON_LOCAL_TERMINAL'
OUTCOME_GPU_READY = 'GPU_READY'
OUTCOME_PREFIX_READY = 'PREFIX_READY'


def source_frame_digest(frame):
    """Immutable original contents; runtime delivery stamps are not input."""
    import hashlib
    raw = {k: v for k, v in frame.items() if not k.startswith('_replay_')}
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode('utf8')).hexdigest()
OUTCOME_ABSTAIN = 'COMMON_ABSTAIN'
# Public first-visible quality judgement (MID1 after the probe, MID2 after the
# optional second stage).  Both consumers observe it; the controller counts
# only the first judgement per task_id for its public intent samples.
OUTCOME_QUALITY = 'SHARED_QUALITY_INTENT'

# Keys the controller adapter must never forward.
CONTROLLER_PRIVATE_KEYS = ('bank_index', 'condition', 'source', 'change_instant',
                           'image_condition', 'blur', 'label')

DEFAULT_DEADLINE_MS = 50.0
PERIOD_NS = 50_000_000

# Frozen Local routing (change C1).  When a frozen table is supplied the source
# still runs the real Local actors for quality, but every routing decision that
# used to depend on the wall clock (initial admission, the dequeue expiry, the
# MID2 afford switch, and the ``candidate_walk`` remaining-time walk) is read
# from a table derived offline from one real run.  The arrival plan is already
# frozen, so this makes the Local->edge stream a pure function of that plan.
FROZEN_ROUTE_ENV = 'T016_FROZEN_ROUTE'
FROZEN_ROUTE_SCHEMA = 'T016_frozen_local_route_v1'


def load_frozen_route(path, expected_sha256=None):
    """Load a routing table produced by ``tools/export_frozen_local_route.py``.

    The document stores one small code per ``task_id`` (dense range) for the
    initial admission verdict, the quality stage and the committed terminal,
    plus the rare side records (abstain reason, GPU candidates)."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError('frozen Local route SHA mismatch: %s != %s' % (digest, expected_sha256))
    doc = json.loads(raw.decode('utf8'))
    if doc.get('schema') != FROZEN_ROUTE_SCHEMA:
        raise RuntimeError('unexpected frozen Local route schema: %r' % (doc.get('schema'),))
    codes = doc['codes']
    return {
        'path': str(path),
        'sha256': digest,
        'source_run': doc.get('source_run'),
        'generated_utc': doc.get('generated_utc'),
        'task_id_min': int(doc['task_id_min']),
        'n_tasks': int(doc['arrivals']),
        'initial': doc['initial'],
        'quality': doc['quality'],
        'terminal': doc['terminal'],
        'abstain_reason': dict(doc.get('abstain_reason') or {}),
        'gpu_candidates': dict(doc.get('gpu_candidates') or {}),
        'initial_names': {int(v): k for k, v in codes['initial'].items()},
        'quality_names': {int(v): k for k, v in codes['quality'].items()},
        'terminal_names': {int(v): k for k, v in codes['terminal'].items()},
        'counts': doc.get('counts'),
        'document': doc,
    }


# --------------------------------------------------------------------------
# pure helpers (unit-testable without torch, without actors)
# --------------------------------------------------------------------------
def initial_admission(cpu_wait_ns, probe_ns, min_continuation_ns, remaining_ns):
    """Original CPU wait / LocalProbe / frozen-minimum-continuation rule."""
    need = float(cpu_wait_ns) + float(probe_ns) + float(min_continuation_ns)
    if need <= float(remaining_ns):
        return 'PROBE'
    return 'PREFIX'


def gpu_intrinsic_frozen(kind, common_intrinsic_ms, ipc_ns):
    """Common frozen zero-queue GPU continuation cost, in ns (legacy prior)."""
    if kind not in common_intrinsic_ms:
        raise KeyError('unknown frozen GPU continuation kind %r' % (kind,))
    return float(common_intrinsic_ms[kind]) * 1e6 + float(ipc_ns)


def min_continuation_ns(local_suffix_ns, common_intrinsic_ms, ipc_ns):
    """Frozen minimum continuation after a Local probe (Local suffix or GPU)."""
    gpu = min(gpu_intrinsic_frozen(k, common_intrinsic_ms, ipc_ns)
              for k in ('ACCURATE', 'FAST_FULL'))
    return min(float(local_suffix_ns), gpu)


def candidate_walk(quality, local_cost_fn, gpu_cost_fn, remaining_ns):
    """Walk ``quality_order`` candidates in order (public, unchanged from N10).

    Local candidates are judged by the PUBLIC real CPU queue and the original
    costs; GPU candidates by the common FROZEN zero-queue intrinsic.  Returns
    ``(chosen_role, feasible_gpu_roles)`` where ``chosen_role`` is the first
    feasible Local role (committed by the source) or None, and
    ``feasible_gpu_roles`` lists the public feasible GPU candidates in the
    original order.
    """
    gpu_roles = []
    for role in quality:
        if role == 'Local':
            if local_cost_fn(role) <= remaining_ns:
                return role, gpu_roles
        else:
            if gpu_cost_fn(role) <= remaining_ns:
                gpu_roles.append(role)
    return None, gpu_roles


def local_quality_route(order, mode='legacy'):
    """Quality failure escalates to Accurate; Fast is a capacity fallback.

    An Accurate-first judgement is committed to that quality role. When Local
    is quality eligible, its original Fast fallback remains available only if
    the Local continuation cannot fit the remaining time. No phase/image
    identity enters this rule.
    """
    if mode == 'legacy':
        return list(order)
    if mode == 'quality_first_eligible':
        # The conditioned-pool scene commits the first public quality role,
        # including Fast; resource pressure never selects another candidate.
        return list(order[:1])
    if mode != 'quality_to_accurate':
        raise ValueError('unknown Local quality routing mode: %s' % mode)
    if order and order[0] == 'Accurate':
        return ['Accurate']
    if 'Local' in order:
        return ['Local'] + [r for r in order if r != 'Local']
    return ['Accurate'] if 'Accurate' in order else []


def whitelist_public(kw):
    """Strip every controller-private key from a public event payload."""
    return {k: v for k, v in kw.items() if k not in CONTROLLER_PRIVATE_KEYS}


def frame_lag_ns(consumer_recv_ns, source_publish_ns):
    return int(consumer_recv_ns) - int(source_publish_ns)


# --------------------------------------------------------------------------
# frame bus: one publisher, N independent non-blocking consumers
# --------------------------------------------------------------------------
class FrameBus(object):
    def __init__(self, ctx, consumers, capacity=32768):
        self.ctx = ctx
        self.capacity = int(capacity)
        self.queues = {name: ctx.Queue(maxsize=self.capacity) for name in consumers}
        self.published = 0
        self.dropped = {name: 0 for name in consumers}
        self.closed = False

    def publish(self, frame):
        """Publish one immutable frame to every consumer; never blocks."""
        if self.closed:
            return
        self.published += 1
        for name, q in self.queues.items():
            try:
                q.put_nowait(frame)
            except Exception:
                # A consumer that cannot keep up loses frames; this is counted
                # and reported, never silently absorbed.
                self.dropped[name] += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        for q in self.queues.values():
            try:
                q.put_nowait({'schema': FRAME_SCHEMA, 'outcome': 'SOURCE_END',
                              'frame_id': -1, 'task_id': -1})
            except Exception:
                pass

    def drain(self, name, limit=128):
        """Consumer-side non-blocking read; returns (frames, end_seen)."""
        out = []
        q = self.queues[name]
        for _ in range(int(limit)):
            try:
                item = q.get_nowait()
            except Exception:
                break
            if item.get('outcome') == 'SOURCE_END':
                return out, True
            out.append(item)
        return out, False


# --------------------------------------------------------------------------
# frozen cost identity for the shared source
# --------------------------------------------------------------------------
class FrozenSourceCosts(object):
    """The frozen CPU Local table plus the common frozen GPU continuation
    intrinsics.  The GPU numbers are the N10 frozen prior, kept under
    ``source_prior_legacy``; the per-card gates used at admission inside the
    groups are measured by the qf4 one-start warmup instead (TASK s4)."""

    def __init__(self, reference=None, evidence_dir=None):
        ev = Path(evidence_dir) if evidence_dir else (HERE / 'evidence')
        self.evidence_dir = ev
        c = json.loads((ev / 'cpu_probe.json').read_text(encoding='utf-8'))['times']
        self.local_probe_ns = (c['LOCAL_PREFIX']['p95_ms'] + c['MID1_HEAD']['p95_ms']) * 1e6
        self.local_mid2_ns = (c['LOCAL_MID2']['p95_ms'] + c['MID2_HEAD']['p95_ms']) * 1e6
        self.local_suffix_ns = {False: c['LOCAL_SUFFIX5']['p95_ms'] * 1e6,
                                True: c['LOCAL_SUFFIX9']['p95_ms'] * 1e6}
        self.local_total_ns = self.local_probe_ns + max(self.local_suffix_ns.values())
        self.local_ipc_ns = 600_000.0
        self.gpu_ipc_ns = 400_000.0
        ref = reference
        if ref is None:
            p = ev / 'REFERENCE.json'
            ref = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
        self.reference = ref
        self.common_intrinsic_ms = dict(ref.get('common_source_intrinsic_ms') or {})
        if not self.common_intrinsic_ms:
            raise RuntimeError('REFERENCE.common_source_intrinsic_ms missing')
        self.c_ref_ms = dict((ref.get('c_ref') or {}).get('reference_ms') or {})

    def provenance(self):
        return {
            'source': 'evidence/cpu_probe.json + REFERENCE common intrinsic prior',
            'gpu_intrinsic_prior': 'N10 frozen common_source_intrinsic_ms (source_prior_legacy);'
                                   ' per-card gates are measured by the qf4 warmup',
            'local_probe_ns': self.local_probe_ns,
            'local_mid2_ns': self.local_mid2_ns,
            'local_suffix_ns': {str(k): v for k, v in self.local_suffix_ns.items()},
            'local_ipc_ns': self.local_ipc_ns,
            'gpu_ipc_ns': self.gpu_ipc_ns,
            'common_source_intrinsic_ms': dict(self.common_intrinsic_ms),
            'common_source_cost_rule': (self.reference.get('common_source_cost_rule')
                                        or 'frozen common intrinsic + CPU table'),
        }


def _local_worker(node, bank, rawbank, commands, results, cpu_graph=False):
    """Spawn target for the same Local stage protocol, using CPU only."""
    from live_runtime_v20_two_gpu import local_actor
    local_actor(node, bank, rawbank, commands, results, cpu_graph=cpu_graph)


# --------------------------------------------------------------------------
# the shared source
# --------------------------------------------------------------------------
class SharedLocalSource(object):
    """Sole plan injector + the ONE physical Local service."""

    def __init__(self, spec, plan, costs, folder=None, duration_s=None,
                 deadline_ms=None, nodes=20, ctx=None):
        import multiprocessing as mp
        self.spec = spec
        self.plan = plan
        self.costs = costs
        self.folder = Path(folder) if folder else (HERE / 'runs' / str(spec.get('name')))
        self.duration_s = float(duration_s if duration_s is not None else spec.get('duration_s', 250))
        self.deadline_ms = float(deadline_ms if deadline_ms is not None else spec.get('deadline_ms', DEFAULT_DEADLINE_MS))
        self.deadline_ns = int(round(self.deadline_ms * 1e6))
        route_path = str(spec.get('frozen_route_path')
                         or spec.get('T016_FROZEN_ROUTE')
                         or os.environ.get(FROZEN_ROUTE_ENV) or '').strip()
        self.frozen_route = load_frozen_route(route_path) if route_path else None
        self.nodes = int(nodes)
        self.ctx = ctx or mp.get_context('spawn')
        self.results = None
        self.commands = {}
        self.ps = []
        self.bank = None
        self.raw = None
        self.meta = None
        self.t0 = None
        self.counters = {
            'arrivals': 0, 'probe': 0, 'prefix': 0, 'local_terminal': 0,
            'gpu_ready': 0, 'abstain': 0, 'local_mid2': 0, 'local_suffix': 0,
            'expired_before_admission': 0, 'common_rejections': 0,
        }
        self.publish_stats = {}
        self._frames = 0
        self._frames_file = None

    # -- actors ----------------------------------------------------------
    def prepare_bank(self):
        from live_runtime_v20_two_gpu import bank_prepare
        self.bank, self.raw, self.meta, _labels = bank_prepare()

    def start_actors(self):
        if self.bank is None:
            self.prepare_bank()
        self.results = self.ctx.Queue()
        for n in range(self.nodes):
            q = self.ctx.Queue()
            self.commands['local:%d' % n] = q
            p = self.ctx.Process(target=_local_worker,
                                 args=(n, self.bank, self.raw, q, self.results,
                                       self.spec.get('local_cpu_graph', False)))
            # Spawn unpickles torch tensors before entering the worker. Inherit
            # the intended CPU before imports, then restore the source binding.
            old_affinity = os.sched_getaffinity(0)
            raw_cpu = os.environ.get('T012_LOCAL_CPU_%d' % n,
                                     os.environ.get('T011_LOCAL_CPU_%d' % n))
            try:
                if raw_cpu is not None:
                    os.sched_setaffinity(0, {int(raw_cpu)})
                p.start()
            finally:
                os.sched_setaffinity(0, old_affinity)
            self.ps.append(p)
        ready = []
        for _ in self.ps:
            r = self.results.get(timeout=600)
            if r['type'] != 'READY':
                raise RuntimeError(r)
            ready.append(r)
        return ready

    def set_t0(self, t0_ns):
        self.t0 = int(t0_ns)

    def stop_actors(self):
        for q in self.commands.values():
            q.put({'kind': 'STOP'})
        for p in self.ps:
            p.join(timeout=30)
        if any(p.is_alive() for p in self.ps):
            raise RuntimeError('shared Local actor did not shut down cleanly')

    # -- admission -------------------------------------------------------
    def _local_queue_wait(self, node, now_ns, state):
        """Original public CPU wait rule for one node (unchanged from N10)."""
        wait = 0.0
        for j in state['q'][node]:
            if j['kind'] == 'LOCAL_PROBE':
                wait += self.costs.local_probe_ns + max(self.costs.local_suffix_ns.values()) + self.costs.local_ipc_ns
            else:
                wait += max(self.costs.local_suffix_ns.values()) + self.costs.local_ipc_ns
        s = state['busy'][node]
        if s:
            wait += max(0.0, state['budget_finish'][node] - now_ns)
            if s['kind'] == 'LOCAL_PROBE':
                wait += max(self.costs.local_total_ns - self.costs.local_probe_ns, 0.0)
            elif s['kind'] == 'LOCAL_MID2':
                wait += self.costs.local_suffix_ns[True] + self.costs.local_ipc_ns
        return wait

    def _frozen_min_continuation(self, node, mid2):
        return min_continuation_ns(self.costs.local_suffix_ns[mid2] + self.costs.local_ipc_ns,
                                   self.costs.common_intrinsic_ms, self.costs.gpu_ipc_ns)

    def _frozen_record(self, task_id):
        """Decoded frozen decision for one task (raises if the table misses it)."""
        fr = self.frozen_route
        k = int(task_id) - fr['task_id_min']
        if not 0 <= k < fr['n_tasks']:
            raise RuntimeError('frozen Local route does not cover task %r' % (task_id,))
        record = {'initial': fr['initial_names'][fr['initial'][k]],
                  'quality': fr['quality_names'][fr['quality'][k]],
                  'terminal': fr['terminal_names'][fr['terminal'][k]]}
        if record['terminal'] == 'GPU_READY':
            record['gpu_candidates'] = list(fr['gpu_candidates'].get(str(int(task_id))) or [])
        if record['terminal'] == 'ABSTAIN':
            record['abstain_reason'] = fr['abstain_reason'].get(str(int(task_id)))
        return record

    def _publish(self, bus, frame):
        """Publish one frame with its unique frame_id and the single buffered
        server-side archive line (TASK.md section 5)."""
        self._frames += 1
        frame['frame_id'] = self._frames
        bus.publish(frame)
        if self._frames_file is not None:
            try:
                self._frames_file.write(json.dumps(frame, sort_keys=True) + '\n')
            except Exception:
                self.counters.setdefault('archive_write_errors', 0)
                self.counters['archive_write_errors'] += 1

    # -- main loop -------------------------------------------------------
    def run(self, bus):
        from live_runtime_v20_two_gpu import quality_order
        import gc
        # T012: `admits_after_prediction` is NOT used here any more.  Its new
        # signature (estimate_ns, deadline_abs_ns, fresh_now_ns) takes an
        # ABSOLUTE deadline, and this source never calls it (the shared source
        # has no arm-local absolute deadline; the arms re-check admission in
        # `group_runtime_t011`).  The old threaded-through parameter was dead.
        if self.t0 is None:
            raise RuntimeError('t0 must be set by the joint barrier before the run')
        spec = self.spec
        nodes = self.nodes
        state = {'q': [deque() for _ in range(nodes)], 'busy': [None] * nodes,
                 'budget_finish': [0] * nodes, 'job': [None] * nodes,
                 'held': [None] * nodes}
        plan = self.plan
        frozen_route = self.frozen_route
        cursor = 0
        pending = {}          # job_id -> task record (in flight on a Local actor)
        done = set()
        now = time.perf_counter_ns() - self.t0
        need_enabled_spec = bool(spec.get('quality_need_enabled', False))
        protect_accurate = bool(spec.get('protect_accurate_deadline', False))
        started = time.perf_counter_ns()
        frames_path = self.folder / 'shared_frames.jsonl'
        self._frames_file = open(frames_path, 'w', encoding='utf8')  # buffered
        gc_started, gc_counts, gc_total, gc_max = {}, Counter(), Counter(), Counter()
        gc_long, loop_long = deque(maxlen=64), deque(maxlen=64)
        last_loop = None
        def note_gc(phase, info):
            gen = info['generation']; stamp = time.perf_counter_ns()
            if phase == 'start':
                gc_started[gen] = stamp
            else:
                elapsed = stamp - gc_started.pop(gen, stamp)
                gc_counts[gen] += 1; gc_total[gen] += elapsed
                gc_max[gen] = max(gc_max[gen], elapsed)
                if elapsed >= 5_000_000:
                    gc_long.append({'generation': gen, 'finish_ns': stamp-self.t0,
                                    'wall_ms': elapsed/1e6, 'collected': info.get('collected')})
        gc.callbacks.append(note_gc)
        try:
            while True:
                now = time.perf_counter_ns() - self.t0
                if last_loop is not None and now-last_loop >= 10_000_000:
                    loop_long.append({'finish_ns': now, 'wall_ms': (now-last_loop)/1e6})
                last_loop = now
                if now >= self.duration_s * 1e9 and cursor >= len(plan) and not pending:
                    break
                if (time.perf_counter_ns() - started) > (self.duration_s + 60) * 1e9:
                    raise RuntimeError('shared source drain timeout, preserve evidence')
                budget_until = time.perf_counter_ns() + 1_000_000
                budget = 32
                # ---- inject arrivals (sole injector) ----
                while cursor < len(plan) and plan[cursor]['scheduled_offset_ns'] <= now \
                        and budget > 0 and time.perf_counter_ns() < budget_until:
                    budget -= 1
                    r = plan[cursor]
                    cursor += 1
                    i = r['task_id']
                    arrival_ns = int(r['scheduled_offset_ns'])
                    deadline_abs = arrival_ns + self.deadline_ns
                    self.counters['arrivals'] += 1
                    publish_ns = time.perf_counter_ns() - self.t0
                    base = {
                        'schema': FRAME_SCHEMA, 'outcome': OUTCOME_ARRIVAL, 'task_id': i,
                        'stage': 'ARRIVAL', 'source_publish_ns': publish_ns,
                        'arrival_ns': arrival_ns, 'deadline_ns': deadline_abs,
                        'origin_node': int(r['origin_node']), 'bank_index': int(r['bank_index']),
                        'source': r.get('source'), 'condition': r.get('condition'),
                    }
                    self._publish(bus, dict(base))
                    node = int(r['origin_node'])
                    wait = self._local_queue_wait(node, publish_ns, state)
                    probe = self.costs.local_probe_ns + self.costs.local_ipc_ns
                    minimum = self._frozen_min_continuation(node, False)
                    remaining = deadline_abs - publish_ns
                    if frozen_route is not None:
                        # C1: the admission verdict is frozen; wait/probe/minimum/
                        # remaining stay as audit fields only.
                        verdict = self._frozen_record(i)['initial']
                        if verdict == 'ABSTAIN_EXPIRED':
                            self.counters['expired_before_admission'] += 1
                            self._publish(bus, dict(base, outcome=OUTCOME_ABSTAIN, stage='ABSTAIN',
                                                    reason='expired before initial admission',
                                                    prediction=None, real_start_ns=None,
                                                    real_finish_ns=None))
                            continue
                    elif publish_ns > deadline_abs:
                        self.counters['expired_before_admission'] += 1
                        self._publish(bus, dict(base, outcome=OUTCOME_ABSTAIN, stage='ABSTAIN',
                                                reason='expired before initial admission',
                                                prediction=None, real_start_ns=None,
                                                real_finish_ns=None))
                        continue
                    else:
                        verdict = initial_admission(wait, probe, minimum, remaining)
                    base['local_admission'] = {
                        'wait_ns': int(wait), 'probe_budget_ns': int(probe),
                        'minimum_continuation_ns': int(minimum),
                        'remaining_ns': int(remaining),
                        'source_injection_lag_ns': int(publish_ns - arrival_ns),
                        'decision': verdict}
                    if verdict == 'PROBE':
                        job = {'task_id': i, 'job_id': i, 'kind': 'LOCAL_PROBE',
                               'bank_index': int(r['bank_index']), 'dispatch_ns': publish_ns,
                               'stage': 'PROBE'}
                        self._enqueue_local(state, node, job)
                        pending[i] = dict(base, stage='PROBE', remaining_ns=int(remaining))
                        self.counters['probe'] += 1
                    else:
                        self.counters['prefix'] += 1
                        self.counters['common_rejections'] += 1
                        self._publish(bus, dict(base, outcome=OUTCOME_PREFIX_READY, stage='PREFIX',
                                                reason='initial CPU admission insufficient',
                                                prediction=None, real_start_ns=None,
                                                real_finish_ns=None))
                # ---- local scheduling ----
                for n in range(nodes):
                    if state['busy'][n] is not None or not state['q'][n]:
                        continue
                    j = state['q'][n].popleft()
                    task = pending.get(j['task_id'])
                    if task is None:
                        continue
                    now2 = time.perf_counter_ns() - self.t0
                    if frozen_route is None and now2 > task['deadline_ns']:
                        self.counters['abstain'] += 1
                        self._publish(bus, dict(task, outcome=OUTCOME_ABSTAIN, stage='ABSTAIN',
                                                source_publish_ns=now2,
                                                reason='expired Local queue',
                                                prediction=None, real_start_ns=None,
                                                real_finish_ns=None))
                        pending.pop(j['task_id'], None)
                        self._release_local(state, n)
                        continue
                    cost = (self.costs.local_probe_ns if j['kind'] == 'LOCAL_PROBE'
                            else self.costs.local_mid2_ns if j['kind'] == 'LOCAL_MID2'
                            else max(self.costs.local_suffix_ns.values()))
                    state['busy'][n] = j
                    state['job'][n] = j
                    state['budget_finish'][n] = now2 + cost + self.costs.local_ipc_ns
                    # a fresh probe must not collide with an unreleased continuation
                    if state['held'][n] != j['task_id']:
                        self._release_local(state, n)
                    self.commands['local:%d' % n].put(j)
                # ---- consume local results ----
                for _ in range(64):
                    try:
                        r = self.results.get_nowait()
                    except Exception:
                        break
                    if r['type'] == 'ERROR':
                        raise RuntimeError(r)
                    if r['type'] != 'DONE':
                        raise RuntimeError(r)
                    rid = r['resource']
                    n = int(rid.split(':')[1])
                    j = r['job']
                    i = j['task_id']
                    state['busy'][n] = None
                    state['job'][n] = None
                    if j['kind'] == 'LOCAL_PROBE':
                        if i not in pending:
                            raise RuntimeError('probe result for unknown task %r' % (i,))
                        state['held'][n] = i
                        self._on_probe(bus, pending, state, r, n, need_enabled_spec,
                                       protect_accurate, quality_order)
                    elif j['kind'] == 'LOCAL_MID2':
                        state['held'][n] = i
                        self._on_mid2(bus, pending, state, r, n, need_enabled_spec,
                                      protect_accurate, quality_order)
                    else:
                        state['held'][n] = None
                        self._on_suffix(bus, pending, r, n)
                        done.add(i)
                time.sleep(0.00008)
        finally:
            gc.callbacks.remove(note_gc)
            try:
                self._frames_file.flush()
            except Exception:
                pass
            self._frames_file.close()
            self._frames_file = None
        self.publish_stats = {'frames': self._frames, 'dropped': dict(bus.dropped),
                              'counters': dict(self.counters),
                              'frames_archive': str(frames_path),
                              'frozen_route': ({
                                  'path': self.frozen_route['path'],
                                  'sha256': self.frozen_route['sha256'],
                                  'source_run': self.frozen_route['source_run'],
                                  'arrivals': self.frozen_route['n_tasks'],
                                  'counts': self.frozen_route['counts'],
                              } if self.frozen_route is not None else None),
                              'gc': {'counts': dict(gc_counts), 'total_ns': dict(gc_total),
                                     'max_ns': dict(gc_max), 'long_events': list(gc_long)},
                              'loop_long_gaps': list(loop_long)}
        return dict(self.publish_stats)

    # -- local helpers ---------------------------------------------------
    def _release_local(self, state, node):
        """Drop an unreleased Local probe continuation on one node.

        The N10 Local actor keeps the MID1 probe state (``held``) until a
        LOCAL_SUFFIX consumes it, and it refuses a new LOCAL_PROBE while a
        continuation is still held.  Whenever the shared source abandons a task
        on a node (GPU continuation, abstain, expiry) the hold must be released
        explicitly, otherwise the next probe on that node raises
        ``Local continuation lost``.
        """
        if state['held'][node] is not None:
            self.commands['local:%d' % node].put({'kind': 'RELEASE'})
            state['held'][node] = None

    def _enqueue_local(self, state, node, job):
        q = state['q'][node]
        if job['kind'] in ('LOCAL_SUFFIX', 'LOCAL_MID2'):
            q.appendleft(job)
        else:
            q.append(job)

    def _on_probe(self, bus, pending, state, r, n, need_enabled_spec,
                  protect_accurate, quality_order):
        i = r['job']['task_id']
        task = pending[i]
        now = time.perf_counter_ns() - self.t0
        p = r['probabilities']
        stage = 'MID1'
        enabled = bool(need_enabled_spec)
        order = local_quality_route(quality_order(p, 'A', r['accurate_need_probability'], enabled),
                                    self.spec.get('local_quality_route_mode', 'legacy'))
        task.update({'stage': stage, 'quality_probabilities': p,
                     'accurate_need_probability': r['accurate_need_probability'],
                     'real_start_ns': int(r['start_ns'] - self.t0),
                     'real_finish_ns': int(r['finish_ns'] - self.t0),
                     'cpu_stages': [dict(s, start_ns=int(s['start_ns'] - self.t0),
                                         finish_ns=int(s['finish_ns'] - self.t0))
                                    for s in r.get('stages', [])]})
        # original public MID2 switch, judged with the frozen minimum continuation
        minimum_after_mid2 = self._frozen_min_continuation(n, True)
        remaining = task['deadline_ns'] - now
        afford_mid2 = (self.costs.local_mid2_ns + self.costs.local_ipc_ns
                       + minimum_after_mid2) <= remaining
        early_quality = order[0] if order else 'Abstain'
        self._publish(bus, dict(task, outcome=OUTCOME_QUALITY, stage='MID1',
                                source_publish_ns=now,
                                quality_role=early_quality,
                                quality_probabilities=p,
                                accurate_need_probability=r['accurate_need_probability']))
        if self.frozen_route is not None:
            # C1: the MID2 switch is frozen (the actor output is deterministic
            # under the frozen models, so MID2_candidate must agree; a mismatch
            # is counted and falls back to the single-stage path).
            want_mid2 = self._frozen_record(i)['quality'] == 'MID2'
            do_mid2 = bool(want_mid2 and r.get('MID2_candidate'))
            if want_mid2 and not r.get('MID2_candidate'):
                self.counters['frozen_mid2_mismatch'] = self.counters.get('frozen_mid2_mismatch', 0) + 1
        else:
            guard = bool(protect_accurate and early_quality == 'Accurate')
            do_mid2 = bool(r['MID2_candidate'] and afford_mid2 and not guard)
        if do_mid2:
            self.counters['local_mid2'] += 1
            job = {'task_id': i, 'job_id': r['job']['job_id'], 'kind': 'LOCAL_MID2',
                   'bank_index': task['bank_index'], 'dispatch_ns': now}
            self._enqueue_local(state, n, job)
            return
        self._walk_and_commit(bus, pending, state, task, n, p,
                              r['accurate_need_probability'], order)

    def _on_mid2(self, bus, pending, state, r, n, need_enabled_spec,
                 protect_accurate, quality_order):
        i = r['job']['task_id']
        task = pending[i]
        now = time.perf_counter_ns() - self.t0
        p = r['probabilities']
        order = local_quality_route(quality_order(p, 'A', r['accurate_need_probability'], bool(need_enabled_spec)),
                                    self.spec.get('local_quality_route_mode', 'legacy'))
        task.update({'stage': 'MID2', 'quality_probabilities': p,
                     'accurate_need_probability': r['accurate_need_probability'],
                     'real_start_ns': int(r['start_ns'] - self.t0),
                     'real_finish_ns': int(r['finish_ns'] - self.t0)})
        self._publish(bus, dict(task, outcome=OUTCOME_QUALITY, stage='MID2',
                                source_publish_ns=now,
                                quality_role=(order[0] if order else 'Abstain'),
                                quality_probabilities=p,
                                accurate_need_probability=r['accurate_need_probability']))
        self._walk_and_commit(bus, pending, state, task, n, p,
                              r['accurate_need_probability'], order)

    def _walk_and_commit(self, bus, pending, state, task, n, p, need_probability, order):
        """Decide the continuation INSIDE the shared source (unchanged rules).

        Under C1 frozen routing the committed outcome is read from the frozen
        table; otherwise the original remaining-time ``candidate_walk`` decides."""
        now = time.perf_counter_ns() - self.t0
        i = task['task_id']
        remaining = task['deadline_ns'] - now
        mid2 = task['stage'] == 'MID2'
        local_cost = self.costs.local_suffix_ns[mid2] + self.costs.local_ipc_ns

        if self.frozen_route is not None:
            record = self._frozen_record(i)
            kind = record['terminal']
            gpu_roles = list(record.get('gpu_candidates') or [])
            abstain_reason = record.get('abstain_reason')
        else:
            def local_fn(_role):
                return local_cost

            def gpu_fn(role):
                gkind = 'ACCURATE' if role == 'Accurate' else 'FAST_FULL'
                return gpu_intrinsic_frozen(gkind, self.costs.common_intrinsic_ms,
                                            self.costs.gpu_ipc_ns)

            chosen, gpu_roles = candidate_walk(order, local_fn, gpu_fn, remaining)
            if chosen == 'Local':
                kind = 'LOCAL'
            elif gpu_roles:
                kind = 'GPU_READY'
            else:
                kind = 'ABSTAIN'
            abstain_reason = 'no feasible Local or GPU continuation'

        if kind == 'LOCAL':
            self.counters['local_suffix'] += 1
            job = {'task_id': i, 'job_id': task.get('job_id', i), 'kind': 'LOCAL_SUFFIX',
                   'bank_index': task['bank_index'], 'dispatch_ns': now}
            self._enqueue_local(state, n, job)
            return
        if kind == 'GPU_READY' and gpu_roles:
            self.counters['gpu_ready'] += 1
            kinds = {r: ('ACCURATE' if r == 'Accurate' else 'FAST_FULL') for r in gpu_roles}
            self._publish(bus, dict(task, outcome=OUTCOME_GPU_READY, stage='GPU_CONT',
                                    source_publish_ns=now, reason='GPU continuation',
                                    gpu_candidates=list(gpu_roles), gpu_candidate_kinds=kinds,
                                    prediction=None))
            pending.pop(i, None)
            self._release_local(state, n)
            return
        self.counters['abstain'] += 1
        self._publish(bus, dict(task, outcome=OUTCOME_ABSTAIN, stage='ABSTAIN',
                                source_publish_ns=now,
                                reason=abstain_reason or 'no feasible Local or GPU continuation',
                                prediction=None))
        pending.pop(i, None)
        self._release_local(state, n)

    def _on_suffix(self, bus, pending, r, n):
        i = r['job']['task_id']
        task = pending.pop(i, None)
        if task is None:
            raise RuntimeError('suffix result for unknown task %r' % (i,))
        self.counters['local_terminal'] += 1
        now = time.perf_counter_ns() - self.t0
        # T012 C3: the four delivery endpoints of a common Local terminal.
        # ``terminal_available_ns`` is the COMMON SOURCE parent-side instant, so
        # BOTH arms observe the identical value (the deadline verdict must not
        # depend on which arm happens to consume the frame).  The compute/actor
        # end is the real Local actor finish; the source publish instant is the
        # parent-availability instant.
        actor_finish_ns = int(r['finish_ns'] - self.t0)
        self._publish(bus, dict(task, outcome=OUTCOME_LOCAL_TERMINAL, stage='LOCAL_TERMINAL',
                                source_publish_ns=now, reason='common Local terminal',
                                prediction=int(r['prediction']),
                                real_start_ns=int(r['start_ns'] - self.t0),
                                real_finish_ns=actor_finish_ns,
                                compute_end_ns=actor_finish_ns,
                                actor_finish_ns=actor_finish_ns,
                                parent_received_ns=now,
                                terminal_available_ns=now))


def load_plan(spec):
    """Read the frozen arrival plan exactly as the runtime does."""
    import hashlib
    raw = Path(spec['input_plan_path']).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if spec.get('input_plan_sha256') and digest != spec['input_plan_sha256']:
        raise ValueError('frozen input plan file SHA mismatch: %s' % digest)
    plan = json.loads(raw.decode('utf8'))
    logical = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':'))
                             .encode()).hexdigest()
    if spec.get('expected_plan_sha256') and logical != spec['expected_plan_sha256']:
        raise ValueError('frozen input plan logical SHA mismatch: %s' % logical)
    if spec.get('expected_arrivals') is not None and len(plan) != int(spec['expected_arrivals']):
        raise ValueError('frozen input plan arrivals %d != %s' % (len(plan), spec['expected_arrivals']))
    return plan, digest, logical


def source_summary_path(folder):
    return Path(folder) / 'shared_source_summary.json'
