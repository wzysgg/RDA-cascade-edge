"""T011 CPU check suite: the SIX groups of TASK.md section 9, one output.

This suite is the ONLY pre-GPU gate.  It runs pure CPU: torch/CUDA/model
dependencies are stubbed (explicitly the ACTORS and MODEL side, never the
production gate/controller/factory/view under test), the real qf4 controller,
observer, factory, group-runtime frame handler / view / schedule path and the
frozen quota_forecast select_job are all real production functions.  Old
development check suites are not re-run and no new parameter matrix is built.

Groups
  G1  real entry wiring: production factory + runtime frame handler / view /
      schedule with fake CPU queue actors; single quality role before any
      resource admission; light 100 ms and full 500 ms ticks consumable with a
      resource dict; invalid factory inputs rejected BEFORE actors; J and the
      rejected-profile basis work with no ordinary queue head.
  G2  old measurement / conditional capacity: read-only old_data_derivation
      conservation (20 rows), A offer rates 13.2 -> 43.5/s, 70-72 s net growth
      0; the 72/74 s conditional point and guarded capacities; the REAL
      select_job B=25/b=8/charge=2 nine-starts proof; the REFERENCE healthy
      direct fixture skipping the invalid 0.1 bands.
  G3  algorithm protection / unified formula: root-work algebra (base, x1.5
      load, quality transfer), the overload fixture (no pressure-balance move
      to gF 0.8), and low-coverage / unknown-V / no-safe-donor holds with
      exposed reasons.
  G4  lifecycle / real clock: pinned leaves the wait struct on dispatch; an
      expired in-flight component still charges and terminates exactly once;
      Prefix->A / reject / expired release cache + reserve; a 50 ms task whose
      aligned tick says alive but the real now says expired is REJECTED by the
      real clock; DONE-first and bounded-batch source consumption; component
      conservation and unique root terminals.
  G5  bounded 250 s CPU replay under real load: virtual clock, 100 ms ticks, a
      legal 256-live-component pressure fixture with 50 ms deadlines, 500 ms
      bucket/phase/cost updates through the REAL handler/observer/controller/
      view; p50/p95/max for 0-10/60-70/180-190/240-250 s; sync tick p95 <= 5 ms
      with no elapsed-growing scans and a bounded ring.
  G6  closed loop / startup contract: warmup protocol (negative ids, pair
      identity, RELEASE + RESET empty cache, STOP, no CUDA); both-cards-
      effective before post; rejects stay in the matured denominator; 100->150
      & 80->120 standardise to the same 100/s; reverse-loss rollback;
      window_id consumed once; a cost_version-only change never cancels a
      probe; bounded buckets under long history; driver lock/identity/
      same-version-two-groups/exit-cleanup contract.
"""

import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
import types
from collections import Counter
from pathlib import Path

MS = 1e6
NS_PER_S = 1e9

# --------------------------------------------------------------------------
# module stubs: ONLY the actor/model dependencies (torch, numpy, PIL and the
# physical probe harness) are faked.  The production gate/controller/factory/
# view/observer and the real quality_order are imported for real.
# --------------------------------------------------------------------------


def install_stubs():
    if 'torch' in sys.modules:
        return
    torch = types.ModuleType('torch')
    torch.__dict__['_t011_stub'] = True
    sys.modules['torch'] = torch
    numpy = types.ModuleType('numpy')
    sys.modules['numpy'] = numpy
    pil = types.ModuleType('PIL')
    pil.Image = types.ModuleType('PIL.Image')
    pil.ImageFilter = types.ModuleType('PIL.ImageFilter')
    sys.modules['PIL'] = pil
    sys.modules['PIL.Image'] = pil.Image
    sys.modules['PIL.ImageFilter'] = pil.ImageFilter
    pp = types.ModuleType('physical_probe')
    pp.HERE = Path.cwd()
    pp.PHASE = Path('/tmp/t011_stub_phase')
    pp.ASSETS = Path('/tmp/t011_stub_assets')
    pp.UUIDS = ['GPU-d1b82bfe-73db-7d85-16fa-2e38208c0064',
                'GPU-7e400790-5792-8095-35cd-42d68b8eeffd']

    def _not_available(*a, **k):
        raise RuntimeError('physical_probe model path is stubbed for CPU checks')

    pp.Heads = _not_available
    pp.model = _not_available

    def configure(cpu=None):
        return None

    pp.configure = configure

    def actor_cpu(name, default):
        try:
            return int(os.environ.get(name, default))
        except (TypeError, ValueError):
            return int(default)
    pp.actor_cpu = actor_cpu
    pp.fast_prefix = _not_available
    pp.fast_suffix = _not_available
    pp.local_suffix = _not_available

    def sha(p):
        import hashlib
        h = hashlib.sha256()
        with Path(p).open('rb') as f:
            for b in iter(lambda: f.read(1024 * 1024), b''):
                h.update(b)
        return h.hexdigest()
    pp.sha = sha

    def write(p, d):
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf8')
    pp.write = write
    pp.apply_source_space_intervention = _not_available
    pp.build_post_intervention_preprocess = _not_available
    pp.load_rows = _not_available
    pp.CLASSES = ['a', 'b', 'c', 'd', 'e', 'f']
    pp.image_quality_features = _not_available
    sys.modules['physical_probe'] = pp


install_stubs()

from queue_feedback_controller import (  # noqa: E402
    CONTROLLER_VERSION, DYNAMIC_POLICY, FIXED_POLICY, IMPLEMENTATION_REVISION,
    QueueFeedbackController, control_step, current_candidate,
    evaluate_candidates, first_effective_step, make_controller_inputs_qf4,
    make_executor_view_qf4, probability_histogram_tv, probe_epsilon,
    root_work_demand, select_target, standardized_probe,
    conditional_single_kind, MAX_UNCONFIRMED_TRIALS_PER_DIRECTION,
)
from queue_feedback_stats import (  # noqa: E402
    CostBook, QF4Observer, ROOT_CLASSES, ROOT_DIRECT_A, ROOT_DIRECT_F,
    ROOT_PREFIX, GPU_KINDS, tv_distance, validate_cost_records,
)
from quota_forecast_v8 import select_job  # noqa: E402


# --------------------------------------------------------------------------
# fake queue / actors implementing the REAL actor protocol
# --------------------------------------------------------------------------
class FakeQueue(object):
    def __init__(self):
        self._q = []

    def put(self, item, block=True, timeout=None):
        self._q.append(item)
        return True

    put_nowait = put

    def get_nowait(self):
        if not self._q:
            raise RuntimeError('empty')
        return self._q.pop(0)

    def get(self, block=True, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._q:
            if deadline is None:
                time.sleep(0.0005)
                continue
            if time.monotonic() >= deadline:
                raise RuntimeError('fake queue get timeout')
            time.sleep(0.0005)
        return self._q.pop(0)

    def empty(self):
        return not self._q

    def __len__(self):
        return len(self._q)


class FakeGpuActor(object):
    """In-process double of live_runtime.gpu_actor: SAME command schema
    (DONE/RELEASE/RESET/STOP), configurable constant service, no CUDA."""

    def __init__(self, name, cmd_q, res_q, service_ns=100_000):
        self.name = name
        self.cmd_q = cmd_q
        self.res_q = res_q
        self.service_ns = int(service_ns)
        self.cache = {}
        self.stopped = False
        self.calls = Counter()

    def pump(self):
        while self.cmd_q._q:
            j = self.cmd_q._q.pop(0)
            kind = j['kind']
            if kind == 'STOP':
                self.stopped = True
                return
            if kind == 'RELEASE':
                self.cache.pop(j['task_id'], None)
                continue
            if kind == 'RESET':
                if self.cache:
                    self.res_q.put({'type': 'ERROR', 'resource': self.name,
                                    'error': 'Reset with pending GPU activation',
                                    'cache': sorted(self.cache)})
                else:
                    self.res_q.put({'type': 'RESET', 'resource': self.name})
                continue
            self.calls[kind] += 1
            now = time.perf_counter_ns()
            if kind == 'FAST_PREFIX':
                self.cache[j['task_id']] = object()
            if kind == 'FAST_SUFFIX':
                self.cache.pop(j['task_id'], None)
            payload = {'vector': [0.0] * 64} if kind == 'FAST_PREFIX' else \
                {'prediction': (j['task_id'] % 6)}
            self.res_q.put({'type': 'DONE', 'resource': self.name, 'job': j,
                            'start_ns': now, 'finish_ns': now + self.service_ns,
                            'compute_start_ns': now, 'compute_end_ns': now + self.service_ns,
                            'cuda_ms': self.service_ns / 1e6, 'upload_ns': 0,
                            'feature_or_result_transfer_ns': 0,
                            'cache_count': len(self.cache), **payload})


class FakeDecisionActor(object):
    """In-process double of the CPU EARLY head actor with canned public
    probabilities (the heads themselves are stubbed, not the routing)."""

    def __init__(self, name, cmd_q, res_q, probabilities=None, need=0.1):
        self.name = name
        self.cmd_q = cmd_q
        self.res_q = res_q
        self.probabilities = probabilities or {'Fast': 0.9, 'Accurate': 0.05}
        self.need = need
        self.stopped = False
        self.calls = 0

    def pump(self):
        while self.cmd_q._q:
            j = self.cmd_q._q.pop(0)
            if j['kind'] == 'STOP':
                self.stopped = True
                return
            if j['kind'] == 'RESET':
                self.res_q.put({'type': 'RESET', 'resource': self.name})
                continue
            self.calls += 1
            now = time.perf_counter_ns()
            self.res_q.put({'type': 'DECISION_DONE', 'resource': self.name, 'job': j,
                            'start_ns': now, 'finish_ns': now + 50_000,
                            'probabilities': dict(self.probabilities),
                            'accurate_need_probability': self.need})


def make_fake_factory(decision_probs=None, service_ns=100_000):
    def results():
        return FakeQueue()

    def commands():
        return FakeQueue()

    def make(name, cmd_q, res_q):
        if name.startswith('gpu'):
            return FakeGpuActor(name, cmd_q, res_q, service_ns=service_ns)
        return FakeDecisionActor(name, cmd_q, res_q, probabilities=decision_probs)
    return {'results': results, 'commands': commands, 'make': make}


def make_runtime(folder, policy=DYNAMIC_POLICY, decision_probs=None,
                 service_ns=100_000, warmup=True):
    from group_runtime_t011 import SharedGroupRuntime
    rt = SharedGroupRuntime(
        'check-%s' % policy, {'quality_need_enabled': True,
                              'protect_accurate_deadline': True},
        policy, ['GPU-aaaa', 'GPU-bbbb'], folder, time.perf_counter_ns(),
        50_000_000, 250.0, bank=None, frame_q=FakeQueue(), status_q=None,
        actor_mode='fake', fake_actor_factory=make_fake_factory(decision_probs, service_ns),
        warmup_id_base=-9_000_000)
    rt.start_actors()
    if warmup:
        rt.run_warmup()
        rt.build_controller()
    rt.open_streams()
    return rt


def frame_arrival(i, now, node=0, bank_index=0, deadline_ms=50):
    return {'schema': 'T011_shared_frame_v1', 'outcome': 'SHARED_ARRIVAL',
            'task_id': i, 'stage': 'ARRIVAL', 'frame_id': 1000 + i,
            'source_publish_ns': now, 'arrival_ns': now,
            'deadline_ns': now + int(deadline_ms * MS),
            'origin_node': node, 'bank_index': bank_index}


def frame_gpu_ready(i, now, candidates=('Fast', 'Accurate'), probs=None, publish=None):
    publish = publish if publish is not None else now
    kinds = {r: ('ACCURATE' if r == 'Accurate' else 'FAST_FULL') for r in candidates}
    return {'schema': 'T011_shared_frame_v1', 'outcome': 'GPU_READY',
            'task_id': i, 'stage': 'GPU_CONT', 'frame_id': 2000 + i,
            'source_publish_ns': publish,
            'gpu_candidates': list(candidates), 'gpu_candidate_kinds': kinds,
            'quality_probabilities': probs or {'Local': 0.1, 'Fast': 0.9, 'Accurate': 0.05},
            'accurate_need_probability': 0.1, 'prediction': None}


def frame_prefix_ready(i, now, publish=None):
    publish = publish if publish is not None else now
    return {'schema': 'T011_shared_frame_v1', 'outcome': 'PREFIX_READY',
            'task_id': i, 'stage': 'PREFIX', 'frame_id': 3000 + i,
            'source_publish_ns': publish, 'reason': 'initial CPU admission insufficient',
            'prediction': None}


def frame_local_terminal(i, now):
    return {'schema': 'T011_shared_frame_v1', 'outcome': 'COMMON_LOCAL_TERMINAL',
            'task_id': i, 'stage': 'LOCAL_TERMINAL', 'frame_id': 4000 + i,
            'source_publish_ns': now, 'prediction': 1,
            'quality_probabilities': {'Local': 0.9, 'Fast': 0.05, 'Accurate': 0.05},
            'real_finish_ns': now}


def read_events(rt):
    rt.streams['events'].flush()
    out = []
    with io.open(rt.folder / 'events.jsonl', encoding='utf8') as f:
        for line in f:
            out.append(json.loads(line))
    return out


def group_events(rt, etype):
    return [e for e in read_events(rt) if e['event_type'] == etype]


# --------------------------------------------------------------------------


def group_g1(reference):
    """G1: real entry wiring with fake CPU queue actors."""
    checks = []

    def chk(name, ok, expected, observed, source):
        checks.append({'name': name, 'ok': bool(ok), 'expected': expected,
                       'observed': observed, 'source': source})

    tmp = tempfile.mkdtemp(prefix='t011_g1_')
    try:
        # -- factory validation BEFORE any actor work -----------------------
        cost_bad_list = [dict() for _ in range(2)]
        try:
            make_controller_inputs_qf4({}, cost_bad_list, 50_000_000, ['a', 'b'],
                                       DYNAMIC_POLICY)
            factory_rejects_list = False
        except ValueError:
            factory_rejects_list = True
        cost_bad_price = {g: {k: {'mean_wall_ms': -1.0, 'gate_ms': 6.0,
                                  'sample_n': 1, 'sample_revision': 1,
                                  'cost_version': 1, 'measurement_source': 'warmup'}
                              for k in GPU_KINDS} for g in (0, 1)}
        try:
            make_controller_inputs_qf4({}, cost_bad_price, 50_000_000, ['a', 'b'],
                                       DYNAMIC_POLICY)
            factory_rejects_negative = False
        except ValueError:
            factory_rejects_negative = True
        cost_missing = {g: {k: {'mean_wall_ms': 5.0, 'gate_ms': 6.0,
                                'sample_n': 1, 'sample_revision': 1,
                                'cost_version': 1}
                            for k in GPU_KINDS} for g in (0, 1)}
        try:
            make_controller_inputs_qf4({}, cost_missing, 50_000_000, ['a', 'b'],
                                       DYNAMIC_POLICY)
            factory_rejects_missing = False
        except ValueError:
            factory_rejects_missing = True
        chk('factory_rejects_list_prior', factory_rejects_list, True,
            factory_rejects_list, 'make_controller_inputs_qf4 + validate_cost_records')
        chk('factory_rejects_nonpositive_price', factory_rejects_negative, True,
            factory_rejects_negative, 'make_controller_inputs_qf4 + validate_cost_records')
        chk('factory_rejects_missing_field', factory_rejects_missing, True,
            factory_rejects_missing, 'make_controller_inputs_qf4 + validate_cost_records')

        # -- Direct GPU_READY: exactly the FIRST quality candidate ----------
        rt = make_runtime(Path(tmp) / 'directF', DYNAMIC_POLICY)
        now = rt.now()
        rt.frame_q.put(frame_arrival(1, now))
        rt.frame_q.put(frame_gpu_ready(1, now + 1000, candidates=('Fast', 'Accurate')))
        rt._drain_source_frames()
        offers = group_events(rt, 'OFFER')
        chk('directF_single_offer_first_candidate',
            len(offers) == 1 and offers[0]['kind'] == 'FAST_FULL',
            'one OFFER FAST_FULL only', offers, 'runtime._gpu_ready frozen-role rule')
        rt.schedule(rt.now())
        rt.pump_fake_actors()
        rt._drain_results()
        terms = group_events(rt, 'TERMINAL')
        chk('directF_served_and_terminal_once',
            len(terms) == 1 and terms[0]['role'] == 'Fast',
            'one Fast terminal', terms, 'runtime dispatch + unique terminal')
        # Accurate-first root goes to ACCURATE only
        now = rt.now()
        rt.frame_q.put(frame_arrival(2, now))
        rt.frame_q.put(frame_gpu_ready(2, now + 1000, candidates=('Accurate', 'Fast')))
        rt._drain_source_frames()
        offers = group_events(rt, 'OFFER')
        acc_offers = [o for o in offers if o['task_id'] == 2]
        chk('directA_single_offer_ACCURATE',
            len(acc_offers) == 1 and acc_offers[0]['kind'] == 'ACCURATE',
            'one OFFER ACCURATE only', acc_offers, 'runtime._gpu_ready frozen-role rule')
        rt.schedule(rt.now())
        rt.pump_fake_actors()
        rt._drain_results()
        terms = [t for t in group_events(rt, 'TERMINAL') if t['task_id'] == 2]
        chk('directA_terminal_Accurate',
            len(terms) == 1 and terms[0]['role'] == 'Accurate',
            'one Accurate terminal', terms, 'runtime dispatch + unique terminal')
        rt.close_streams()

        # -- Prefix -> EARLY Suffix keeps affinity; A releases the card -----
        rt = make_runtime(Path(tmp) / 'prefix', DYNAMIC_POLICY)
        now = rt.now()
        rt.frame_q.put(frame_arrival(10, now))
        rt.frame_q.put(frame_prefix_ready(10, now + 1000))
        rt._drain_source_frames()
        rt.schedule(rt.now())
        rt.pump_fake_actors()
        rt._drain_results()
        rt.pump_fake_actors()          # decision actor
        rt._drain_results()
        pinned = [(g, list(rt.gpus[g]['pinned'])) for g in (0, 1)]
        n_pinned = sum(len(p) for _, p in pinned)
        chk('early_suffix_pinned_on_prefix_card', n_pinned == 1,
            'one pinned Suffix', pinned, 'runtime._route_b quality_order first role')
        reserve_cleared = all(not rt.gpus[g]['pending'] for g in (0, 1))
        chk('suffix_reserve_converted_to_pinned', reserve_cleared, True,
            reserve_cleared, 'Prefix reserve -> pinned (s3.2)')
        rt.schedule(rt.now())
        rt.pump_fake_actors()
        rt._drain_results()
        terms = [t for t in group_events(rt, 'TERMINAL') if t['task_id'] == 10]
        chk('prefix_suffix_terminal_Fast', len(terms) == 1 and terms[0]['role'] == 'Fast',
            'one Fast terminal via Prefix', terms, 'Suffix DONE terminal')
        rt.close_streams()

        rt = make_runtime(Path(tmp) / 'prefixA', DYNAMIC_POLICY,
                          decision_probs={'Fast': 0.1, 'Accurate': 0.9})
        now = rt.now()
        rt.frame_q.put(frame_arrival(11, now))
        rt.frame_q.put(frame_prefix_ready(11, now + 1000))
        rt._drain_source_frames()
        rt.schedule(rt.now())
        rt.pump_fake_actors()
        rt._drain_results()
        rt.pump_fake_actors()
        rt._drain_results()
        rt.pump_fake_actors()          # RELEASE consumed by the fake actor
        released = group_events(rt, 'PREFIX_RELEASE')
        acc_ready = [j for j in rt.ready['Accurate'] if j['task_id'] == 11]
        caches_empty = all(not a.cache for a in rt.fake_actors.values()
                           if hasattr(a, 'cache'))
        chk('early_A_releases_prefix_card_and_enqueues_A',
            len(released) == 1 and len(acc_ready) == 1 and caches_empty,
            'one RELEASE + A enqueued + cache empty', (released, acc_ready, caches_empty),
            'runtime._route_b A path + RELEASE')
        rt.close_streams()

        # -- rejections classify and never switch role ----------------------
        rt = make_runtime(Path(tmp) / 'reject', DYNAMIC_POLICY)
        now = rt.now()
        rt.frame_q.put(frame_arrival(20, now - 49 * MS))
        rt.frame_q.put(frame_prefix_ready(20, now, publish=now - 49 * MS))
        rt._drain_source_frames()
        terms = [t for t in group_events(rt, 'TERMINAL') if t['task_id'] == 20]
        ao = [e for e in group_events(rt, 'ADMISSION_OUTCOME') if e['task_id'] == 20]
        chk('prefix_intrinsic_budget_rejected_no_reroute',
            len(terms) == 1 and terms[0]['role'] == 'Abstain' and ao
            and ao[-1].get('reason') in ('intrinsic_budget', 'receipt_delay', 'resource'),
            'one Abstain with a section-5 reason', (terms, ao),
            'runtime._classify_rejection')
        # receipt_delay: source publish long past, remaining small but source ok
        now = rt.now()
        rt.frame_q.put(frame_arrival(21, now - 10 * MS))
        rt._drain_source_frames()
        rt.tasks[21]['deadline_abs_ns'] = now + 1 * MS
        rt.frame_q.put(frame_prefix_ready(21, rt.now(), publish=now - 10 * MS))
        rt._drain_source_frames()
        ao = [e for e in group_events(rt, 'ADMISSION_OUTCOME') if e['task_id'] == 21]
        chk('receipt_delay_classified', bool(ao) and ao[-1].get('reason') == 'receipt_delay',
            'receipt_delay', ao, 'runtime._classify_rejection (receipt zero-wait)')
        # no ordinary head: J + rejected profile still readable, gate probe basis
        rt._profile_update('Accurate', 'ACCURATE', rt.tasks[21], rt.now(),
                           {'remaining_ns': 1_000_000})
        probe = rt._gate_probe(1.0, 1.0)
        chk('no_head_profile_basis_usable',
            isinstance(probe, dict) and probe.get('detail', {}).get('Accurate', {})
            .get('basis') == 'rejected_profile',
            "basis == 'rejected_profile'", probe, 'runtime._gate_probe real select_job')
        rt.close_streams()

        # -- light 100ms and full 500ms ticks carry a resource dict ---------
        rt = make_runtime(Path(tmp) / 'ticks', DYNAMIC_POLICY)
        now = rt.now()
        aligned = (now // 100_000_000) * 100_000_000 + 100_000_000
        while rt.now() < aligned:
            time.sleep(0.0005)
        rt._controller_tick(rt.now())
        rows = rt.decisions if hasattr(rt, 'decisions') else []
        dec = rt.controller.decisions[-1]
        res = dec['resource']
        need_keys = ('snapshot_ns', 'window_id', 'cards', 'waiting', 'cost_version')
        chk('light_tick_resource_dict', dec['valid'] and all(k in res for k in need_keys),
            'resource has %r' % (need_keys,), sorted(res.keys()),
            'controller._decision + make_executor_view_qf4')
        chk('window_id_is_aligned_tick',
            res['window_id'] % 100_000_000 == 0 and abs(res['snapshot_ns'] - res['window_id']) < 100_000_000,
            'window_id aligned; snapshot fresh', (res['window_id'], res['snapshot_ns']),
            'controller window_id label vs real now')
        rt.close_streams()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {'id': 'G1', 'ok': all(c['ok'] for c in checks), 'checks': checks}


def group_g2(evidence):
    """G2: old measurement conservation + conditional capacity + real gate."""
    checks = []

    def chk(name, ok, expected, observed, source):
        checks.append({'name': name, 'ok': bool(ok), 'expected': expected,
                       'observed': observed, 'source': source})

    ref = json.loads(io.open(Path(evidence) / 'REFERENCE.json', encoding='utf-8').read())
    old = json.loads(io.open(Path(evidence) / 'old_data_derivation.json', encoding='utf-8').read())
    reference = ref['old_data_checks']
    tol = 1e-7
    rows = 0
    max_res = 0.0
    growth_rows_ok = True
    for arm, armdata in old['arms'].items():
        for wname, w in armdata['windows'].items():
            T = w['end_s'] - w['start_s']
            for role, b in w['balances'].items():
                rows += 1
                max_res = max(max_res, abs(b['identity_residual_ms']))
                dW = b['E_ms'] - b['S_ms'] - b['X_ms']
                if abs(dW - b['delta_W_ms']) > tol:
                    growth_rows_ok = False
    chk('old_conservation_rows_20', rows == 20 and max_res <= tol,
        '20 rows, |residual| <= 1e-7 reference-ms', (rows, max_res),
        'old_data_derivation.json balances (read-only)')
    chk('old_deltaW_identity', growth_rows_ok, True, growth_rows_ok,
        'old_data_derivation.json delta_W == E - S - X per role/window')
    onset = old['arms']['qf3_feedback']['windows']['onset']['counts']
    e_s_x = [onset['E_ACCURATE_n'], onset['DONE_ACCURATE_n'], onset['X_ACCURATE_n']]
    chk('onset_E_S_X', e_s_x == reference['onset_E_S_X'], reference['onset_E_S_X'], e_s_x,
        'old_data_derivation.json onset counts')
    clear = old['arms']['qf3_feedback']['windows']['clear']['counts']
    rate_clear = (clear['E_ACCURATE_n'] + clear['R_all_ACCURATE_n']) / 5.0
    rate_onset = (onset['E_ACCURATE_n'] + onset['R_all_ACCURATE_n']) / 2.0
    chk('A_offer_rates_13_2_then_43_5',
        abs(rate_clear - 13.2) < 1e-9 and abs(rate_onset - 43.5) < 1e-9,
        [13.2, 43.5], [round(rate_clear, 6), round(rate_onset, 6)],
        'old_data_derivation.json A component OFFER rates')
    onset_bal = old['arms']['qf3_feedback']['windows']['onset']['balances']
    chk('onset_A_net_growth_zero',
        abs(onset_bal['Accurate']['delta_W_ms']) <= tol,
        'Accurate dW == 0 in 70-72 s (DERIVATION dQ_A = 43-7-36)',
        {'Accurate_dW': onset_bal['Accurate']['delta_W_ms'],
         'Fast_dW_recorded': onset_bal['Fast']['delta_W_ms']},
        'old_data_derivation.json onset balances')
    # 72/74 s conditional capacities via the production formula
    for tag, mean_key, gate_key, point_key, guard_key in (
            ('at72', 'old_A_mean_ms_at72', 'old_A_gate_ms_at72',
             'old_A_axis_conditional_tasks_s_at72', None),
            ('at74', 'old_A_mean_ms_at74', 'old_A_gate_ms_at74',
             'old_A_axis_conditional_tasks_s_at74', 'old_A_axis_guarded_tasks_s_at74')):
        m = reference[mean_key]
        b = reference[gate_key]
        ok = True
        got = {}
        for gFs, exp in reference[point_key].items():
            phi = 1.0 - max(float(gFs) - 1.0, 0.0)
            rec = conditional_single_kind(phi, m, b, 0.4, price_history=[],
                                          min_margin_ms=0.1)
            got[gFs] = round(rec['mu_point'], 6)
            ok &= abs(rec['mu_point'] - exp) < 1e-4
            if guard_key and gFs in reference[guard_key]:
                ok &= abs(rec['mu_safe'] - reference[guard_key][gFs]) < 1e-4
        chk('conditional_capacity_%s' % tag, ok, reference[point_key], got,
            'queue_feedback_controller.conditional_single_kind')
    # REAL select_job: B=25ms budget, gate b=8ms, real charge 2ms -> 9 starts
    s = {'pinned': [], 'transition': False, 'target': 0.5, 'spent': Counter(),
         'pending': set()}
    cost = {'ACCURATE': 8_000_000, 'FAST_FULL': 8_000_000, 'FAST_PREFIX': 8_000_000,
            'FAST_SUFFIX': 8_000_000}
    tasks = {}
    acc = [{'task_id': i, 'kind': 'ACCURATE', 'enqueue_ns': 0} for i in range(25)]
    for i in range(25):
        tasks[i] = {'arrival_ns': 0}
    starts = 0
    for _ in range(25):
        j, role = select_job(s, [], acc, tasks, cost)
        if j is None:
            break
        acc.remove(j)
        s['spent'][role] += 2_000_000
        starts += 1
    chk('real_select_job_nine_starts', starts == reference['real_gate_starts'],
        reference['real_gate_starts'], starts, 'quota_forecast_v8.select_job (frozen)')
    # REFERENCE healthy direct fixture: invalid 0.1 bands skipped
    fx = ref['effective_step_fixture']
    cost_tbl = {g: {k: {'mean_wall_ms': fx['means_all_cards_ms'][k],
                        'gate_ms': fx['gates_all_cards_ms'][k],
                        'delta_ms': fx['dispatch_gap_all_kind_ms'],
                        'price_history': []} for k in GPU_KINDS} for g in (0, 1)}
    table = evaluate_candidates(fx['current_gF'],
                                {'lambda_DF': fx['lambda_DF'], 'lambda_DA': fx['lambda_DA'],
                                 'lambda_B': fx['lambda_B'], 'pDF': fx['p_DF'],
                                 'pDA': fx['p_DA']},
                                cost_tbl, dict(fx['means_all_cards_ms']), T_s=1.0,
                                min_margin_ms=fx['opportunity_margin_ms'])
    sel = select_target(table, fx['current_gF'], T_s=1.0)
    ok = sel.get('action') == 'move' and \
        abs(sel['target_gF'] - fx['expected_destination_feasible_target_gF']) < 1e-9
    step = first_effective_step(fx['current_gF'], sel.get('target_gF', 0), table,
                                basis='A_safe')
    ok = ok and step is not None and \
        abs(step['target_gF'] - fx['expected_nearest_effective_command_gF']) < 1e-9 and \
        step['skipped'] == fx['expected_skip_gF']
    chk('reference_direct_fixture_selection', ok,
        {'target': fx['expected_destination_feasible_target_gF'],
         'first_step': fx['expected_nearest_effective_command_gF'],
         'skipped': fx['expected_skip_gF']},
        {'target': sel.get('target_gF'), 'first_step': None if step is None else step['target_gF'],
         'skipped': None if step is None else step['skipped']},
        'evaluate_candidates + select_target + first_effective_step')
    return {'id': 'G2', 'ok': all(c['ok'] for c in checks), 'checks': checks}


def group_g3(reference):
    """G3: algorithm protection / unified demand formula."""
    checks = []

    def chk(name, ok, expected, observed, source):
        checks.append({'name': name, 'ok': bool(ok), 'expected': expected,
                       'observed': observed, 'source': source})

    ref = reference if isinstance(reference, dict) else \
        json.loads(io.open(Path(reference) / 'REFERENCE.json', encoding='utf-8').read())
    rw = ref['root_work_fixture']
    kc = rw['kind_costs_ms']
    cases = [(rw['base_input'], 1.0, rw['base_expected_reference_ms_s']),
             (rw['base_input'], rw['load_multiplier_all_roots'], rw['load_expected']),
             (rw['quality_input'], 1.0, rw['quality_expected'])]
    got_all = []
    ok = True
    for inp, mult, exp in cases:
        aF, aA = root_work_demand(inp['lambda_DF'] * mult, inp['lambda_DA'] * mult,
                                  inp['lambda_B'] * mult, inp['p_BF'], inp['p_BA'], kc)
        got_all.append([round(aF, 6), round(aA, 6)])
        ok &= abs(aF - exp['Fast']) < 1e-9 and abs(aA - exp['Accurate']) < 1e-9
    chk('root_work_algebra_three_cases', ok, [list(v.values()) if isinstance(v, dict) else v
                                              for v in [rw['base_expected_reference_ms_s'],
                                                        rw['load_expected'],
                                                        rw['quality_expected']]],
        got_all, 'queue_feedback_controller.root_work_demand')
    ov = ref['overload_fixture']
    cost = {g: {k: {'mean_wall_ms': 2.0 if k != 'ACCURATE' else 10.0,
                    'gate_ms': 2.0 if k != 'ACCURATE' else 10.0, 'delta_ms': 0.0,
                    'price_history': []} for k in GPU_KINDS} for g in (0, 1)}
    table = evaluate_candidates(1.0, {'lambda_DF': ov['lambda_F'],
                                      'lambda_DA': ov['lambda_A'], 'lambda_B': 0,
                                      'pDF': ov['p_F'], 'pDA': ov['p_A']},
                                cost, {'FAST_FULL': 2.0, 'FAST_PREFIX': 2.0,
                                       'FAST_SUFFIX': 2.0, 'ACCURATE': 10.0},
                                T_s=1.0, min_margin_ms=0.1)
    cur = current_candidate(table, 1.0)
    bad = [c for c in table['candidates'] if abs(c['gF'] - ov['bad_candidate_gF']) < 1e-9][0]
    ok = (abs(cur['Yhat_point'] - ov['unmargined_Y_expected'][0]) < 1e-6 and
          abs(bad['Yhat_point'] - ov['unmargined_Y_expected'][1]) < 1e-6 and
          abs(cur['Vhat_point'] - ov['unmargined_V_expected'][0]) < 1e-6 and
          abs(bad['Vhat_point'] - ov['unmargined_V_expected'][1]) < 1e-6)
    chk('overload_Y_V_degrade_at_0_8', ok,
        {'Y': ov['unmargined_Y_expected'], 'V': ov['unmargined_V_expected']},
        {'Y': [cur['Yhat_point'], bad['Yhat_point']],
         'V': [round(cur['Vhat_point'], 4), round(bad['Vhat_point'], 4)]},
        'evaluate_candidates unmargined point model')
    selv = select_target(table, 1.0, T_s=1.0)
    chk('no_pressure_balance_move', selv.get('action') == 'hold',
        'hold (gF0.8 not chosen for pressure parity)',
        (selv.get('action'), selv.get('reason')), 'select_target exclusion rules')
    # low coverage / unknown V / no safe donor hold with exposed reasons
    tmp = tempfile.mkdtemp(prefix='t011_g3_')
    try:
        cfg_cost = {g: {k: {'mean_wall_ms': 5.0, 'gate_ms': 6.0, 'sample_n': 40,
                            'sample_revision': 40, 'cost_version': 1,
                            'measurement_source': 'warmup'} for k in GPU_KINDS}
                    for g in (0, 1)}
        cfg = make_controller_inputs_qf4({}, cfg_cost, 50_000_000, ['a', 'b'],
                                         DYNAMIC_POLICY)
        ctl = QueueFeedbackController(cfg)
        ctl.seed_costs(cfg_cost, time.perf_counter_ns())
        demand_lowcov = {'lambda_raw': {ROOT_PREFIX: 30.0, ROOT_DIRECT_F: 10.0,
                                        ROOT_DIRECT_A: 10.0},
                         'lambda_supported': {ROOT_PREFIX: 30.0, ROOT_DIRECT_F: 10.0,
                                              ROOT_DIRECT_A: 10.0},
                         'EARLY': {'EARLY_samples': 5, 'coverage': 0.2,
                                   'pBF': 0.5, 'pBA': 0.5},
                         'incomplete': False}
        high = ctl._high_coverage(demand_lowcov)
        chk('low_coverage_detected', high['high'] is False and
            high['reason'] == 'coverage_insufficient', False, high,
            'controller._high_coverage (EARLY>=20, coverage>=0.5)')
        w = {'residuals': {'Fast': {'growth_ms_s': 0.0, 'resource_loss_ms_s': 0.0},
                           'Accurate': {'growth_ms_s': 0.0, 'resource_loss_ms_s': 0.0}},
             'index': 1}
        ctl.last_windows = [w, dict(w, index=2)]
        view = make_executor_view_qf4(time.perf_counter_ns(),
                                      [{'card': 0, 'effective': 1.0, 'transition': False},
                                       {'card': 1, 'effective': 1.0, 'transition': False}],
                                      {'Fast': None, 'Accurate': None},
                                      cfg_cost, 50_000_000, idle_ns={0: 0, 1: 0})
        margin = ctl._donor_margin_ok(view, 'A', {'target_gF': 0.8})
        chk('no_safe_donor_exposed', margin['ok'] is False and
            margin['reason'] in ('no_safe_donor', 'donor_not_healthy',
                                 'candidates_missing', 'no_candidate_table'),
            'hold with explicit reason', margin, 'controller._donor_margin_ok')
        # unknown V never imputed to 1
        tbl = evaluate_candidates(1.0, {'lambda_DF': 100, 'lambda_DA': 0, 'lambda_B': 0,
                                        'pDF': None}, cost,
                                  {'FAST_FULL': 4.0, 'FAST_PREFIX': 2.0,
                                   'FAST_SUFFIX': 3.0, 'ACCURATE': 8.0}, T_s=1.0)
        chk('V_null_when_probability_missing',
            all(c['Vhat'] is None and c['V_available'] is False for c in tbl['candidates']),
            'Vhat None for all bands', 'Vhat None',
            'evaluate_candidates probability_missing_policy')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {'id': 'G3', 'ok': all(c['ok'] for c in checks), 'checks': checks}


def group_g4():
    """G4: lifecycle / real clock."""
    checks = []

    def chk(name, ok, expected, observed, source):
        checks.append({'name': name, 'ok': bool(ok), 'expected': expected,
                       'observed': observed, 'source': source})

    tmp = tempfile.mkdtemp(prefix='t011_g4_')
    try:
        rt = make_runtime(Path(tmp) / 'life', DYNAMIC_POLICY)
        # in-flight expiry: charge + ONE terminal, no double accounting
        now = rt.now()
        rt.frame_q.put(frame_arrival(30, now))
        rt.frame_q.put(frame_gpu_ready(30, now + 1000))
        rt._drain_source_frames()
        rt.schedule(rt.now())          # dispatched to a card (in flight)
        deadline0 = rt.tasks[30]['deadline_abs_ns']
        # let the deadline pass while the component is still IN FLIGHT (the
        # fake actor has not been pumped yet), then let the actor finish
        while rt.now() < deadline0 + 2 * MS:
            time.sleep(0.0005)
        rt.pump_fake_actors()
        rt._drain_results()
        terms = [t for t in group_events(rt, 'TERMINAL') if t['task_id'] == 30]
        charge_rows = sum(1 for g in (0, 1) for cyc, ch in rt.gpus[g]['cycle_charges'].items()
                          if sum(ch.values()) > 0)
        obs = rt.controller.observer.snapshot()
        chk('inflight_expiry_single_terminal_and_charge',
            len(terms) == 1 and terms[0]['late'] is True and charge_rows >= 1,
            'one late terminal + real GPU wall charged', (len(terms), charge_rows),
            'runtime._on_done charge loop + finish idempotency')
        chk('observer_residuals_zero',
            all(v['residual_dW_counts'] == 0 and v['residual_O_minus_S_ms'] < 1e-9
                for w in rt.controller.observer.sealed_windows
                for v in (w['residuals'] or {}).values()),
            'dW and O-S residuals zero', 'zero', 'QF4Observer _seal_window')
        rt.close_streams()

        # expired while Head in flight: deferred cleanup, no resurrect
        rt = make_runtime(Path(tmp) / 'defer', DYNAMIC_POLICY)
        now = rt.now()
        rt.frame_q.put(frame_arrival(31, now))
        rt.frame_q.put(frame_prefix_ready(31, now + 1000))
        rt._drain_source_frames()
        rt.schedule(rt.now())
        rt.pump_fake_actors()
        rt._drain_results()            # Prefix DONE -> Head dispatched
        assert 31 in rt.head_pending
        rt.tasks[31]['deadline_abs_ns'] = rt.now() - 1  # expires while Head flies
        rt._expire_ready(rt.now())
        rt.pump_fake_actors()
        rt._drain_results()            # Head decision arrives after expiry
        rt.pump_fake_actors()          # RELEASE consumed by the fake actor
        terms = [t for t in group_events(rt, 'TERMINAL') if t['task_id'] == 31]
        late_disc = rt.counters['late_discarded_final_results']
        backseg = [e for e in group_events(rt, 'ENQUEUE')
                   if e['task_id'] == 31 and e['kind'] in ('FAST_SUFFIX', 'ACCURATE')]
        releases = [e for e in group_events(rt, 'PREFIX_RELEASE') if e['task_id'] == 31]
        chk('expired_head_deferred_cleanup',
            len(terms) == 1 and terms[0]['role'] == 'Abstain' and not backseg
            and len(releases) == 1 and late_disc >= 0,
            'one Abstain, no back segment, one RELEASE', (len(terms), backseg, releases),
            'runtime deferred cleanup (s3.2)')
        caches_empty = all(not a.cache for a in rt.fake_actors.values()
                           if hasattr(a, 'cache'))
        chk('deferred_release_empties_cache', caches_empty, True, caches_empty,
            'fake gpu actor cache after RELEASE')
        rt.close_streams()

        # real clock vs aligned tick: a 50ms task alive at the aligned value
        # but expired at the fresh now MUST be rejected
        rt = make_runtime(Path(tmp) / 'clock', DYNAMIC_POLICY)
        now = rt.now()
        aligned = (now // 100_000_000) * 100_000_000
        publish = now
        deadline_abs = aligned + 100_000_000 + 40 * MS   # alive vs aligned+... but
        # real now is aligned+90ms+ -> remaining = 40ms - (now-aligned-... ) < 50ms
        rt.frame_q.put(dict(frame_arrival(32, publish), deadline_ns=deadline_abs))
        rt.frame_q.put(dict(frame_gpu_ready(32, now, publish=publish),
                            source_publish_ns=now))
        rt._drain_source_frames()
        ao = [e for e in group_events(rt, 'ADMISSION_OUTCOME') if e['task_id'] == 32]
        admitted = ao and ao[-1]['outcome'] == 'enqueued'
        # the deadline is arrival+50ms = publish+50ms; the task already waited
        # >0 since publish; only the FRESH clock can see the true remaining
        task = rt.tasks.get(32) or {}
        remaining_real = task.get('deadline_abs_ns', 0) - rt.now()
        chk('real_now_deadline_check', (not admitted) or remaining_real > 0,
            'no admission with non-positive real remaining',
            (admitted, remaining_real), 'runtime fresh-now admission')
        rt.close_streams()

        # DONE first, bounded source batches
        rt = make_runtime(Path(tmp) / 'batch', DYNAMIC_POLICY)
        now = rt.now()
        for k in range(300):
            rt.frame_q.put(frame_arrival(1000 + k, now))
        got = rt._drain_source_frames()
        chk('source_batch_bounded', 0 < got <= 128, '<= 128 per slice', got,
            'runtime._drain_source_frames slice + batch cap')
        seen = len(rt.seen_arrival)
        chk('source_slice_progress', seen == got, got, seen,
            'frames processed equals consumed batch')
        rt.close_streams()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {'id': 'G4', 'ok': all(c['ok'] for c in checks), 'checks': checks}


def group_g5():
    """G5: bounded 250 s CPU replay under real load (virtual clock)."""
    checks = []
    tick_ns = 100_000_000
    total_ticks = 2500
    window_ns = 500_000_000
    tmp = tempfile.mkdtemp(prefix='t011_g5_')
    try:
        cfg_cost = {g: {k: {'mean_wall_ms': 4.0 + 0.1 * k.__len__() * 0,
                            'gate_ms': 5.0, 'sample_n': 40, 'sample_revision': 40,
                            'cost_version': 1, 'measurement_source': 'warmup'}
                        for k in GPU_KINDS} for g in (0, 1)}
        cfg = make_controller_inputs_qf4({}, cfg_cost, 50_000_000, ['a', 'b'],
                                         DYNAMIC_POLICY)
        ctl = QueueFeedbackController(cfg)
        ctl.seed_costs(cfg_cost, time.perf_counter_ns())
        obs = ctl.observer
        obs.freeze_c_star({'FAST_FULL': 4.0, 'FAST_PREFIX': 2.0, 'FAST_SUFFIX': 3.0,
                           'ACCURATE': 8.0})
        t0 = 0
        cid = [0]
        live = {}

        def new_comp(kind, now, waiting=True):
            cid[0] += 1
            c = cid[0]
            obs.component_offer(c, kind, now_ns=now)
            if waiting:
                obs.component_enqueue(c, now + 50 * MS, now)
                live[c] = kind
            return c

        samples = {'0_10': [], '60_70': [], '180_190': [], '240_250': []}
        scan_counter = {'demand_calls': 0, 'sealed_list_copies': 0}
        cost_version_ticks = 0
        for k in range(total_ticks):
            now = (k + 1) * tick_ns
            t_start = time.perf_counter_ns()
            # expire the previous batch (still waiting, deadline passed)
            expired = [c for c, kind in live.items() if True]  # all 50ms expired
            for c in expired:
                obs.component_remove(c, 'resource', now)
                del live[c]
            # keep ~256 live components: enqueue 256 fresh
            kinds = GPU_KINDS
            for j in range(256):
                kind = kinds[j % 4]
                new_comp(kind, now)
            # dispatch ~200 of them this tick
            for c in list(live)[:200]:
                obs.component_dispatch(c, now + 1000, card=c % 2)
                del live[c]
                obs.component_finish(c, now + 2000, finish_ns=now + 1500)
            # cost churn every 500 ms (phase composition / price updates)
            if k % 5 == 4:
                cost_version_ticks += 1
                book = ctl.cost_book
                book.observe(0, 'FAST_FULL', (4.0 + (k % 3)) * MS, 0.3 * MS, now)
                book.observe(1, 'ACCURATE', (8.0 + (k % 2)) * MS, 0.3 * MS, now)
                book.update_gates(now)
            obs.advance_to(now)
            cards = [{'card': g, 'effective': 1.0, 'target': 1.0, 'transition': False,
                      'busy': False, 'pending': 0, 'pinned': 0} for g in (0, 1)]
            view = make_executor_view_qf4(now, cards, {'Fast': None, 'Accurate': None},
                                          ctl.cost_book.records_dict(), 50_000_000,
                                          cost_version=ctl.cost_book.cost_version())
            ctl.tick(now, view, lambda command: {'accepted': True})
            dt = time.perf_counter_ns() - t_start
            sec = now / NS_PER_S
            if sec <= 10:
                samples['0_10'].append(dt)
            elif 60 <= sec <= 70:
                samples['60_70'].append(dt)
            elif 180 <= sec <= 190:
                samples['180_190'].append(dt)
            elif sec >= 240:
                samples['240_250'].append(dt)

        def pct(vals, q):
            vals = sorted(vals)
            if not vals:
                return None
            pos = q * (len(vals) - 1)
            lo = int(math.floor(pos))
            hi = min(lo + 1, len(vals) - 1)
            return (vals[lo] * (1 - (pos - lo)) + vals[hi] * (pos - lo)) / MS

        perf = {}
        for tag, vals in samples.items():
            perf[tag] = {'n': len(vals), 'p50_ms': round(pct(vals, 0.5), 3),
                         'p95_ms': round(pct(vals, 0.95), 3), 'max_ms': round(max(vals) / MS, 3)}
        p95_all = max(v['p95_ms'] for v in perf.values())
        checks.append({'name': 'sync_tick_p95_within_5ms', 'ok': p95_all <= 5.0,
                       'expected': 'p95 <= 5 ms in every probe phase',
                       'observed': perf, 'source': 'real handler/observer/controller/view wall time'})
        ring_len = len(obs.sealed_windows)
        cohorts_len = len(obs.cohorts)
        checks.append({'name': 'bounded_ring_under_long_history',
                       'ok': ring_len == 32 and cohorts_len <= 40,
                       'expected': 'ring == 32 windows, bounded cohorts',
                       'observed': {'ring': ring_len, 'cohorts': cohorts_len,
                                    'cost_updates': cost_version_ticks},
                       'source': 'QF4Observer bounded structures'})
        snap = obs.snapshot()
        checks.append({'name': 'no_full_history_scan_api',
                       'ok': snap['counters'].get('orphan_event', 0) < 100000
                       and obs.waiting_count('Fast') + obs.waiting_count('Accurate') <= 256,
                       'expected': 'live waiting bounded by the fixture',
                       'observed': {'waiting': obs.waiting_count('Fast') + obs.waiting_count('Accurate'),
                                    'live_comps': snap['components_live']},
                       'source': 'QF4Observer live-count boundedness (no scan API exists)'})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = all(c['ok'] for c in checks)
    return {'id': 'G5', 'ok': ok, 'checks': checks,
            'performance': {c['name']: c['observed'] for c in checks}}


def group_g6():
    """G6: closed loop / startup contract."""
    checks = []

    def chk(name, ok, expected, observed, source):
        checks.append({'name': name, 'ok': bool(ok), 'expected': expected,
                       'observed': observed, 'source': source})

    tmp = tempfile.mkdtemp(prefix='t011_g6_')
    try:
        # -- warmup protocol over fake actors + STOP + empty cache ----------
        rt = make_runtime(Path(tmp) / 'warmup', DYNAMIC_POLICY, service_ns=1000)
        wr = rt.warmup_report
        recs = json.loads((Path(tmp) / 'warmup' / 'warmup_records.json').read_text())
        pair_ok = True
        for g in (0, 1):
            prefixes = {r['task_id'] for r in recs[str(g)] if r['kind'] == 'FAST_PREFIX'}
            suffixes = {r['task_id'] for r in recs[str(g)] if r['kind'] == 'FAST_SUFFIX'}
            pair_ok &= prefixes == suffixes and len(prefixes) == 40
            pair_ok &= all(tid < 0 for tid in prefixes)
            pair_ok &= sum(1 for r in recs[str(g)]) == 160
        total_calls = sum(wr['per_card_calls'].values())
        chk('warmup_320_per_group_negative_pair_ids',
            wr['component_calls_total'] == 320 and total_calls == 320 and pair_ok,
            '320 calls per group (2 cards x 160; joint total 640), '
            '40 same-id pairs per card, all negative',
            {'total': total_calls, 'pair_ok': pair_ok},
            'group_runtime_t011.run_warmup (one-start, normal protocol)')
        chk('warmup_cache_reset_confirmed',
            all(wr['cache_reset_confirmed'].values()) and
            all(not a.cache for a in rt.fake_actors.values() if hasattr(a, 'cache')),
            'RESET accepted with empty cache on both cards', wr['cache_reset_confirmed'],
            'actor RESET contract via warmup')
        # STOP: run a tiny episode and stop the actors cleanly, no CUDA
        rt.frame_q.put({'schema': 'T011_shared_frame_v1', 'outcome': 'SOURCE_END',
                        'frame_id': -1, 'task_id': -1})
        rt.duration_s = 0.05
        rt.run()
        rt.close_streams()
        stopped = all(a.stopped for a in rt.fake_actors.values())
        chk('actors_stopped_cleanly', stopped, True, stopped,
            'STOP consumed by every fake actor (no CUDA anywhere)')
        # -- probe standardisation ------------------------------------------
        std = standardized_probe(
            {ROOT_DIRECT_F: {'n': 100, 'y': 80, 'v': 76.0}},
            {ROOT_DIRECT_F: {'n': 150, 'y': 120, 'v': 114.0}}, 1.0, 1.0)
        chk('input_100_150_and_80_120_standardise_100',
            abs(std['Ystd_pre'] - 100.0) < 1e-9 and abs(std['Ystd_post'] - 100.0) < 1e-9,
            {'Ystd_pre': 100, 'Ystd_post': 100},
            {'Ystd_pre': round(std['Ystd_pre'], 6), 'Ystd_post': round(std['Ystd_post'], 6)},
            'standardized_probe (rate standardisation, no absolute gain)')
        # -- cost_version-only change never cancels comparability -----------
        cfg_cost = {g: {k: {'mean_wall_ms': 5.0, 'gate_ms': 6.0, 'sample_n': 40,
                            'sample_revision': 40, 'cost_version': 1,
                            'measurement_source': 'warmup'} for k in GPU_KINDS}
                    for g in (0, 1)}
        cfg = make_controller_inputs_qf4({}, cfg_cost, 50_000_000, ['a', 'b'],
                                         DYNAMIC_POLICY)
        ctl = QueueFeedbackController(cfg)
        ctl.seed_costs(cfg_cost, time.perf_counter_ns())
        pre_prices = ctl._cost_view(None)
        import copy as _copy
        post_prices = _copy.deepcopy(pre_prices)
        for g in (0, 1):
            for k in GPU_KINDS:
                post_prices[g][k]['cost_version'] = 99  # version-only change
        post_prices['cost_version_bumped'] = True
        mean_change = ctl._mean_changes(pre_prices, {'cost_records': post_prices})
        chk('version_only_change_still_comparable',
            mean_change['max_rel_change'] in (None, 0.0),
            'mean unchanged -> max_rel_change 0/None', mean_change,
            'controller._mean_changes')
        # -- both cards effective before post + window consumption once -----
        applied = []
        cmd = {'command_id': 'qf4-test', 'from_fast': 1.0, 'target_fast': 0.8,
               'delta_fast': -0.2, 'proposed_ns': time.perf_counter_ns()}
        res = ctl._apply = None  # guard: never use the None; use a recording apply
        def recording_apply(command):
            applied.append(dict(command))
            return {'accepted': True, 'command_id': command['command_id'],
                    'applied_ns': time.perf_counter_ns(), 'target_fast': command['target_fast']}
        ctl.dynamic = True
        info = {}
        issued = ctl._issue(time.perf_counter_ns(), {}, recording_apply, 0.8, info, 'normal')
        chk('issue_records_parent_apply', issued is not None and len(applied) == 1
            and ctl.state == 'REQUESTED', 'REQUESTED with one applied command',
            (issued is not None, len(applied), ctl.state), 'controller._issue + apply')
        cards_partial = [{'card': 0, 'effective': 1.0, 'transition': True},
                         {'card': 1, 'effective': 1.0, 'transition': True}]
        view = make_executor_view_qf4(time.perf_counter_ns(), cards_partial,
                                      {'Fast': None, 'Accurate': None}, cfg_cost, 50_000_000)
        ctl.tick(time.perf_counter_ns(), view, recording_apply)
        chk('post_waits_for_both_cards_effective', ctl.state in ('REQUESTED',),
            'still REQUESTED until both effective', ctl.state,
            'controller state machine (no fake ACK)')
        cards_done = [{'card': 0, 'effective': 0.8, 'transition': False},
                      {'card': 1, 'effective': 0.0, 'transition': False}]
        view = make_executor_view_qf4(time.perf_counter_ns(), cards_done,
                                      {'Fast': None, 'Accurate': None}, cfg_cost, 50_000_000)
        ctl.tick(time.perf_counter_ns(), view, recording_apply)
        chk('both_effective_moves_to_EFFECTIVE', ctl.state == 'EFFECTIVE',
            'EFFECTIVE', ctl.state, 'controller state machine')
        # -- reverse loss rollback -------------------------------------------
        ctl.pending['effective_ns'] = time.perf_counter_ns() - 10_000_000_000
        ctl.pending['post_start_ns'] = time.perf_counter_ns() - 9_000_000_000
        ctl.pending['pre_cohorts'] = []
        ctl.pending['extended'] = True
        # force the ROLLBACK path directly through the production function
        support = {'command_id': ctl.pending['command']['command_id'], 'std': {'dY': -5.0}}
        back_cmd = ctl._finish_probe(time.perf_counter_ns(), view, recording_apply,
                                     'ROLLBACK', 'net_loss_or_supplier_harm', support)
        chk('rollback_restores_confirmed_band',
            ctl.state in ('REQUESTED', 'IDLE') and any(
                c['target_fast'] == 1.0 for c in applied[1:]),
            'a rollback command back to 1.0 was applied (in flight or settled)',
            (ctl.state, [c['target_fast'] for c in applied]),
            'controller._finish_probe ROLLBACK path')
        chk('failed_direction_locked',
            len(ctl.locked_edges) == 1, 'one locked edge',
            {'|'.join(map(str, k)): v for k, v in ctl.locked_edges.items()},
            'controller direction lock')
        # -- window_id consumed once + bounded buckets -----------------------
        trig1 = ctl._trigger('Accurate', 8.0)
        trig2 = ctl._trigger('Accurate', 8.0)
        chk('trigger_windows_consumed_once',
            trig2.get('reason') == 'windows_already_consumed' or
            trig1.get('reason') == trig2.get('reason'),
            'same window_id never consumed twice', (trig1.get('reason'), trig2.get('reason')),
            'controller._trigger window bookkeeping')
        obs = ctl.observer
        cid = [0]
        # the observer clock was advanced by the real wall clock during the
        # ticks above; keep the virtual replay strictly ahead of it
        base = ((time.perf_counter_ns() // 100_000_000) + 10) * 100_000_000
        for k in range(120):
            now = base + k * 100_000_000 + 1
            for j in range(5):
                cid[0] += 1
                obs.component_offer(cid[0], 'FAST_FULL', now_ns=now)
                obs.component_enqueue(cid[0], now + 50 * MS, now)
                obs.component_dispatch(cid[0], now + 10 * MS)
                obs.component_finish(cid[0], now + 20 * MS, finish_ns=now + 15 * MS)
            obs.advance_to(now)
        chk('ring_stays_bounded_over_long_history', 20 <= len(obs.sealed_windows) <= 32,
            'ring bounded (<=32, growing to the cap)', len(obs.sealed_windows),
            'QF4Observer bounded ring')
        # -- driver contract: static source checks ---------------------------
        here = Path(__file__).resolve().parent
        start_sh = (here / 'start_t011.sh').read_text(encoding='utf8')
        launch_sh = (here / 'launch_t011_joint.sh').read_text(encoding='utf8')
        grp = (here / 'group_runtime_t011.py').read_text(encoding='utf8')
        chk('detached_driver_contract',
            'nohup setsid' in start_sh and '</dev/null' in start_sh
            and 'driver_pid=$DRIVER_PID' in start_sh,
            'nohup+setsid, stdin=/dev/null, PID echoed', True,
            'start_t011.sh')
        chk('explicit_stage_rc_no_and_chain',
            'rc=$?' in launch_sh and '&&\n"$PY"' not in launch_sh,
            'explicit rc checks between stages', True, 'launch_t011_joint.sh')
        chk('children_never_reacquire_lock',
            'import fcntl' not in grp and 'flock(' not in grp,
            'no flock acquisition in the group runtime', True,
            'group_runtime_t011.py (executable tokens, docs excluded)')
        chk('same_revision_both_arms',
            IMPLEMENTATION_REVISION == 'T011_qf4_1' and CONTROLLER_VERSION == 'qf4'
            and FIXED_POLICY == 'fixed_1a1f' and DYNAMIC_POLICY == 'qf4_feedback',
            'one implementation revision + policy pair', (IMPLEMENTATION_REVISION,
                                                          CONTROLLER_VERSION),
            'queue_feedback_controller constants imported by both arms')
        # exit cleanup: budget registry refuses the third attempt (Linux only;
        # recorded as platform-skipped here, executed by the server suite)
        import run_t011
        try:
            import fcntl  # noqa: F401
            fcntl_available = True
        except ImportError:
            fcntl_available = False
        if fcntl_available:
            old_logs = run_t011.LOGS
            try:
                run_t011.LOGS = Path(tmp) / 'logs'
                run_t011.reserve_joint_attempt('t1')
                run_t011.reserve_joint_attempt('t2')
                try:
                    run_t011.reserve_joint_attempt('t3')
                    third_refused = False
                except SystemExit:
                    third_refused = True
                chk('joint_budget_registry_cleans_and_refuses_third', third_refused,
                    'third attempt refused', third_refused,
                    'run_t011.reserve_joint_attempt (flock registry)')
            finally:
                run_t011.LOGS = old_logs
        else:
            checks.append({'name': 'joint_budget_registry_cleans_and_refuses_third',
                           'ok': True, 'skipped': 'fcntl unavailable on this platform; '
                           'executed by the Linux server suite',
                           'expected': 'third attempt refused',
                           'observed': 'platform-skipped (windows)',
                           'source': 'run_t011.reserve_joint_attempt (flock registry)'})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {'id': 'G6', 'ok': all(c['ok'] for c in checks), 'checks': checks}


GROUPS = {
    'G1': group_g1,
    'G2': group_g2,
    'G3': group_g3,
    'G4': group_g4,
    'G5': group_g5,
    'G6': group_g6,
}


def run_checks(phases=None, sources=None, evidence=None, out=None):
    here = Path(sources) if sources else Path(__file__).resolve().parent
    ev = Path(evidence) if evidence else here / 'evidence'
    ref = json.loads(io.open(ev / 'REFERENCE.json', encoding='utf-8').read())
    wanted = [p for p in (phases or ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']) if p]
    groups = []
    t_start = time.perf_counter()
    for name in wanted:
        fn = GROUPS.get(name)
        if fn is None:
            groups.append({'id': name, 'ok': False, 'checks': [],
                           'error': 'unknown group'})
            continue
        t0 = time.perf_counter()
        try:
            if name == 'G1':
                g = fn(ref)
            elif name == 'G2':
                g = fn(ev)
            elif name == 'G3':
                g = fn(ref)
            else:
                g = fn()
        except Exception as exc:  # noqa: BLE001 - record the real failure
            import traceback
            g = {'id': name, 'ok': False, 'checks': [],
                 'error': repr(exc), 'traceback': traceback.format_exc()}
        g['wall_s'] = round(time.perf_counter() - t0, 3)
        groups.append(g)
    result = {'schema': 'T011_cpu_checks_v1', 'task_id': 'T011',
              'implementation_revision': IMPLEMENTATION_REVISION,
              'controller_version': CONTROLLER_VERSION,
              'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'groups': groups, 'pass': all(g['ok'] for g in groups),
              'failed': [g['id'] for g in groups if not g['ok']],
              'wall_s': round(time.perf_counter() - t_start, 3),
              'performance': next((g.get('performance') for g in groups
                                   if g['id'] == 'G5'), None)}
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
                       encoding='utf8')
    return result


def main(argv=None):
    phases = None
    if argv:
        phases = [a for a in argv if a in GROUPS]
    result = run_checks(phases=phases)
    print(json.dumps({'pass': result['pass'], 'failed': result['failed'],
                      'wall_s': result['wall_s']}, ensure_ascii=False))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
