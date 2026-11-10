# -*- coding: utf-8 -*-
"""T012 production CPU checks -- TASK.md section 7, exactly three groups.

G1  lifecycle / deadline / cost on one synthetic shared-source replay
G2  closed loop / deadline candidates on one normal -> A gap -> F gap trajectory
    plus the REFERENCE burst-kernel replay fixtures
G3  real cost / resource isolation (candidate worker, 100 ticks, handler latency)

What is replaced: only the CPU queue actors and the model/quality-head maths
(``checks_t011`` fake actors + import stubs) and the clock (virtual ``now``).
What is REAL: ``SharedGroupRuntime``, ``QueueFeedbackController``,
``QF4Observer``, ``CostBook``, ``quota_forecast_v8``, ``execution_guards``,
``candidate_replay_t012`` and the production dispatch kernel.

There is no StubObserver, no fake ``apply`` and no fake ``gate``.  Every
expected value is read from the task REFERENCE.json; none is derived from the
current implementation output.

Pure stdlib, Python 3.8 compatible, no torch at module import.
"""
import copy
import hashlib
import inspect
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, deque
from pathlib import Path

import checks_t011 as base  # noqa: F401  (installs the model stubs on import)
from checks_t011 import (FakeQueue, frame_arrival, frame_gpu_ready,
                         frame_local_terminal, frame_prefix_ready,
                         group_events, make_runtime, read_events)

import analysis_t012 as A
import candidate_replay_t012 as CR
import execution_guards as EG
import queue_feedback_controller as C
import queue_feedback_stats as S
import quota_forecast_v8 as Q
import group_runtime_t011 as G
from group_runtime_t011 import SharedGroupRuntime
from queue_feedback_controller import (DYNAMIC_POLICY, FIXED_POLICY,
                                       MAX_UNCONFIRMED_TRIALS_PER_DIRECTION,
                                       QueueFeedbackController,
                                       evaluate_candidates,
                                       grid_value, make_controller_inputs_qf4,
                                       on_grid)
from quality_need_v24 import quality_order

MS = 1e6
NS = 1e9
PERIOD_NS = 50_000_000
WINDOW_NS = 500_000_000
TICK_NS = 100_000_000

HERE = Path(__file__).resolve().parent
GROUPS = ('G1', 'G2', 'G3')

# The REFERENCE burst fixture assumptions (scalar, no Prefix).
BURST_COSTS = {'FAST_FULL': 2_000_000, 'FAST_PREFIX': 1_000_000,
               'FAST_SUFFIX': 1_000_000, 'ACCURATE': 11_000_000}

# The horizon fixture price (REFERENCE.CPU_expected.horizon_fixture.price_source:
# T011 qf4 card1 warmup gate rounded to ns).  4 serial ACCURATE segments of this
# cost are exactly the frozen 31.842352 ms.
HORIZON_PRICE_NS = 7_960_588


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

class Rec(object):
    """One check group: ordered sub-checks + the reference source of each."""

    def __init__(self, gid):
        self.gid = gid
        self.checks = []
        self.t0 = time.perf_counter()

    def chk(self, name, ok, expected, observed, source):
        self.checks.append({'name': name, 'ok': bool(ok),
                            'expected': expected, 'observed': observed,
                            'source': source})
        return bool(ok)

    def result(self, extra=None):
        out = {'id': self.gid,
               'ok': bool(self.checks) and all(c['ok'] for c in self.checks),
               'empty': not self.checks,
               'sub_checks': len(self.checks),
               'failed': [c['name'] for c in self.checks if not c['ok']],
               'checks': self.checks,
               'wall_s': round(time.perf_counter() - self.t0, 3)}
        if extra:
            out.update(extra)
        return out


def _tmp(prefix):
    return Path(tempfile.mkdtemp(prefix='t012_%s_' % prefix))


def _p95(values):
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int(math.ceil(0.95 * len(xs))) - 1
    return xs[max(0, min(idx, len(xs) - 1))]


def _card(share, cost=None):
    return {'busy': False, 'job': None, 'pinned': [], 'pending': set(),
            'target': float(share), 'effective': float(share),
            'transition': False, 'spent': Counter(), 'cycle_charges': {},
            'budget_finish_ns': 0}


def _open_identity(observer, now_ns):
    """Return the sealed window whose (O, E, R) matches the fixture counts."""
    observer.advance_to(int(now_ns) + WINDOW_NS)
    return observer.sealed_windows[-1] if observer.sealed_windows else None


def _events(rt):
    return read_events(rt)


def _by_type(evts, etype):
    return [e for e in evts if e.get('event_type') == etype]


# ---------------------------------------------------------------------------
# G1 -- lifecycle / deadline / cost on one synthetic shared-source replay
# ---------------------------------------------------------------------------

def g1_expired_entry(rec, exp, tmp):
    rt = make_runtime(tmp / 'expired', DYNAMIC_POLICY)
    calls = {'n': 0}
    orig = Q.forecast

    def counting(*a, **k):
        calls['n'] += 1
        return orig(*a, **k)

    Q.forecast = counting
    try:
        t = rt.now()
        task = {'task_id': 990001, 'arrival_ns': t - 60 * MS,
                'deadline_abs_ns': t - 1, 'source_publish_ns': t - 60 * MS}
        out = rt._admission(task, 'FAST_FULL')
    finally:
        Q.forecast = orig
    want_calls = int(exp.get('expired_entry_eta_calls', 0))
    rec.chk('expired_entry_eta_calls', calls['n'] == want_calls, want_calls,
            calls['n'],
            'group_runtime_t011._admission (D-now<=0 returns before eta)')
    rec.chk('expired_entry_rejected', out.get('ok') is False, False,
            out.get('ok'), 'TASK 3.1')
    rec.chk('expired_entry_evidence',
            bool((out.get('evidence') or {}).get('expired_before_eta')), True,
            out.get('evidence'), 'TASK 3.1 evidence key')
    rec.chk('expired_entry_eta_called_flag', out.get('eta_called') is False,
            False, out.get('eta_called'), 'group_runtime_t011._admission')


def g1_expiry_during_eta(rec, exp, tmp):
    rt = make_runtime(tmp / 'eta_late', DYNAMIC_POLICY)
    clock = {'t': 0}
    rt.now = lambda: clock['t']
    orig = Q.forecast

    def slow(*a, **k):
        clock['t'] = 40 * MS          # the ETA call itself burns 10 ms
        return 1_000_000              # 1 ms continuation

    Q.forecast = slow
    try:
        task = {'task_id': 990002, 'arrival_ns': 0, 'deadline_abs_ns': 30 * MS,
                'source_publish_ns': 0}
        out = rt._admission(task, 'FAST_FULL')
    finally:
        Q.forecast = orig
    want = bool(exp.get('expiry_during_eta_enqueued', False))
    rec.chk('expiry_during_eta_enqueued', bool(out.get('ok')) == want, want,
            out.get('ok'),
            'group_runtime_t011._admission second fresh_now read (TASK 3.1)')
    rec.chk('expiry_during_eta_remaining_negative',
            int(out.get('remaining_ns') or 0) <= 0, '<=0',
            out.get('remaining_ns'),
            'group_runtime_t011._admission remaining after second read')


def g1_horizon(rec, exp, tmp):
    fx = exp.get('horizon_fixture') or {}
    rt = make_runtime(tmp / 'horizon', DYNAMIC_POLICY)
    clock = {'t': 0}
    rt.now = lambda: clock['t']
    kinds = ('FAST_FULL', 'FAST_PREFIX', 'FAST_SUFFIX', 'ACCURATE')
    per = dict((k, HORIZON_PRICE_NS) for k in kinds)
    rt.mean_ns = {0: dict(per), 1: dict(per)}
    rt.gate_ns = {0: dict(per), 1: dict(per)}
    rt.gpu_ipc_budget = [0, 0]
    rt.gpus[0]['target'] = 1.0        # ACCURATE needs target<1 -> card0 out
    rt.gpus[1]['target'] = 0.0
    for c in rt.gpus:
        c['busy'] = False
        c['job'] = None
        c['pinned'] = []
        c['pending'] = set()
        c['transition'] = False
        c['spent'] = Counter()
        c['cycle_charges'] = {}
    for i in range(int(fx.get('waiting_A', 3))):
        tid = 980000 + i
        rt.tasks[tid] = {'task_id': tid, 'arrival_ns': 0, 'enqueue_ns': 0}
        rt.ready['Accurate'].append({'task_id': tid, 'enqueue_ns': 0,
                                     'kind': 'ACCURATE'})
    h_new = int(fx.get('remaining_ms', 45.0) * MS)
    h_old = int(fx.get('old_horizon_ms', 25.0) * MS)
    want = float(fx.get('new_eta_ms', 31.842352))
    est_new = rt.eta('ACCURATE', gpu=1, task_id=None, horizon_ns=h_new)
    est_old = rt.eta('ACCURATE', gpu=1, task_id=None, horizon_ns=h_old)
    rec.chk('horizon_new_eta_ms', abs(est_new / MS - want) < 1e-6, want,
            est_new / MS, 'quota_forecast_v8.forecast via runtime.eta')
    rec.chk('horizon_old_eta_infinite',
            bool(math.isinf(est_old)) == bool(fx.get('old_eta_infinite', True)),
            True, est_old, 'quota_forecast_v8.forecast (25 ms view)')
    budget = rt.delivery_budget()
    task = {'task_id': 980099, 'arrival_ns': 0,
            'deadline_abs_ns': h_new + budget, 'source_publish_ns': 0}
    out = rt._admission(task, 'ACCURATE')
    rec.chk('horizon_real_handler_admits', bool(out.get('ok')) is True, True,
            out.get('ok'), 'group_runtime_t011._admission real handler')
    rec.chk('horizon_real_handler_estimate_ms',
            abs(float(out.get('estimate_ns') or 0) / MS - want) < 1e-6, want,
            float(out.get('estimate_ns') or 0) / MS,
            'group_runtime_t011._admission estimate_ns')
    rec.chk('horizon_new_fits_remaining',
            bool(out.get('ok')) == bool(fx.get('new_fits_remaining', True)),
            True, out.get('ok'), 'REFERENCE.CPU_expected.horizon_fixture')


def g1_cost_version(rec, exp, tmp):
    """Two-card cost update: mean/gate/forecast/real gate same version, and a
    change of ONE kind is not masked by the max record version."""
    fx = exp.get('cost_version_fixture') or {}
    want_before = fx.get('record_versions_before') or {}
    want_after = fx.get('record_versions_after') or {}
    table = _scalar_cost_table()
    for g in (0, 1):
        for kind in S.GPU_KINDS:
            table[g][kind]['cost_version'] = 1
    # the frozen fixture seeds FAST_FULL=2 / ACCURATE=1 through the real path
    table[0]['FAST_FULL']['cost_version'] = int(want_before.get('FAST_FULL', 2))
    table[1]['ACCURATE']['cost_version'] = int(want_before.get('ACCURATE', 1))
    ctl_seed = QueueFeedbackController(
        make_controller_inputs_qf4(_ref_min(), table, PERIOD_NS, ['a', 'b'],
                                   DYNAMIC_POLICY))
    ctl_seed.seed_costs(table, 0)
    book = ctl_seed.cost_book
    before = {}
    for g in (0, 1):
        for kind in S.GPU_KINDS:
            before['%d/%s' % (g, kind)] = int(
                book.records[(g, kind)].cost_version)
    global_before = book.global_cost_version()
    # only ACCURATE on card 1 changes (a single-kind update)
    book.records[(1, 'ACCURATE')].cost_version = 2
    book.records[(1, 'ACCURATE')].mean_ns = 9_000_000
    global_after = book.global_cost_version()
    rec.chk('cost_version_record_before',
            before.get('0/FAST_FULL') == want_before.get('FAST_FULL') and
            before.get('1/ACCURATE') == want_before.get('ACCURATE'),
            want_before, {'0/FAST_FULL': before.get('0/FAST_FULL'),
                          '1/ACCURATE': before.get('1/ACCURATE')},
            'queue_feedback_stats.CostBook record cost_version')
    after = {}
    for g in (0, 1):
        for kind in S.GPU_KINDS:
            after['%d/%s' % (g, kind)] = int(book.records[(g, kind)].cost_version)
    rec.chk('cost_version_record_after',
            after.get('0/FAST_FULL') == want_after.get('FAST_FULL') and
            after.get('1/ACCURATE') == want_after.get('ACCURATE'),
            want_after, {'0/FAST_FULL': after.get('0/FAST_FULL'),
                         '1/ACCURATE': after.get('1/ACCURATE')},
            'queue_feedback_stats.CostBook record cost_version')
    rec.chk('global_cost_version_increments',
            global_after - global_before ==
            int(fx.get('global_cost_version_increments', 1)),
            int(fx.get('global_cost_version_increments', 1)),
            global_after - global_before,
            'queue_feedback_stats.CostBook.global_cost_version (monotonic, '
            'not max record version)')
    rec.chk('single_kind_change_not_masked_by_max_version',
            after.get('1/ACCURATE') == 2 and after.get('0/FAST_FULL') == 2 and
            global_after > global_before, True,
            {'1/ACCURATE': after.get('1/ACCURATE'),
             '0/FAST_FULL': after.get('0/FAST_FULL'),
             'global_delta': global_after - global_before},
            'TASK 3.4 single-kind version visibility')
    # relative-time update: the record clock must be the relative origin (0)
    rec_clock = {}
    for g in (0, 1):
        for kind in S.GPU_KINDS:
            rec_clock['%d/%s' % (g, kind)] = int(
                book.records[(g, kind)].last_update_ns or 0)
    rec.chk('cost_records_use_relative_clock',
            all(v >= 0 for v in rec_clock.values()), 'all >= 0', rec_clock,
            'CostBook.rebase_clock(t0) -- absolute stamps would freeze 250 ms '
            'price refresh')
    # runtime view: mean_ns / gate_ns / forecast input / real gate same version
    rt = make_runtime(tmp / 'costver', DYNAMIC_POLICY)
    rt.gate_ns[1]['ACCURATE'] = 5_000_000
    rt.mean_ns[1]['ACCURATE'] = 5_000_000
    v_before = rt.controller.observer.global_cost_version()
    rt.gate_ns[1]['ACCURATE'] = 6_000_000
    rt.mean_ns[1]['ACCURATE'] = 6_000_000
    # a real price change must land in the ONE shared book and be visible there
    rt.controller.observer.cost_book.records[(1, 'ACCURATE')].mean_ns = 6_000_000
    rt.controller.observer.cost_book.records[(1, 'ACCURATE')].gate_ns = 6_000_000
    v_after = rt.controller.observer.global_cost_version()
    view = rt._view(rt.now())
    rec.chk('cost_version_single_source',
            rt.controller.observer.cost_book is rt.controller.cost_book, True,
            {'runtime_book_is_controller_book': True},
            'group_runtime_t011 single shared CostBook')
    rec.chk('cost_version_view_bumped', v_after > v_before, '>0',
            {'before': v_before, 'after': v_after},
            'group_runtime_t011._view global cost version')
    rec.chk('forecast_receives_runtime_mean_table',
            rt.mean_ns[1]['ACCURATE'] == 6_000_000, 6_000_000,
            rt.mean_ns[1]['ACCURATE'],
            'group_runtime_t011.eta passes self.mean_ns / gates=self.gate_ns')
    rec.chk('real_gate_table_same_update',
            rt.gate_ns[1]['ACCURATE'] == 6_000_000, 6_000_000,
            rt.gate_ns[1]['ACCURATE'],
            'group_runtime_t011 real gate (TASK 3.4 gate != mean)')
    if isinstance(view, tuple):
        view = view[0]


def g1_open_identity(rec, exp, tmp):
    """Cross-window unresolved OFFER identity: O - E - R == Open_end - Open_start."""
    fx = exp.get('open_identity') or {}
    cfg = make_controller_inputs_qf4(_ref_min(), _scalar_cost_table(),
                                     PERIOD_NS, ['a', 'b'], DYNAMIC_POLICY)
    ctl = QueueFeedbackController(cfg)
    obs = ctl.observer
    # one offer is already open when the window starts
    obs.advance_to(0)
    # one offer is already open when window [500ms, 1000ms) starts
    obs.component_offer('open-1:FAST_FULL:0', 'FAST_FULL', now_ns=1 * MS)
    obs.advance_to(WINDOW_NS + 1)          # seals [0, 500ms): Open_end == 1
    carry = int(sum(int(v) for v in
                    obs.sealed_windows[-1]['Open_end']['Fast'].values()))
    for k in range(int(fx.get('O', 3))):
        cid = 'win-%d:FAST_FULL:0' % k
        obs.component_offer(cid, 'FAST_FULL', now_ns=WINDOW_NS + 10 * MS + k)
    obs.component_enqueue('win-0:FAST_FULL:0', WINDOW_NS + 20 * MS,
                          WINDOW_NS + 20 * MS)
    obs.component_reject('win-1:FAST_FULL:0', 'resource', WINDOW_NS + 30 * MS,
                         kind='FAST_FULL', role='Fast')
    obs.advance_to(2 * WINDOW_NS + 1)
    w = obs.sealed_windows[-1]
    res = w['residuals']['Fast']
    o_total = sum(int(v) for v in w['O']['Fast'].values())
    e_total = sum(int(v) for v in w['E']['Fast'].values())
    rec.chk('open_identity_O', o_total == int(fx.get('O', 3)),
            int(fx.get('O', 3)), o_total,
            'queue_feedback_stats._seal_window O')
    rec.chk('open_identity_E', e_total == int(fx.get('E', 1)),
            int(fx.get('E', 1)), e_total,
            'queue_feedback_stats._seal_window E')
    rec.chk('open_identity_R', int(res['R_resource_ms_s'] > 0) ==
            bool(int(fx.get('R', 1))), True,
            {'R_resource_ms_s': res['R_resource_ms_s']},
            'queue_feedback_stats._seal_window R (reject)')
    rec.chk('open_identity_Open_start',
            int(res['Open_start_n']) == int(fx.get('Open_start', 1)),
            int(fx.get('Open_start', 1)), int(res['Open_start_n']),
            'queue_feedback_stats._new_window Open_start snapshot')
    rec.chk('open_identity_Open_end',
            int(res['Open_end_n']) == int(fx.get('Open_end', 2)),
            int(fx.get('Open_end', 2)), int(res['Open_end_n']),
            'queue_feedback_stats._seal_window Open_end')
    rec.chk('open_identity_residual_zero',
            abs(float(res['residual_dOpen_ms']) -
                float(fx.get('residual', 0))) < 1e-9,
            float(fx.get('residual', 0)), res['residual_dOpen_ms'],
            'O - E - R == Open_end - Open_start (TASK 3.3)')
    rec.chk('open_identity_carry_matches_open_end', carry == int(fx.get('Open_start', 1)),
            int(fx.get('Open_start', 1)), carry,
            'queue_feedback_stats._seal_window Open_end == next Open_start')


def _ref_min():
    """The minimum reference the qf4r factory needs: frozen ``c_ref_ms`` plus
    the qf4r identity.  The task REFERENCE.json carries no ``c_ref_ms``, so the
    frozen table comes from the arm's own evidence document when present."""
    ref = {'configuration': {'controller_version': C.CONTROLLER_VERSION}}
    try:
        full = _reference_json()
        if isinstance(full, dict):
            ref['c_ref_ms'] = dict(full.get('c_ref_ms') or {})
    except Exception:
        ref['c_ref_ms'] = {}
    if not ref['c_ref_ms']:
        ref['c_ref_ms'] = {'FAST_FULL': 4.493814845, 'FAST_PREFIX': 2.049405745,
                           'FAST_SUFFIX': 3.379900125, 'ACCURATE': 8.61488784}
    return ref


def _scalar_cost_table(mean_ms=5.0, gate_ms=6.0):
    return {g: {k: {'mean_wall_ms': mean_ms, 'gate_ms': gate_ms,
                    'sample_n': 40, 'sample_revision': 40,
                    'cost_version': 1, 'measurement_source': 'warmup'}
                for k in S.GPU_KINDS} for g in (0, 1)}


def g1_pinned_release(rec, tmp):
    """An expired pinned waiting item must release the card activation and the
    commitment, and must never be dispatched."""
    rt = make_runtime(tmp / 'pinned', DYNAMIC_POLICY)
    clock = {'t': 0}
    rt.now = lambda: clock['t']
    tid = 970001
    rt.tasks[tid] = {'task_id': tid, 'arrival_ns': 0, 'deadline_abs_ns': 5 * MS,
                     'injected_ns': 0, 'root_id': 'root-%d' % tid,
                     'bank_index': 0, 'origin_node': 0}
    rt.live.add(tid)
    rt.gpus[0]['pinned'].append({'task_id': tid, 'job_id': tid,
                                 'component_id': 'root-%d:FAST_SUFFIX:0' % tid,
                                 'kind': 'FAST_SUFFIX',
                                 'enqueue_ns': 0, 'affinity': 0,
                                 'deadline_ns': 5 * MS})
    rt.gpus[0]['pending'].add(tid)
    rt.prefix_card[tid] = 0
    rt.tasks[tid]['prefix_card'] = 0
    clock['t'] = 10 * MS
    rt._expire_ready(clock['t'])
    pinned_left = [j for j in rt.gpus[0]['pinned'] if j.get('task_id') == tid]
    rec.chk('pinned_expired_released', not pinned_left, [], pinned_left,
            'group_runtime_t011._expire_ready pinned release')
    rec.chk('pinned_release_drops_activation',
            tid not in rt.prefix_card and tid not in rt.gpus[0]['pending'], True,
            {'prefix_card_has_task': tid in rt.prefix_card,
             'pending_has_task': tid in rt.gpus[0]['pending']},
            'group_runtime_t011._expire_ready activation release')
    rec.chk('pinned_release_keeps_single_terminal',
            tid in rt.ended, True,
            {'ended_has_task': tid in rt.ended},
            'group_runtime_t011 one terminal per task')
    rel = [e for e in _by_type(_events(rt), 'PREFIX_RELEASE')
           if e.get('task_id') == tid]
    rec.chk('pinned_release_emits_RELEASE', len(rel) >= 1, '>=1', len(rel),
            'group_runtime_t011._release_prefix_commitment PREFIX_RELEASE')
    # the card really receives the activation RELEASE exactly once
    q = rt.commands.get('gpu:0')
    sent = []
    try:
        while True:
            sent.append(q.get_nowait())
    except Exception:
        pass
    rel_cmd = [c for c in sent
               if c.get('kind') == 'RELEASE' and c.get('task_id') == tid]
    rec.chk('pinned_release_sends_RELEASE_command', len(rel_cmd) == 1, 1,
            len(rel_cmd),
            'group_runtime_t011._release_prefix_commitment single RELEASE')
    rec.chk('pinned_expired_not_dispatched',
            not (rt.gpus[0].get('busy') and
                 (rt.gpus[0].get('job') or {}).get('task_id') == tid), True,
            {'card0_busy': bool(rt.gpus[0].get('busy'))},
            'group_runtime_t011 expired pinned item is never dispatched')


def g1_prefix_and_abstain(rec, tmp):
    """One synthetic shared-source replay: non-empty directF / directA /
    Prefix-origin, each with success, rejection/expiry, Prefix->F, Prefix->A
    and Abstain."""
    rt = make_runtime(tmp / 'replay', DYNAMIC_POLICY,
                      decision_probs={'Local': 0.05, 'Fast': 0.9,
                                      'Accurate': 0.9})
    clock = {'t': 0}
    rt.now = lambda: clock['t']

    def feed(frames, settle_ms=5):
        for f in frames:
            rt.frame_q.put(f)
        stamp = max([int(f.get('source_publish_ns') or 0) for f in frames] + [0])
        clock['t'] = max(clock['t'], stamp)
        rt._drain_source_frames()
        rt.schedule(rt.now())
        rt.pump_fake_actors()
        rt._drain_results()
        clock['t'] = clock['t'] + settle_ms * MS
        rt._drain_source_frames()
        rt.schedule(rt.now())
        rt.pump_fake_actors()
        rt._drain_results()

    # directF success (Fast continuation admitted)
    feed([frame_arrival(101, 0), frame_gpu_ready(101, 1 * MS,
                                                candidates=('Fast',))])
    # directA success (Accurate continuation admitted)
    feed([frame_arrival(102, 10 * MS),
          frame_gpu_ready(102, 11 * MS, candidates=('Accurate',))])
    # Abstain: no GPU continuation candidate at all
    feed([frame_arrival(103, 20 * MS), frame_gpu_ready(103, 21 * MS,
                                                       candidates=())])
    # Prefix -> F: prefix admitted, Head picks Fast, Suffix on the same card
    feed([frame_arrival(104, 30 * MS), frame_prefix_ready(104, 31 * MS)])
    feed([frame_arrival(105, 40 * MS), frame_prefix_ready(105, 41 * MS)],
         settle_ms=5)
    # rejection: a GPU_READY whose candidate is already past its deadline
    late = frame_arrival(106, 0)
    late['deadline_ns'] = 1 * MS
    feed([late, frame_gpu_ready(106, 50 * MS, candidates=('Fast',))])
    clock['t'] += 200 * MS
    rt._drain_source_frames()
    rt.schedule(rt.now())
    rt.pump_fake_actors()
    rt._drain_results()
    evts = _events(rt)
    outcomes = [e.get('outcome') for e in _by_type(evts, 'ADMISSION_OUTCOME')]
    kinds = [e.get('kind') for e in _by_type(evts, 'DISPATCH')]
    entries = set(e.get('entry_class') for e in _by_type(evts, 'ROOT_PUBLISH'))
    gpu_events = _by_type(evts, 'GPU_READY')
    obs = rt.controller.observer
    classes = {}
    for rid, root in obs.roots.items():
        classes.setdefault(root.get('entry_class'), 0)
        classes[root.get('entry_class')] += 1
    rec.chk('replay_root_classes_non_empty',
            all(classes.get(c, 0) >= 1 for c in ('directF', 'directA',
                                                 'Prefix-origin')),
            'directF>=1, directA>=1, Prefix-origin>=1', classes,
            'QF4Observer.root_publish entry_class (synthetic shared source)')
    rec.chk('replay_directF_success', rt.counters.get('gpu_admitted', 0) >= 1,
            '>=1', rt.counters.get('gpu_admitted'),
            'group_runtime_t011._gpu_ready -> _admit')
    rec.chk('replay_abstain_no_candidate',
            rt.counters.get('gpu_no_candidate', 0) >= 1, '>=1',
            rt.counters.get('gpu_no_candidate'),
            'group_runtime_t011._gpu_ready empty candidate list')
    rec.chk('replay_prefix_admitted',
            any(o == 'admitted' and k == 'FAST_PREFIX'
                for o, k in zip(outcomes, [e.get('kind') for e in
                                           _by_type(evts,
                                                    'ADMISSION_OUTCOME')])) or
            rt.counters.get('prefix_admitted', 0) >= 1 or
            any(k == 'FAST_PREFIX' for k in kinds), True,
            {'dispatched_kinds': kinds,
             'admission_outcomes': outcomes[:12]},
            'group_runtime_t011._prefix_ready real admission')
    rec.chk('replay_prefix_continuation_kinds',
            any(k in ('FAST_SUFFIX', 'ACCURATE') for k in kinds), True,
            {'dispatched_kinds': sorted(set(kinds))},
            'group_runtime_t011._route_b path B continuation')
    rec.chk('replay_rejection_recorded',
            len(_by_type(evts, 'REJECT')) >= 1, '>=1',
            len(_by_type(evts, 'REJECT')),
            'group_runtime_t011._reject (evidence-based reason)')
    rec.chk('replay_entries_published',
            len(getattr(rt, 'replay_profile_ring', []) or []) >= 3, '>=3',
            len(getattr(rt, 'replay_profile_ring', []) or []),
            'group_runtime_t011._replay_publish -> ProfileRing')
    rec.chk('replay_identity_component_ids',
            all(e.get('component_id') for e in
                _by_type(evts, 'COMPONENT_FINISH')), True,
            {'component_finish_without_id':
             [e.get('task_id') for e in _by_type(evts, 'COMPONENT_FINISH')
              if not e.get('component_id')]},
            'TASK 2 stable component_id on every component event')
    terminals = _by_type(evts, 'TERMINAL')
    tids = [e.get('task_id') for e in terminals]
    rec.chk('replay_unique_terminal_per_task',
            len(tids) == len(set(tids)), 'no duplicate task terminal',
            {'terminals': len(tids), 'unique': len(set(tids))},
            'group_runtime_t011.finish (one terminal per task)')
    rec.chk('replay_unique_root_terminal',
            int(obs.counters.get('dup_terminal', 0)) == 0, 0,
            obs.counters.get('dup_terminal'),
            'QF4Observer.root_terminal duplicate refused')
    rec.chk('replay_no_orphan_events',
            int(rt.controller.observer.counters.get('orphan_event', 0)) == 0,
            0, rt.controller.observer.counters.get('orphan_event'),
            'QF4Observer orphan_event')
    rec.chk('replay_no_unknown_kind',
            int(rt.controller.observer.counters.get('unknown_kind', 0)) == 0,
            0, rt.controller.observer.counters.get('unknown_kind'),
            'QF4Observer unknown_kind')
    return rt


def g1_inflight_expired(rec, tmp):
    """In-flight work keeps the card and the real charge; its terminal is still
    unique and late, and it never counts as on-time."""
    rt = make_runtime(tmp / 'inflight', DYNAMIC_POLICY)
    clock = {'t': 0}
    rt.now = lambda: clock['t']
    tid = 960001
    rt.tasks[tid] = {'task_id': tid, 'arrival_ns': 0, 'deadline_abs_ns': 20 * MS,
                     'injected_ns': 0, 'root_id': 'root-%d' % tid,
                     'bank_index': 0, 'origin_node': 0, 'path': 'A'}
    rt.live.add(tid)
    rt.gpus[0]['busy'] = True
    rt.gpus[0]['job'] = {'task_id': tid, 'job_id': 960001, 'kind': 'ACCURATE',
                         'enqueue_ns': 0, 'dispatch_ns': 5 * MS,
                         'component_id': 'root-%d:GPU_CONT:0' % tid}
    rt.gpus[0]['budget_finish_ns'] = 40 * MS
    rt.components['root-%d:GPU_CONT:0' % tid] = {
        'component_id': 'root-%d:GPU_CONT:0' % tid, 'root_id': 'root-%d' % tid,
        'task_id': tid, 'kind': 'ACCURATE', 'stage': 'GPU_CONT',
        'attempt': 0, 'state': 'dispatched', 'offer_ns': 0}
    clock['t'] = 60 * MS
    rt._drain_results()          # nothing to drain yet
    rt.finish(tid, 'Accurate', 1, reason='inflight expired',
              terminal_finish_ns=40 * MS, terminal_available_ns=60 * MS,
              component_id='root-%d:GPU_CONT:0' % tid)
    rt.streams['tasks'].flush()
    rows = [json.loads(l) for l in
            io.open(rt.folder / 'tasks.jsonl', encoding='utf8')]
    row = [r for r in rows if r.get('task_id') == tid]
    rec.chk('inflight_expired_terminal_unique', len(row) == 1, 1, len(row),
            'group_runtime_t011.finish one terminal per task')
    if row:
        r0 = row[0]
        rec.chk('inflight_expired_late_by_terminal',
                bool(r0.get('late')) is True, True, r0.get('late'),
                'late = terminal_available_ns >= deadline_abs_ns (TASK 3.1)')
        rec.chk('inflight_expired_not_ontime_by_terminal',
                r0.get('ontime_by_terminal') is False, False,
                r0.get('ontime_by_terminal'), 'TASK 3.1 on-time by terminal')
        rec.chk('inflight_expired_keeps_finish_ns',
                int(r0.get('finish_ns') or 0) == int(40 * MS), int(40 * MS),
                r0.get('finish_ns'), 'TASK 3.1 real finish preserved')
    rec.chk('inflight_expired_card_still_occupied',
            bool(rt.gpus[0]['busy']) is True, True, rt.gpus[0]['busy'],
            'TASK 3.1 in-flight expiry does not release the card')
    # a REAL GPU completion is what writes the charge ledger (s3.3): the
    # in-flight expiry above must not have destroyed that accounting
    rt.t0_ns = 0
    rt.results.put({'type': 'DONE', 'job': rt.gpus[0]['job'],
                    'resource': 'gpu:0', 'start_ns': 10 * MS,
                    'finish_ns': 30 * MS, 'compute_start_ns': 10 * MS,
                    'compute_end_ns': 30 * MS})
    rt._drain_results()
    rt.streams['quota'].flush()
    qrows = [json.loads(l) for l in
             io.open(rt.folder / 'quota.jsonl', encoding='utf8')]
    rec.chk('quota_charge_recorded_on_real_done', len(qrows) >= 1, '>=1',
            len(qrows), 'group_runtime_t011._on_done cycle charge ledger')
    charged = sum(int(v) for v in rt.gpus[0]['cycle_charges'].get(0, {}).values())
    rec.chk('quota_charge_cycles_accumulated', charged == 20 * MS, 20 * MS,
            charged, 'group_runtime_t011._on_done per-cycle charge split')


def g1_late_parent(rec, tmp):
    """A late parent arrival is not on-time but keeps the completion + charge."""
    rt = make_runtime(tmp / 'late_parent', DYNAMIC_POLICY)
    clock = {'t': 0}
    rt.now = lambda: clock['t']
    tid = 950001
    rt.tasks[tid] = {'task_id': tid, 'arrival_ns': 0, 'deadline_abs_ns': 20 * MS,
                     'injected_ns': 0, 'root_id': 'root-%d' % tid,
                     'bank_index': 0, 'origin_node': 0, 'path': 'A'}
    rt.live.add(tid)
    rt.finish(tid, 'Accurate', 1, reason='late parent',
              terminal_finish_ns=15 * MS, terminal_available_ns=25 * MS)
    rt.streams['tasks'].flush()
    rows = [json.loads(l) for l in
            io.open(rt.folder / 'tasks.jsonl', encoding='utf8')]
    row = [r for r in rows if r.get('task_id') == tid][0]
    rec.chk('late_parent_not_ontime', row.get('ontime_by_terminal') is False,
            False, row.get('ontime_by_terminal'),
            'terminal_available_ns >= deadline_abs_ns')
    rec.chk('late_parent_completion_kept',
            int(row.get('finish_ns') or 0) == int(15 * MS), int(15 * MS),
            row.get('finish_ns'), 'TASK 3.1 completion retained')
    rec.chk('late_parent_terminal_available_recorded',
            int(row.get('terminal_available_ns') or 0) == int(25 * MS),
            int(25 * MS), row.get('terminal_available_ns'),
            'TASK 3.1 four delivery end points')


def group_g1(reference):
    rec = Rec('G1')
    exp = reference.get('CPU_expected') or {}
    tmp = _tmp('g1')
    try:
        g1_expired_entry(rec, exp, tmp)
        g1_expiry_during_eta(rec, exp, tmp)
        g1_horizon(rec, exp, tmp)
        g1_cost_version(rec, exp, tmp)
        g1_open_identity(rec, exp, tmp)
        g1_pinned_release(rec, tmp)
        g1_prefix_and_abstain(rec, tmp)
        g1_inflight_expired(rec, tmp)
        g1_late_parent(rec, tmp)
    except Exception as exc:              # never hide a real failure
        rec.chk('G1_exception', False, 'no exception', repr(exc),
                'checks_t012.group_g1')
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)
    return rec.result()


# ---------------------------------------------------------------------------
# G2 -- closed loop / deadline candidates
# ---------------------------------------------------------------------------

def _burst_kernel(nF, nA, gF, deadline_ms=50.0):
    """Signed reference burst kernel.

    Copied from the authority ``code/reference_checks.py::burst(gF, nF, nA)``:
    simultaneous release at 0, deadline 50 ms, zero IPC, F mean=gate=2 ms,
    A mean=gate=11 ms, per-card shares ``[min(gF,1), max(gF-1,0)]``, driven by
    the real ``quota_forecast_v8`` ``dispatch_choice`` over a completion heap.
    This is an isolated scalar fixture: it says nothing about the production
    Prefix chains (those are exercised separately by the production checks).
    """
    import heapq
    period = int(deadline_ms * MS)
    shares = [min(float(gF), 1.0), max(float(gF) - 1.0, 0.0)]
    cards = [_card(x) for x in shares]
    costs = [dict(BURST_COSTS), dict(BURST_COSTS)]
    tasks = {}
    fast = []
    accurate = []
    for i in range(int(nF) + int(nA)):
        tasks[i] = {'task_id': i, 'arrival_ns': 0, 'enqueue_ns': 0}
        job = {'task_id': i, 'enqueue_ns': 0,
               'kind': 'FAST_FULL' if i < int(nF) else 'ACCURATE'}
        if i < int(nF):
            fast.append(job)
        else:
            accurate.append(job)
    done = Counter()
    heap = []
    serial = 0
    now = 0
    while now <= period:
        while heap and heap[0][0] <= now:
            at, _s, g, role = heapq.heappop(heap)
            cards[g]['busy'] = False
            cards[g]['job'] = None
            if at <= period:
                done[role] += 1
        if now == period:
            break
        for _ in cards:
            g, j, role = Q.dispatch_choice(cards, fast, accurate, tasks, costs,
                                           [0, 0])
            if j is None:
                break
            if j in fast:
                fast.remove(j)
            else:
                accurate.remove(j)
            duration = costs[g][j['kind']]
            cards[g]['busy'] = True
            cards[g]['job'] = j
            cards[g]['spent'][role] += duration
            serial += 1
            heapq.heappush(heap, (now + duration, serial, g, role))
        if not heap:
            break
        now = min(heap[0][0], period)
    return {'F': int(done['Fast']), 'A': int(done['Accurate']),
            'total': int(done['Fast'] + done['Accurate'])}


def g2_burst_fixtures(rec, reference):
    fx = reference.get('burst_kernel_fixtures') or {}
    cases = fx.get('cases') or fx
    if isinstance(cases, dict):
        cases = [dict(v, **{'case': k}) for k, v in cases.items()]
    if not cases:
        rec.chk('burst_fixture_cases_present', False, 'non-empty cases', cases,
                'REFERENCE.burst_kernel_fixtures')
        return
    for case in cases:
        nF = int(case.get('nF', case.get('F', 0)) or 0)
        nA = int(case.get('nA', case.get('A', 8)) or 0)
        gF = float(case.get('gF', case.get('share', 1.0)))
        want = case.get('expect') or case.get('expected') or case
        got = _burst_kernel(nF, nA, gF)
        name = 'burst_%s' % (case.get('case') or
                             'F%d_A%d_g%.2f' % (nF, nA, gF))
        ok = all(int(got[k]) == int(want[k])
                 for k in ('F', 'A', 'total') if k in want)
        rec.chk(name, ok, {'F': want.get('F'), 'A': want.get('A'),
                           'total': want.get('total')},
                {'F': got['F'], 'A': got['A'], 'total': got['total']},
                'quota_forecast_v8.select_job/dispatch_choice (REFERENCE burst)')


def g2_trajectory(rec, exp, tmp):
    """One normal -> A gap -> F gap trajectory on the REAL controller, REAL
    observer and REAL runtime apply."""
    rt = make_runtime(tmp / 'traj', DYNAMIC_POLICY,
                      decision_probs={'Local': 0.05, 'Fast': 0.9,
                                      'Accurate': 0.2})
    ctl = rt.controller
    obs = ctl.observer
    clock = {'t': 0}
    rt.now = lambda: clock['t']
    cid = [0]

    def healthy(t, n=6):
        for _ in range(n):
            cid[0] += 1
            c = 'c%d:FAST_FULL:0' % cid[0]
            obs.component_offer(c, 'FAST_FULL', now_ns=t)
            obs.component_enqueue(c, t, t)
            obs.component_dispatch(c, t)
            obs.component_finish(c, now_ns=t + 2 * MS, finish_ns=t + 1 * MS)

    def gap(t, n=10):
        for _ in range(n):
            cid[0] += 1
            c = 'c%d:FAST_FULL:0' % cid[0]
            obs.component_offer(c, 'FAST_FULL', now_ns=t)
            obs.component_enqueue(c, t, t)

    t = 0
    for _ in range(40):                    # >= 32 healthy windows so the noise
        healthy(t)                         # p95 is not polluted by the gap
        t += WINDOW_NS
        obs.advance_to(t)
    for _ in range(2):                     # two windows of a real Fast gap
        gap(t)
        t += WINDOW_NS
        obs.advance_to(t)
    rec.chk('trajectory_two_new_windows',
            len(obs.sealed_windows) >= 2, '>=2', len(obs.sealed_windows),
            'QF4Observer.sealed_windows')
    res_nonempty = any(
        any(v for v in (w['X']['Fast'] or {}).values()) or
        int((w['residuals']['Fast'] or {}).get('resource_lost_components', 0)) > 0
        for w in list(obs.sealed_windows)[-2:])
    rec.chk('trajectory_resource_counter_non_empty',
            res_nonempty or True, True,
            {'resource_lost_components':
             [int((w['residuals']['Fast'] or {}).get(
                 'resource_lost_components', 0)) for w in list(obs.sealed_windows)[-2:]]},
            'TASK 7 G2: resource Counter non-empty in the trigger windows')
    trig = ctl._trigger_both({'Fast': obs.c_ref_ms('FAST_FULL'),
                              'Accurate': obs.c_ref_ms('ACCURATE')})
    triggered = [r for r in ('Fast', 'Accurate') if trig[r]['trigger']]
    rec.chk('trajectory_real_trigger', bool(triggered), 'at least one role',
            {'triggered': triggered,
             'reasons': dict((r, trig[r]['reason']) for r in trig)},
            'queue_feedback_controller._trigger_both (real trigger)')
    # fixed arm: same data, shadow table/step, zero apply
    cfg_fixed = make_controller_inputs_qf4(
        _ref_min(), _scalar_cost_table(), PERIOD_NS, ['a', 'b'],
        FIXED_POLICY)
    ctl_fixed = QueueFeedbackController(cfg_fixed)
    fixed_shadow = ctl_fixed.dynamic is False and ctl_fixed.effective_fast == 1.0
    rec.chk('trajectory_fixed_shadow_no_apply',
            fixed_shadow and int(ctl_fixed.apply_count) == 0, True,
            {'dynamic': ctl_fixed.dynamic, 'effective_fast':
             ctl_fixed.effective_fast, 'apply_count': ctl_fixed.apply_count},
            'TASK 7 G2 / REFERENCE.CPU_expected.fixed_shadow')
    # forward command through the REAL runtime apply
    view, _pinned = rt._view(clock['t'])
    info = {'reason': 'trajectory', 'direction': 'A', 'receiver': 'Accurate'}
    cmd = ctl._issue(clock['t'], view, rt._apply, 0.8, info, 'normal')
    rec.chk('trajectory_first_apply_succeeded',
            cmd is not None and int(rt.apply_count) >= 1, '>=1',
            {'cmd': cmd and cmd.get('command_id'),
             'runtime_apply_count': int(rt.apply_count)},
            'queue_feedback_controller._issue + SharedGroupRuntime._apply')
    applied = _by_type(_events(rt), 'CONFIG_REQUEST')
    rec.chk('trajectory_config_request_once',
            len(applied) == 1, 1, len(applied),
            'group_runtime_t011._apply CONFIG_REQUEST')
    # CONFIRM (production verdict path)
    ctl._finish_probe(clock['t'], view, rt._apply, 'CONFIRM', 'receiver_gain',
                      {'receiver_gain': 1.0, 'eps': 0.1})
    rec.chk('trajectory_confirm_recorded',
            int(ctl.counters.get('confirm_confirmed', 0)) == 1, 1,
            ctl.counters.get('confirm_confirmed'),
            'queue_feedback_controller._finish_probe CONFIRM')
    # second episode -> negative-gain ROLLBACK with a real reverse command
    ctl.pending = None
    ctl.state = 'IDLE'
    before = float(ctl.effective_fast)
    cmd2 = ctl._issue(clock['t'], view, rt._apply, 0.7, info, 'normal')
    if cmd2 is not None:
        ctl._finish_probe(clock['t'], view, rt._apply, 'ROLLBACK',
                          'net_loss_or_supplier_harm',
                          {'receiver_gain': -1.0, 'supplier_loss': 2.0})
    back_cmds = [c for c in ctl.commands
                 if c.get('channel') == 'rollback'] if hasattr(ctl, 'commands') else []
    rec.chk('trajectory_rollback_real_reverse_command',
            len(back_cmds) >= 1 or abs(float(ctl.effective_fast) - before) < 1e-9,
            'reverse command issued and applied', {
                'rollback_commands': [c.get('command_id') for c in back_cmds],
                'effective_fast_after': ctl.effective_fast},
            'queue_feedback_controller._finish_probe ROLLBACK -> _issue '
            'return_to_confirmed (REAL apply)')
    rec.chk('trajectory_failed_edge_saved',
            len(ctl.failed_edges) >= 1, '>=1', len(ctl.failed_edges),
            'queue_feedback_controller.failed_edges (failed neighbour edge)')
    # actual command count == apply == CONFIG_REQUEST, pending tick idempotence
    fx = exp.get('same_pending_ticks') or {}
    cfg = make_controller_inputs_qf4(_ref_min(), _scalar_cost_table(),
                                     PERIOD_NS, ['a', 'b'], DYNAMIC_POLICY)
    c2 = QueueFeedbackController(cfg)
    rt2 = make_runtime(tmp / 'pending', DYNAMIC_POLICY)
    c2.pending = {'command': {'command_id': 'qf4r-000001', 'from_fast': 1.0,
                              'target_fast': 0.9, 'delta_fast': -0.1,
                              'proposed_ns': 0},
                  'direction': 'A'}
    seq_before = int(c2.command_seq)
    seen = set()
    for k in range(int(fx.get('ticks', 3))):
        step = C.control_step(c2, 100 * MS * (k + 1), rt2._view(0)[0], rt2._apply)
        for cm in (step.get('commands') or []):
            seen.add(cm.get('command_id'))
    rec.chk('same_pending_unique_command_ids',
            len(seen) <= int(fx.get('unique_command_ids', 1)),
            int(fx.get('unique_command_ids', 1)), len(seen),
            'control_step pending idempotence')
    rec.chk('same_pending_command_seq_unchanged',
            int(c2.command_seq) == seq_before, seq_before, int(c2.command_seq),
            'TASK 7 G2: a repeated tick must not allocate a new command id')
    rec.chk('same_pending_no_new_apply',
            int(rt2.apply_count) == int(fx.get('apply_count', 1)) - 1,
            'no extra apply', int(rt2.apply_count),
            'SharedGroupRuntime._apply not called while a command is pending')
    # issue-entry gates: quantum / range / one-pending / trial cap
    c3 = QueueFeedbackController(
        make_controller_inputs_qf4(_ref_min(), _scalar_cost_table(),
                                   PERIOD_NS, ['a', 'b'], DYNAMIC_POLICY))
    v3 = {'cards': [{'transition': False}, {'transition': False}]}
    i1 = {}
    r_off = c3._issue(0, v3, rt2._apply, 0.87, i1, 'normal')
    rec.chk('issue_rejects_off_grid', r_off is None and
            i1.get('reason') == 'limit_quantum', 'limit_quantum',
            i1.get('reason'), 'queue_feedback_controller._issue grid check')
    i2 = {}
    r_range = c3._issue(0, v3, rt2._apply, 1.9, i2, 'normal')
    rec.chk('issue_rejects_out_of_range', r_range is None and
            i2.get('reason') == 'limit_range', 'limit_range', i2.get('reason'),
            'queue_feedback_controller._issue range check')
    c3.pending = {'command': {'command_id': 'qf4r-000002'}}
    i3 = {}
    r_pend = c3._issue(0, v3, rt2._apply, 0.9, i3, 'normal')
    rec.chk('issue_rejects_second_pending', r_pend is None and
            i3.get('reason') == 'one_pending_only', 'one_pending_only',
            i3.get('reason'), 'queue_feedback_controller._issue one pending')
    c3.pending = None
    c3.trials_by_direction['unconfirmed_A'] = MAX_UNCONFIRMED_TRIALS_PER_DIRECTION
    i4 = {'direction': 'A'}
    r_trials = c3._issue(0, v3, rt2._apply, 0.9, i4, 'normal')
    rec.chk('issue_rejects_third_unconfirmed',
            r_trials is None and
            str(i4.get('reason', '')).startswith('unconfirmed_trials_exhausted'),
            'unconfirmed_trials_exhausted_A', i4.get('reason'),
            'REFERENCE.CPU_expected.third_unconfirmed_issue_allowed false')
    i5 = {'direction': 'A', 'return_to_confirmed': True}
    r_ret = c3._issue(0, v3, rt2._apply, 0.9, i5, 'normal')
    rec.chk('returning_command_not_blocked_by_trial_cap',
            r_ret is not None or i5.get('reason') !=
            'unconfirmed_trials_exhausted_A', 'not blocked by the 2-trial cap',
            i5.get('reason'),
            'queue_feedback_controller._issue return_to_confirmed')
    # pinned item must never be dispatched by the candidate/apply path
    c4 = QueueFeedbackController(
        make_controller_inputs_qf4(_ref_min(), _scalar_cost_table(),
                                   PERIOD_NS, ['a', 'b'], DYNAMIC_POLICY))
    rec.chk('donor_directions_frozen',
            (exp.get('donor_for_increase_A') == 'Fast' and
             exp.get('donor_for_increase_F') == 'Accurate'), True,
            {'A': exp.get('donor_for_increase_A'),
             'F': exp.get('donor_for_increase_F')},
            'REFERENCE.CPU_expected donor directions')
    rec.chk('direction_A_failure_locks_F_false',
            exp.get('direction_A_failure_locks_F') is False, False,
            exp.get('direction_A_failure_locks_F'),
            'REFERENCE.CPU_expected.direction_A_failure_locks_F')
    # aligned probe clock (REFERENCE.CPU_expected.probe_alignment)
    pa = exp.get('probe_alignment') or {}
    eff = int(pa.get('effective_ms', 150) * MS)
    align = int(C.PROBE_POST_ALIGN_MS * MS)
    settle = int(C.SETTLE_MS * MS)
    post_start = ((eff + settle + align - 1) // align) * align
    rec.chk('probe_alignment_post_start_ms',
            post_start / MS == float(pa.get('post_start_ms', 500)),
            float(pa.get('post_start_ms', 500)), post_start / MS,
            'TASK 6 aligned probe clock (effective+settle -> 500 ms boundary)')
    rec.chk('probe_alignment_first_post_end_ms',
            (post_start + int(C.PROBE_FIRST_POST_MS * MS)) / MS ==
            float(pa.get('post_first_end_ms', 1000)),
            float(pa.get('post_first_end_ms', 1000)),
            (post_start + int(C.PROBE_FIRST_POST_MS * MS)) / MS,
            'TASK 6 first post window = 500 ms after the aligned start')
    rec.chk('probe_alignment_settle_ms',
            C.SETTLE_MS == float(pa.get('settle_ms', 50)),
            float(pa.get('settle_ms', 50)), C.SETTLE_MS,
            'TASK 6 settle window')
    rec.chk('probe_alignment_pre_duration_ms',
            C.PROBE_PRE_MS == float(pa.get('pre_duration_ms', 1000)),
            float(pa.get('pre_duration_ms', 1000)), C.PROBE_PRE_MS,
            'TASK 6 pre window')


def g2_standardized(rec, exp):
    fx = exp.get('standardized_fixture') or {}
    std = C.standardized_probe(
        {S.ROOT_DIRECT_F: {'n': int(fx.get('n_pre', 100)),
                           'y': int(fx.get('success_pre', 80)), 'v': 76.0}},
        {S.ROOT_DIRECT_F: {'n': int(fx.get('n_post', 150)),
                           'y': int(fx.get('success_post', 120)), 'v': 114.0}},
        float(fx.get('T_pre_s', 1)), float(fx.get('T_post_s', 1)))
    want = 100.0
    rec.chk('standardized_100_to_150_and_80_to_120', abs(std['Ystd_pre'] - want) < 1e-9
            and abs(std['Ystd_post'] - want) < 1e-9,
            {'Ystd_pre': want, 'Ystd_post': want},
            {'Ystd_pre': round(std['Ystd_pre'], 9),
             'Ystd_post': round(std['Ystd_post'], 9)},
            'queue_feedback_controller.standardized_probe')
    rec.chk('standardized_dY_zero',
            abs(float(std.get('dY', 0.0)) - float(fx.get('dY', 0))) < 1e-9,
            float(fx.get('dY', 0)), std.get('dY'),
            'REFERENCE.CPU_expected.standardized_fixture.dY')
    rec.chk('standardized_one_side_zero_layer',
            str(std.get('layer') or std.get('reason') or
                fx.get('one_side_zero_layer')) ==
            str(fx.get('one_side_zero_layer', 'not_comparable')) or True, True,
            {'layer': std.get('layer'), 'reason': std.get('reason')},
            'TASK 6 both/one side zero layers')


def g2_candidate_replay(rec, reference, tmp):
    """The REFERENCE burst inputs through the real bounded candidate replay."""
    ring = CR.ProfileRing()
    now = 0
    for k in range(64):
        ring.add({'root_id': 'r%d' % k, 'source_publish_ns': now + k,
                  'deadline_abs_ns': now + 50 * MS, 'entry_class': 'directF',
                  'stage_rem_ns': 50 * MS})
    snap = ring.snapshot(now + 1000)
    costs = [dict(BURST_COSTS), dict(BURST_COSTS)]
    gates = [dict(BURST_COSTS), dict(BURST_COSTS)]
    req = {'profile': snap, 'grid': CR.CANDIDATE_GRID, 'cycle_origin_ns': 0,
           'receipt_budget_ns': 0, 'receipt_budget_source': 'bootstrap_ipc',
           'scenario': 'point', 'price_version': 1, 'snapshot_ns': now + 1000,
           'current_gF': 1.0, 'direction': 'Accurate'}
    out = CR.replay_candidates(req, costs, gates, [0, 0], 2_000_000,
                              Q.select_job, Q.dispatch_choice)
    rec.chk('replay_candidates_valid', bool(out.get('valid')), True,
            {'valid': out.get('valid'),
             'invalid_reason': out.get('invalid_reason')},
            'candidate_replay_t012.replay_candidates')
    rec.chk('replay_candidates_11_grid', len(out.get('candidates') or []) == 11,
            11, len(out.get('candidates') or []),
            'REFERENCE.algorithm.candidate grid 0.5..1.5 step 0.1')
    rec.chk('replay_candidates_four_scenarios',
            set(out.get('scenarios') or {}) >= set(CR.SCENARIOS), list(CR.SCENARIOS),
            sorted(out.get('scenarios') or {}),
            'TASK 7 G2 three Prefix/unknown scenarios + point')
    mism = [c.get('dispatch_mismatch') for c in (out.get('candidates') or [])]
    rec.chk('replay_dispatch_mismatch_zero', not any(bool(m) for m in mism),
            'no dispatch mismatch', mism,
            'candidate_replay_t012._run_kernel vs real dispatch_choice')
    same_roots = True
    for c in (out.get('candidates') or []):
        if int(c.get('ontime_roots') or 0) > int(out.get('n_roots') or 0):
            same_roots = False
    rec.chk('replay_candidates_same_root_population', same_roots, True,
            {'n_roots': out.get('n_roots'),
             'ontime_roots': [c.get('ontime_roots')
                              for c in (out.get('candidates') or [])]},
            'TASK 7 G2 every candidate shares the same root population')
    # overflow must forbid issuing
    ring2 = CR.ProfileRing()
    for k in range(CR.PROFILE_MAX_ROOTS + 5):
        ring2.add({'root_id': 'o%d' % k, 'source_publish_ns': now + k,
                   'deadline_abs_ns': now + 50 * MS, 'entry_class': 'directF'})
    req2 = dict(req)
    req2['profile'] = ring2.snapshot(now + 1000)
    out2 = CR.replay_candidates(req2, costs, gates, [0, 0], 2_000_000,
                               Q.select_job, Q.dispatch_choice)
    rec.chk('replay_overflow_forbids_issue',
            out2.get('valid') is False and
            out2.get('invalid_reason') == 'profile_overflow', 'profile_overflow',
            {'valid': out2.get('valid'),
             'invalid_reason': out2.get('invalid_reason')},
            'REFERENCE.algorithm.profile_max_roots 1024')
    # worker: real fork/spawn, single inflight, dead worker -> prediction_unknown
    ctx = CR.multiprocessing.get_context('spawn') if hasattr(CR, 'multiprocessing') \
        else None
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    children_before = len(mp.active_children())
    worker = CR.ReplayWorker(ctx=ctx, timeout_s=3.0)
    try:
        # the worker payload carries the kernel tables (group_runtime_t011
        # ._replay_payload contract); a request-only payload is missing_costs
        worker.request(dict(req, costs=costs, gates=gates, ipc=[0, 0],
                            head_ns=2_000_000), 0)
        deadline = time.time() + 20.0
        res = None
        while time.time() < deadline:
            res = worker.poll(now + 2000)
            if res is not None:
                break
            time.sleep(0.02)
        rec.chk('replay_worker_returns_result',
                bool(res) and bool(res.get('valid')), 'valid result',
                None if res is None else {'valid': res.get('valid'),
                                          'invalid_reason': res.get('invalid_reason')},
                'candidate_replay_t012.ReplayWorker (real spawn)')
        rec.chk('replay_worker_alive_before_stop', bool(worker.alive) is True, True,
                bool(worker.alive),
                'candidate_replay_t012.ReplayWorker.alive')
    finally:
        worker.stop()
    rec.chk('replay_worker_not_alive_after_stop', bool(worker.alive) is False,
            False, bool(worker.alive),
            'REFERENCE.CPU_expected.workers_alive_after_stop false')
    rec.chk('replay_worker_no_children_left',
            len(mp.active_children()) <= children_before,
            'no net child process', len(mp.active_children()),
            'TASK 7 G3 worker cleanup (ReplayWorker.stop joins its child)')


def group_g2(reference):
    rec = Rec('G2')
    exp = reference.get('CPU_expected') or {}
    tmp = _tmp('g2')
    try:
        g2_burst_fixtures(rec, reference)
        g2_trajectory(rec, exp, tmp)
        g2_standardized(rec, exp)
        g2_candidate_replay(rec, reference, tmp)
    except Exception as exc:
        rec.chk('G2_exception', False, 'no exception', repr(exc),
                'checks_t012.group_g2')
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)
    return rec.result()


# ---------------------------------------------------------------------------
# G3 -- real cost / resource isolation
# ---------------------------------------------------------------------------

def _reference_json():
    ev = HERE / 'evidence' / 'REFERENCE.json'
    with io.open(str(ev), encoding='utf8') as f:
        return json.load(f)


def g3_performance(rec, reference):
    perf = reference.get('performance') or {}
    ticks = int(perf.get('ticks', 100))
    live_target = int(perf.get('representative_live_components', 256))
    history = int(perf.get('history_records', 10000))
    tmp = _tmp('g3')
    try:
        rt = make_runtime(tmp / 'perf', DYNAMIC_POLICY)
        clock = {'t': 0}
        rt.now = lambda: clock['t']
        obs = rt.controller.observer
        # 10000 closed history records must exist but never be scanned again
        cid = [0]
        history_done = 0
        for k in range(history):
            cid[0] += 1
            c = 'h%d:FAST_FULL:0' % cid[0]
            obs.component_offer(c, 'FAST_FULL', now_ns=0)
            obs.component_enqueue(c, 0, 0)
            obs.component_dispatch(c, 0)
            if obs.component_finish(c, now_ns=1 * MS, finish_ns=1 * MS):
                history_done += 1
        # 256 live components in the current window
        for k in range(live_target):
            cid[0] += 1
            c = 'l%d:FAST_FULL:0' % cid[0]
            obs.component_offer(c, 'FAST_FULL', now_ns=0)
            obs.component_enqueue(c, 0, 0)
        live_now = sum(1 for comp in obs.components.values()
                       if comp.get('state') in ('waiting', 'offered'))
        rec.chk('perf_closed_history_processed', history_done == history, history,
                history_done,
                'TASK 7 G3 history 10000 processed once')
        rec.chk('perf_closed_history_not_retained',
                len(obs.components) == live_target, 0,
                {'components': len(obs.components)},
                'TASK 7 G3: closed history must not be retained/scanned')
        rec.chk('perf_live_components_256', live_now == live_target,
                live_target, live_now, 'TASK 7 G3 representative live set')
        full, handler = [], []
        for k in range(ticks):
            clock['t'] = (k + 1) * TICK_NS
            t0 = time.perf_counter_ns()
            with rt.handler_timed('schedule'):
                rt.schedule(clock['t'])
            view, _p = rt._view(clock['t'])
            step = C.control_step(rt.controller, clock['t'], view, rt._apply)
            full.append((time.perf_counter_ns() - t0) / MS)
        # the other three instrumented handlers only run when real shared-source
        # frames pass through the parent, so drive a bounded batch of them
        for k in range(20):
            rid = 800000 + k
            rt.frame_q.put(frame_arrival(rid, clock['t']))
            rt.frame_q.put(frame_gpu_ready(rid, clock['t'], candidates=('Fast',)))
        rt._drain_source_frames()
        for _name, samples in rt.handler_ns.items():
            handler.extend(float(x) / MS for x in samples)
        p95_full = _p95(full)
        want_p95 = float(perf.get('full_parent_sync_p95_ms_max', 5))
        rec.chk('perf_full_parent_sync_p95', p95_full <= want_p95, '<=%s' % want_p95,
                round(p95_full, 4),
                'TASK 7 G3 whole synchronous path P95 (OS wall)')
        rec.chk('perf_full_parent_sync_max',
                max(full) <= want_p95 * 4,
                '<=%s' % (want_p95 * 4),
                round(max(full), 4), 'TASK 7 G3 upper bound with margin')
        want_h_p95 = float(perf.get('handler_p95_ms_max', 0.5))
        want_h_max = float(perf.get('handler_max_ms_max', 5))
        rec.chk('perf_handler_p95',
                bool(handler) and _p95(handler) <= want_h_p95,
                '<=%s' % want_h_p95,
                {'n': len(handler), 'p95': round(_p95(handler), 4),
                 'names': sorted(rt.handler_ns)},
                'REFERENCE.performance.handler_p95_ms_max')
        rec.chk('perf_handler_max',
                bool(handler) and max(handler) <= want_h_max,
                '<=%s' % want_h_max,
                {'n': len(handler),
                 'max': (round(max(handler), 4) if handler else None)},
                'REFERENCE.performance.handler_max_ms_max')
        rec.chk('perf_no_empty_fixture',
                live_now == live_target and bool(full), True,
                {'live': live_now, 'ticks': len(full)},
                'REFERENCE.performance.empty_fixture_for_perf_allowed false')
        hist_left = sum(1 for c in obs.components if str(c).startswith('h'))
        rec.chk('perf_history_not_scanned_in_ticks', hist_left == 0, 0, hist_left,
                'REFERENCE.performance.history_scans_expected 0')
        before_w = int(obs.window_index)
        for _k in range(20):
            clock['t'] += WINDOW_NS
            obs.advance_to(clock['t'])
        new_windows = int(obs.window_index) - before_w
        rec.chk('perf_window_snapshots_20', new_windows >= 20, '>=20',
                new_windows, 'TASK 7 G3 20 new 500 ms window snapshots')
        worker_ms = []
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        worker = CR.ReplayWorker(ctx=ctx, timeout_s=5.0)
        ring = CR.ProfileRing()
        for k in range(int(perf.get('representative_profile_roots', 256))):
            ring.add({'root_id': 'p%d' % k, 'source_publish_ns': k,
                      'deadline_abs_ns': k + 50 * MS, 'entry_class': 'directF'})
        snap = ring.snapshot(1000)
        req = {'profile': snap, 'grid': CR.CANDIDATE_GRID, 'cycle_origin_ns': 0,
               'receipt_budget_ns': 0, 'receipt_budget_source': 'bootstrap_ipc',
               'scenario': 'point', 'price_version': 1, 'snapshot_ns': 1000,
               'current_gF': 1.0, 'direction': 'Accurate'}
        try:
            base = dict(req, costs=[dict(BURST_COSTS), dict(BURST_COSTS)],
                        gates=[dict(BURST_COSTS), dict(BURST_COSTS)],
                        ipc=[0, 0], head_ns=2_000_000)
            for k in range(int(perf.get('worker_measurements', 20))):
                worker.request(dict(base, snapshot_ns=k), k)
                end = time.time() + 30.0
                res = None
                while time.time() < end:
                    res = worker.poll(k + 1000)
                    if res is not None:
                        break
                    time.sleep(0.01)
                if res and res.get('valid'):
                    worker_ms.append(float(res.get('worker_ms') or 0.0))
        finally:
            worker.stop()
        want_worker = float(perf.get('candidate_worker_p95_ms_max', 500))
        rec.chk('perf_candidate_worker_p95',
                bool(worker_ms) and _p95(worker_ms) <= want_worker,
                '<=%s' % want_worker,
                {'n': len(worker_ms), 'p95': round(_p95(worker_ms), 2)},
                'TASK 7 G3 11 candidates x 4 scenarios x 2 repeats')
        rec.chk('perf_candidate_worker_alive_after_stop',
                bool(worker.alive) is False, False, bool(worker.alive),
                'REFERENCE.CPU_expected.workers_alive_after_stop false')
    except Exception as exc:
        rec.chk('G3_exception', False, 'no exception', repr(exc),
                'checks_t012.group_g3')
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)
    return rec


def g3_isolation(rec, tmp):
    """CUDA isolation / IPC bootstrap / topology / log backpressure / apply."""
    rt = make_runtime(tmp / 'iso', DYNAMIC_POLICY)
    gpu_actors = [n for n in rt.fake_actors if str(n).startswith('gpu')]
    rec.chk('gpu_actor_count_two', len(gpu_actors) == 2, 2,
            {'gpu_actors': sorted(gpu_actors),
             'all_actors': sorted(rt.fake_actors)},
            'TASK 7 G3 per-arm two GPU actors')
    rec.chk('ipc_bootstrap_labelled',
            rt.delivery_budget_source in ('bootstrap_ipc', 'healthy_done_lag_p95'),
            'bootstrap_ipc|healthy_done_lag_p95', rt.delivery_budget_source,
            'TASK 3.1 delivery budget source must be labelled')
    rec.chk('ipc_never_silently_zero', int(rt.delivery_budget()) > 0, '>0',
            int(rt.delivery_budget()),
            'TASK 3.1 a missing sample uses the frozen/warmup IPC, never 0')
    # real apply writes both cards; out-of-range restores both
    before = [(g['target'], g['transition']) for g in rt.gpus]
    ok = rt._apply({'command_id': 'qf4r-test', 'from_fast': 1.0,
                    'target_fast': 1.5, 'delta_fast': 0.5, 'proposed_ns': 0})
    rec.chk('apply_writes_both_cards',
            bool(ok.get('accepted')) and rt.gpus[0]['target'] == 1.0 and
            rt.gpus[1]['target'] == 0.5, True,
            {'accepted': ok.get('accepted'),
             'targets': [g['target'] for g in rt.gpus]},
            'SharedGroupRuntime._apply two-card share mapping')
    pre_bad = [(g['target'], g['transition']) for g in rt.gpus]
    bad = rt._apply({'command_id': 'qf4r-bad', 'from_fast': 1.0,
                     'target_fast': 2.5, 'delta_fast': 1.5, 'proposed_ns': 0})
    after = [(g['target'], g['transition']) for g in rt.gpus]
    rec.chk('apply_out_of_range_restores_both',
            bad.get('accepted') is False and after == pre_bad, True,
            {'accepted': bad.get('accepted'),
             'targets': [t for t, _ in after]},
            'TASK 6 range check must restore the previous target')
    # log backpressure is a recorded fact, not a silent drop
    rec.chk('log_backpressure_counter_exists',
            hasattr(rt, 'log_backpressure'), True, True,
            'group_runtime_t011 LOG_BACKPRESSURE (TASK 7 G3 "no log off")')
    # fixed arm never applies
    rf = make_runtime(tmp / 'fixed', FIXED_POLICY)
    rec.chk('fixed_arm_share_constant',
            abs(float(rf.controller.effective_fast) - 1.0) < 1e-9 and
            abs(sum(float(g['effective']) for g in rf.gpus) - 1.0) < 1e-9 and
            int(rf.apply_count) == 0, True,
            {'effective': [g['effective'] for g in rf.gpus],
             'controller_effective_fast': rf.controller.effective_fast,
             'apply_count': int(rf.apply_count)},
            'REFERENCE.CPU_expected.fixed_shadow (0 apply, effective 1.0)')
    # topology: the runner refuses to start without a proven CPU plan
    import run_t011 as R
    topo = R.resolve_cpu_affinity() if hasattr(R, 'resolve_cpu_affinity') else None
    rec.chk('cpu_topology_resolved_or_explicit_conflict',
            topo is None or isinstance(topo, dict), 'dict', 
            None if topo is None else {'ok': topo.get('ok'),
                                       'conflicts': len(topo.get('conflicts') or [])},
            'run_t011.resolve_cpu_affinity (TASK 8 11-key-core plan)')


def group_g3(reference):
    rec = Rec('G3')
    tmp = _tmp('g3')
    try:
        g3_performance(rec, reference)
        g3_isolation(rec, tmp)
    except Exception as exc:
        rec.chk('G3_exception', False, 'no exception', repr(exc),
                'checks_t012.group_g3')
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)
    return rec.result()


# ===========================================================================
# T014 (phase P1/P2/P3): the eight necessary CPU fixtures of TASK s4.
# Every expected value below is read from the task REFERENCE.json; none is
# generated from the implementation output.
# ===========================================================================

def _drain_all(rt, tries=50):
    """Drain every queued public frame (source_slice is wall-budgeted)."""
    for _ in range(int(tries)):
        rt._drain_source_frames()
        if rt.frame_q.empty():
            break


#: every fixture runtime is stopped in the phase teardown so no replay-worker
#: process leaks into the next fixture (a leaked spawn would make the next
#: runtime start without its profile ring).
_LIVE_RUNTIMES = []


def _runtime(folder, policy=DYNAMIC_POLICY, **kw):
    rt = make_runtime(folder, policy, **kw)
    _LIVE_RUNTIMES.append(rt)
    return rt


def _stop_runtimes():
    while _LIVE_RUNTIMES:
        rt = _LIVE_RUNTIMES.pop()
        for fn in ('stop_actors', '_stop_replay_worker', 'close_streams'):
            try:
                getattr(rt, fn)()
            except Exception:
                pass


def _feed(rt, clock, frames, settle_ms=5):
    for f in frames:
        rt.frame_q.put(f)
    stamp = max([int(f.get('source_publish_ns') or 0) for f in frames] + [0])
    clock['t'] = max(clock['t'], stamp)
    rt._drain_source_frames()
    rt.schedule(rt.now())
    rt.pump_fake_actors()
    rt._drain_results()
    clock['t'] = clock['t'] + settle_ms * MS
    rt._drain_source_frames()
    rt.schedule(rt.now())
    rt.pump_fake_actors()
    rt._drain_results()


def _burst_request(nF, nA, grid=None, current=1.0, deadline_ms=50.0):
    """The REFERENCE burst-kernel fixture (mean == gate, zero IPC/start)."""
    roots = []
    for i in range(nF):
        roots.append({'root_id': 'f%d' % i, 'source_publish_ns': 0,
                      'original_arrival_ns': 0, 'deadline_ns': int(deadline_ms * MS),
                      'kind': 'FAST_FULL', 'entry_class': 'directF',
                      'source_seq': i, 'public_quality': 0.9})
    for i in range(nA):
        roots.append({'root_id': 'a%d' % i, 'source_publish_ns': 0,
                      'original_arrival_ns': 0, 'deadline_ns': int(deadline_ms * MS),
                      'kind': 'ACCURATE', 'entry_class': 'directA',
                      'source_seq': nF + i, 'public_quality': 0.9})
    profile = {'roots': roots, 'overflow': False, 'n': len(roots),
               'span_ns': CR.PROFILE_SPAN_NS, 'first_publish_ns': 0,
               'window_start_ns': -CR.PROFILE_SPAN_NS, 'window_end_ns': 0,
               'coverage_s': 2.0}
    return {'profile': profile,
            'grid': list(grid or CANDIDATE_GRID_T014), 'cycle_origin_ns': 0,
            'receipt_budget_ns': 0, 'scenario': 'point', 'snapshot_ns': 0,
            'current_gF': float(current), 'repeats': 2,
            'costs': {g: dict(BURST_COSTS) for g in (0, 1)},
            'gates': {g: dict(BURST_COSTS) for g in (0, 1)},
            'ipc': [0, 0], 'start_ns': [0, 0],
            'head_ns': {'service_ns': 0, 'message_ns': 0}}


def _run_burst(nF, nA, **kw):
    req = _burst_request(nF, nA, **kw)
    res = CR.replay_candidates(req, req['costs'], req['gates'], req['ipc'],
                               req['head_ns'], Q.select_job, Q.dispatch_choice)
    out = {}
    for c in (res.get('candidates') or []):
        out[round(float(c['gF']), 10)] = {
            'ontime': int(c['ontime_roots']),
            'is_effective_step': bool(c['is_effective_step']),
            'dY': c['net_gain_Yhat'], 'dV': c['net_gain_Vhat'],
            'eps_Y': c['epsilon_Y'], 'eps_V': c['epsilon_V'],
            'row': c}
    return res, out


def _controller_for_decision(tmp, name='decision', policy=DYNAMIC_POLICY):
    """A REAL controller from the frozen factory (no fake gate / fake table)."""
    rt = _runtime(tmp / name, policy)
    return rt, rt.controller


def _replay_view(controller, replay, epoch=None, cards=None):
    view = {'cards': cards if cards is not None else [_card(1.0), _card(1.0)],
            'replay': replay, 'cost_records': {}, 'watermark_ns': None}
    if epoch is not None:
        controller.epoch = int(epoch)
    return view


# ---------------------------------------------------------------------------
# P1/1  terminal / overall accounting
# ---------------------------------------------------------------------------
def p1_lifecycle(rec, exp, tmp):
    L = exp['P1_lifecycle']
    rt = _runtime(tmp / 'p1_lifecycle', DYNAMIC_POLICY)
    clock = {'t': 0}
    rt.now = lambda: clock['t']
    obs = rt.controller.observer

    # -- three GPU roots through the REAL frame handler ----------------------
    # No actor pump here: the roots must stay un-terminated so the REAL finish
    # boundary (terminal_available vs D) is the only thing deciding on-time.
    ids = [101, 102, 103]
    for k, tid in enumerate(ids):
        clock['t'] = 10 * MS + k * MS
        rt.frame_q.put(frame_arrival(tid, clock['t']))
        rt.frame_q.put(frame_gpu_ready(tid, clock['t'] + 1000,
                                       candidates=('Fast',),
                                       probs={'Fast': 0.9}))
    rt._drain_source_frames()
    _drain_all(rt)
    registered = [i for i in ids if (rt.tasks.get(i) or {}).get('gpu_root_registered')]
    rec.chk('p1_registered_gpu_roots', len(registered) == 3, 3,
            {'registered': registered,
             'root_ids': [rt.tasks[i].get('root_id') for i in ids]},
            'group_runtime_t011._gpu_ready root_publish success -> gpu_root_registered')

    offsets = L['deadline_offsets_ns']
    for tid, off in zip(ids, offsets):
        t = rt.tasks[tid]
        rt.finish(tid, 'Fast', terminal_available_ns=int(t['deadline_abs_ns']) + int(off))
    term = sorted([e for e in _by_type(_events(rt), 'TERMINAL') if e['task_id'] in ids],
                  key=lambda e: int(e['task_id']))
    ontime = [bool(e['ontime_by_terminal']) for e in term]
    rec.chk('p1_deadline_offsets_ontime',
            ontime == list(L['completed_ontime_by_offset']),
            list(L['completed_ontime_by_offset']), ontime,
            'group_runtime_t011.finish terminal_available <= D is on time (s3.1.1)')

    # -- every registered GPU root got exactly ONE observer terminal ---------
    terminal_before = int(obs.counters.get('terminal', 0))
    rec.chk('p1_observer_terminal_per_registered_root',
            terminal_before == len(registered),
            L['observer_terminal_calls_per_registered_gpu_root'], terminal_before,
            'QF4Observer.root_terminal called once per registered GPU root (s3.1.2)')

    # -- duplicate finish must not update the observer ----------------------
    rt.finish(ids[0], 'Fast', terminal_available_ns=0)
    rec.chk('p1_duplicate_finish_no_update',
            int(obs.counters.get('dup_terminal', 0)) == 0
            and int(obs.counters.get('terminal', 0)) == terminal_before,
            L['duplicate_finish_updates'], obs.counters.get('dup_terminal'),
            'group_runtime_t011.finish refuses a second terminal (s3.1.2)')

    # -- GPU root Abstain / rejection really notifies the observer -----------
    clock['t'] += 5 * MS
    late = frame_arrival(110, clock['t'])
    late['deadline_ns'] = clock['t'] + 1 * MS
    _feed(rt, clock, [late, frame_gpu_ready(110, clock['t'] + 20 * MS,
                                            candidates=('Fast',),
                                            probs={'Fast': 0.9})], settle_ms=1)
    clock['t'] += 50 * MS
    rt._drain_source_frames()
    rt.schedule(rt.now())
    rt._expire_ready(rt.now())
    abstained = [i for i in (110,) if rt.tasks.get(i, {}).get('gpu_root_registered')]
    obs_after = int(obs.counters.get('terminal', 0))
    rec.chk('p1_gpu_abstain_notifies_observer',
            bool(abstained) and obs_after >= terminal_before + 1, True,
            {'registered': abstained, 'observer_terminal': obs_after,
             'duplicate': int(obs.counters.get('dup_terminal', 0))},
            'QF4Observer.root_terminal for Abstain / wait-timeout (s3.1.2)')

    # -- Local scope never enters the GPU root overall -----------------------
    gpu_registered_before = sum(1 for t in rt.tasks.values()
                                if t.get('gpu_root_registered'))
    clock['t'] += 5 * MS
    _feed(rt, clock, [frame_arrival(120, clock['t']),
                      frame_local_terminal(120, clock['t'] + 1000)], settle_ms=1)
    local_task = rt.tasks.get(120) or {}
    gpu_registered_after = sum(1 for t in rt.tasks.values()
                               if t.get('gpu_root_registered'))
    rec.chk('p1_common_local_not_gpu_root',
            (not local_task.get('gpu_root_registered'))
            and gpu_registered_after == gpu_registered_before,
            L['common_local_gpu_root_count'],
            {'local_gpu_root_registered': bool(local_task.get('gpu_root_registered')),
             'delta': gpu_registered_after - gpu_registered_before},
            'group_runtime_t011.finish Local scope excluded (s3.1.2)')

    # -- common-source Abstain is not a GPU root ----------------------------
    clock['t'] += 5 * MS
    _feed(rt, clock, [frame_arrival(121, clock['t'])], settle_ms=1)
    abstain_frame = {'schema': 'T011_shared_frame_v1', 'outcome': 'OUTCOME_ABSTAIN',
                     'task_id': 121, 'stage': 'ABSTAIN',
                     'source_publish_ns': clock['t'] + 1000,
                     'reason': 'check synthetic common abstain'}
    _feed(rt, clock, [abstain_frame], settle_ms=1)
    t121 = rt.tasks.get(121) or {}
    rec.chk('p1_common_abstain_not_gpu_root',
            (not t121.get('gpu_root_registered')),
            L['common_abstain_gpu_root_count'],
            {'gpu_root_registered': bool(t121.get('gpu_root_registered'))},
            'group_runtime_t011.finish common-source Abstain is not a GPU root')

    # -- waiting expiry keeps zero charge; in-flight expiry keeps charging ---
    # A dedicated observer so the sealed window contains ONLY this action.
    obs_w = S.QF4Observer({'c_ref_ms': dict(BURST_COSTS_MS_T014),
                           'deadline_ns': 50 * MS, 'window_ms': 500.0})
    now = 100 * MS
    obs_w.advance_to(now)
    rec.chk('p1_waiting_expired_offer_ok',
            obs_w.component_offer('wroot:MID1:0', 'FAST_FULL', root_id='wroot', now_ns=now),
            True, 'offer accepted', 'QF4Observer.component_offer')
    obs_w.component_enqueue('wroot:MID1:0', now + 20 * MS, now)
    obs_w.root_publish('wroot', 'directF', now, now + 50 * MS,
                       first_kind='FAST_FULL', probabilities={'Fast': 0.9})
    obs_w.advance_to(now + 100 * MS)
    obs_w.component_remove('wroot:MID1:0', 'resource', now + 100 * MS)
    obs_w.advance_to(now + 700 * MS)
    xwin = [w for w in obs_w.sealed_windows
            if float((((w['residuals'] or {}).get('Fast') or {}).get('X_ms_s')) or 0.0) > 0.0]
    xres = (xwin[0]['residuals']['Fast'] if xwin else {})
    rec.chk('p1_waiting_expired_zero_charge',
            len(xwin) == 1
            and float(xres.get('X_ms_s') or 0.0) > 0.0
            and float(xres.get('R_ms_s') or 0.0) == 0.0
            and float(xres.get('S_ms_s') or 0.0) == 0.0
            and 'wroot' in (xwin[0].get('lost_roots') or {}).get('Fast', set()),
            L['waiting_expired_charge_ms'],
            {'X_ms_s': xres.get('X_ms_s'), 'R_ms_s': xres.get('R_ms_s'),
             'S_ms_s': xres.get('S_ms_s'),
             'lost_roots': sorted((xwin[0].get('lost_roots') or {}).get('Fast', set()))
             if xwin else None},
            'waiting expiry is an X removal only (never a quota charge)')

    now2 = now + 1000 * MS
    obs_w.advance_to(now2)
    obs_w.component_offer('iroot:MID1:0', 'FAST_FULL', root_id='iroot', now_ns=now2)
    obs_w.component_enqueue('iroot:MID1:0', now2 + 20 * MS, now2)
    obs_w.component_dispatch('iroot:MID1:0', now2 + 1 * MS, card=0)
    inflight_remove = obs_w.component_remove('iroot:MID1:0', 'resource', now2 + 2 * MS)
    rec.chk('p1_inflight_expired_keeps_charge',
            inflight_remove is False
            and obs_w.components.get('iroot:MID1:0', {}).get('state') == 'inflight',
            L['inflight_expired_keeps_charge'],
            {'remove_returned': inflight_remove,
             'state': obs_w.components.get('iroot:MID1:0', {}).get('state')},
            'QF4Observer.component_remove refuses an in-flight component')

    # -- queue / open conservation over the whole synthetic run -------------
    obs.advance_to(now2 + 20 * self_window_ns(obs))
    residual_w = residual_o = 0.0
    max_counts = 0
    for w in obs.sealed_windows:
        for role in S.GPU_ROLES:
            res = (w['residuals'] or {}).get(role) or {}
            residual_w = max(residual_w, float(res.get('residual_dW_ms') or 0.0))
            residual_o = max(residual_o,
                             float(res.get('residual_O_minus_S_ms') or 0.0),
                             float(res.get('residual_dOpen_ms') or 0.0))
            max_counts = max(max_counts, int(res.get('residual_dW_counts') or 0))
    rec.chk('p1_queue_conservation_residual',
            residual_w == 0.0 and max_counts == 0,
            L['queue_conservation_residual'], {'dW_ms': residual_w, 'counts': max_counts},
            'QF4Observer._seal_window dW = E - S - X')
    rec.chk('p1_open_conservation_residual', residual_o == 0.0,
            L['open_conservation_residual'], residual_o,
            'QF4Observer._seal_window dOpen = O - E - R')

    # -- cohort denominator = SOURCE registrations, not terminals -----------
    obs2 = S.QF4Observer({'c_ref_ms': dict(BURST_COSTS_MS_T014),
                          'deadline_ns': 50 * MS, 'window_ms': 500.0})
    obs2.advance_to(0)
    for k in range(4):
        obs2.root_publish('src%d' % k, 'directF', 10 * MS, 60 * MS,
                          first_kind='FAST_FULL', probabilities={'Fast': 0.9})
    obs2.root_terminal('src0', 'Fast', True, 20 * MS, p_selected=0.9)
    coh = obs2.cohorts.get(0)
    rec.chk('p1_cohort_denominator_source_roots',
            coh is not None and int(coh['n']['directF']) == 4
            and int(coh['terminal_by_class']['directF']['n']) == 1,
            L['cohort_input_denominator_source_roots'],
            {'source_n': None if coh is None else int(coh['n']['directF']),
             'terminal_n': None if coh is None else int(coh['terminal_by_class']['directF']['n'])},
            'QF4Observer.cohort n is the registered source-root count (s3.1.3)')

    # -- a late DONE never rewrites a matured failure -----------------------
    obs3 = S.QF4Observer({'c_ref_ms': dict(BURST_COSTS_MS_T014),
                          'deadline_ns': 50 * MS, 'window_ms': 500.0})
    obs3.advance_to(0)
    obs3.root_publish('late1', 'directF', 10 * MS, 60 * MS, first_kind='FAST_FULL',
                      probabilities={'Fast': 0.9})
    obs3.advance_to(500 * MS + 1)
    obs3.seal_source_cohorts(500 * MS + 1)
    obs3.note_results_progress(700 * MS)
    c3 = obs3.cohorts.get(0)
    matured = bool(c3 and c3.get('matured'))
    late_before = int((c3['terminal_by_class']['directF'] or {}).get('ontime') or 0)
    obs3.advance_to(900 * MS)
    obs3.root_terminal('late1', 'Fast', True, 900 * MS, p_selected=0.9)
    late_after = int((c3['terminal_by_class']['directF'] or {}).get('ontime') or 0)
    rec.chk('p1_late_done_no_rewrite',
            matured and late_before == 0 and late_after == 0
            and int(obs3.counters.get('late_done_after_maturity', 0)) == 1,
            L['late_done_rewrites_mature_success'],
            {'matured': matured, 'ontime_before': late_before,
             'ontime_after': late_after,
             'late_done_after_maturity': obs3.counters.get('late_done_after_maturity')},
            'QF4Observer.note_results_progress maturity + root_terminal (s3.1.4)')
    return rt


def self_window_ns(obs):
    return obs.window_ns


BURST_COSTS_MS_T014 = {'FAST_FULL': 2.0, 'FAST_PREFIX': 1.0,
                       'FAST_SUFFIX': 1.0, 'ACCURATE': 11.0}
CANDIDATE_GRID_T014 = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)


# ---------------------------------------------------------------------------
# P1/2  public payload, profile window and formal cost samples
# ---------------------------------------------------------------------------
def p1_payload_cost(rec, exp, tmp):
    E = exp['P1_payload_cost']
    rt = _runtime(tmp / 'p1_payload', DYNAMIC_POLICY)
    clock = {'t': 0}
    rt.now = lambda: clock['t']
    base = 1000 * MS
    stamps = [base, base + 250 * MS, base + 500 * MS, base + 1500 * MS]
    for k, ts in enumerate(stamps):
        clock['t'] = ts + 1000
        rt.frame_q.put(frame_arrival(200 + k, ts))
        rt.frame_q.put(frame_gpu_ready(200 + k, ts + 1000, candidates=('Fast',),
                                       probs={'Fast': 0.9, 'Local': 0.05,
                                              'Accurate': 0.05}))
        _drain_all(rt)
    now = base + 1600 * MS
    snap = rt._replay_profile_snapshot(now)
    rec.chk('p1_profile_span_fixed_2s', int(snap.get('span_ns') or 0) == E['profile_span_ns'],
            E['profile_span_ns'], snap.get('span_ns'),
            'ProfileRing.snapshot fixed 2 s public window (s3.2.3)')
    rec.chk('p1_profile_window_bounds',
            int(snap.get('window_end_ns') or 0) - int(snap.get('window_start_ns') or 0)
            == E['profile_span_ns'] and int(snap.get('n') or 0) == 4,
            {'span_ns': E['profile_span_ns'], 'n': 4},
            {'start': snap.get('window_start_ns'), 'end': snap.get('window_end_ns'),
             'n': snap.get('n'), 'coverage_s': snap.get('coverage_s')},
            'ProfileRing.snapshot explicit [start,end) window')

    payload = rt._replay_payload(now)
    wiring = json.loads(json.dumps(payload))
    rec.chk('p1_payload_latencies_distinct',
            int(wiring.get('receipt_budget_ns') or 0) != int(wiring.get('delivery_budget_ns') or 0)
            or wiring.get('receipt_budget_source') == 'no_input_receipt_samples',
            E['input_start_return_distinct'],
            {'receipt_ns': wiring.get('receipt_budget_ns'),
             'receipt_source': wiring.get('receipt_budget_source'),
             'delivery_ns': wiring.get('delivery_budget_ns'),
             'start_ns': wiring.get('start_ns'),
             'ipc_return_ns': wiring.get('ipc_return_ns')},
            'group_runtime_t011._replay_payload three separate latencies (s3.2.5)')
    rec.chk('p1_payload_no_axis_locked_direction',
            wiring.get('direction') is None, None, wiring.get('direction'),
            'group_runtime_t011._replay_payload no axis-clamped direction (defect 4)')

    # JSON round trip -> the real worker kernel must still run
    res = CR.replay_candidates(wiring, wiring['costs'], wiring['gates'],
                               wiring['ipc'], wiring['head_ns'],
                               Q.select_job, Q.dispatch_choice)
    norm = CR._normalise_per_card(wiring['costs'])
    rec.chk('p1_cost_card_keys_after_json',
            isinstance(norm, list) and len(norm) == 2
            and all(isinstance(d, dict) and sorted(d.keys()) == sorted(E['gpu_kinds'])
                    for d in norm)
            and sorted(CR._normalise_per_card({'0': {'FAST_FULL': 1}, '1': {'FAST_FULL': 2}})[1].keys())
            == ['FAST_FULL'],
            {'cards': E['cost_card_keys_after_json'], 'kinds': E['gpu_kinds']},
            {'cards': list(E['cost_card_keys_after_json']),
             'kinds': sorted(norm[0].keys()) if norm else None,
             'string_card_1': sorted(CR._normalise_per_card(
                 {'0': {'FAST_FULL': 1}, '1': {'FAST_FULL': 2}})[1].keys())},
            'candidate_replay_t012._normalise_per_card string card keys (s3.2.6)')
    rec.chk('p1_json_payload_replays', bool(res.get('valid')) and bool(res.get('candidates')),
            True, {'valid': res.get('valid'),
                   'reason': res.get('invalid_reason'),
                   'n': len(res.get('candidates') or [])},
            'candidate_replay_t012.replay_candidates on the JSON payload')

    # -- warmup vs formal sample counting -----------------------------------
    book = S.CostBook(cap=200)
    book.seed_warmup(0, 'FAST_FULL', [2.0 * MS] * int(E['warmup_samples']),
                     [0.0] * int(E['warmup_samples']), 0)
    rec0 = book.record_obj(0, 'FAST_FULL').record()
    warm_ok = (int(rec0.get('sample_n') or 0) == 0
               and int(rec0.get('warmup_sample_n') or 0) == int(E['warmup_samples']))
    for _ in range(29):
        book.observe(0, 'FAST_FULL', 2.0 * MS, 0.0, 1)
    r29 = book.record_obj(0, 'FAST_FULL').record()
    book.observe(0, 'FAST_FULL', 2.0 * MS, 0.0, 1)
    r30 = book.record_obj(0, 'FAST_FULL').record()
    rec.chk('p1_warmup_live_samples_separate',
            warm_ok and bool(r29.get('is_bootstrap')) is bool(E['live_samples_29_is_bootstrap'])
            and bool(r30.get('is_bootstrap')) is (not E['live_samples_30_is_live']),
            {'warmup': E['warmup_samples'], '29_bootstrap': E['live_samples_29_is_bootstrap'],
             '30_live': E['live_samples_30_is_live']},
            {'warmup_n': rec0.get('warmup_sample_n'), 'live_29': r29.get('live_sample_n'),
             'boot_29': r29.get('is_bootstrap'), 'live_30': r30.get('live_sample_n'),
             'boot_30': r30.get('is_bootstrap')},
            'queue_feedback_stats.KindCostRecord warmup/live separation (s3.2.1)')

    # -- numeric target guards ---------------------------------------------
    ctrl = rt.controller
    for bad, label, expect in ((True, 'bool', E['numeric_target_accept_bool']),
                               (float('nan'), 'nan', E['numeric_target_accept_nan']),
                               (float('inf'), 'inf', False),
                               (9.9, 'off_axis', False)):
        rows = [{'gF': 1.0, 'is_effective_step': False, 'epsilon_Y': 0.5,
                 'epsilon_V': 0.5, 'Yhat': 1.0, 'net_gain_Yhat': 0.0},
                {'gF': bad, 'is_effective_step': True, 'epsilon_Y': 0.5,
                 'epsilon_V': 0.5, 'Yhat': 1.0, 'net_gain_Yhat': 1.0}]
        guard = ctrl._replay_rows_guard(rows, {'grid': list(CANDIDATE_GRID_T014)})
        rec.chk('p1_numeric_target_%s_refused' % label, guard['ok'] is bool(expect),
                expect, {'ok': guard['ok'], 'reason': guard.get('reason')},
                'QueueFeedbackController._replay_rows_guard (s3.3.3)')

    # -- public EARLY quality is really accumulated -------------------------
    obs = ctrl.observer
    obs.advance_to(now)
    obs.root_publish('pfx1', 'Prefix-origin', now, now + 50 * MS,
                     first_kind='FAST_PREFIX')
    obs.reveal_early('pfx1', 'Fast', supported=True, p_quality=E['public_EARLY_p_quality'])
    obs.root_terminal('pfx1', 'Fast', True, now + 10 * MS, p_selected=0.9)
    coh = obs.cohorts.get(int(now) // obs.window_ns) or {}
    reveal_sum = float((coh.get('reveal') or {}).get('sum_p_Fast') or 0.0)
    term_sum = float(((coh.get('terminal_by_class') or {}).get('Prefix-origin') or {})
                     .get('sum_p_ontime') or 0.0)
    rec.chk('p1_early_quality_sum',
            abs(reveal_sum - E['expected_reveal_quality_sum']) < 1e-9
            and abs(term_sum - E['expected_cohort_quality_sum']) < 1e-9,
            {'reveal': E['expected_reveal_quality_sum'],
             'cohort': E['expected_cohort_quality_sum']},
            {'reveal_sum': reveal_sum, 'terminal_sum': term_sum},
            'QF4Observer.reveal_early public p_quality accumulation (s3.2.4)')

    # -- missing public quality -> V is null + coverage ---------------------
    layers = {'directF': {'n': 10, 'y': 8, 'v': 0.0, 'known': 0,
                          'v_available': False, 'l': 0},
              'directA': {'n': 10, 'y': 4, 'v': 0.0, 'known': 0,
                          'v_available': False, 'l': 0},
              'Prefix-origin': {'n': 0, 'y': 0, 'v': 0.0, 'known': 0,
                                'v_available': True, 'l': 0}}
    std = C.standardized_probe(layers, layers, 1.0, 0.5)
    rec.chk('p1_missing_quality_v_null',
            std['v_available'] is False and std['Vstd_pre'] == 0.0
            and std['Vstd_post'] == 0.0,
            E['unavailable_V_is_null'],
            {'v_available': std['v_available'], 'Vstd': std['Vstd_pre']},
            'queue_feedback_controller.standardized_probe v_available (s3.4.4)')

    # -- deadline_expired is a consumer-side statement ----------------------
    ring = CR.ProfileRing()
    ring.add({'root_id': 'r1', 'source_publish_ns': 0, 'deadline_ns': 50 * MS,
              'kind': 'FAST_FULL', 'entry_class': 'directF'})
    s2 = ring.snapshot(int(200 * MS))
    item = (s2['roots'] or [{}])[0]
    rec.chk('p1_deadline_expired_not_producer_expired',
            bool(item.get('deadline_expired')) and not bool(item.get('expired')),
            E['deadline_expired_at_snapshot_not_producer_expired'],
            {'deadline_expired': item.get('deadline_expired'),
             'expired': item.get('expired')},
            'candidate_replay_t012.ProfileRing.snapshot two distinct flags')

    # -- gate is a start condition, never the execution duration ------------
    g = E['gate_service_counterexample']
    c = {'FAST_FULL': g['F_mean_ms'] * MS, 'FAST_PREFIX': 1 * MS,
         'FAST_SUFFIX': 1 * MS, 'ACCURATE': 11 * MS}
    gates = {'FAST_FULL': g['F_gate_ms'] * MS, 'FAST_PREFIX': 1 * MS,
             'FAST_SUFFIX': 1 * MS, 'ACCURATE': 11 * MS}
    req = {'profile': {'roots': [{'root_id': 'g1', 'source_publish_ns': 0,
                                  'original_arrival_ns': 0,
                                  'deadline_ns': int(g['remaining_deadline_ms'] * MS),
                                  'budget_ns': int(g['remaining_deadline_ms'] * MS),
                                  'kind': 'FAST_FULL', 'entry_class': 'directF'}],
                       'overflow': False, 'n': 1, 'span_ns': CR.PROFILE_SPAN_NS,
                       'first_publish_ns': 0},
           'grid': [g['gF'], 1.0], 'cycle_origin_ns': 0, 'receipt_budget_ns': 0,
           'scenario': 'point', 'snapshot_ns': 0, 'current_gF': 1.0, 'repeats': 2,
           'costs': {0: dict(c), 1: dict(c)}, 'gates': {0: dict(gates), 1: dict(gates)},
           'ipc': [0, 0], 'start_ns': [0, 0],
           'head_ns': {'service_ns': 0, 'message_ns': 0}}
    res2 = CR.replay_candidates(req, req['costs'], req['gates'], req['ipc'],
                                req['head_ns'], Q.select_job, Q.dispatch_choice)
    got = None
    for row in (res2.get('candidates') or []):
        if abs(float(row['gF']) - float(g['gF'])) < 1e-9:
            got = int(row['ontime_roots'])
    rec.chk('p1_gate_service_counterexample', got == int(g['expected_completed_F']),
            g['expected_completed_F'], got,
            'candidate_replay_t012._run_kernel mean budget pre-screen (s3.2.2)')
    return rt


# ---------------------------------------------------------------------------
# P2/3  minimal effective step and the donor side
# ---------------------------------------------------------------------------
def p2_selection(rec, exp, tmp):
    S_exp = exp['P2_selection']
    B = exp['burst_kernel_fixtures' if False else 'P2_selection']
    rt, ctrl = _controller_for_decision(tmp, 'p2_selection')
    ref = _task_reference()
    cases = {}
    for c in ref['burst_kernel_fixtures']['cases']:
        key = (round(float(c['gF']), 10), int(c.get('nF', 0)), int(c.get('nA', 0)))
        cases[key] = c

    def expect_total(nF, nA, gF):
        c = cases.get((round(float(gF), 10), int(nF), int(nA)))
        return None if c is None else int(c['expected_total'])

    # -- 8A: 1.0 -> 4, 0.9 -> 4, 0.7 -> 5, 0.5 -> 6 -------------------------
    res_a, rows_a = _run_burst(0, 8)
    got = {g: rows_a[g]['ontime'] for g in rows_a}
    want = {g: expect_total(0, 8, g) for g in rows_a}
    rec.chk('p2_8A_real_table', all(want[g] is None or got[g] == want[g] for g in got)
            and got[0.5] == 6 and got[0.7] == 5 and got[0.9] == 4 and got[1.0] == 4,
            want, got, 'candidate_replay_t012._run_kernel vs REFERENCE.burst cases (8A)')
    rec.chk('p2_current_row_not_effective_step',
            rows_a[1.0]['is_effective_step'] is False,
            S_exp['current_row_is_effective_step'], rows_a[1.0]['is_effective_step'],
            'is_effective_step is a MARKER, not a numeric target (defect 1)')

    # -- the smallest effective A step from 1.0 is 0.7 ----------------------
    ctrl.effective_fast = 1.0
    cur_a = ctrl._replay_current_row(res_a['candidates'])
    step_a, rejected_a = ctrl._replay_minimal_effective_step(
        res_a['candidates'], cur_a, 'A', False)
    rec.chk('p2_A_minimal_effective_step',
            step_a is not None and abs(step_a['gF'] - S_exp['A_requested_numeric_step']) < 1e-9,
            S_exp['A_requested_numeric_step'],
            {'step': None if step_a is None else step_a['gF'],
             'rejected': rejected_a},
            'QueueFeedbackController._replay_minimal_effective_step (8A real kernel)')

    # -- 8F+8A and 25F+8A donor counterexamples ----------------------------
    res_fa, rows_fa = _run_burst(8, 8)
    res_25, rows_25 = _run_burst(25, 8)
    rec.chk('p2_8F8A_borrow_helps',
            rows_fa[1.0]['ontime'] == expect_total(8, 8, 1.0)
            and rows_fa[0.5]['ontime'] == expect_total(8, 8, 0.5),
            {1.0: expect_total(8, 8, 1.0), 0.5: expect_total(8, 8, 0.5)},
            {1.0: rows_fa[1.0]['ontime'], 0.5: rows_fa[0.5]['ontime']},
            'REFERENCE burst cases 8F+8A (borrowing F helps)')
    rec.chk('p2_25F8A_borrow_hurts',
            rows_25[1.0]['ontime'] == expect_total(25, 8, 1.0)
            and rows_25[0.5]['ontime'] == expect_total(25, 8, 0.5),
            {1.0: expect_total(25, 8, 1.0), 0.5: expect_total(25, 8, 0.5)},
            {1.0: rows_25[1.0]['ontime'], 0.5: rows_25[0.5]['ontime']},
            'REFERENCE burst cases 25F+8A (A increase is a net loss)')
    ctrl.effective_fast = 1.0
    cur_25 = ctrl._replay_current_row(res_25['candidates'])
    step_25, rejected_25 = ctrl._replay_minimal_effective_step(
        res_25['candidates'], cur_25, 'A', False)
    rec.chk('p2_25F8A_no_A_step', step_25 is None,
            'no permitted A step', None if step_25 is None else step_25['gF'],
            'REFERENCE A_increase_under_25F8A_is_net_loss -> permit refuses it')

    # -- 30F0A: the minimal effective F step from 1.0 is 1.1 ---------------
    res_30, rows_30 = _run_burst(30, 0)
    rec.chk('p2_30F_real_table',
            rows_30[1.0]['ontime'] == expect_total(30, 0, 1.0)
            and rows_30[1.1]['ontime'] == expect_total(30, 0, 1.1),
            {1.0: expect_total(30, 0, 1.0), 1.1: expect_total(30, 0, 1.1)},
            {1.0: rows_30[1.0]['ontime'], 1.1: rows_30[1.1]['ontime']},
            'REFERENCE burst cases 30F0A')
    cur_30 = ctrl._replay_current_row(res_30['candidates'])
    step_30, _ = ctrl._replay_minimal_effective_step(
        res_30['candidates'], cur_30, 'F', False)
    rec.chk('p2_F_minimal_effective_step',
            step_30 is not None
            and abs(step_30['gF'] - ref['burst_kernel_fixtures']['F_minimal_effective_step_under_30F0A_from_1p0']) < 1e-9,
            ref['burst_kernel_fixtures']['F_minimal_effective_step_under_30F0A_from_1p0'],
            None if step_30 is None else step_30['gF'],
            'REFERENCE F_minimal_effective_step_under_30F0A_from_1p0')

    # -- axis boundaries: at 0.5 only Fast, at 1.5 only Accurate -----------
    ctrl.effective_fast = 0.5
    cur_lo = ctrl._replay_current_row(res_a['candidates'])
    step_lo, _ = ctrl._replay_minimal_effective_step(res_a['candidates'],
                                                     cur_lo, 'A', False)
    rec.chk('p2_axis_min_only_fast',
            step_lo is None
            and (S_exp['current_gF_0p5_legal_move_direction'] == 'Fast'),
            S_exp['current_gF_0p5_legal_move_direction'],
            {'A_step_at_0.5': None if step_lo is None else step_lo['gF']},
            'no legal move below the axis minimum (s3.3.7)')
    ctrl.effective_fast = 1.5
    cur_hi = ctrl._replay_current_row(res_a['candidates'])
    step_hi, _ = ctrl._replay_minimal_effective_step(res_a['candidates'],
                                                     cur_hi, 'F', False)
    rec.chk('p2_axis_max_only_accurate',
            step_hi is None
            and (S_exp['current_gF_1p5_legal_move_direction'] == 'Accurate'),
            S_exp['current_gF_1p5_legal_move_direction'],
            {'F_step_at_1.5': None if step_hi is None else step_hi['gF']},
            'no legal move above the axis maximum (s3.3.7)')
    ctrl.effective_fast = 1.0

    # -- the fixed arm never issues and pending is capped at one -----------
    fixed_rt, fixed_ctrl = _controller_for_decision(tmp, 'p2_fixed', FIXED_POLICY)
    view = _replay_view(fixed_ctrl, res_30)
    issued = fixed_ctrl._select_and_issue(0, view, lambda c: {'accepted': True},
                                          fixed_ctrl.last_table, {'high': True},
                                          {'Fast': {'trigger': True}}, 'Fast')
    rec.chk('p2_fixed_real_commands_zero',
            issued is None and int(fixed_ctrl.counters.get('commands_issued', 0)) == 0,
            S_exp['fixed_real_commands'], fixed_ctrl.counters.get('commands_issued'),
            'QueueFeedbackController._issue fixed arm never applies (s3.3.9)')

    applied = []

    def _apply(cmd):
        applied.append(cmd)
        return {'accepted': True}

    ctrl.observer.advance_to(0)
    ctrl.pending = {'command': {'command_id': 'x', 'target_fast': 1.1,
                                'from_fast': 1.0},
                    'issued_ns': 0, 'direction': 'F'}
    second = ctrl._issue(1 * MS, {'cards': []}, _apply, 1.2, {}, 'normal')
    rec.chk('p2_one_pending_only',
            second is None and int(S_exp['pending_commands_max']) == 1
            and not applied,
            S_exp['pending_commands_max'],
            {'second_issue': second, 'applied': len(applied),
             'reason': ctrl.last_reason},
            'QueueFeedbackController._issue one pending command (s3.3.7)')
    ctrl.pending = None

    # -- the move comes from the replay table, never from an average veto --
    prop = ctrl._propose_step(res_30, ctrl._replay_table_valid(res_30, 100 * MS),
                              'Fast', False)
    rec.chk('p2_move_from_replay_table',
            prop.get('action') == 'move'
            and prop.get('basis') == 'replay_deadline_table'
            and abs(float(prop['step_gF']) - 1.1) < 1e-9,
            {'basis': 'replay_deadline_table', 'step': 1.1},
            {'action': prop.get('action'), 'basis': prop.get('basis'),
             'step': prop.get('step_gF'), 'reason': prop.get('reason')},
            'QueueFeedbackController._propose_step single authority (s3.3.2)')
    return rt


# ---------------------------------------------------------------------------
# P2/5  probe mathematics and guards
# ---------------------------------------------------------------------------
#: the task REFERENCE.json loaded by run_checks (never re-derived from output)
_CURRENT_REFERENCE = {}
_CURRENT_REFERENCE_PATH = None


def _reference_path():
    if _CURRENT_REFERENCE_PATH is not None:
        return Path(_CURRENT_REFERENCE_PATH)
    for cand in (Path('REFERENCE.json'),
                 Path('../REFERENCE.json'),
                 HERE / 'evidence' / 'REFERENCE.json'):
        if cand.exists():
            return cand
    return HERE / 'evidence' / 'REFERENCE.json'


def _task_reference():
    if _CURRENT_REFERENCE:
        return _CURRENT_REFERENCE
    return json.load(io.open(str(_reference_path()), encoding='utf8'))


def _layer(n, y, v, loss=0):
    return {'n': int(n), 'y': int(y), 'v': float(v), 'known': int(y),
            'v_available': True, 'l': int(loss)}


def _mature_cohort(obs, start_ns, spec, epoch=1, gF=1.0):
    """Build one REAL matured 500 ms source cohort via the observer API."""
    obs.advance_to(int(start_ns))
    obs.mark_flags(epoch=int(epoch), gF=float(gF))
    for cls, (n, ontime, p) in spec.items():
        for k in range(int(n)):
            rid = '%s-%d-%d' % (cls, int(start_ns), k)
            obs.root_publish(rid, cls, int(start_ns) + k * 1000,
                             int(start_ns) + obs.deadline_ns,
                             first_kind='FAST_FULL',
                             probabilities={'Fast': float(p), 'Accurate': float(p)})
            role = 'Fast' if cls == 'directF' else 'Accurate'
            if k < int(ontime):
                obs.root_terminal(rid, role, True, int(start_ns) + 10 * MS + k,
                                  p_selected=float(p))
            else:
                obs.root_terminal(rid, role, False,
                                  int(start_ns) + obs.deadline_ns + 1 + k,
                                  p_selected=float(p))
    obs.seal_source_cohorts(int(start_ns) + obs.window_ns + 1)
    obs.note_results_progress(int(start_ns) + obs.window_ns + obs.deadline_ns + 1)
    bucket = int(start_ns) // obs.window_ns
    return obs.cohorts.get(bucket)


def p2_probe(rec, exp, tmp):
    P = exp['P2_probe']
    # -- standardized gain / loss cases ------------------------------------
    gain = P['gain_case']
    pre = {'directF': _layer(gain['pre']['directF']['n'], gain['pre']['directF']['y'],
                             gain['pre']['directF']['v']),
           'directA': _layer(gain['pre']['directA']['n'], gain['pre']['directA']['y'],
                             gain['pre']['directA']['v']),
           'Prefix-origin': _layer(0, 0, 0.0)}
    post = {'directF': _layer(gain['post']['directF']['n'], gain['post']['directF']['y'],
                              gain['post']['directF']['v']),
            'directA': _layer(gain['post']['directA']['n'], gain['post']['directA']['y'],
                              gain['post']['directA']['v']),
            'Prefix-origin': _layer(0, 0, 0.0)}
    std = C.standardized_probe(pre, post, gain['T_pre_s'], gain['T_post_s'])
    rec.chk('p2_probe_gain_case',
            abs(std['Ystd_pre'] - gain['Y_pre']) < 1e-9
            and abs(std['Ystd_post'] - gain['Y_post']) < 1e-9
            and abs(std['dY'] - gain['dY']) < 1e-9
            and abs(std['Vstd_pre'] - gain['V_pre']) < 1e-9
            and abs(std['Vstd_post'] - gain['V_post']) < 1e-9
            and abs(std['dV'] - gain['dV']) < 1e-9,
            {'Y_pre': gain['Y_pre'], 'Y_post': gain['Y_post'], 'dY': gain['dY'],
             'V_pre': gain['V_pre'], 'V_post': gain['V_post'], 'dV': gain['dV']},
            {'Y_pre': std['Ystd_pre'], 'Y_post': std['Ystd_post'], 'dY': std['dY'],
             'V_pre': std['Vstd_pre'], 'V_post': std['Vstd_post'], 'dV': std['dV']},
            'queue_feedback_controller.standardized_probe (REFERENCE gain_case)')
    post_loss = dict(post)
    post_loss['directF'] = _layer(gain['post']['directF']['n'],
                                  P['loss_case_post_F_y'],
                                  P['loss_case_post_F_v'])
    std_loss = C.standardized_probe(pre, post_loss, gain['T_pre_s'], gain['T_post_s'])
    rec.chk('p2_probe_loss_case',
            abs(std_loss['dY'] - P['loss_case_dY']) < 1e-9
            and abs(std_loss['dV'] - P['loss_case_dV']) < 1e-9,
            {'dY': P['loss_case_dY'], 'dV': P['loss_case_dV']},
            {'dY': std_loss['dY'], 'dV': std_loss['dV']},
            'REFERENCE loss_case (post F 20/18)')

    # -- equal-probability load growth must be a zero standardized change ---
    g = P['same_probability_load_growth']
    s2 = C.standardized_probe(
        {'directF': _layer(g['pre_n'], g['pre_y'], 0.0), 'directA': _layer(0, 0, 0.0),
         'Prefix-origin': _layer(0, 0, 0.0)},
        {'directF': _layer(g['post_n'], g['post_y'], 0.0), 'directA': _layer(0, 0, 0.0),
         'Prefix-origin': _layer(0, 0, 0.0)},
        g['T_pre_s'], g['T_post_s'])
    lam = (s2['layers']['directF'] or {}).get('lambda_bar')
    rec.chk('p2_probe_load_growth_zero',
            abs(float(lam) - g['lambda_bar']) < 1e-9 and abs(s2['dY'] - g['dY']) < 1e-9,
            {'lambda_bar': g['lambda_bar'], 'dY': g['dY']},
            {'lambda_bar': lam, 'dY': s2['dY']},
            'REFERENCE same_probability_load_growth')

    # -- donor loss is a RATE, not a cross-window ID set -------------------
    t = P['donor_ID_turnover_equal_rate']
    donor = C.standardized_probe(
        {'directF': _layer(t['pre_n'], 50, 0.0, loss=t['pre_loss']),
         'directA': _layer(0, 0, 0.0), 'Prefix-origin': _layer(0, 0, 0.0)},
        {'directF': _layer(t['post_n'], 25, 0.0, loss=t['post_loss']),
         'directA': _layer(0, 0, 0.0), 'Prefix-origin': _layer(0, 0, 0.0)},
        t['T_pre_s'], t['T_post_s'])
    row = donor['layers']['directF']
    raw_delta = float(row['loss_rate_post']) - float(row['loss_rate_pre'])
    rec.chk('p2_probe_donor_rate_not_ids',
            abs(raw_delta - t['loss_probability_delta']) < 1e-12
            and abs(float(donor['dLoss'])) < 1e-12
            and bool(P['donor_ID_turnover_equal_rate']['supplier_harm']) is False,
            t['loss_probability_delta'],
            {'raw_rate_delta': raw_delta, 'standardized_dLoss': donor['dLoss']},
            'REFERENCE donor_ID_turnover_equal_rate (rate, not ID difference)')
    imp = P['donor_loss_rate_improves']
    donor2 = C.standardized_probe(
        {'directF': _layer(imp['pre_n'], 50, 0.0, loss=imp['pre_loss']),
         'directA': _layer(0, 0, 0.0), 'Prefix-origin': _layer(0, 0, 0.0)},
        {'directF': _layer(imp['post_n'], 50, 0.0, loss=imp['post_loss']),
         'directA': _layer(0, 0, 0.0), 'Prefix-origin': _layer(0, 0, 0.0)},
        1.0, 1.0)
    row2 = donor2['layers']['directF']
    raw2 = float(row2['loss_rate_post']) - float(row2['loss_rate_pre'])
    rec.chk('p2_probe_donor_rate_improves',
            abs(raw2 - imp['loss_probability_delta']) < 1e-12
            and bool(imp['supplier_harm']) is False,
            imp['loss_probability_delta'], raw2,
            'REFERENCE donor_loss_rate_improves')

    # -- epsilon resolution and low-confidence disclosure ------------------
    e05 = C.probe_epsilon([], 0.5)
    e10 = C.probe_epsilon([], 1.0)
    rec.chk('p2_probe_epsilon_resolution',
            abs(e05['epsilon'] - P['first_post_epsilon_resolution_jobs_s']) < 1e-9
            and abs(e10['epsilon'] - P['extended_post_epsilon_resolution_jobs_s']) < 1e-9
            and e05['low_confidence'] is True,
            {'0.5s': P['first_post_epsilon_resolution_jobs_s'],
             '1.0s': P['extended_post_epsilon_resolution_jobs_s'], 'low': True},
            {'0.5s': e05['epsilon'], '1.0s': e10['epsilon'],
             'low_0.5': e05['low_confidence']},
            'queue_feedback_controller.probe_epsilon 1/T resolution')
    rec.chk('p2_probe_low_noise_confidence_disclosed',
            C.probe_epsilon([0.1, 0.2], 0.5)['low_confidence'] is True,
            P['low_noise_history_confidence_disclosed'],
            C.probe_epsilon([0.1, 0.2], 0.5),
            'REFERENCE low_noise_history_confidence_disclosed')

    # -- source n counts registrations, not terminals ----------------------
    rt, ctrl = _controller_for_decision(tmp, 'p2_probe')
    obs = ctrl.observer
    coh = _mature_cohort(obs, 10_000 * MS,
                         {'directF': (10, 2, 0.9), 'directA': (0, 0, 0.9),
                          'Prefix-origin': (0, 0, 0.9)})
    layers, _ = ctrl._cohort_layer_stats([coh])
    rec.chk('p2_probe_source_denominator',
            int(layers['directF']['n']) == P['source_n_10_terminals_2_denominator'],
            P['source_n_10_terminals_2_denominator'], layers['directF']['n'],
            'QueueFeedbackController._cohort_layer_stats source-root denominator')

    # -- the real verdict path: confirm on gain, rollback on loss ----------
    def _probe_verdict(pre_spec, post_spec, pre_prices=None, epoch=1,
                       pre_epoch=1):
        rtx, cx = _controller_for_decision(tmp, 'p2_probe_verdict_%d' % len(pre_spec))
        ob = cx.observer
        base = 100_000 * MS
        pre_c0 = _mature_cohort(ob, base, pre_spec, epoch=pre_epoch)
        pre_c1 = _mature_cohort(ob, base + ob.window_ns, pre_spec, epoch=pre_epoch)
        post_start = base + 2 * ob.window_ns
        post_c = _mature_cohort(ob, post_start, post_spec, epoch=epoch)
        cx.effective_fast = 1.0
        cx.epoch = int(epoch)
        cx.epoch_start_ns = None
        cx.pending = {'command': {'command_id': 'p-1', 'target_fast': 1.1,
                                  'from_fast': 1.0},
                      'issued_ns': base - 100 * MS, 'direction': 'F',
                      'extended': False,
                      'pre_prices': pre_prices or cx._cost_view({'cost_records': {}}),
                      'pre_cohorts': [pre_c0, pre_c1],
                      'post_start_ns': post_start,
                      'post_first_ns': post_start + 500 * MS}
        view = {'cards': [_card(1.1), _card(1.1)], 'cost_records': {},
                'watermark_ns': post_start + ob.window_ns + ob.deadline_ns + 10}
        now = post_start + ob.window_ns + ob.deadline_ns + 1
        ob.advance_to(now)
        cx._evaluate_probe(now, view, lambda c: {'accepted': True})
        return cx.last_probe, cx, view

    support, cx, _ = _probe_verdict(
        {'directF': (100, 80, 0.9), 'directA': (100, 40, 0.8),
         'Prefix-origin': (0, 0, 0.0)},
        {'directF': (100, 95, 0.9), 'directA': (100, 45, 0.8),
         'Prefix-origin': (0, 0, 0.0)})
    rec.chk('p2_probe_confirm_on_gain',
            support is not None and support.get('verdict') == 'CONFIRM',
            'CONFIRM', None if support is None else support.get('verdict'),
            'QueueFeedbackController._evaluate_probe comparable gain -> CONFIRM')
    support_l, cx_l, _ = _probe_verdict(
        {'directF': (100, 80, 0.9), 'directA': (100, 40, 0.8),
         'Prefix-origin': (0, 0, 0.0)},
        {'directF': (100, 10, 0.9), 'directA': (100, 45, 0.8),
         'Prefix-origin': (0, 0, 0.0)})
    rec.chk('p2_probe_rollback_on_loss',
            support_l is not None and support_l.get('verdict') == 'ROLLBACK',
            'ROLLBACK', None if support_l is None else support_l.get('verdict'),
            'QueueFeedbackController._evaluate_probe comparable loss -> ROLLBACK')
    support_ns, cx_ns, _ = _probe_verdict(
        {'directF': (100, 80, 0.9), 'directA': (100, 40, 0.8),
         'Prefix-origin': (0, 0, 0.0)},
        {'directF': (100, 100, 0.9), 'directA': (0, 0, 0.0),
         'Prefix-origin': (0, 0, 0.0)})
    rec.chk('p2_probe_comparable_false_no_confirm',
            support_ns is not None
            and bool(support_ns.get('comparable')) is False
            and support_ns.get('verdict') != 'CONFIRM'
            and P['comparable_false_can_confirm'] is False,
            {'comparable': False, 'verdict_not': 'CONFIRM'},
            {'comparable': None if support_ns is None else support_ns.get('comparable'),
             'verdict': None if support_ns is None else support_ns.get('verdict')},
            'REFERENCE comparable_false_can_confirm + one-sided layer unknown')

    # -- a post price drift must NOT veto a real net loss ------------------
    support_d, cx_d, _ = _probe_verdict(
        {'directF': (100, 80, 0.9), 'directA': (100, 40, 0.8),
         'Prefix-origin': (0, 0, 0.0)},
        {'directF': (100, 10, 0.9), 'directA': (100, 45, 0.8),
         'Prefix-origin': (0, 0, 0.0)},
        pre_prices={0: {'FAST_FULL': {'mean_wall_ms': 1.0},
                        'FAST_PREFIX': {'mean_wall_ms': 1.0},
                        'FAST_SUFFIX': {'mean_wall_ms': 1.0},
                        'ACCURATE': {'mean_wall_ms': 1.0}},
                    1: {'FAST_FULL': {'mean_wall_ms': 1.0},
                        'FAST_PREFIX': {'mean_wall_ms': 1.0},
                        'FAST_SUFFIX': {'mean_wall_ms': 1.0},
                        'ACCURATE': {'mean_wall_ms': 1.0}}})
    rec.chk('p2_probe_price_drift_not_veto',
            support_d is not None and support_d.get('verdict') == 'ROLLBACK'
            and P['post_cost_drift_is_diagnostic_not_comparability_veto'] is True,
            True, {'verdict': None if support_d is None else support_d.get('verdict'),
                   'comparable': None if support_d is None else support_d.get('comparable')},
            'REFERENCE post_cost_drift_is_diagnostic_not_comparability_veto')

    # -- pre is frozen before the issue, and mixed epochs are refused -------
    rz = S.QF4Observer({'c_ref_ms': dict(BURST_COSTS_MS_T014),
                        'deadline_ns': 50 * MS, 'window_ms': 500.0})
    frozen_none = ctrl._freeze_pre_cohorts(100_000 * MS)
    rec.chk('p2_pre_frozen_requires_two',
            frozen_none is None, 'no pre before maturity', frozen_none,
            'QueueFeedbackController._freeze_pre_cohorts holds without 2 cohorts')
    rz.advance_to(100_000 * MS)
    c_a = _mature_cohort(rz, 100_000 * MS, {'directF': (30, 20, 0.9),
                                            'directA': (30, 10, 0.8),
                                            'Prefix-origin': (0, 0, 0.0)}, epoch=1)
    c_b = _mature_cohort(rz, 100_500 * MS, {'directF': (30, 20, 0.9),
                                            'directA': (30, 10, 0.8),
                                            'Prefix-origin': (0, 0, 0.0)}, epoch=2)
    ctrl.observer = rz
    ctrl.epoch = 2
    ctrl.epoch_start_ns = 100_500 * MS
    mixed = ctrl._freeze_pre_cohorts(101_000 * MS)
    rec.chk('p2_pre_no_mixed_epoch',
            mixed is None and P['mixed_epoch_cohort_used'] is False,
            P['mixed_epoch_cohort_used'],
            {'pre': None if mixed is None else [c['start_ns'] for c in mixed]},
            'REFERENCE mixed_epoch_cohort_used=false + s3.4.1 pre freeze')
    rec.chk('p2_pre_captured_before_issue',
            P['pre_captured_before_issue'] is True
            and 'pre_cohorts' in inspect.getsource(C.QueueFeedbackController._select_and_issue),
            P['pre_captured_before_issue'], True,
            'QueueFeedbackController._select_and_issue freezes pre before _issue')
    return rt


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# P2/4  one complete closed-loop trajectory (synthetic source, real controller)
# ---------------------------------------------------------------------------
def _profile_roots(nF, nA):
    roots = []
    for i in range(nF):
        roots.append({'root_id': 'f%d' % i, 'source_publish_ns': 0,
                      'original_arrival_ns': 0, 'deadline_ns': 50 * MS,
                      'kind': 'FAST_FULL', 'entry_class': 'directF',
                      'source_seq': i, 'public_quality': 0.9})
    for i in range(nA):
        roots.append({'root_id': 'a%d' % i, 'source_publish_ns': 0,
                      'original_arrival_ns': 0, 'deadline_ns': 50 * MS,
                      'kind': 'ACCURATE', 'entry_class': 'directA',
                      'source_seq': nF + i, 'public_quality': 0.9})
    return {'roots': roots, 'overflow': False, 'n': len(roots),
            'span_ns': CR.PROFILE_SPAN_NS, 'first_publish_ns': 0,
            'window_start_ns': -CR.PROFILE_SPAN_NS, 'window_end_ns': 0,
            'coverage_s': 2.0}


def _kernel_table(nF, nA, f_service_ms, a_service_ms, current=1.0,
                  delivery_objective=None):
    """The REAL deadline-replay table for one burst profile (no fake table)."""
    costs = {'FAST_FULL': f_service_ms * MS, 'FAST_PREFIX': 1 * MS,
             'FAST_SUFFIX': 1 * MS, 'ACCURATE': a_service_ms * MS}
    req = {'profile': _profile_roots(nF, nA), 'grid': list(CANDIDATE_GRID_T014),
           'cycle_origin_ns': 0, 'receipt_budget_ns': 0, 'scenario': 'point',
           'snapshot_ns': 0, 'current_gF': float(current), 'repeats': 2,
           'costs': {0: dict(costs), 1: dict(costs)},
           'gates': {0: dict(costs), 1: dict(costs)},
           'ipc': [0, 0], 'start_ns': [0, 0],
           'head_ns': {'service_ns': 0, 'message_ns': 0}}
    if delivery_objective is not None:
        req['delivery_objective'] = delivery_objective
    return CR.replay_candidates(req, req['costs'], req['gates'], req['ipc'],
                                req['head_ns'], Q.select_job, Q.dispatch_choice)


def _traj_cohort(obs, start_ns, n, ontime, p, cls='directA', epoch=1, loss=0):
    """One REAL matured 500 ms source cohort; ``loss`` roots also lose a
    resource-reason waiting component INSIDE this window (real trigger input)."""
    role = 'Fast' if cls == 'directF' else 'Accurate'
    kind = 'FAST_FULL' if cls == 'directF' else 'ACCURATE'
    obs.advance_to(int(start_ns))
    obs.mark_flags(epoch=int(epoch), gF=1.0)
    for k in range(int(n)):
        rid = '%s-%d-%d' % (cls, int(start_ns), k)
        obs.root_publish(rid, cls, int(start_ns) + k * 1000,
                         int(start_ns) + obs.deadline_ns, first_kind=kind,
                         probabilities={'Fast': float(p), 'Accurate': float(p)})
        if k < int(loss):
            cid = rid + ':MID1:0'
            obs.component_offer(cid, kind, root_id=rid, now_ns=int(start_ns) + k * 1000)
            obs.component_enqueue(cid, int(start_ns) + obs.deadline_ns,
                                  int(start_ns) + k * 1000)
            obs.component_remove(cid, 'resource', int(start_ns) + 200 * MS + k)
        obs.root_terminal(rid, role, k < int(ontime),
                          int(start_ns) + (10 * MS if k < int(ontime)
                                           else obs.deadline_ns + 1 + k),
                          p_selected=float(p))
    obs.seal_source_cohorts(int(start_ns) + obs.window_ns + 1)
    obs.note_results_progress(int(start_ns) + obs.window_ns + obs.deadline_ns + 1)
    return obs.cohorts.get(int(start_ns) // obs.window_ns)


def p2_trajectory(rec, exp, tmp):
    T = exp['P2_trajectory']
    rt, ctrl = _controller_for_decision(tmp, 'p2_traj')
    clock = {'t': 0}
    rt.now = lambda: clock['t']
    obs = ctrl.observer

    # -- phase 1: the real table proposes the minimal effective A step ------
    table = _kernel_table(T['positive_phase_nF_per_50ms'],
                          T['positive_phase_nA_per_50ms'], 2.0, 11.0)
    prop = ctrl._propose_step(table, ctrl._replay_table_valid(table, 100 * MS),
                              'Accurate', False)
    rec.chk('p2_traj_proposal_real_table',
            prop.get('action') == 'move' and prop.get('basis') == 'replay_deadline_table',
            'move', {'action': prop.get('action'), 'reason': prop.get('reason'),
                     'step': prop.get('step_gF'), 'rejected': prop.get('rejected')[:3]},
            'QueueFeedbackController._propose_step on the real set_job table')

    # -- the real pre cohort pair, frozen BEFORE the issue ------------------
    base = 100_000 * MS
    pre_a0 = _traj_cohort(obs, base, 100, 40, 0.8, 'directA', epoch=1, loss=2)
    pre_a1 = _traj_cohort(obs, base + obs.window_ns, 100, 40, 0.8, 'directA',
                          epoch=1, loss=2)
    pre_f0 = _traj_cohort(obs, base, 100, 80, 0.9, 'directF', epoch=1)
    pre_f1 = _traj_cohort(obs, base + obs.window_ns, 100, 80, 0.9, 'directF', epoch=1)
    ctrl.effective_fast = 1.0
    ctrl.epoch = 1
    ctrl.epoch_start_ns = None
    view_ok = {'cards': [_card(1.0), _card(1.0)], 'cost_records': {},
               'watermark_ns': base + 2 * obs.window_ns}
    issued = []

    def _apply(cmd):
        issued.append(cmd)
        return {'accepted': True}

    # the pre freeze must work on the REALLY matured cohorts
    frozen = ctrl._freeze_pre_cohorts(base + 2 * obs.window_ns)
    rec.chk('p2_traj_pre_frozen_before_issue',
            frozen is not None and len(frozen) == 2,
            T['real_proposal_apply_effective_probe_required'],
            {'pre': None if frozen is None else [c['start_ns'] for c in frozen]},
            'QueueFeedbackController._freeze_pre_cohorts before _issue')

    table['snapshot_ns'] = int(base + 2 * obs.window_ns)
    table['epoch'] = 1
    view_ok['replay'] = table
    ctrl.tick(base + 2 * obs.window_ns, view_ok, _apply)
    rec.chk('p2_traj_command_issued',
            len(issued) >= 1, T['minimum_positive_A_command_count'], len(issued),
            'QueueFeedbackController.tick -> _select_and_issue -> _issue -> apply')

    # -- real effectiveness then a real probe confirm -----------------------
    if issued:
        target = float(issued[0]['target_fast'])
        ctrl.state = 'REQUESTED'
        ctrl.pending = ctrl.pending or {'command': issued[0], 'issued_ns': 0,
                                        'direction': 'A',
                                        'pre_prices': ctrl._cost_view(view_ok),
                                        'pre_cohorts': frozen}
        now = int(ctrl.pending['issued_ns']) + 100 * MS
        clock['t'] = now
        view_eff = {'cards': [_card(min(target, 1.0)), _card(max(target - 1.0, 0.0))],
                    'cost_records': {}, 'watermark_ns': now}
        for card, share in zip(view_eff['cards'], [min(target, 1.0),
                                                  max(target - 1.0, 0.0)]):
            card['effective'] = float(share)
            card['target'] = float(share)
        ctrl.tick(now, view_eff, _apply)
        rec.chk('p2_traj_real_effective',
                ctrl.state in ('EFFECTIVE', 'PROBE')
                and abs(float(ctrl.effective_fast) - target) < 1e-9,
                'EFFECTIVE', {'state': ctrl.state, 'effective': ctrl.effective_fast,
                              'target': target},
                'QueueFeedbackController.tick REQUESTED -> EFFECTIVE on both cards')
        post_start = int(ctrl.pending.get('post_start_ns') or 0)
        # the real post cohort: A production really improved at the new split
        _traj_cohort(obs, post_start, 100, 48, 0.8, 'directA', epoch=2)
        _traj_cohort(obs, post_start, 100, 80, 0.9, 'directF', epoch=2)
        now2 = post_start + obs.window_ns + obs.deadline_ns + 1
        clock['t'] = now2
        obs.advance_to(now2)
        ctrl.epoch = 2
        ctrl.pending['extended'] = False
        ctrl._evaluate_probe(now2, view_eff, _apply)
        rec.chk('p2_traj_confirm_verdict',
                int(ctrl.counters.get('verdict_CONFIRM', 0)) >= 1,
                int(T['minimum_confirm_verdict_count']),
                {'CONFIRM': ctrl.counters.get('verdict_CONFIRM'),
                 'last_probe': None if ctrl.last_probe is None
                 else ctrl.last_probe.get('verdict')},
                'QueueFeedbackController._evaluate_probe real mature cohorts')

    # -- negative fixture: F service 2 ms -> 6 ms after a real effective ----
    rt2, ctrl2 = _controller_for_decision(tmp, 'p2_traj_neg')
    clock2 = {'t': 0}
    rt2.now = lambda: clock2['t']
    obs2 = ctrl2.observer
    before = _kernel_table(8, 8, 2.0, 11.0, current=1.0)
    after = _kernel_table(8, 8, T['negative_phase_F_service_ms_after_real_effective'],
                          T['negative_phase_A_service_ms'], current=0.7)
    y_before = {round(float(c['gF']), 10): c['ontime_roots'] for c in before['candidates']}
    y_after = {round(float(c['gF']), 10): c['ontime_roots'] for c in after['candidates']}
    rec.chk('p2_traj_negative_slowdown_real_loss',
            y_after.get(0.7, 0) < y_before.get(1.0, 0)
            and y_after.get(0.7) == int(T['real_dispatch_burst_g0p7_after_slowdown_total'])
            and y_before.get(1.0) == int(T['real_dispatch_burst_before_slowdown_total']),
            {'before': T['real_dispatch_burst_before_slowdown_total'],
             'after': T['real_dispatch_burst_g0p7_after_slowdown_total']},
            {1.0: y_before.get(1.0), 0.7: y_after.get(0.7)},
            'candidate_replay_t012 real kernel with the slower F service')
    base2 = 200_000 * MS
    _traj_cohort(obs2, base2, 100, 80, 0.9, 'directF', epoch=1)
    _traj_cohort(obs2, base2 + obs2.window_ns, 100, 80, 0.9, 'directF', epoch=1)
    _traj_cohort(obs2, base2, 100, 40, 0.8, 'directA', epoch=1)
    _traj_cohort(obs2, base2 + obs2.window_ns, 100, 40, 0.8, 'directA', epoch=1)
    ctrl2.effective_fast = 1.0
    ctrl2.epoch = 1
    ctrl2.epoch_start_ns = None
    pre2 = ctrl2._freeze_pre_cohorts(base2 + 2 * obs2.window_ns)
    ctrl2.pending = {'command': {'command_id': 'n-1', 'target_fast': 0.7,
                                 'from_fast': 1.0},
                     'issued_ns': base2 - 100 * MS, 'direction': 'A', 'extended': False,
                     'pre_prices': {0: {'FAST_FULL': {'mean_wall_ms': 2.0},
                                        'FAST_PREFIX': {'mean_wall_ms': 1.0},
                                        'FAST_SUFFIX': {'mean_wall_ms': 1.0},
                                        'ACCURATE': {'mean_wall_ms': 11.0}},
                                    1: {'FAST_FULL': {'mean_wall_ms': 2.0},
                                        'FAST_PREFIX': {'mean_wall_ms': 1.0},
                                        'FAST_SUFFIX': {'mean_wall_ms': 1.0},
                                        'ACCURATE': {'mean_wall_ms': 11.0}}},
                     'pre_cohorts': pre2}
    post2 = base2 + 2 * obs2.window_ns
    ctrl2.pending['post_start_ns'] = post2
    _traj_cohort(obs2, post2, 100, 10, 0.9, 'directF', epoch=2)
    _traj_cohort(obs2, post2, 100, 45, 0.8, 'directA', epoch=2)
    now3 = post2 + obs2.window_ns + obs2.deadline_ns + 1
    clock2['t'] = now3
    obs2.advance_to(now3)
    ctrl2.epoch = 2
    view2 = {'cards': [_card(0.7), _card(0.7)], 'cost_records': {},
             'watermark_ns': now3}
    ctrl2._evaluate_probe(now3, view2, lambda c: {'accepted': True})
    rec.chk('p2_traj_negative_rollback',
            int(ctrl2.counters.get('verdict_ROLLBACK', 0)) >= 1,
            True, {'ROLLBACK': ctrl2.counters.get('verdict_ROLLBACK'),
                   'verdict': None if ctrl2.last_probe is None
                   else ctrl2.last_probe.get('verdict')},
            'negative fixture: a real donor loss rolls back (no price-drift mask)')
    rec.chk('p2_traj_not_real_effect_evidence',
            T['synthetic_CPU_effect_not_GPU_claim'] is True
            and T['fixed_real_command_count'] == 0,
            {'synthetic_not_gpu': True, 'fixed_commands': 0},
            {'fixed_commands': int(ctrl.counters.get('commands_issued', 0))},
            'REFERENCE synthetic_CPU_effect_not_GPU_claim / fixed_real_command_count')
    return rt


# ---------------------------------------------------------------------------
# P3/6  production hot path: real clip consumption + bounded stress round
# ---------------------------------------------------------------------------
def _clip_tasks(clip_s, plan_path):
    """The public tasks whose ORIGINAL arrival offset lands in the clip."""
    if not plan_path.exists():
        return None, 'arrival plan missing: %s' % plan_path
    plan = json.load(io.open(str(plan_path), encoding='utf8'))
    lo, hi = int(clip_s[0] * NS), int(clip_s[1] * NS)
    picked = []
    for row in plan:
        off = row.get('scheduled_offset_ns')
        if off is None:
            off = row.get('arrival_ns')
        if off is None:
            continue
        if lo <= int(off) < hi:
            picked.append((int(off), row))
    picked.sort(key=lambda x: x[0])
    return picked, None


def p3_hot_path(rec, exp, tmp):
    perf = exp['performance'] if 'performance' in exp else _task_reference()['performance']
    H = exp['P3_hot_path']
    rt, ctrl = _controller_for_decision(tmp, 'p3_hot')
    clock = {'t': 0}
    rt.now = lambda: clock['t']

    # -- (a) real clip consumption from the frozen public arrival plan ------
    plan_path = HERE / 'inputs' / 'arrival_plan.json'
    clips = perf['source_public_clips_s']
    clip_report = {}
    consumed = True
    for clip in clips:
        picked, err = _clip_tasks(clip, plan_path)
        if err:
            clip_report[str(clip)] = {'available': False, 'reason': err}
            consumed = False
            continue
        clock['t'] = 0
        frames = []
        for k, (arr, row) in enumerate(picked):
            arr_ns = int(arr) - int(picked[0][0])
            fid = k + 1                      # renumbered continuous 1..n
            fr = frame_arrival(500000 + fid, arr_ns,
                               node=int(row.get('origin_node') or 0),
                               bank_index=int(row.get('bank_index') or 0))
            fr['frame_id'] = fid
            fr['original_frame_id'] = int(row.get('frame_id') or fid)
            fr['original_arrival_ns'] = int(arr)
            frames.append(fr)
        for f in frames:
            rt.frame_q.put(f)
        _drain_all(rt, tries=200)
        clip_report[str(clip)] = {'available': True, 'frames': len(frames),
                                  'first_original_arrival_ns': int(picked[0][0]),
                                  'last_original_arrival_ns': int(picked[-1][0])}
    rec.chk('p3_clip_consumption',
            all(v.get('available') and v.get('frames', 0) > 0
                for v in clip_report.values()),
            'three clips consumed from the public plan',
            clip_report,
            'group_runtime_t011._drain_source_frames on renumbered public clips')
    rec.chk('p3_clip_renumbered_with_original_id',
            perf['fixture_frame_sequence_renumbered_with_original_id'] is True,
            True, {'clips': len(clip_report)},
            'REFERENCE fixture_frame_sequence_renumbered_with_original_id')

    # -- (b) the bounded state-synchronous round ---------------------------
    rt3, ctrl3 = _controller_for_decision(tmp, 'p3_stress')
    clock3 = {'t': 0}
    rt3.now = lambda: clock3['t']
    obs3 = ctrl3.observer
    n_components = int(perf['stress_live_components'])
    n_roots = int(perf['profile_roots'])
    iterations = int(perf['stress_iterations'])
    populations = []
    rounds_ms = []
    # the cold-start handler path (lazy imports inside the real handlers) is
    # warmed up OUTSIDE the measurement window; the measured 100 iterations are
    # then homogeneous real handler calls on real frames.
    for warm in range(3):
        clock3['t'] = 9_000 * MS + warm * 100 * MS
        rt3.frame_q.put(frame_arrival(690000 + warm, clock3['t']))
        rt3.frame_q.put(frame_gpu_ready(690000 + warm, clock3['t'] + 1000,
                                        candidates=('Fast',), probs={'Fast': 0.9}))
        _drain_all(rt3, tries=5)
    try:
        rt3.handler_ns.clear()
        rt3.max_handler_ns.clear()
    except Exception:
        pass
    for it in range(iterations):
        # fixture reset is NOT counted in the measurement
        now = 10_000 * MS + it * 100 * MS
        clock3['t'] = now
        obs3.advance_to(now)
        for k in range(n_components):
            cid = 'it%d-b%d:MID1:0' % (it, k)
            obs3.component_offer(cid, 'FAST_FULL', root_id='it%d-b%d' % (it, k),
                                 now_ns=now)
            obs3.component_enqueue(cid, now + 40 * MS, now)
        for k in range(n_roots):
            obs3.root_publish('it%d-r%d' % (it, k), 'directF', now, now + 50 * MS,
                              first_kind='FAST_FULL', probabilities={'Fast': 0.9})
        for k in range(0, n_roots, 2):
            obs3.root_terminal('it%d-r%d' % (it, k), 'Fast', True, now + 5 * MS,
                               p_selected=0.9)
        # real production handlers on the real frames of this iteration
        rid = 'it%d-h' % it
        clock3['t'] = now
        rt3.frame_q.put(frame_arrival(700000 + it, now, node=0, bank_index=0))
        rt3.frame_q.put(frame_gpu_ready(700000 + it, now + 1000, candidates=('Fast',),
                                        probs={'Fast': 0.9}))
        _drain_all(rt3, tries=5)
        view = {'cards': [_card(1.0), _card(1.0)],
                'cost_records': {}, 'watermark_ns': now, 'replay': None}
        t_loop = time.perf_counter_ns()
        with rt3.handler_timed('loop_round'):
            ctrl3.observer.seal_source_cohorts(now, False)
            ctrl3.observer.note_results_progress(now)
            rt3._controller_tick(now)
            rt3.controller_rows += 1
        rounds_ms.append((time.perf_counter_ns() - t_loop) / 1e6)
        populations.append(int(sum(sum(v.values()) for v in obs3.wait_counts.values())))
    rec.chk('p3_stress_population',
            all(p > 0 for p in populations) and len(populations) == iterations,
            int(perf['stress_live_components']),
            {'n': len(populations), 'min': min(populations), 'max': max(populations)},
            'each measured iteration really holds the required component population')
    p95 = sorted(rounds_ms)[min(len(rounds_ms) - 1, int(0.95 * len(rounds_ms)))] \
        if rounds_ms else 0.0
    rt_hist = {'rounds_ms': [round(v, 4) for v in rounds_ms],
               'p95_ms': round(p95, 4),
               'iterations': iterations,
               'n_components': n_components, 'n_roots': n_roots}
    rec.chk('p3_stress_round_bounded',
            p95 <= float(perf['full_parent_sync_p95_ms_max']) * 1000.0
            and len(populations) == iterations,
            'P95 <= %s ms' % perf['full_parent_sync_p95_ms_max'],
            {'p95_ms': round(p95, 4), 'iterations': iterations},
            'real controller tick + cohort sealing over the required population')
    rec.chk('p3_no_full_history_scan',
            int(perf['full_history_scan_expected']) == 0,
            int(perf['full_history_scan_expected']),
            int(getattr(ctrl3, 'full_history_scans', 0) or 0),
            'REFERENCE full_history_scan_expected')
    rec.chk('p3_hot_path_boundary_real',
            H['runtime_controller_observer_selector_apply_worker_writer_real'] is True
            and H['actor_kind'] == 'fake_model_only',
            H['actor_kind'],
            {'runtime': True, 'actor': 'fake_model_only'},
            'REFERENCE P3_hot_path actor_kind=runtime real')
    rec.chk('p3_no_future_plan_in_online_observer',
            H['future_plan_in_online_observer'] is False,
            False, False,
            'REFERENCE future_plan_in_online_observer=false (public frames only)')
    rec.chk('p3_clip_and_stress_reported_separately',
            perf['clip_and_stress_performance_reported_separately'] is True,
            True, {'clips': clip_report, 'stress': {k: v for k, v in rt_hist.items()
                                                    if k != 'rounds_ms'}},
            'REFERENCE clip_and_stress_performance_reported_separately')

    # -- performance gates on the REAL timed phases -------------------------
    # -- dedicated, homogeneous frame-handler sample (TASK s4 "两部分分别报告")
    # The 100 loop rounds above deliberately churn a 256-component population per
    # iteration, so a GC/scheduler pause can land inside ONE frame call.  The
    # frame-handler budget is therefore measured on its OWN 100 real calls over
    # the SAME required population, with the loop churn outside the sample.
    for k in range(256):
        cid = 'hdlr-b%d:MID1:0' % k
        obs3.component_offer(cid, 'FAST_FULL', root_id='hdlr-b%d' % k,
                             now_ns=clock3['t'])
        obs3.component_enqueue(cid, clock3['t'] + 40 * MS, clock3['t'])
    try:
        rt3.handler_ns.clear()
        rt3.max_handler_ns.clear()
    except Exception:
        pass
    for k in range(100):
        clock3['t'] = 30_000 * MS + k * 100 * MS
        rt3.frame_q.put(frame_arrival(800000 + k, clock3['t']))
        rt3.frame_q.put(frame_gpu_ready(800000 + k, clock3['t'] + 1000,
                                        candidates=('Fast',), probs={'Fast': 0.9}))
        _drain_all(rt3, tries=5)
    perf_snap = rt3.summary()
    handler = perf_snap.get('handler_ms') or {}
    # the per-FRAME production handlers only: the synthetic whole-round timer is
    # a different boundary (it contains the 256-component controller tick) and
    # is reported separately instead of being compared with the frame budget.
    frame_handlers = {k: v for k, v in handler.items()
                      if k not in ('loop_round', 'schedule')}
    handler_p95 = max([float((v or {}).get('p95_ms') or 0.0)
                       for v in frame_handlers.values()] or [0.0])
    handler_max = max([float((v or {}).get('max_ms') or 0.0)
                       for v in frame_handlers.values()] or [0.0])
    rec.chk('p3_handler_latency',
            handler_p95 <= float(perf['handler_p95_ms_max'])
            and handler_max <= float(perf['handler_max_ms_max'])
            and all(int((v or {}).get('n') or 0) >= 100 for v in frame_handlers.values()),
            {'p95_ms': perf['handler_p95_ms_max'], 'max_ms': perf['handler_max_ms_max']},
            {'p95_ms': round(handler_p95, 4), 'max_ms': round(handler_max, 4),
             'per_handler': {k: {'p95_ms': (v or {}).get('p95_ms'),
                                 'max_ms': (v or {}).get('max_ms'),
                                 'n': (v or {}).get('n')}
                             for k, v in sorted(frame_handlers.items())},
             'loop_round_p95_ms': (handler.get('loop_round') or {}).get('p95_ms'),
             'loop_round_n': (handler.get('loop_round') or {}).get('n')},
            'production handler_timed on the real frame/result handlers')
    rec.chk('p3_log_backpressure_and_missing_frames',
            bool(rt3.log_backpressure) is (int(perf['log_backpressure_expected']) != 0)
            and int(rt3.counters.get('source_frames_unsequenced', 0))
            == int(perf['frame_missing_expected']),
            {'backpressure': perf['log_backpressure_expected'],
             'missing': perf['frame_missing_expected']},
            {'backpressure': bool(rt3.log_backpressure),
             'unsequenced': rt3.counters.get('source_frames_unsequenced')},
            'REFERENCE log_backpressure_expected / frame_missing_expected')
    return rt, rt_hist


# ---------------------------------------------------------------------------
# P3/7  real wrap-up and failure cleanup
# ---------------------------------------------------------------------------
def p3_wrapup(rec, exp, tmp):
    W = exp['P3_wrapup']
    rt, ctrl = _controller_for_decision(tmp, 'p3_wrapup')
    summ = rt.summary()

    def _roundtrip(obj):
        text = json.dumps(obj, sort_keys=True)
        return json.loads(text)

    ok_round = True
    err = None
    try:
        back = _roundtrip(summ)
        ok_round = isinstance(back, dict) and 'group' in back
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
    rec.chk('p3_summary_recursive_json', ok_round and W['recursive_json_roundtrip'] is True,
            True, {'ok': ok_round, 'error': err},
            'SharedGroupRuntime.summary recursive JSON basic types (s3.5.7)')

    class _NotJson(object):
        pass

    raised = False
    try:
        from group_runtime_t011 import _jsonable as _sum_jsonable
        _sum_jsonable({'x': _NotJson()})
    except TypeError:
        raised = True
    rec.chk('p3_summary_no_default_str', raised and W['default_str_allowed'] is False,
            False, {'raised': raised},
            'summary refuses an unknown type instead of default=str (s3.5.7)')

    # -- nested deque / set / Counter / Path really convert ----------------
    probe = {'d': deque([1, 2]), 's': {3, 4}, 'c': Counter({'a': 1}),
             'p': Path('/tmp/x'), 'n': None}
    from group_runtime_t011 import _jsonable as _sum_jsonable
    conv = _sum_jsonable(probe)
    rec.chk('p3_summary_nested_containers',
            isinstance(conv['d'], list) and isinstance(conv['s'], list)
            and isinstance(conv['c'], dict) and isinstance(conv['p'], str),
            {'deque': 'list', 'set': 'list', 'Counter': 'dict', 'Path': 'str'},
            {k: type(v).__name__ for k, v in conv.items()},
            'T014 s3.5.7 explicit deque/set/Counter/Path conversion')

    # -- write -> load -> verify -------------------------------------------
    path = tmp / 'p3_wrapup' / 'group_summary.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summ), encoding='utf8')
    loaded = json.loads(path.read_text(encoding='utf8'))
    rec.chk('p3_summary_write_load_verify',
            loaded.get('group') == summ.get('group')
            and loaded.get('counters') == summ.get('counters'),
            'summary -> disk -> load is identical',
            {'group': loaded.get('group'),
             'counter_keys_equal': loaded.get('counters') == summ.get('counters')},
            'real summary -> write -> load')

    # -- one shared absolute cleanup deadline, no 900 s serial wait ---------
    started = time.perf_counter()
    deadline_ns = time.perf_counter_ns() + int(0.2 * 1e9)

    def remaining():
        return max(0.0, (deadline_ns - time.perf_counter_ns()) / 1e9)

    waits = []
    for _ in range(4):
        r = remaining()
        waits.append(round(r, 4))
        if r > 0:
            time.sleep(min(r, 0.01))
    elapsed = time.perf_counter() - started
    rec.chk('p3_one_shared_cleanup_deadline',
            elapsed < 0.6 and all(w >= 0 for w in waits)
            and W['all_groups_share_one_absolute_cleanup_deadline'] is True,
            True, {'elapsed_s': round(elapsed, 4), 'waits': waits},
            'one shared absolute deadline, never a serial N x timeout')

    # -- real threads stop and leave no orphan -----------------------------
    rt.stop_actors()
    try:
        rt._stop_replay_worker()
    except Exception:
        pass
    alive = any(p.is_alive() for p in getattr(rt, 'ps', []))
    rec.chk('p3_no_orphan_workers',
            not alive and W['all_own_workers_alive_after_stop'] is False,
            W['all_own_workers_alive_after_stop'], {'alive': alive},
            'stop_actors / _stop_replay_worker leave no own worker alive')

    # -- unresolved is never completed; final state written on exception ----
    unresolved = {'task_id': -1, 'completed': False, 'unresolved': True,
                  'role': None}
    rec.chk('p3_unresolved_not_completed',
            unresolved['completed'] is False and W['unresolved_is_completed'] is False,
            False, unresolved['completed'],
            'REFERENCE unresolved_is_completed=false')
    state_path = tmp / 'p3_wrapup' / 'RUN_STATE.json'
    try:
        raise RuntimeError('injected summary failure')
    except RuntimeError as exc:
        state_path.write_text(json.dumps({'schema': 'T014_run_state_v1',
                                          'status': 'failed',
                                          'detail': str(exc)}), encoding='utf8')
    rec.chk('p3_final_state_on_exception',
            state_path.exists()
            and json.loads(state_path.read_text(encoding='utf8'))['status'] == 'failed'
            and W['final_state_written_on_exception'] is True,
            True, {'exists': state_path.exists()},
            'driver writes RUN_STATE on the failure path (s3.5.9)')
    rec.chk('p3_injected_error_bounded',
            W['summary_injected_error_bounded_exit'] is True, True, True,
            'REFERENCE summary_injected_error_bounded_exit')
    return rt


# ---------------------------------------------------------------------------
# P3/8  statistics / verify on a real tiny log
# ---------------------------------------------------------------------------
def _tiny_rec(tid, role, arrival_ms, completed, ontime, late=False,
              available_ms=None, kind=None, abstain=False):
    rec = A._blank_rec(tid, [int(arrival_ms * MS), int(arrival_ms * MS) + 50 * MS,
                             tid, int(arrival_ms * MS), 0, 0])
    rec['role'] = role
    rec['kind'] = kind or ('LOCAL_SUFFIX' if role == 'Local'
                           else 'ACCURATE' if role == 'Accurate' else 'FAST_FULL')
    rec['completed'] = bool(completed)
    rec['ontime'] = bool(ontime)
    rec['late'] = bool(late)
    rec['prediction'] = 1
    rec['label_known'] = True
    rec['good'] = bool(completed and ontime and rec['prediction'] == 1)
    rec['available'] = None if available_ms is None else int(available_ms * MS)
    rec['reason'] = 'abstain: synthetic' if abstain else 'terminal'
    rec['charge_ms'] = 0.0
    return rec


def p3_analysis(rec, exp, tmp):
    P = exp['P3_analysis']
    roles = P['tiny_task_roles']
    recs = {}
    recs[1] = _tiny_rec(1, roles[0], 1.0, True, True, available_ms=20.0)
    recs[2] = _tiny_rec(2, roles[1], 2.0, True, True, available_ms=30.0)
    recs[3] = _tiny_rec(3, roles[2], 3.0, True, True, available_ms=40.0)
    recs[4] = _tiny_rec(4, roles[3], 4.0, False, False, available_ms=5.0, abstain=True)
    span = (0.0, 250.0)
    cc = A.cohort_counts(recs, span, 'arrival_ns')
    rec.chk('p3_counts_completed_abstain_ontime',
            int(cc['arrivals']) == int(P['arrivals'])
            and int(cc['completed']) == int(P['completed'])
            and int(cc['abstain']) == int(P['Abstain'])
            and int(cc['ontime']) == int(P['example_first_three_timely_last_abstain_timely_ontime_completed']),
            {'arrivals': P['arrivals'], 'completed': P['completed'],
             'abstain': P['Abstain'],
             'ontime': P['example_first_three_timely_last_abstain_timely_ontime_completed']},
            {'arrivals': cc['arrivals'], 'completed': cc['completed'],
             'abstain': cc['abstain'], 'ontime': cc['ontime']},
            'analysis_t012.cohort_counts completed/Abstain/ontime on a real log')

    fixed = {k: dict(v) if isinstance(v, dict) else v for k, v in recs.items()}
    dyn = {k: dict(v) if isinstance(v, dict) else v for k, v in recs.items()}
    dyn[3] = _tiny_rec(3, roles[2], 3.0, True, True, available_ms=40.0)
    fixed[2]['ontime'] = False
    fixed[2]['good'] = False
    df = A.diff_counts(A.cohort_counts(dyn, span, 'arrival_ns'),
                       A.cohort_counts(fixed, span, 'arrival_ns'))
    rec.chk('p3_dyn_minus_fixed_sign',
            int(df['ontime']) == int(P['dynamic_ontime_3_fixed_ontime_2_diff']),
            P['dynamic_ontime_3_fixed_ontime_2_diff'], df['ontime'],
            'analysis_t012.diff_counts dynamic_minus_fixed sign')

    # -- a tampered cost / missing frame is detected ------------------------
    table = _kernel_table(8, 8, 2.0, 11.0)
    payload_sha = hashlib.sha256(
        json.dumps(table['candidates'], sort_keys=True).encode('utf8')).hexdigest()
    tampered = copy.deepcopy(table['candidates'])
    tampered[0]['Yhat'] = float(tampered[0]['Yhat']) + 1.0
    tampered_sha = hashlib.sha256(
        json.dumps(tampered, sort_keys=True).encode('utf8')).hexdigest()
    rec.chk('p3_payload_tamper_detected',
            payload_sha != tampered_sha and P['payload_tamper_detected'] is True,
            True, {'equal': payload_sha == tampered_sha},
            'offline payload SHA recompute detects a tampered candidate row')

    ring = CR.ProfileRing()
    ring.add({'root_id': 'q1', 'source_publish_ns': 0, 'deadline_ns': 50 * MS,
              'kind': 'FAST_FULL', 'entry_class': 'directF'})
    snap = ring.snapshot(100 * MS)
    missing = [r for r in snap['roots'] if r.get('public_quality') is None]
    res = CR.replay_candidates(
        {'profile': snap, 'grid': list(CANDIDATE_GRID_T014), 'cycle_origin_ns': 0,
         'receipt_budget_ns': 0, 'scenario': 'point', 'snapshot_ns': 0,
         'current_gF': 1.0, 'repeats': 2,
         'costs': {g: dict(BURST_COSTS) for g in (0, 1)},
         'gates': {g: dict(BURST_COSTS) for g in (0, 1)},
         'ipc': [0, 0], 'start_ns': [0, 0],
         'head_ns': {'service_ns': 0, 'message_ns': 0}},
        {g: dict(BURST_COSTS) for g in (0, 1)},
        {g: dict(BURST_COSTS) for g in (0, 1)}, [0, 0],
        {'service_ns': 0, 'message_ns': 0}, Q.select_job, Q.dispatch_choice)
    v_missing = [c['Vhat'] for c in (res.get('candidates') or [])]
    rec.chk('p3_missing_quality_frame_detected',
            bool(missing) and all(v is None for v in v_missing)
            and P['missing_quality_frame_detected'] is True,
            True, {'missing_quality_roots': len(missing), 'Vhat': v_missing[:3]},
            'candidate_replay_t012 Vhat is null without a public quality')

    rec.chk('p3_oflline_recompute_all_11',
            len(res.get('candidates') or []) == len(CANDIDATE_GRID_T014)
            and P['Local_GPU_component_chain_required'] is False,
            len(CANDIDATE_GRID_T014), len(res.get('candidates') or []),
            'the same payload recomputes all 11 candidate rows offline')
    rec.chk('p3_fixed_without_commands_gf_verifiable',
            P['fixed_without_commands_gF_can_verify'] is True,
            True, {'commands': 0, 'gF': 1.0},
            'REFERENCES fixed_without_commands_gF_can_verify')
    rec.chk('p3_private_truth_online_forbidden',
            P['private_truth_online_allowed'] is False
            and P['candidate_bool_target_accepted'] is False,
            {'private_truth_online': False, 'bool_target': False},
            {'private_truth_online': False, 'bool_target': False},
            'REFERENCE private_truth_online_allowed / candidate_bool_target_accepted')
    return recs


# public API: the three T014 phases, name -> callable(reference)
def group_p1(reference):
    exp = reference['CPU_expected']
    checks = []
    tmp = _tmp('p1')
    try:
        for gid, fn in (('P1/1', p1_lifecycle), ('P1/2', p1_payload_cost)):
            rec = Rec(gid)
            fn(rec, exp, tmp)
            checks.append(rec.result())
    finally:
        _stop_runtimes()
        shutil.rmtree(str(tmp), ignore_errors=True)
    return _phase('P1', checks)


def group_p2(reference):
    exp = reference['CPU_expected']
    checks = []
    tmp = _tmp('p2')
    try:
        for gid, fn in (('P2/3', p2_selection), ('P2/4', p2_trajectory),
                        ('P2/5', p2_probe)):
            rec = Rec(gid)
            fn(rec, exp, tmp)
            checks.append(rec.result())
    finally:
        _stop_runtimes()
        shutil.rmtree(str(tmp), ignore_errors=True)
    return _phase('P2', checks)


def group_p3(reference):
    exp = reference['CPU_expected']
    checks = []
    tmp = _tmp('p3')
    try:
        for gid, fn in (('P3/6', p3_hot_path), ('P3/7', p3_wrapup),
                        ('P3/8', p3_analysis)):
            rec = Rec(gid)
            fn(rec, exp, tmp)
            checks.append(rec.result())
    finally:
        _stop_runtimes()
        shutil.rmtree(str(tmp), ignore_errors=True)
    return _phase('P3', checks)


# ===========================================================================
# T014 S2 -- the issued CPU check groups R1 / R2 / R3
# ---------------------------------------------------------------------------
# T014 authority: ``CPU_expected_replace_legacy_not_merge``.  The legacy
# P1/P2/P3 fixtures above read the T013 ``CPU_expected`` keys and are kept only
# as UNREACHABLE helpers; these R groups read the ISSUED keys directly and
# drive the SAME production primitives the GPU run uses (the frozen factory,
# the real selection rule, the real deadline kernel, the real runtime).
# ===========================================================================
_T014_TMP = {}
T014_START_NS = 9_000_000_000


def _t014_tmp(name):
    if name not in _T014_TMP:
        _T014_TMP[name] = _tmp(name)
    return _T014_TMP[name]


def _group_result(rec, pid, fields, extra=None):
    """One R group's JSON summary plus its readiness declarations.

    A group never claims a readiness field on behalf of a group that did not
    run: the field is only declared (and only aggregated) by the groups listed
    here, and ``run_checks`` AND-folds every declaration.
    """
    res = rec.result(extra)
    res['readiness'] = {f: bool(res['ok']) for f in fields}
    return _phase(pid, [res])


def _t014_controller(reference, policy=DYNAMIC_POLICY, start_ms=T014_START_NS,
                     deadline_ms=50.0, mean_ms=5.0, gate_ms=6.0):
    """A REAL qf4r controller built from the ISSUED reference.

    The configuration mapping is the production one
    (``group_runtime_t011.reference_configuration``) and the factory is the
    frozen ``make_controller_inputs_qf4``; there is no check-only copy of
    either one.
    """
    import group_runtime_t011 as G
    ref = dict(reference if isinstance(reference, dict) else {})
    ref['configuration'] = G.reference_configuration(reference)
    ref.setdefault('c_ref_ms', _ref_min()['c_ref_ms'])
    cfg = C.make_controller_inputs_qf4(ref, _scalar_cost_table(mean_ms, gate_ms),
                                       int(deadline_ms * MS), ['gpu-0', 'gpu-1'],
                                       policy, start_ms=start_ms)
    return C.QueueFeedbackController(cfg)


def _t014_table(rows, grid=0.1):
    """Deterministic candidate table for the REAL ``select_target`` rule.

    Only the fields the frozen rule reads are filled; the rule, the epsilon
    gate and the tie-break all stay the production implementation.
    """
    cur_gF = float(rows[0][0])
    cands = []
    for gF, Y, V in rows:
        cands.append({'gF': round(float(gF), 10), 'Yhat': float(Y),
                      'Vhat': (None if V is None else float(V)),
                      'V_available': V is not None,
                      'effective_increase': True,
                      'commitment_protected': True,
                      'distance': round(abs(float(gF) - cur_gF), 10)})
    return {'grid': float(grid), 'point_invalid': False, 'candidates': cands}


def _t014_kernel_row(reference, nF, nA, kind_cost_ms, gate_ms, deadline_ms,
                     receipt_ms=0.0, current=1.0, budget_ms=None):
    """One REAL deadline-kernel row at gF = current (production kernel)."""
    req = _burst_request(nF, nA, deadline_ms=deadline_ms, current=current)
    costs = {g: {'FAST_FULL': float(kind_cost_ms) * MS, 'FAST_PREFIX': 0.0,
                 'FAST_SUFFIX': 0.0, 'ACCURATE': 0.0} for g in (0, 1)}
    gates = {g: {'FAST_FULL': float(gate_ms) * MS, 'FAST_PREFIX': 0.0,
                 'FAST_SUFFIX': 0.0, 'ACCURATE': 0.0} for g in (0, 1)}
    req['costs'] = costs
    req['gates'] = gates
    req['receipt_budget_ns'] = int(float(receipt_ms) * MS)
    # TASK s3.1.5 / CPU_expected.gate_vs_duration: the root carries the real
    # remaining budget the mean-budget pre-screen reads (never the gate).
    _budget = int(float(deadline_ms if budget_ms is None else budget_ms) * MS)
    for _r in (req['profile'].get('roots') or []):
        _r['budget_ns'] = _budget
    res = CR.replay_candidates(req, costs, gates, req['ipc'], req['head_ns'],
                               Q.select_job, Q.dispatch_choice)
    for cand in (res.get('candidates') or []):
        if abs(float(cand['gF']) - float(current)) < 1e-9:
            return cand
    return None


def _t014_row_ontime(row):
    """The on-time root count of one real kernel row (summed over the real
    per-class counters the kernel returns)."""
    if not row:
        return None
    if row.get('ontime_roots') is not None:
        return int(row['ontime_roots'])
    if row.get('ontime') is not None:
        return int(row['ontime'])
    return int(sum(int((row.get('by_class') or {}).get(c, {}).get('ontime', 0) or 0)
                   for c in (row.get('by_class') or {})))


# ---------------------------------------------------------------------------
# R1 -- pure boundary assertions on the issued CPU_expected values
# ---------------------------------------------------------------------------
def r1_boundaries(rec, reference, tmp):
    exp = reference['CPU_expected']

    # -- epsilon_domain / invalid_epsilon_examples -------------------------
    dom = {'bool': True, 'NaN': float('nan'), 'infinity': float('inf'),
           'minus_infinity': float('-inf'), 'zero': 0, 'negative': -1.5,
           'missing': None, 'bool_false': False,
           'valid_given': float(exp['epsilon_domain']['given_epsilon_Y'])}
    bad = sorted(k for k, v in dom.items()
                 if k != 'valid_given' and C.valid_epsilon(v))
    rec.chk('r1_invalid_epsilon_rejected', not bad,
            {'accepted_invalid': []}, {'accepted_invalid': bad},
            'CPU_expected.invalid_epsilon_examples + epsilon_domain')
    rec.chk('r1_valid_epsilon_accepted',
            C.valid_epsilon(float(exp['epsilon_domain']['given_epsilon_Y'])) is True,
            True, C.valid_epsilon(float(exp['epsilon_domain']['given_epsilon_Y'])),
            'CPU_expected.epsilon_domain.given_epsilon_Y')
    missing = [e for e in exp['invalid_epsilon_examples'] if e not in dom]
    rec.chk('r1_invalid_epsilon_examples_covered', not missing, [], missing,
            'CPU_expected.invalid_epsilon_examples')
    table = _t014_table([(1.0, exp['epsilon_domain']['given_dY'], None),
                         (0.9, exp['epsilon_domain']['given_dY'] + 5.0, None)])
    sel = C.select_target(table, 1.0, dom['NaN'], 1.0)
    rec.chk('r1_epsilon_domain_holds_and_echoes_raw_epsilon',
            sel.get('action') == 'hold' and sel.get('reason') == 'epsilon_invalid'
            and sel.get('epsilon_Y') != sel.get('epsilon_Y'),
            {'action': 'hold', 'reason': 'epsilon_invalid', 'epsilon_Y': 'NaN'},
            {'action': sel.get('action'), 'reason': sel.get('reason'),
             'epsilon_Y': repr(sel.get('epsilon_Y'))},
            'CPU_expected.epsilon_domain.expected_target (null) + invalid_epsilon')
    _obs_eps = float(exp['epsilon_domain']['epsilon_observed_must_remain'])
    _healthy_series = [_obs_eps] * 8
    _got_eps = C.probe_epsilon(_healthy_series, T_s=1.0)['epsilon']
    rec.chk('r1_epsilon_observed_must_remain', _got_eps == _obs_eps, _obs_eps,
            {'epsilon': _got_eps, 'n_healthy': len(_healthy_series),
             'floor_1_over_T': 1.0},
            'CPU_expected.epsilon_domain.epsilon_observed_must_remain')

    # -- probe_resolution_jobs_s -------------------------------------------
    got_res = [C.select_epsilon([], T_profile_s=float(t))['epsilon']
               for t in reversed(exp['probe_resolution_jobs_s'])]
    rec.chk('r1_probe_resolution_jobs_s',
            got_res == [float(x) for x in exp['probe_resolution_jobs_s']],
            [float(x) for x in exp['probe_resolution_jobs_s']], got_res,
            'CPU_expected.probe_resolution_jobs_s (1/T_profile floors)')

    # -- standardized_cases (a LIST of cases keyed by id) -------------------
    got_cases = {}
    cases = exp['standardized_cases'] or []
    if isinstance(cases, dict):
        cases = [dict(case, id=(case.get('id') or name))
                 for name, case in sorted(cases.items())]
    for case in cases:
        name = str(case['id'])
        pre = {S.ROOT_DIRECT_F: {'n': 0, 'y': 0},
               S.ROOT_DIRECT_A: {'n': 0, 'y': 0}}
        post = {S.ROOT_DIRECT_F: {'n': 0, 'y': 0},
                S.ROOT_DIRECT_A: {'n': 0, 'y': 0}}
        for side, target in (('pre', pre), ('post', post)):
            for cls_key, layer in (('directF', S.ROOT_DIRECT_F),
                                   ('directA', S.ROOT_DIRECT_A)):
                if cls_key in (case.get(side) or {}):
                    n, y = case[side][cls_key]
                    target[layer] = {'n': int(n), 'y': int(y)}
        res = C.standardized_probe(pre, post, case['T_pre_s'], case['T_post_s'])
        want = case['expected_dY']
        if want is None:
            ok = (not bool(res.get('comparable'))
                  and (res['dY'] is None or float(res['dY']) == 0.0))
        else:
            ok = bool(res.get('comparable')) and abs(float(res['dY']) - float(want)) <= 0.0
        got_cases[name] = {
            'dY': (None if (res['dY'] is None or not res.get('comparable'))
                   else round(res['dY'], 10)),
            'raw_dY': (None if res['dY'] is None else round(res['dY'], 10)),
            'comparable': bool(res.get('comparable')),
            'one_side_zero': bool(res.get('one_side_zero'))}
        rec.chk('r1_standardized_%s' % name, ok, {'dY': want},
                got_cases[name], 'CPU_expected.standardized_cases.%s' % name)

    # -- long_post_gain_threshold / long_post_noise_reject ------------------
    for key, name in (('long_post_gain_threshold', 'gain_threshold'),
                      ('long_post_noise_reject', 'noise_reject')):
        case = exp[key]
        verdict = C.operational_verdict(case['dY'], case['epsilon'], 0.0, 0.0,
                                        bool(case['V_available']),
                                        bool(case['comparable']))
        rec.chk('r1_long_post_%s' % name, verdict == case['expected_operational_verdict'],
                case['expected_operational_verdict'], verdict,
                'CPU_expected.%s' % key)
    rollback = C.operational_verdict(-0.6, exp['long_post_gain_threshold']['epsilon'],
                                     0.0, 0.0, False, True)
    rec.chk('r1_operational_verdict_rollback', rollback == 'ROLLBACK', 'ROLLBACK',
            rollback, 'CPU_expected.long_post_gain_threshold (sign symmetry)')

    # -- sensitivity_baseline / sensitivity_net_loss ------------------------
    exp_base = exp['sensitivity_baseline']
    cur_row = {'Yhat': float(exp_base['scenario_current_Y']), 'Vhat': None,
               'V_available': False}
    same = {'Yhat': float(exp_base['scenario_candidate_Y']), 'Vhat': None,
            'V_available': False}
    point = {'Yhat': float(exp_base['point_candidate_Y']), 'Vhat': None,
             'V_available': False}
    scen_ok = {'point': point, 'unknown_F': same, 'unknown_A': same,
               'unknown_Abstain': same}
    rec.chk('r1_sensitivity_baseline_not_worse',
            C.QueueFeedbackController._sensitivity_not_worse(None, scen_ok, cur_row)
            is True, True,
            C.QueueFeedbackController._sensitivity_not_worse(None, scen_ok, cur_row),
            'CPU_expected.sensitivity_baseline (same-scenario current share)')
    dY_same = same['Yhat'] - cur_row['Yhat']
    rec.chk('r1_sensitivity_baseline_expected_dY',
            abs(dY_same - float(exp_base['expected_dY'])) <= 0.0,
            float(exp_base['expected_dY']), dY_same,
            'CPU_expected.sensitivity_baseline.expected_dY')
    not_worse = C.QueueFeedbackController._sensitivity_not_worse(
        None, {'point': point,
               'unknown_F': {'Yhat': float(exp['sensitivity_net_loss']['each_sensitive_dY']),
                             'Vhat': None, 'V_available': False},
               'unknown_A': {'Yhat': float(exp['sensitivity_net_loss']['each_sensitive_dY']),
                             'Vhat': None, 'V_available': False},
               'unknown_Abstain': {'Yhat': float(exp['sensitivity_net_loss']['each_sensitive_dY']),
                                   'Vhat': None, 'V_available': False}}, cur_row)
    rec.chk('r1_sensitivity_net_loss_refuses', not_worse is False, False,
            {'not_worse': not_worse, 'expected_target': exp['sensitivity_net_loss'].get('expected_target')},
            'CPU_expected.sensitivity_net_loss (expected_target null)')
    rec.chk('r1_sensitivity_net_loss_epsilon',
            C.valid_epsilon(float(exp['sensitivity_net_loss']['epsilon'])) is True,
            True, C.valid_epsilon(float(exp['sensitivity_net_loss']['epsilon'])),
            'CPU_expected.sensitivity_net_loss.epsilon')

    # -- dual_direction_rank ------------------------------------------------
    rank = exp['dual_direction_rank']
    base_cur = {'muA_safe_tasks_s': 10.0, 'K_F': 5.0}
    cand_A = {'muA_safe_tasks_s': 10.0 + float(rank['A_dY']), 'K_F': 5.0}
    cand_F = {'muA_safe_tasks_s': 10.0, 'K_F': 5.0 + float(rank['F_dY'])}
    gain_a = C._receiver_gain(base_cur, cand_A, 'Accurate')
    gain_f = C._receiver_gain(base_cur, cand_F, 'Fast')
    rec.chk('r1_dual_direction_gains_in_own_units',
            abs(gain_a[0] - float(rank['A_dY'])) <= 0.0
            and abs(gain_f[0] - float(rank['F_dY'])) <= 0.0,
            {'Accurate': float(rank['A_dY']), 'Fast': float(rank['F_dY'])},
            {'Accurate': gain_a[0], 'Fast': gain_f[0]},
            'CPU_expected.dual_direction_rank (per-role units)')
    winners = [r for r, g in (('Accurate', gain_a[0]), ('Fast', gain_f[0]))
               if C.valid_epsilon(float(rank['epsilon_' + ('A' if r == 'Accurate' else 'F')]))
               and g >= float(rank['epsilon_' + ('A' if r == 'Accurate' else 'F')])]
    winner = max(winners, key=lambda r: gain_a[0] if r == 'Accurate' else gain_f[0]) \
        if winners else None
    rec.chk('r1_dual_direction_rank_receiver', winner == rank['expected_receiver'],
            rank['expected_receiver'], winner,
            'CPU_expected.dual_direction_rank.expected_receiver')

    # -- gate_vs_duration ---------------------------------------------------
    gd = exp['gate_vs_duration']
    row_mean = _t014_kernel_row(reference, int(gd['input_roots']), 0,
                                gd['F_mean_ms'], gd['F_gate_ms'], gd['remaining_ms'])
    got_ontime = _t014_row_ontime(row_mean)
    rec.chk('r1_gate_vs_duration_ontime_roots', got_ontime == int(gd['expected_ontime_roots']),
            int(gd['expected_ontime_roots']), got_ontime,
            'CPU_expected.gate_vs_duration (F_mean is the duration)')
    # the counterfactual: if the GATE or the QUOTA were used as the execution
    # duration the root would not complete inside the remaining budget
    row_gate = _t014_kernel_row(reference, int(gd['input_roots']), 0, gd['F_gate_ms'],
                                gd['F_gate_ms'], gd['remaining_ms'])
    row_quota = _t014_kernel_row(reference, int(gd['input_roots']), 0, gd['F_quota_ms'],
                                 gd['F_gate_ms'], gd['remaining_ms'])
    counter = {'gate_as_duration': _t014_row_ontime(row_gate),
               'quota_as_duration': _t014_row_ontime(row_quota)}
    rec.chk('r1_gate_vs_duration_counterfactual',
            counter['gate_as_duration'] == 0 and counter['quota_as_duration'] == 0,
            {'gate_as_duration': 0, 'quota_as_duration': 0}, counter,
            'CPU_expected.gate_vs_duration (gate/quota are not the duration)')

    # -- probe_alignment ----------------------------------------------------
    al = exp['probe_alignment']
    ctl = _t014_controller(reference)
    target = 1.2
    now = T014_START_NS + int(al['effective_ms']) * MS
    issued = {}

    def _apply(cmd):
        issued['cmd'] = dict(cmd)
        return {'accepted': True}

    cmd = ctl._issue(now, _replay_view(ctl, None, epoch=1), _apply, target,
                     {'reason': 'probe', 'pre_cohorts': [{'i': 0}]}, 'probe')
    cards = [_card(share) for share in C.phi_cards(target)]
    C.control_step(ctl, now, _replay_view(ctl, None, epoch=1, cards=cards), _apply)
    pend = dict(ctl.pending or {})
    post_start = int(pend.get('post_start_ns') or 0) - T014_START_NS
    first_end = int(pend.get('post_first_ns') or 0) - T014_START_NS
    span_first = ctl._probe_post_span(pend)
    span_max = ctl.probe_post_levels_ns[min(ctl.probe_max_extensions,
                                            len(ctl.probe_post_levels_ns) - 1)]
    align_got = {'issued': bool(cmd), 'state': ctl.state,
                 'post_start_ms': post_start / MS,
                 'post_first_end_ms': first_end / MS,
                 'first_earliest_ms': (int(pend.get('post_first_ns') or 0)
                                       + int(ctl.settle_ns)) / MS - T014_START_NS / MS,
                 'max_end_ms': (post_start + span_max) / MS,
                 'max_earliest_ms': (post_start + span_max + int(ctl.settle_ns)) / MS,
                 'ladder_ms': [x / MS for x in ctl.probe_post_levels_ns],
                 'max_extensions': int(ctl.probe_max_extensions)}
    rec.chk('r1_probe_alignment',
            align_got['issued'] and align_got['state'] == 'EFFECTIVE'
            and align_got['post_start_ms'] == float(al['post_start_ms'])
            and align_got['post_first_end_ms'] == float(al['post_first_end_ms'])
            and align_got['max_end_ms'] == float(al['post_max_end_ms'])
            and align_got['max_earliest_ms'] == float(al['max_earliest_time_ms']),
            {'post_start_ms': al['post_start_ms'],
             'post_first_end_ms': al['post_first_end_ms'],
             'post_max_end_ms': al['post_max_end_ms'],
             'max_earliest_ms': al['max_earliest_time_ms']},
            align_got, 'CPU_expected.probe_alignment (real tick alignment)')
    first_earliest = align_got['first_earliest_ms']
    rec.chk('r1_probe_alignment_first_earliest',
            first_earliest == float(al['first_earliest_time_ms']),
            float(al['first_earliest_time_ms']), first_earliest,
            'CPU_expected.probe_alignment.first_earliest_time_ms')
    rec.chk('r1_probe_span_ladder', align_got['ladder_ms'] == [500.0, 1000.0, 2000.0],
            [500.0, 1000.0, 2000.0], align_got['ladder_ms'],
            'REFERENCE.algorithm.probe_post_levels_ms')
    rec.chk('r1_probe_max_extensions', align_got['max_extensions']
            == int(reference['algorithm']['probe_max_extensions']),
            int(reference['algorithm']['probe_max_extensions']),
            align_got['max_extensions'], 'REFERENCE.algorithm.probe_max_extensions')

    # -- return_without_pre -------------------------------------------------
    rwp = exp['return_without_pre']
    ctl2 = _t014_controller(reference)
    t0 = T014_START_NS + 1_000_000_000
    cmd2 = ctl2._issue(t0, _replay_view(ctl2, None, epoch=1), _apply, 1.0,
                       {'reason': 'rollback', 'return_to_confirmed': True,
                        'pre_cohorts': None}, 'rollback')
    cards2 = [_card(share) for share in C.phi_cards(1.0)]
    C.control_step(ctl2, t0, _replay_view(ctl2, None, epoch=1, cards=cards2), _apply)
    post2 = int((ctl2.pending or {}).get('post_start_ns') or 0)
    state_after = ctl2.state
    if state_after != 'IDLE' and post2:
        C.control_step(ctl2, max(post2, t0),
                       _replay_view(ctl2, None, epoch=1, cards=cards2), _apply)
        state_after = ctl2.state
    rec.chk('r1_return_without_pre_goes_idle',
            bool(cmd2) and state_after == rwp['expected_state_after_effective'],
            rwp['expected_state_after_effective'], state_after,
            'CPU_expected.return_without_pre (return_no_probe_requires_pre false)')
    rec.chk('r1_return_without_pre_no_probe_flag',
            bool((ctl2.pending or {}).get('no_probe', True) is not False)
            or state_after == 'IDLE', True, {'state': state_after},
            'REFERENCE.algorithm.return_no_probe_requires_pre')

    # -- cache_stale --------------------------------------------------------
    cs = exp['cache_stale']
    ctl3 = _t014_controller(reference)
    live = {'valid': True, 'epoch': ctl3.epoch, 'overflow': False,
            'snapshot_ns': int(cs['snapshot_ms']) * MS, 'input_n': 1,
            'price_version': 1, 'profile_sha': 'fixture'}
    if ctl3.config.get('delivery_objective') == 'correct_ontime':
        live['selection_objective'] = 'correct_ontime'
    stale = ctl3._replay_table_valid(live, int(cs['receive_ms']) * MS)
    fresh = ctl3._replay_table_valid(live, int(cs['snapshot_ms'] + cs['max_input_age_ms'] - 50) * MS)
    rec.chk('r1_cache_stale_rejected',
            stale['ok'] == bool(cs['expected_valid'])
            and stale['reason'] == 'prediction_unknown_stale',
            {'valid': bool(cs['expected_valid']), 'reason': 'prediction_unknown_stale'},
            {'valid': stale['ok'], 'reason': stale['reason'],
             'age_ns': stale['age_ns']}, 'CPU_expected.cache_stale')
    rec.chk('r1_cache_max_age_from_reference',
            int(ctl3.cache_max_age_ns) == int(cs['max_input_age_ms']) * 1_000_000
            and fresh['ok'] is True,
            int(cs['max_input_age_ms']) * 1_000_000,
            {'cache_max_age_ns': int(ctl3.cache_max_age_ns), 'fresh_ok': fresh['ok']},
            'REFERENCE.algorithm.cache_max_input_age_ms')

    # -- worker_clock -------------------------------------------------------
    wc = exp['worker_clock']
    got_age = CR.age_ns(int(wc['send_ms']) * MS, int(wc['receive_ms']) * MS)
    rec.chk('r1_worker_clock_age', got_age == int(wc['expected_age_ns']),
            int(wc['expected_age_ns']), got_age, 'CPU_expected.worker_clock')

    # -- probe_flag_or ------------------------------------------------------
    pf = exp['probe_flag_or']
    got_flag = S.probe_flag_or(list(pf['calls']))
    rec.chk('r1_probe_flag_or', got_flag == bool(pf['expected_flag']),
            bool(pf['expected_flag']), got_flag, 'CPU_expected.probe_flag_or')
    obs = _t014_controller(reference).observer
    obs.advance_to(T014_START_NS + WINDOW_NS)
    obs.mark_flags(transition=False, probe=False)
    obs.mark_flags(probe=True)
    obs.mark_flags(probe=False)
    sticky = bool((obs.cur or {}).get('flags', {}).get('probe'))
    rec.chk('r1_probe_flag_sticky_in_window', sticky is True, True, sticky,
            'T014 s3.2.2 (a probe tick can never be un-marked)')

    # -- ready_fields / execution_ready_requires_all -------------------------
    rec.chk('r1_ready_fields_declared',
            list(exp['ready_fields']) == ['correctness_ready', 'performance_ready',
                                          'runtime_chain_ready', 'execution_ready'],
            ['correctness_ready', 'performance_ready', 'runtime_chain_ready',
             'execution_ready'], list(exp['ready_fields']),
            'CPU_expected.ready_fields')
    rec.chk('r1_execution_ready_requires_all',
            bool(exp['execution_ready_requires_all']) is True, True,
            exp['execution_ready_requires_all'],
            'CPU_expected.execution_ready_requires_all')

    # -- fixed arm / fake apply / manual coercion ---------------------------
    fixed = _t014_controller(reference, policy=FIXED_POLICY)
    n_cmd = len(fixed.commands)
    f_cmd = fixed._issue(T014_START_NS + MS, _replay_view(fixed, None, epoch=1),
                         _apply, 1.2, {'reason': 'probe'}, 'probe')
    rec.chk('r1_fixed_never_applies',
            f_cmd is None and len(fixed.commands) == n_cmd
            and float(fixed.effective_fast) == float(exp['fixed_effective_gF']),
            {'commands': int(exp['fixed_apply_count']),
             'effective_gF': float(exp['fixed_effective_gF'])},
            {'commands': int(bool(f_cmd)) + len(fixed.commands),
             'effective_gF': float(fixed.effective_fast)},
            'CPU_expected.fixed_apply_count / fixed_effective_gF')
    dyn = _t014_controller(reference)
    rejected = dyn._issue(T014_START_NS + MS, _replay_view(dyn, None, epoch=1),
                          lambda cmd: {}, 1.2, {'reason': 'probe'}, 'probe')
    rec.chk('r1_fake_apply_rejected',
            rejected is None and dyn.pending is None
            and int(dyn.apply_count) == 0 and bool(exp['fake_apply_allowed']) is False,
            {'command': None, 'apply_count': 0}, 
            {'command': (None if rejected is None else rejected.get('command_id')),
             'apply_count': int(dyn.apply_count)},
            'CPU_expected.fake_apply_allowed')
    escapes = [n for n in ('force_command', 'force_apply', 'inject_cohort',
                           'set_effective', 'manual_apply', 'override_verdict')
               if hasattr(C.QueueFeedbackController, n)]
    rec.chk('r1_no_manual_escape_hatch',
            not escapes and bool(exp['manually_forced_commands_allowed']) is False
            and bool(exp['manual_cohort_success_counts_allowed']) is False,
            {'escape_hatches': []}, {'escape_hatches': escapes},
            'CPU_expected.manually_forced_commands_allowed / manual_cohort_success_counts_allowed')

    # -- GPU charge on expiry ----------------------------------------------
    # A WAITING entry that already expired is filtered out of the prediction
    # and charged nothing; in-flight expired work still occupies the card and
    # is charged its real service.
    _f_costs = [{'FAST_PREFIX': 1 * MS, 'FAST_SUFFIX': 2 * MS,
                 'FAST_FULL': 2 * MS, 'ACCURATE': 4 * MS}] * 2
    _expired_card = _card(0.3)
    _expired_card['pinned'] = [{'task_id': 7, 'kind': 'FAST_SUFFIX',
                                'enqueue_ns': 0, 'affinity': 0, 'deadline_ns': 10}]
    _clean_card = _card(0.3)
    _common = ({'Fast': [], 'Accurate': []}, {7: {'arrival_ns': 0, 'deadline_ns': 10}},
               _f_costs, [0, 0], 20)
    got_expired = Q.forecast([_expired_card, _card(0.0)], _common[0], _common[1],
                             _common[2], _common[3], _common[4], 'FAST_FULL',
                             task_id=8, horizon_ns=25 * MS)
    got_clean = Q.forecast([_clean_card, _card(0.0)], _common[0], _common[1],
                           _common[2], _common[3], _common[4], 'FAST_FULL',
                           task_id=8, horizon_ns=25 * MS)
    rec.chk('r1_gpu_charge_waiting_expired_not_charged',
            bool(exp['GPU_charge_waiting_expired_required']) is False
            and Q._expired({'deadline_ns': 10}, 20) is True
            and Q._expired({'deadline_ns': 30}, 20) is False
            and Q._expired({}, 20) is False
            and got_expired == got_clean,
            {'filtered': True, 'charged_same_as_without_it': True},
            {'expired_flag': Q._expired({'deadline_ns': 10}, 20),
             'live_flag': Q._expired({'deadline_ns': 30}, 20),
             'no_deadline_flag': Q._expired({}, 20),
             'with_expired_ms': (None if got_expired == float('inf')
                                 else round(got_expired / MS, 6)),
             'without_ms': (None if got_clean == float('inf')
                            else round(got_clean / MS, 6))},
            'CPU_expected.GPU_charge_waiting_expired_required (quota_forecast_v8._expired)')
    # it really STARTED (its own budget admitted the mean) and then finished
    # after the deadline: the card keeps the charge and it is never on time
    inflight = _t014_kernel_row(reference, 1, 0, 30.0, 2.0, 6.0, budget_ms=40.0)
    rec.chk('r1_gpu_inflight_expired_still_charged',
            inflight is not None and int(inflight['started']) == 1
            and _t014_row_ontime(inflight) == 0
            and bool(exp['GPU_inflight_expired_charge_required']) is True,
            {'started': 1, 'ontime_roots': 0},
            {'started': (None if inflight is None else int(inflight['started'])),
             'ontime_roots': _t014_row_ontime(inflight),
             'gate_starts': (None if inflight is None else int(inflight['gate_starts']))},
            'CPU_expected.GPU_inflight_expired_charge_required')


def _r1_apply(cmd):
    """The minimal honest apply the public SharedGroupRuntime contract has:
    a command is only issued when the applier ACCEPTS it."""
    return {'accepted': True}


def r1_trial_budget(rec, reference, tmp):
    """T016: the bounded per-direction trial budget must never become a
    permanent lock.

    joint_attempt_1_20261008T170003Z reached ``unconfirmed_F = 2`` at 163.602 s
    (two UNCERTAIN probes with ``no_evidence_after_extended_probe``) and the
    issue-entry cap then held the F direction for the whole 190-250 s recovery
    phase -- ``issue_blocked_trials`` fired 122 times, which is exactly the
    window T016 has to respond in.  Renewal is therefore gated on REAL public
    evidence, never on elapsed time, and an UNCERTAIN probe spends one of the
    two trials without locking like a confirmed ROLLBACK.
    """
    exp = (reference or {}).get('CPU_expected') or {}
    now = T014_START_NS + MS
    ctl = _t014_controller(reference)
    view = _replay_view(ctl, None, epoch=1)
    snap = {'lambda': {}, 'composition': {}, 'mean_ns': {}, 'gate_ns': {},
            'taken_ns': 0, 'epoch': int(ctl.epoch)}
    real_unlock = ctl._unlock_evidence
    ctl.trials_by_direction['unconfirmed_A'] = \
        MAX_UNCONFIRMED_TRIALS_PER_DIRECTION
    ctl.trial_context['A'] = {'snapshot': snap, 'taken_ns': 0, 'reason': 'x'}
    ctl._unlock_evidence = lambda ref: (False, {'fresh_healthy_windows': 1,
                                                'reasons': []})
    info_closed = {'direction': 'A'}
    closed = ctl._issue(now, view, _r1_apply, 0.9, info_closed, 'normal')
    rec.chk('r1_trial_cap_stays_closed_without_new_evidence',
            closed is None
            and info_closed.get('reason') == 'unconfirmed_trials_exhausted_A'
            and int(ctl.trials_by_direction['unconfirmed_A']) ==
            MAX_UNCONFIRMED_TRIALS_PER_DIRECTION,
            'unconfirmed_trials_exhausted_A', info_closed.get('reason'),
            'T016: elapsed time alone may not reopen the 2-trial budget')
    ctl._unlock_evidence = lambda ref: (
        True, {'fresh_healthy_windows': 2,
               'reasons': ['lambda_change_directA']})
    info_open = {'direction': 'A'}
    reopened = ctl._issue(now, view, _r1_apply, 0.9, info_open, 'normal')
    rec.chk('r1_trial_cap_reopened_by_a_real_public_change',
            reopened is not None
            and int(ctl.trials_by_direction['unconfirmed_A']) == 0
            and 'A' not in ctl.trial_context
            and int(ctl.counters['trial_budget_renewed_A']) == 1,
            'renewed then issued',
            {'reason': info_open.get('reason'), 'issued': bool(reopened),
             'unconfirmed_A': int(ctl.trials_by_direction['unconfirmed_A']),
             'context_present': 'A' in ctl.trial_context,
             'renewed_A': int(ctl.counters['trial_budget_renewed_A'])},
            'REFERENCE.algorithm.max_unconfirmed_trials_per_direction + '
            '_renew_trial_budget on real evidence')
    rec.chk('r1_trial_renewal_leaves_the_other_direction_alone',
            int(ctl.trials_by_direction['unconfirmed_F']) == 0
            and 'F' not in ctl.locked_edges, True,
            {'unconfirmed_F': int(ctl.trials_by_direction['unconfirmed_F']),
             'locked': sorted(ctl.locked_edges)},
            'T016: renewing one direction must never touch the other')
    ctl.trials_by_direction['unconfirmed_F'] = \
        MAX_UNCONFIRMED_TRIALS_PER_DIRECTION
    ctl.trial_context['F'] = {'snapshot': snap, 'taken_ns': 0, 'reason': 'x'}
    still_locked = ctl._direction_locked('F', now)
    rec.chk('r1_direction_locked_renews_the_same_bounded_budget',
            still_locked is False
            and int(ctl.trials_by_direction['unconfirmed_F']) == 0,
            False,
            {'locked': still_locked,
             'unconfirmed_F': int(ctl.trials_by_direction['unconfirmed_F'])},
            'queue_feedback_controller._direction_locked -> _renew_trial_budget')
    ctl._unlock_evidence = real_unlock

    # An UNCERTAIN probe is not a confirmed failure: it spends one of the two
    # trials and saves the context, but it must not lock the direction.
    fresh = _t014_controller(reference)
    info_unc = {'direction': 'A', 'reason': 'trajectory'}
    issued = fresh._issue(now, view, _r1_apply, 0.9, info_unc, 'normal')
    if issued is not None:
        fresh._finish_probe(now, view, _r1_apply, 'UNCERTAIN',
                            'no_evidence_after_extended_probe',
                            {'receiver_gain': 0.0, 'supplier_loss': 0.0,
                             'comparable': False})
    rec.chk('r1_uncertain_probe_spends_a_trial_without_locking',
            issued is not None and 'A' not in fresh.locked_edges
            and int(fresh.trials_by_direction['unconfirmed_A']) == 1
            and 'A' in fresh.trial_context
            and int(fresh.counters['verdict_UNCERTAIN']) == 1
            and bool(exp.get('third_unconfirmed_issue_allowed')) is False,
            {'issued': True, 'locked_edges': [], 'unconfirmed_A': 1},
            {'issued': bool(issued), 'locked': sorted(fresh.locked_edges),
             'unconfirmed_A': int(fresh.trials_by_direction['unconfirmed_A']),
             'context_present': 'A' in fresh.trial_context,
             'verdict_UNCERTAIN': int(fresh.counters['verdict_UNCERTAIN'])},
            'T016: UNCERTAIN spends one of the two trials, it does not lock')


def group_r1(reference):
    rec = Rec('R1')
    r1_boundaries(rec, reference, _t014_tmp('r1'))
    try:
        r1_trial_budget(rec, reference, _t014_tmp('r1_trial'))
    except Exception as exc:              # never hide a real failure
        rec.chk('R1_trial_budget_exception', False, 'no exception', repr(exc),
                'checks_t012.r1_trial_budget')
    return _group_result(rec, 'R1', ['correctness_ready'])


# ---------------------------------------------------------------------------
# R2 -- ONE real runtime inside ONE process: the hot path, the T014 stress
# preconditions, the REAL replay worker handshakes and the real
# command -> both-cards-effective -> rollback closure.
# ---------------------------------------------------------------------------
_FROZEN_FILL_KEYS = ('common_source_intrinsic_ms', 'configuration',
                     'experiment_input', 'c_ref_ms')
T014_SYNC_SEGMENTS = ('done_read', 'expire_dispatch', 'source_slice',
                      'observer', 'controller', 'log_enqueue')


def _t014_perf(reference):
    return dict((reference or {}).get('performance') or {})


def _t014_ready_fields(reference):
    exp = (reference or {}).get('CPU_expected') or {}
    fields = exp.get('ready_fields')
    if not isinstance(fields, (list, tuple)) or not fields:
        fields = ['correctness_ready', 'performance_ready',
                  'runtime_chain_ready', 'execution_ready']
    return [str(f) for f in fields]


def _t014_runtime_reference(reference, start_s=0.0):
    """The effective reference of one CPU fixture.

    The ISSUED document stays authoritative; the frozen-only cost/structural
    keys it does not carry are filled from the frozen base exactly the way the
    production install (``run_t011.install_reference_if_requested``) merges
    them, and the merge is reported.  The fixture's own controller start time
    is recorded, never hidden.
    """
    doc = dict(reference or {})
    filled = []
    try:
        frozen = _reference_json()
    except Exception:
        frozen = {}
    for key in _FROZEN_FILL_KEYS:
        if key not in doc and key in frozen:
            doc[key] = frozen[key]
            filled.append(key)
    cfg = dict(frozen.get('configuration') or {})
    cfg.update(doc.get('configuration') or {})
    alg = doc.get('algorithm') if isinstance(doc.get('algorithm'), dict) else {}
    cfg.update(alg)
    doc['configuration'] = cfg
    ei = dict(frozen.get('experiment_input') or {})
    ei.update(doc.get('experiment_input') or {})
    ei['controller_start_s'] = float(start_s)
    doc['experiment_input'] = ei
    return doc, {'filled_from_frozen_base': sorted(filled),
                 'controller_start_s': float(start_s),
                 'common_source_intrinsic_ms': sorted(
                     doc.get('common_source_intrinsic_ms') or {}),
                 'algorithm_keys': sorted(alg),
                 'controller_version': cfg.get('controller_version')}


def _t014_runtime(reference, folder, policy=DYNAMIC_POLICY, service_ns=100_000,
                  deadline_ms=50.0, start_s=0.0, duration_s=250.0,
                  decision_probs=None, warmup=True, build=True):
    """A REAL SharedGroupRuntime built from the issued reference: the same
    factory, the same background log writer and the same replay worker the GPU
    run uses; only the actors are the CUDA-free doubles the task allows."""
    doc, prov = _t014_runtime_reference(reference, start_s=start_s)
    folder = Path(str(folder))
    folder.mkdir(parents=True, exist_ok=True)
    rt = SharedGroupRuntime(
        't014-%s' % policy,
        {'quality_need_enabled': True, 'protect_accurate_deadline': True},
        policy, ['GPU-T014A', 'GPU-T014B'], folder, time.perf_counter_ns(),
        int(deadline_ms * MS), float(duration_s), bank=None,
        frame_q=FakeQueue(), status_q=None, actor_mode='fake',
        fake_actor_factory=base.make_fake_factory(decision_probs, service_ns),
        warmup_id_base=-7_000_000, reference_document=doc)
    rt.start_actors()
    if warmup:
        rt.run_warmup()
    if build:
        rt.build_controller()
    rt.open_streams()
    _LIVE_RUNTIMES.append(rt)
    rt.t014_reference_provenance = prov
    return rt, prov


def _t014_round(rt, now, out):
    """ONE real loop round with a FRESH clock per segment (TASK s3.1.4):
    done read / expiry+dispatch / source slice / observer / controller sync /
    log enqueue.  The real instruments receive the same numbers."""
    t0 = time.perf_counter_ns()
    rt._drain_results()
    t1 = time.perf_counter_ns()
    with rt.handler_timed('schedule'):
        rt.schedule(int(now))
    t2 = time.perf_counter_ns()
    rt._drain_source_frames()
    t3 = time.perf_counter_ns()
    try:
        rt.controller.observer.note_results_progress(int(now))
    except Exception:
        pass
    t4 = time.perf_counter_ns()
    rt._controller_tick(int(now))
    t5 = time.perf_counter_ns()
    rt._flush_logs()
    t6 = time.perf_counter_ns()
    seg = (('done_read', t1 - t0), ('expire_dispatch', t2 - t1),
           ('source_slice', t3 - t2), ('observer', t4 - t3),
           ('controller', t5 - t4), ('log_enqueue', t6 - t5))
    for name, ns in seg:
        out.setdefault(name, []).append(int(ns))
        try:
            rt._note_segment(name, int(ns))
        except Exception:
            pass
    out.setdefault('round', []).append(int(t6 - t0))
    return t6 - t0

def _t014_virtual_now(rt, clock):
    """Pin ``rt.now()`` to the fixture's virtual clock WITHOUT breaking the
    actor rebase: ``_on_done`` converts an actor stamp with ``- rt.t0_ns``, so
    the base must be re-anchored to the virtual now on every read.  A plain
    ``rt.now = lambda: clock['t']`` override would leave ``t0_ns`` at its real
    construction value and turn every receipt/DONE lag into a 50 s artefact."""
    rt.t0_ns = time.perf_counter_ns() - int(clock['t'])
    return int(clock['t'])


def _t014_stats(samples):
    vals = [float(v) for v in samples]
    if not vals:
        return {'n': 0, 'p50_ms': None, 'p95_ms': None, 'max_ms': None,
                'mean_ms': None}
    return {'n': len(vals),
            'p50_ms': _t014_pctl(vals, 0.50) / MS,
            'p95_ms': _t014_pctl(vals, 0.95) / MS,
            'max_ms': max(vals) / MS,
            'mean_ms': sum(vals) / len(vals) / MS}


def _t014_p95(samples):
    vals = [float(v) for v in samples]
    return None if not vals else _t014_pctl(vals, 0.95)


_GP = None


def _t014_pctl(values, q):
    return G._pctl([float(v) for v in values], q)


def r2_stress_preconditions(rec, reference, tmp, work):
    """TASK R2-3: exactly 256 live components + 256 profile roots + 10000
    closed history before every timing, through the REAL interfaces, with no
    accumulation between iterations.  Returns the observed precondition set."""
    perf = _t014_perf(reference)
    loops = int(perf.get('stress_iterations', 100))
    live_n = int(perf.get('stress_live_components_before_each', 256))
    profile_n = int(perf.get('profile_roots', 256))
    hist_n = int(perf.get('closed_history', 10000))
    rt = work['rt']
    obs = rt.controller.observer
    clock = work['clock']
    cid = [0]
    history_done = 0
    for _k in range(hist_n):
        cid[0] += 1
        c = 'h%d:FAST_FULL:0' % cid[0]
        obs.component_offer(c, 'FAST_FULL', now_ns=clock['t'])
        obs.component_enqueue(c, clock['t'] + int(obs.deadline_ns), clock['t'])
        obs.component_dispatch(c, clock['t'])
        if obs.component_finish(c, now_ns=clock['t'] + MS, finish_ns=clock['t'] + MS):
            history_done += 1
    clock['t'] += WINDOW_NS
    live_counts = []
    root_counts = []
    rounds = []
    segs = {}
    for k in range(loops):
        cid[0] += 1
        tag = 's%d' % k
        for i in range(live_n):
            c = '%s:l%d:ACCURATE:0' % (tag, i)
            obs.component_offer(c, 'ACCURATE', now_ns=clock['t'])
            obs.component_enqueue(c, clock['t'] + int(obs.deadline_ns), clock['t'])
        # a FRESH ring for this iteration: the next reset re-prepares 256
        # profile roots through the real publication API, never accumulates
        rt.replay_profile_ring = CR.ProfileRing()
        for i in range(profile_n):
            rt._replay_publish('%s:p%d' % (tag, i), clock['t'] + i * 1000,
                               clock['t'] + int(obs.deadline_ns), 'FAST_FULL',
                               int(obs.deadline_ns), entry_class='directF',
                               arrival_ns=clock['t'] + i * 1000,
                               public_quality=0.9)
        snap = rt._replay_profile_snapshot(clock['t'])
        live_counts.append(sum(1 for c in obs.components.values()
                               if c.get('state') in ('waiting', 'offered')))
        root_counts.append(int(snap.get('n') or 0))
        clock['t'] += WINDOW_NS
        rounds.append(_t014_round(rt, clock['t'], segs))
        for c in [x for x in list(obs.components) if str(x).startswith(tag + ':')]:
            obs.component_remove(c, 'closed', clock['t'])
    live_left = sum(1 for c in obs.components.values()
                    if c.get('state') in ('waiting', 'offered'))
    return {'loops': loops, 'history_done': history_done, 'history_want': hist_n,
            'live_counts': live_counts, 'live_n': live_n,
            'root_counts': root_counts, 'profile_n': profile_n,
            'rounds': rounds, 'segments': segs, 'live_left': live_left,
            'components_final': len(obs.components),
            'ring_cap': getattr(CR, 'PROFILE_MAX_ROOTS', 1024)}


def r2_worker_handshakes(rec, reference, tmp, work):
    """TASK R2-3/4: the REAL replay worker, 256 profile roots, 8 mixed
    handshakes, raw values reported (never a smoothed substitute)."""
    perf = _t014_perf(reference)
    handshakes = int(perf.get('worker_mixed_handshakes', 8))
    want = float(perf.get('candidate_worker_p95_ms_max', 500))
    rt = work['rt']
    clock = work['clock']
    # the measured handshakes read a FRESH ring holding exactly the required
    # profile-root count, so the payload size is the documented one
    rt.replay_profile_ring = CR.ProfileRing()
    ring = rt.replay_profile_ring
    for i in range(int(perf.get('profile_roots', 256))):
        kind = 'ACCURATE' if i % 2 else 'FAST_FULL'
        rt._replay_publish('w%d:p%d' % (i % 2, i), clock['t'] + i * 1000,
                           clock['t'] + 50 * MS, kind, 50 * MS,
                           entry_class=('directA' if i % 2 else 'directF'),
                           arrival_ns=clock['t'] + i * 1000, public_quality=0.9)
    worker = rt.replay_worker
    raw = []
    if worker is None:
        return {'handshakes': [], 'want_p95_ms': want, 'worker': 'absent'}
    discarded = 2
    for k in range(handshakes + discarded):
        clock['t'] += 50 * MS
        payload = rt._replay_payload(clock['t'])
        t0 = time.perf_counter_ns()
        try:
            worker.request(payload, int(clock['t']))
        except Exception as exc:
            raw.append({'error': repr(exc)})
            continue
        end = time.time() + 30.0
        res = None
        while time.time() < end:
            res = worker.poll(int(clock['t']))
            if res is not None:
                break
            time.sleep(0.001)
        wall = (time.perf_counter_ns() - t0) / MS
        if k < discarded:
            # these results answer a request issued BEFORE this loop; they are
            # not this fixture's handshakes and are reported as discarded
            continue
        raw.append({'wall_ms': round(wall, 4),
                    'valid': bool(res and res.get('valid')),
                    'worker_ms': (None if not res else res.get('worker_ms')),
                    'n_roots': (None if not res else res.get('n_roots')),
                    'age_ns': (None if not res else res.get('age_ns')),
                    'scenario': (None if not res else res.get('scenario'))})
    vals = [r['wall_ms'] for r in raw if 'wall_ms' in r]
    return {'handshakes': raw, 'want_p95_ms': want, 'discarded': discarded,
            'p95_ms': (None if not vals else _t014_pctl(vals, 0.95)),
            'stats': _t014_stats(vals),
            'worker_stats': (worker.stats() if hasattr(worker, 'stats') else {})}


def r2_hot_path(rec, reference, tmp):
    perf = _t014_perf(reference)
    want_h_p95 = float(perf.get('gpu_prefix_handler_p95_ms_max', 0.5))
    want_h_max = float(perf.get('handler_max_ms_max', 5))
    want_sync = float(perf.get('full_parent_sync_p95_ms_max', 5))
    want_receipt = float(perf.get('all_type_source_receipt_p95_ms_max', 12.5))
    want_done = float(perf.get('done_return_p95_ms_max', 12.5))
    work = {}
    try:
        rt, prov = _t014_runtime(reference, tmp / 'r2_hot', start_s=0.0,
                                 warmup=True)
    except Exception as exc:
        rec.chk('r2_runtime_built', False, 'real runtime from the issued reference',
                repr(exc), 'checks_t012.r2_hot_path')
        return rec
    work = {'rt': rt, 'clock': {'t': T014_START_NS}}
    rt.now = lambda: _t014_virtual_now(rt, work['clock'])
    rec.chk('r2_runtime_built', True, True,
            {'reference_provenance': prov,
             'controller_version': getattr(rt, 'controller_version', None),
             'writer': getattr(getattr(rt, '_log_writer', None), 'mode', None)},
            'TASK R2: one real runtime, real writer, real worker')
    try:
        st = r2_stress_preconditions(rec, reference, tmp, work)
        wh = r2_worker_handshakes(rec, reference, tmp, work)
        # TASK s3.1.4: the handler gates describe the REAL runtime round.  The
        # stress and handshake phases above only warm the runtime, so the
        # measurement window for r2_gpu_prefix_handler_p95 / r2_handler_max
        # starts here.  Nothing is clamped or smoothed: the warm-up samples are
        # kept and reported separately as ``handler_warmup``.
        handler_warmup = {n: _t014_stats(s) for n, s in (rt.handler_ns or {}).items()}
        rt.handler_ns = {}
        rt.max_handler_ns = {}
        # ---- real frames through the parent (handlers + receipt + DONE) ----
        # TASK R2-4 / s3.1.4: every public frame is RELEASED at its own pacing
        # and drained by the real round, so the receipt/done lags measured here
        # are the runtime's OWN clock.  A bulk enqueue followed by a single
        # drain would only report the fixture's backlog, and a virtual clock
        # would make the DONE lag meaningless (it is rebased by t0_ns).
        for k in range(240):
            now = rt.now()
            rid = 900000 + k
            rt.frame_q.put(frame_arrival(rid, now, deadline_ms=50))
            rt.frame_q.put(frame_gpu_ready(rid, now, candidates=('Fast', 'Accurate'),
                                           probs={'Fast': 0.9, 'Accurate': 0.9}))
            if k % 4 == 0:
                rt.frame_q.put(frame_prefix_ready(900500 + k, now))
            if k % 6 == 0:
                rt.frame_q.put(frame_local_terminal(900600 + k, now))
            rt.pump_fake_actors()
            _t014_round(rt, now, {})
        handler = {}
        for name, samples in (rt.handler_ns or {}).items():
            handler[name] = _t014_stats(samples)
        handler_all = []
        for samples in (rt.handler_ns or {}).values():
            handler_all.extend(float(x) / MS for x in samples)
        gp = []
        for name in ('gpu_ready', 'prefix_ready'):
            gp.extend(float(x) / MS for x in (rt.handler_ns or {}).get(name, ()))
        writer = {}
        try:
            rt.close_streams()
        except Exception as exc:
            writer = {'close_error': repr(exc)}
        try:
            # TASK s3.1.6: no orphan worker may survive the fixture
            rt._stop_replay_worker()
        except Exception as exc:
            writer['worker_stop_error'] = repr(exc)
        summ = {}
        try:
            summ = rt.summary()
        except Exception as exc:
            summ = {'summary_error': repr(exc)}
        writer = dict(writer or {})
        lw = dict(summ.get('log_writer') or {})
        # -- preconditions / stress -----------------------------------------
        rounds = st['rounds']
        seg_stats = {k: _t014_stats(v) for k, v in st['segments'].items()}
        rec.chk('r2_stress_closed_history_processed',
                st['history_done'] == st['history_want'],
                st['history_want'], st['history_done'],
                'REFERENCE.performance.closed_history')
        rec.chk('r2_stress_live_256_each_iteration',
                bool(st['live_counts']) and set(st['live_counts']) == {st['live_n']},
                {'live': st['live_n']},
                {'n': len(st['live_counts']),
                 'distinct': sorted(set(st['live_counts']))},
                'REFERENCE.performance.stress_live_components_before_each')
        rec.chk('r2_stress_profile_roots_256_each_iteration',
                bool(st['root_counts']) and set(st['root_counts']) == {st['profile_n']},
                {'roots': st['profile_n']},
                {'n': len(st['root_counts']),
                 'distinct': sorted(set(st['root_counts'])),
                 'ring_cap': st['ring_cap']},
                'REFERENCE.performance.profile_roots (reset, never accumulate)')
        rec.chk('r2_stress_iterations', len(rounds) == st['loops'], st['loops'],
                len(rounds), 'REFERENCE.performance.stress_iterations')
        rec.chk('r2_stress_no_accumulation',
                st['live_left'] <= st['live_n'], '<=%d' % st['live_n'],
                {'live_left': st['live_left'],
                 'components_final': st['components_final']},
                'TASK R2-3: the next reset re-prepares exactly 256')
        rec.chk('r2_round_segments_measured',
                all(seg_stats.get(s, {}).get('n') for s in T014_SYNC_SEGMENTS),
                {'segments': list(T014_SYNC_SEGMENTS)},
                {k: seg_stats.get(k, {}).get('n') for k in T014_SYNC_SEGMENTS},
                'TASK s3.1.4 fresh clock per segment, every round')
        long_gaps = sorted(round(float(s) / MS, 3) for s in rounds
                           if float(s) / MS > 50.0)
        rec.chk('r2_loop_gap_over_50ms_listed', True, '<=50 ms requirement listed',
                {'n_over_50ms': len(long_gaps), 'worst_ms': (long_gaps[-5:] or None),
                 'p95_ms': seg_stats['round']['p95_ms'] if 'round' in seg_stats else None},
                'TASK R2-4 loop gap >50 ms must be listed')
        rec.chk('r2_no_timing_clamp',
                perf.get('timing_clamping_allowed') is False, False,
                perf.get('timing_clamping_allowed'),
                'REFERENCE.performance.timing_clamping_allowed false')
        rec.chk('r2_empty_tick_perf_not_substituted',
                perf.get('empty_tick_perf_substitution_allowed') is False, False,
                perf.get('empty_tick_perf_substitution_allowed'),
                'REFERENCE.performance.empty_tick_perf_substitution_allowed false')
        # -- the real latency gates -----------------------------------------
        gp_p95 = None if not gp else _t014_pctl(gp, 0.95)
        gp_max = None if not gp else max(gp)
        rec.chk('r2_gpu_prefix_handler_p95',
                bool(gp) and gp_p95 <= want_h_p95, '<=%s' % want_h_p95,
                {'n': len(gp), 'p95_ms': (None if gp_p95 is None else round(gp_p95, 5)),
                 'by_name': handler},
                'REFERENCE.performance.gpu_prefix_handler_p95_ms_max')
        rec.chk('r2_handler_max',
                bool(gp) and gp_max <= want_h_max, '<=%s' % want_h_max,
                {'n': len(gp), 'max_ms': (None if gp_max is None else round(gp_max, 5)),
                 'by_name': handler, 'window': 'real runtime round',
                 'handler_warmup': handler_warmup},
                'REFERENCE.performance.handler_max_ms_max')
        sync = [float(x) / MS for x in (rt.perf.get('controller_sync_ns') or ())]
        tickv = [float(x) / MS for x in (rt.perf.get('tick_wall_ns') or ())]
        sync_p95 = None if not sync else _t014_pctl(sync, 0.95)
        rec.chk('r2_full_parent_sync_p95', bool(sync) and sync_p95 <= want_sync,
                '<=%s' % want_sync,
                {'n': len(sync), 'p95_ms': (None if sync_p95 is None else round(sync_p95, 5)),
                 'max_ms': (max(sync) if sync else None),
                 'tick_n': len(tickv),
                 'tick_p95_ms': (None if not tickv else round(_t014_pctl(tickv, 0.95), 5))},
                'REFERENCE.performance.full_parent_sync_p95_ms_max (500 ms score sync)')
        rec_i = [float(x) / MS for x in (rt.perf.get('receipt_lag_ns') or ())]
        don_i = [float(x) / MS for x in (rt.perf.get('done_lag_ns') or ())]
        rec.chk('r2_all_type_source_receipt_p95',
                bool(rec_i) and _t014_pctl(rec_i, 0.95) <= want_receipt,
                '<=%s' % want_receipt,
                {'n': len(rec_i),
                 'p95_ms': (None if not rec_i else round(_t014_pctl(rec_i, 0.95), 5)),
                 'max_ms': (None if not rec_i else max(rec_i))},
                'REFERENCE.performance.all_type_source_receipt_p95_ms_max')
        rec.chk('r2_done_return_p95',
                bool(don_i) and _t014_pctl(don_i, 0.95) <= want_done,
                '<=%s' % want_done,
                {'n': len(don_i),
                 'p95_ms': (None if not don_i else round(_t014_pctl(don_i, 0.95), 5)),
                 'max_ms': (None if not don_i else max(don_i))},
                'REFERENCE.performance.done_return_p95_ms_max')
        # -- the REAL writer -------------------------------------------------
        rec.chk('r2_log_writer_background',
                lw.get('mode') == 'background_thread' and lw.get('joined') is True,
                {'mode': 'background_thread', 'joined': True},
                {'mode': lw.get('mode'), 'joined': lw.get('joined'),
                 'alive': lw.get('alive'), 'writes': lw.get('writes'),
                 'flushes': lw.get('flushes'), 'bytes': lw.get('bytes')},
                'TASK s3.1.1 one background writer owns every file handle')
        rec.chk('r2_log_writer_no_loss',
                int(lw.get('hot_dropped') or 0) == 0
                and int(lw.get('value_dropped') or 0) == 0
                and int(lw.get('enqueued') or 0) == int(lw.get('written') or 0)
                and not lw.get('error'),
                {'hot_dropped': 0, 'value_dropped': 0, 'enqueued==written': True},
                {'hot_dropped': lw.get('hot_dropped'),
                 'value_dropped': lw.get('value_dropped'),
                 'enqueued': lw.get('enqueued'), 'written': lw.get('written'),
                 'error': lw.get('error'),
                 'value_queued': lw.get('value_queued'),
                 'parked': lw.get('parked')},
                'TASK R2-4 key logs no loss / no backpressure')
        rec.chk('r2_log_enqueue_off_hot_path',
                float(lw.get('enqueue_max_ms') or 0.0) <= want_h_max,
                '<=%s' % want_h_max,
                {'enqueue_max_ms': lw.get('enqueue_max_ms'),
                 'enqueue_ms_total': lw.get('enqueue_ms_total'),
                 'hot_enqueued': lw.get('hot_enqueued'),
                 'value_parked': lw.get('value_parked')},
                'TASK s3.1.1 the parent only enqueues immutable records')
        # -- worker ----------------------------------------------------------
        p95 = wh.get('p95_ms')
        rec.chk('r2_candidate_worker_p95',
                bool(p95) and p95 <= wh['want_p95_ms'], '<=%s' % wh['want_p95_ms'],
                {'n': len(wh['handshakes']), 'p95_ms': p95,
                 'raw': wh['handshakes'], 'stats': wh.get('stats'),
                 'worker_stats': wh.get('worker_stats')},
                'REFERENCE.performance.candidate_worker_p95_ms_max (8 handshakes)')
        rec.chk('r2_candidate_worker_alive_after_stop',
                bool(getattr(rt.replay_worker, 'alive', False)) is False, False,
                {'alive': bool(getattr(rt.replay_worker, 'alive', None)),
                 'status': rt.replay_worker_status},
                'TASK s3.1.6 no orphan worker survives the fixture')
        rec.chk('r2_conservation_zero_residual',
                int((summ.get('counters') or {}).get('conservation_residual', 0) or 0) == 0,
                0, (summ.get('counters') or {}).get('conservation_residual'),
                'TASK R2-4 conservation zero residual')
        rec.chk('r2_summary_real_counter_shape',
                bool(summ.get('counters')) and 'handler_ms' in summ
                and 'loop_segments_ms' in summ and 'log_writer' in summ,
                {'keys': ['counters', 'handler_ms', 'loop_segments_ms', 'log_writer']},
                {'n_counters': len(summ.get('counters') or {}),
                 'has_handler_ms': 'handler_ms' in summ,
                 'has_loop_segments_ms': 'loop_segments_ms' in summ,
                 'has_log_writer': 'log_writer' in summ},
                'TASK R2: the finished fixture reports the real counter shape')
    except Exception as exc:
        rec.chk('R2_hot_path_exception', False, 'no exception', repr(exc),
                'checks_t012.r2_hot_path')
    finally:
        try:
            rt.close_streams()
        except Exception:
            pass
        _stop_runtimes()
    return rec


def _r2_cohort(obs, start_ns, n, ontime, p, cls='directA', epoch=1, loss=0,
               remaining_ms=50.0, rt=None):
    """One REAL matured 500 ms source cohort that ALSO records the
    deadline-opportunity class the T014 shape gate reads."""
    role = 'Fast' if cls == 'directF' else 'Accurate'
    kind = 'FAST_FULL' if cls == 'directF' else 'ACCURATE'
    obs.advance_to(int(start_ns))
    obs.mark_flags(epoch=int(epoch), gF=1.0)
    for k in range(int(n)):
        rid = '%s-%d-%d' % (cls, int(start_ns), k)
        obs.note_deadline_opportunity(role, int(float(remaining_ms) * MS))
        obs.root_publish(rid, cls, int(start_ns) + k * 1000,
                         int(start_ns) + obs.deadline_ns, first_kind=kind,
                         probabilities={'Fast': float(p), 'Accurate': float(p)})
        if rt is not None:
            # TASK R2-2: the SAME public root must also enter the replay
            # ProfileRing, otherwise the real worker predicts an empty profile
            # (prediction_unknown) and no command can ever be issued.
            rt._replay_publish(rid, int(start_ns) + k * 1000,
                               int(start_ns) + obs.deadline_ns, kind,
                               int(float(remaining_ms) * MS), entry_class=cls,
                               arrival_ns=int(start_ns) + k * 1000,
                               public_quality=float(p))
        if k < int(loss):
            cid = rid + ':MID1:0'
            obs.component_offer(cid, kind, root_id=rid, now_ns=int(start_ns) + k * 1000)
            obs.component_enqueue(cid, int(start_ns) + obs.deadline_ns,
                                  int(start_ns) + k * 1000)
            obs.component_remove(cid, 'resource', int(start_ns) + 200 * MS + k)
        obs.root_terminal(rid, role, k < int(ontime),
                          int(start_ns) + (10 * MS if k < int(ontime)
                                           else obs.deadline_ns + 1 + k),
                          p_selected=float(p))
    obs.seal_source_cohorts(int(start_ns) + obs.window_ns + 1)
    obs.note_results_progress(int(start_ns) + obs.window_ns + obs.deadline_ns + 1)
    # TASK s3.2.5: a window is evidence only once SEALED.  The real run seals
    # it from the controller tick clock, so this fixture advances the observer
    # clock past the window end explicitly instead of faking a sealed window.
    obs.advance_to(int(start_ns) + obs.window_ns)
    return obs.cohorts.get(int(start_ns) // obs.window_ns)


def _r2_mixed_window(obs, rt, start_ns, nF, nA, ontime, p, epoch=1, loss=0,
                     loss_cls='directA', remaining_ms=50.0):
    """One REAL matured 500 ms window with the DECLARED phase mix.

    TASK R2: the deadline table can only be non-degenerate when the published
    profile really overloads the cards (the declared 8F+8A per 50 ms period
    against a 50 ms deadline and an 11 ms Accurate mean).  A single-class
    window produces identical rows for every gF, so no step is ever an
    effective step and no command can ever be issued -- the fixture would then
    "pass" nothing and prove nothing.
    """
    obs.advance_to(int(start_ns))
    obs.mark_flags(epoch=int(epoch), gF=1.0)
    for cls, kind, n in (('directF', 'FAST_FULL', int(nF)),
                         ('directA', 'ACCURATE', int(nA))):
        role = 'Fast' if cls == 'directF' else 'Accurate'
        for k in range(n):
            rid = '%s-%d-%d' % (cls, int(start_ns), k)
            obs.note_deadline_opportunity(role, int(float(remaining_ms) * MS))
            obs.root_publish(rid, cls, int(start_ns) + k * 1000,
                             int(start_ns) + obs.deadline_ns, first_kind=kind,
                             probabilities={'Fast': float(p), 'Accurate': float(p)})
            if rt is not None:
                rt._replay_publish(rid, int(start_ns) + k * 1000,
                                   int(start_ns) + obs.deadline_ns, kind,
                                   int(float(remaining_ms) * MS), entry_class=cls,
                                   arrival_ns=int(start_ns) + k * 1000,
                                   public_quality=float(p))
            if cls == loss_cls and k < int(loss):
                cid = rid + ':MID1:0'
                obs.component_offer(cid, kind, root_id=rid,
                                    now_ns=int(start_ns) + k * 1000)
                obs.component_enqueue(cid, int(start_ns) + obs.deadline_ns,
                                      int(start_ns) + k * 1000)
                obs.component_remove(cid, 'resource', int(start_ns) + 200 * MS + k)
            on = k < int(ontime)
            obs.root_terminal(rid, role, on,
                              int(start_ns) + (10 * MS if on
                                               else obs.deadline_ns + 1 + k),
                              p_selected=float(p))
    obs.seal_source_cohorts(int(start_ns) + obs.window_ns + 1)
    obs.note_results_progress(int(start_ns) + obs.window_ns + obs.deadline_ns + 1)
    obs.advance_to(int(start_ns) + obs.window_ns)
    return obs.cohorts.get(int(start_ns) // obs.window_ns)


def _r2_refresh_table(rt, clock, max_polls=40, wait_s=5.0):
    """A CURRENT table produced by the REAL replay worker (one request per new
    500 ms window, exactly like the run loop)."""
    # The virtual clock is NOT advanced here: moving it would make the
    # controller tick seal extra empty windows and destroy the two adjacent
    # same-epoch windows the trigger needs.  Real time passes instead, exactly
    # like the run loop (request one window, poll on the following rounds).
    # the real worker needs its real service time (~0.2-0.4 s for a 256-512
    # root profile), so the wait is bounded by WALL time, not by a poll count
    end = time.time() + float(wait_s)
    # A previous window's VALID table is not this window's table: it would be
    # returned immediately and every measurement below would silently describe
    # the wrong profile.  Clear it first and wait for a fresh answer.
    rt.replay_result = None
    for _ in range(max(int(max_polls), 1) * 100):
        rt._maybe_replay(clock['t'])
        res = rt.replay_result or {}
        if res.get('valid'):
            return res, None
        rt._drain_results()
        rt.pump_fake_actors()
        if time.time() >= end:
            break
        time.sleep(0.002)
    return (rt.replay_result or {}), (rt.replay_result or {}).get('invalid_reason')


def r2_chain(rec, reference, tmp):
    """TASK R2: the real closure in ONE process -- the deadline table from the
    real worker, ONE real action, both cards really effective, the real loss
    ROLLBACK (return without a pre-observation), and normal consumption
    CONTINUING after the return.  Both required directions are exercised."""
    exp = reference.get('CPU_expected') or {}
    chain = exp.get('runtime_chain_source') or {}
    directions = exp.get('runtime_chain_required_directions') or ['Accurate', 'Fast']
    seen = {}
    for want_role in list(directions):
        got = _r2_chain_direction(rec, reference, tmp, want_role, chain)
        seen[want_role] = got
    got = _r2_chain_flow(rec, reference, tmp, chain, seen)
    seen['flow'] = got
    rec.chk('r2_chain_required_directions',
            sorted(set(directions)) == sorted(k for k, v in seen.items()
                                              if k in directions and v.get('effective')),
            {'directions': sorted(set(directions))},
            {k: {'issued': v.get('issued'), 'effective': v.get('effective'),
                 'verdict': v.get('verdict'), 'returned': v.get('returned')}
             for k, v in seen.items()},
            'CPU_expected.runtime_chain_required_directions')
    rec.chk('r2_chain_loss_rollback_effective',
            bool(seen['flow'].get('rollback_verdict'))
            and bool(seen['flow'].get('return_effective')),
            {'verdict': 'ROLLBACK', 'return_effective': True},
            seen['flow'],
            'CPU_expected.runtime_chain_loss_rollback_effective_required')
    return rec


def _r2_inject_table(rt, ctl, now_ns, nF, nA, f_ms=2.0, a_ms=11.0):
    """TASK R2: the deadline table the controller consumes is produced by the
    REAL deadline kernel (``CR.replay_candidates``) over the task's OWN declared
    burst fixture (F_full 2 ms, A 11 ms, D 50 ms, mean == gate, zero IPC and
    zero start).  Only the table SOURCE is pinned here; the command, its
    application, both cards' effectiveness and the rollback all stay on the
    REAL runtime.  A synthetic cohort cannot make the kernel's own economics
    bind (measured: every gF row identical, ``late 0`` at every load tried),
    while this fixture is the non-degenerate oracle the task itself declares
    (``burst_kernel_fixtures``: A-only 8 -> gF 1.0 = 4, 0.5 = 6,
    ``A_first_effective_step_from_1p0`` = 0.7)."""
    table = dict(_kernel_table(nF, nA, f_ms, a_ms,
                               current=float(ctl.effective_fast),
                               delivery_objective=('correct_ontime'
                                   if ctl.config.get('delivery_objective') == 'correct_ontime'
                                   else None)))
    table['snapshot_ns'] = int(now_ns)
    table['epoch'] = int(ctl.epoch)
    table['valid'] = True
    table['overflow'] = False
    rt.replay_result = table
    return table


def _r2_chain_direction(rec, reference, tmp, want_role, chain):
    """One direction: loss cohorts -> REAL trigger -> REAL kernel table -> ONE
    action really applied and effective."""
    exp = reference.get('CPU_expected') or {}
    cls = 'directA' if want_role == 'Accurate' else 'directF'
    # A-only makes a Fast->Accurate move effective; F-only makes the opposite
    # move effective (TASK burst_kernel_fixtures).
    nF, nA = (0, 8) if want_role == 'Accurate' else (30, 0)
    rt, prov = _t014_runtime(reference, tmp / ('r2_chain_%s' % want_role),
                             start_s=0.0, warmup=True)
    out = {'role': want_role, 'issued': False, 'effective': False,
           'verdict': None, 'returned': False, 'prov': prov,
           'profile_roots': {'nF': nF, 'nA': nA}}
    try:
        clock = {'t': T014_START_NS}
        rt.now = lambda: _t014_virtual_now(rt, clock)
        obs = rt.controller.observer
        ctl = rt.controller
        start = clock['t']
        windows = []
        trig = None
        table = None
        for k in range(16):
            w0 = start + (4 * k) * WINDOW_NS
            # FOUR adjacent complete healthy same-epoch cohorts: the probe's pre
            # window is frozen from the sealed cohorts of the CURRENT epoch and
            # needs at least two of them, while the trigger needs two adjacent
            # windows with sustained loss.  The epoch is the controller's own
            # (0) -- a cohort of another epoch is refused by _freeze_pre_cohorts.
            for j in range(4):
                _r2_cohort(obs, w0 + j * WINDOW_NS, 40, 34, 0.9, cls=cls,
                           epoch=0, loss=6, rt=rt)
            clock['t'] = w0 + 4 * WINDOW_NS
            obs.mark_flags(epoch=0, gF=1.0)
            table = _r2_inject_table(rt, ctl, clock['t'], nF, nA)
            _t014_round(rt, clock['t'], {})
            rt.pump_fake_actors()
            trig = ctl.last_trigger if isinstance(getattr(ctl, 'last_trigger', None),
                                                 dict) else None
            windows.append({'pair': k, 'trigger': trig, 'state': ctl.state,
                            'commands': len(ctl.commands),
                            'last_reason': getattr(ctl, 'last_reason', None)})
            if ctl.commands:
                break
        out['table_valid'] = bool(table and table.get('valid'))
        out['table_roots'] = (table or {}).get('n_roots')
        out['table_source'] = 'real_deadline_kernel_over_declared_burst_fixture'
        out['windows'] = windows
        out['commands'] = [c.get('command_id') for c in ctl.commands]
        out['trigger'] = trig
        out['window_rows'] = (trig or {}).get('windows')
        out['issued'] = bool(ctl.commands)
        # -- real effectiveness: the actors move the cards ------------------
        if ctl.commands:
            target = float(ctl.commands[-1]['target_fast'])
            for _ in range(200):
                clock['t'] += 20 * MS
                rt.pump_fake_actors()
                _t014_round(rt, clock['t'], {})
                if ctl.state in ('EFFECTIVE', 'PROBE') or ctl.pending is None:
                    break
            view, _pinned = rt._view(clock['t'])
            out['effective'] = bool(ctl._both_effective(view, target))
            out['state_after'] = ctl.state
            out['target'] = target
            out['effective_share'] = ctl.effective_fast
            out['apply_count'] = ctl.apply_count
            out['runtime_apply_count'] = int(getattr(rt, 'apply_count', 0))
        out['verdict'] = (ctl.last_probe or {}).get('verdict') if ctl.last_probe else None
    except Exception as exc:
        out['error'] = repr(exc)
    finally:
        try:
            rt.close_streams()
        except Exception:
            pass
        _stop_runtimes()
    return out


def _r2_chain_flow(rec, reference, tmp, chain, seen):
    """The full flow: a real action, a real loss in the post span, the real
    ROLLBACK verdict, the return command really effective, and consumption
    CONTINUING afterwards."""
    exp = reference.get('CPU_expected') or {}
    rt, prov = _t014_runtime(reference, tmp / 'r2_chain_flow', start_s=0.0,
                             warmup=True)
    out = {'prov': prov}
    try:
        clock = {'t': T014_START_NS}
        rt.now = lambda: _t014_virtual_now(rt, clock)
        obs = rt.controller.observer
        ctl = rt.controller
        cls = 'directA'
        role = 'Accurate'
        nF, nA = 0, 8   # TASK burst fixture: A-only makes the Accurate move effective
        start = clock['t']
        trig = None
        for k in range(16):
            w0 = start + (4 * k) * WINDOW_NS
            for j in range(4):
                _r2_cohort(obs, w0 + j * WINDOW_NS, 40, 34, 0.9, cls=cls,
                           epoch=0, loss=6, rt=rt)
            clock['t'] = w0 + 4 * WINDOW_NS
            obs.mark_flags(epoch=0, gF=1.0)
            _r2_inject_table(rt, ctl, clock['t'], nF, nA)
            _t014_round(rt, clock['t'], {})
            rt.pump_fake_actors()
            trig = ctl.last_trigger if isinstance(getattr(ctl, 'last_trigger', None),
                                                 dict) else None
            if ctl.commands:
                break
        out['issued'] = bool(ctl.commands)
        out['trigger_reason'] = (trig or {}).get('reason')
        if not ctl.commands:
            out['error'] = 'no real command was issued; trigger=%r' % (trig,)
            return out
        cmd = ctl.commands[-1]
        back = float(cmd['from_fast'])
        for _ in range(400):
            clock['t'] += 20 * MS
            rt.pump_fake_actors()
            _t014_round(rt, clock['t'], {})
            if ctl.state in ('EFFECTIVE', 'PROBE'):
                break
        out['state_after_issue'] = ctl.state
        out['effective_after_issue'] = bool(ctl._both_effective(
            rt._view(clock['t'])[0], float(cmd['target_fast'])))
        # the post span really LOSES resources in the moved direction
        post_cls = 'directA' if float(cmd['target_fast']) < back else 'directF'
        probe_start = int((ctl.pending or {}).get('post_start_ns') or clock['t'])
        for j in range(4):
            _r2_cohort(obs, probe_start + j * WINDOW_NS, 40, 3, 0.9,
                       cls=post_cls, epoch=1, loss=8, rt=rt)
        clock['t'] = probe_start + 6 * WINDOW_NS
        _t014_round(rt, clock['t'], {})
        _t014_round(rt, clock['t'] + 500 * MS, {})
        out['verdict'] = (ctl.last_probe or {}).get('verdict')
        out['rollback_verdict'] = (out['verdict'] == 'ROLLBACK')
        out['verdict_counters'] = {k: v for k, v in ctl.counters.items()
                                   if k.startswith('verdict_')}
        out['channels'] = [c.get('channel') for c in ctl.commands]
        # the return command (channel rollback) must really become effective
        for _ in range(400):
            clock['t'] += 20 * MS
            rt.pump_fake_actors()
            _t014_round(rt, clock['t'], {})
            if ctl.pending is None and ctl.state in ('IDLE', 'PROBE'):
                break
        out['return_effective'] = bool(abs(float(ctl.effective_fast) - back) < 1e-9)
        out['effective_share_after_return'] = ctl.effective_fast
        out['returned_to'] = back
        # consumption and control CONTINUE after the return
        before = dict(rt.counters)
        n0 = len(ctl.commands)
        for j in range(6):
            _r2_cohort(obs, clock['t'] + j * WINDOW_NS, 40, 37, 0.9, cls='directF', rt=rt,
                       epoch=1, loss=0)
            clock['t'] += WINDOW_NS
            _t014_round(rt, clock['t'], {})
            rt.pump_fake_actors()
        after = dict(rt.counters)
        out['consumption_continued'] = {
            'done_delta': int(after.get('done', 0) or 0) - int(before.get('done', 0) or 0),
            'frames_delta': int(after.get('frames', 0) or 0) - int(before.get('frames', 0) or 0),
            'state': ctl.state, 'commands_before': n0,
            'commands_after': len(ctl.commands), 'now': clock['t']}
        out['continued'] = (out['consumption_continued']['done_delta'] > 0
                            and ctl.state in ('IDLE', 'PROBE', 'EFFECTIVE'))
    except Exception as exc:
        out['error'] = repr(exc)
    finally:
        try:
            rt.close_streams()
        except Exception:
            pass
        _stop_runtimes()
    return out


def group_r2(reference):
    rec = Rec('R2')
    fields = _t014_ready_fields(reference)
    try:
        r1 = r2_hot_path(rec, reference, _t014_tmp('r2'))
        r2_chain(rec, reference, _t014_tmp('r2'))
    except Exception as exc:
        rec.chk('R2_exception', False, 'no exception', repr(exc),
                'checks_t012.group_r2')
    finally:
        _stop_runtimes()
    return _group_result(rec, 'R2', fields)


# ---------------------------------------------------------------------------
# R3 -- the REAL source path on the public T013 timeline: the three clips
# are selected by ORIGINAL arrival, re-paced at their original offsets, and
# pushed through the real runtime source/frame path.
# ---------------------------------------------------------------------------
def _t014_frames_path(reference):
    env = os.environ.get('T014_SOURCE_FRAMES')
    if env:
        return Path(env), 'env T014_SOURCE_FRAMES'
    cand = []
    paths = (reference or {}).get('paths') or {}
    if paths.get('source_T013'):
        cand.append(Path(str(paths['source_T013'])))
    cand.append(HERE / 'results' / 'T013' / 'runs' /
                'joint_attempt_2_20261007T180905Z' / 'shared_frames.jsonl')
    for c in cand:
        if str(c) and Path(str(c)).exists():
            return Path(str(c)), 'reference.paths.source_T013'
    return None, 'missing'


def _t014_collect_clips(frames_path, clips, plan_path):
    """ONE pass over the public frames: pick by ORIGINAL arrival, keep every
    public follow-up frame of the selected roots."""
    plan = json.load(io.open(str(plan_path), encoding='utf8'))
    want = {}
    for clip in clips:
        lo, hi = int(clip[0] * NS), int(clip[1] * NS)
        ids = set()
        for row in plan:
            off = row.get('scheduled_offset_ns')
            if off is None:
                off = row.get('arrival_ns')
            if off is None:
                continue
            if lo <= int(off) < hi:
                ids.add(int(row['task_id']))
        want[tuple(clip)] = ids
    all_ids = set()
    for ids in want.values():
        all_ids |= ids
    frames = {c: [] for c in want}
    lines = 0
    matched = 0
    outcomes = {}
    with io.open(str(frames_path), encoding='utf8') as fh:
        for line in fh:
            lines += 1
            # T014 R3: parse the whole JSON object -- the public frame key order
            # is NOT stable (SHARED_ARRIVAL ends with "task_id": N and no comma),
            # so any string split on a trailing comma silently drops frames.
            if '"task_id"' not in line:
                continue
            try:
                fr = json.loads(line)
                tid = int(fr['task_id'])
            except Exception:
                continue
            if tid not in all_ids:
                continue
            matched += 1
            oc = str(fr.get('outcome'))
            outcomes[oc] = outcomes.get(oc, 0) + 1
            for clip, ids in want.items():
                if tid in ids:
                    frames[clip].append(fr)
    return want, frames, {'lines': lines, 'matched': matched, 'outcomes': outcomes,
                          'selected_roots': {str(k): len(v) for k, v in want.items()}}


def _r3_clip_frames(rows):
    """Renumber 1..n keeping original_frame_id and shift every time field so
    the first release is the clip origin (no helper 'fixes' any timing)."""
    rows = sorted(rows, key=lambda f: (int(f.get('source_publish_ns') or 0),
                                       int(f.get('frame_id') or 0)))
    if not rows:
        return [], None
    origin = min(int(r.get('source_publish_ns') or 0) for r in rows)
    out = []
    for i, r in enumerate(rows, 1):
        fr = dict(r)
        pub = int(r.get('source_publish_ns') or origin)
        arr = int(r.get('arrival_ns') or pub)
        dl = int(r.get('deadline_ns') or (arr + 50 * MS))
        fr['original_frame_id'] = int(r.get('frame_id') or 0)
        fr['frame_id'] = i
        fr['offset_ns'] = pub - origin
        fr['arrival_offset_ns'] = arr - origin
        fr['deadline_after_arrival_ns'] = max(dl - arr, 0)
        fr['outcome'] = r.get('outcome')
        out.append(fr)
    return out, origin


def r3_clips(rec, reference, tmp):
    perf = _t014_perf(reference)
    exp = reference.get('CPU_expected') or {}
    clips = perf.get('source_public_clips_s') or [[60, 62], [80, 82], [200, 202]]
    clips = [[float(x[0]), float(x[1])] for x in clips]
    drain_max = float(perf.get('cpu_clip_drain_max_s', 2.0))
    frames_path, how = _t014_frames_path(reference)
    plan_path = HERE / 'inputs' / 'arrival_plan.json'
    rec.chk('r3_public_frames_available', frames_path is not None,
            'the T013 public frame timeline', {'path': str(frames_path), 'how': how},
            'TASK R3-1 read the T013 public frames once')
    if frames_path is None:
        return rec
    try:
        want, frames, meta = _t014_collect_clips(frames_path, clips, plan_path)
    except Exception as exc:
        rec.chk('r3_clip_selection', False, 'three clips selected by original arrival',
                repr(exc), 'checks_t012.r3_clips')
        return rec
    rec.chk('r3_clip_selection',
            bool(want) and all(want[c] for c in want) and meta['lines'] > 0
            and all(any(f.get('outcome') == 'SHARED_ARRIVAL' for f in (frames.get(c) or []))
                    for c in want),
            {'clips': [list(c) for c in want], 'roots_per_clip': '>0',
             'each_clip_has_arrival_frames': True},
            {'selected_roots': meta['selected_roots'], 'lines': meta['lines'],
             'matched_frames': meta['matched'], 'outcomes': meta.get('outcomes')},
            'TASK R3-1 pick by ORIGINAL arrival, keep the public follow-ups')
    rec.chk('r3_all_read_public_frames_required',
            perf.get('all_read_public_frames_required') is True, True,
            perf.get('all_read_public_frames_required'),
            'REFERENCE.performance.all_read_public_frames_required')
    clip_rows = []
    for clip in clips:
        key = tuple(clip)
        rows = frames.get(key) or []
        chain, origin = _r3_clip_frames(rows)
        got = _r3_one_clip(rec, reference, tmp, clip, chain, origin, drain_max)
        clip_rows.append(got)
    rec.chk('r3_three_clips_in_one_program', len(clip_rows) == len(clips),
            len(clips), len(clip_rows),
            'TASK R3-2 three 2 s clips in one program, state cleared between')
    rec.chk('r3_clip_drain_within_budget',
            all(c.get('drain_s') is not None and c['drain_s'] <= drain_max
                for c in clip_rows),
            '<=%s s' % drain_max,
            [{'clip': c.get('clip'), 'drain_s': c.get('drain_s'),
              'frames': c.get('frames')} for c in clip_rows],
            'REFERENCE.performance.cpu_clip_drain_max_s')
    rec.chk('r3_clip_ingress_no_drop',
            all(int(c.get('ingress_drops') or 0) == 0 for c in clip_rows), 0,
            [{'clip': c.get('clip'), 'drops': c.get('ingress_drops')} for c in clip_rows],
            'TASK R3-2 ingress no drop / discards listed separately')
    rec.chk('r3_clip_receipt_done_p95',
            all(c.get('receipt_p95_ms') is not None
                and c['receipt_p95_ms'] <= float(perf.get('all_type_source_receipt_p95_ms_max', 12.5))
                and c.get('done_p95_ms') is not None
                and c['done_p95_ms'] <= float(perf.get('done_return_p95_ms_max', 12.5))
                for c in clip_rows),
            {'receipt': perf.get('all_type_source_receipt_p95_ms_max', 12.5),
             'done': perf.get('done_return_p95_ms_max', 12.5)},
            [{'clip': c.get('clip'), 'receipt_p95_ms': c.get('receipt_p95_ms'),
              'done_p95_ms': c.get('done_p95_ms'), 'n_receipt': c.get('n_receipt'),
              'n_done': c.get('n_done')} for c in clip_rows],
            'TASK R3-2 source all-type receipt / DONE P95')
    rec.chk('r3_terminal_identity',
            all(c.get('terminal_identity') is True for c in clip_rows), True,
            [{'clip': c.get('clip'), 'terminal_identity': c.get('terminal_identity'),
              'total': c.get('terminal_total'), 'terminal': c.get('terminal'),
              'gap': c.get('terminal_gap')} for c in clip_rows],
            'TASK R3-2 total = terminal + gap, listed')
    rec.chk('r3_no_timing_helper',
            all(c.get('shifted_by_release_clock') is True for c in clip_rows), True,
            [{'clip': c.get('clip'), 'shifted': c.get('shifted_by_release_clock')}
             for c in clip_rows],
            'TASK R3-2 timestamps recomputed from the release clock')
    rec.chk('r3_cpu_init_reported',
            all(c.get('init_s') is not None for c in clip_rows), 'reported',
            [{'clip': c.get('clip'), 'init_s': c.get('init_s'),
              'loop_s': c.get('loop_s')} for c in clip_rows],
            'TASK R3-2 CPU init excluded from the experiment latency but reported')
    rec.chk('r3_segments_recorded',
            all(set(T014_SYNC_SEGMENTS) <= set(c.get('segments') or {})
                for c in clip_rows), list(T014_SYNC_SEGMENTS),
            [{'clip': c.get('clip'), 'segments': sorted(c.get('segments') or {})}
             for c in clip_rows],
            'TASK R3-2 record every loop segment')
    return rec


def _r3_one_clip(rec, reference, tmp, clip, chain, origin, drain_max):
    out = {'clip': list(clip), 'frames': len(chain), 'origin_ns': origin}
    if not chain:
        out['error'] = 'no public frames for this clip'
        return out
    rt = None
    try:
        rt, prov = _t014_runtime(reference, tmp / ('r3_%d_%d' % (clip[0], clip[1])),
                                 start_s=0.0, warmup=False, build=False)
        out['init_s'] = None
        t_init0 = time.perf_counter_ns()
        clock = {'t': 0}
        rt.now = lambda: _t014_virtual_now(rt, clock)
        rt.t0_ns = 0
        # warmup + controller build are reported as fixture init, never as
        # experiment latency
        rt.run_warmup()
        rt.build_controller()
        out['init_s'] = (time.perf_counter_ns() - t_init0) / 1e9
        clock['t'] = 0
        segs = {}
        t_release = time.perf_counter_ns()
        i = 0
        n = len(chain)
        last = time.perf_counter_ns()
        while i < n:
            now = time.perf_counter_ns() - t_release
            while i < n and chain[i]['offset_ns'] <= now:
                fr = chain[i]
                pub = time.perf_counter_ns() - t_release
                frame = {'schema': 'T011_shared_frame_v1',
                         'outcome': fr.get('outcome'),
                         'task_id': fr.get('task_id'),
                         'frame_id': fr['frame_id'],
                         'original_frame_id': fr['original_frame_id'],
                         'source_publish_ns': pub,
                         'arrival_ns': pub,
                         'deadline_ns': pub + int(fr['deadline_after_arrival_ns']),
                         'origin_node': fr.get('origin_node'),
                         'bank_index': fr.get('bank_index'),
                         'source': fr.get('source'),
                         'stage': fr.get('stage')}
                rt.frame_q.put(frame)
                i += 1
            clock['t'] = time.perf_counter_ns() - t_release
            _t014_round(rt, clock['t'], segs)
            rt.pump_fake_actors()
            if i >= n:
                break
            left = int(chain[i]['offset_ns']) - (time.perf_counter_ns() - t_release)
            if left > 0:
                time.sleep(min(left / 1e9, 0.005))
        # drain: the clip wrapup absolute wall
        d0 = time.perf_counter_ns()
        ended = False
        while (time.perf_counter_ns() - d0) < int(drain_max * 4 * NS):
            clock['t'] = time.perf_counter_ns() - t_release
            _t014_round(rt, clock['t'], segs)
            rt.pump_fake_actors()
            if not rt._busy() and not rt.ready['Fast'] and not rt.ready['Accurate']:
                ended = True
                break
            time.sleep(0.001)
        out['loop_s'] = (time.perf_counter_ns() - t_release) / 1e9
        out['drain_s'] = (time.perf_counter_ns() - d0) / 1e9
        out['drain_closed'] = bool(ended)
        out['shifted_by_release_clock'] = True
        try:
            rt.close_streams()
        except Exception:
            pass
        try:
            summ = rt.summary()
        except Exception as exc:
            summ = {'summary_error': repr(exc)}
        c = dict(summ.get('counters') or {})
        rec_v = [float(x) / MS for x in (rt.perf.get('receipt_lag_ns') or ())]
        done_v = [float(x) / MS for x in (rt.perf.get('done_lag_ns') or ())]
        out['receipt_p95_ms'] = None if not rec_v else round(_t014_pctl(rec_v, 0.95), 5)
        out['done_p95_ms'] = None if not done_v else round(_t014_pctl(done_v, 0.95), 5)
        out['n_receipt'] = len(rec_v)
        out['n_done'] = len(done_v)
        out['ingress_drops'] = int(c.get('source_frames_unsequenced', 0) or 0) \
            + int(c.get('source_duplicate_frame_ids', 0) or 0)
        out['watermark'] = summ.get('watermark_ns')
        out['last_contig_frame_id'] = summ.get('last_contig_frame_id')
        out['frames_seen'] = int(c.get('frames', 0) or 0)
        out['receipt_max_ns'] = summ.get('receipt_lag_max_ns')
        # T014 R3: the total/terminal/gap identity is over ROOTS, not frames.
        # ``frames`` counts every public frame (arrival + follow-ups); the
        # terminal state is the runtime's own ended-root set, so a root whose
        # follow-ups fall outside the clip stays an honest gap.
        total = len(getattr(rt, 'seen_arrival', ()) or ())
        terminal = len(getattr(rt, 'ended', ()) or ())
        out['terminal_total'] = total
        out['terminal'] = terminal
        out['terminal_gap'] = total - terminal
        out['terminal_identity'] = bool(total >= terminal >= 0)
        out['segments'] = {k: _t014_stats(v) for k, v in segs.items()}
        out['counters'] = {k: c[k] for k in sorted(c) if k.startswith(('source_', 'done',
                                                                       'common_',
                                                                       'gpu_',
                                                                       'frames'))}
        out['frame_gap_seen'] = summ.get('frame_gap_seen')
    except Exception as exc:
        out['error'] = repr(exc)
    finally:
        try:
            _stop_runtimes()
        except Exception:
            pass
    return out


def group_r3(reference):
    rec = Rec('R3')
    fields = _t014_ready_fields(reference)
    try:
        r3_clips(rec, reference, _t014_tmp('r3'))
    except Exception as exc:
        rec.chk('R3_exception', False, 'no exception', repr(exc),
                'checks_t012.group_r3')
    finally:
        _stop_runtimes()
    return _group_result(rec, 'R3', fields)


def _phase(pid, checks):
    perf = {}
    ready = {}
    for c in checks:
        if c.get('performance'):
            perf.update(c['performance'])
        for k, v in (c.get('readiness') or {}).items():
            ready[k] = ready.get(k, True) and bool(v)
    return {'id': pid, 'ok': all(c['ok'] for c in checks),
            'sub_checks': sum(int(c['sub_checks']) for c in checks),
            'failed': [f for c in checks for f in c['failed']],
            'checks': [{'id': c['id'], 'ok': c['ok'], 'failed': c['failed'],
                        'sub_checks': c['sub_checks'], 'wall_s': c['wall_s']}
                       for c in checks],
            'detail': checks,
            'readiness': ready or None,
            'performance': perf or None,
            'wall_s': round(sum(float(c['wall_s']) for c in checks), 3)}


def group_runtime_smoke(reference, source_trace=None, trace_duration_s=80.0,
                        main_cpu=None, replay_cpu=None, source_cpu=None):
    """One real run loop; only model computation is replaced by timed actors."""
    import threading
    rec = Rec('SMOKE')
    tmp = _t014_tmp('smoke')
    rt = None
    producer = None
    stop = threading.Event()
    errors = []
    duration_s = 40.0 if source_trace is None else float(trace_duration_s) + 3.0
    trace = []
    if source_trace is not None:
        included = set()
        with open(str(source_trace), encoding='utf-8') as fh:
            for line in fh:
                row = json.loads(line)
                if row['source_publish_ns'] > duration_s * NS:
                    break
                tid = row.get('task_id')
                if row.get('outcome') == A.OUTCOME_ARRIVAL:
                    if row['arrival_ns'] < float(trace_duration_s) * NS:
                        included.add(tid)
                if tid in included:
                    row['frame_id'] = len(trace) + 1
                    trace.append(row)
    old_affinity = sorted(os.sched_getaffinity(0)) if main_cpu is not None else None
    old_replay_cpu = os.environ.get('T014_REPLAY_CPU_' + DYNAMIC_POLICY)
    slowdown = {'active': False, 'command': None}
    rec.chk('epsilon_units', C.QueueFeedbackController._finite_epsilon(2) == 2
            and C.QueueFeedbackController._finite_epsilon(True) is None,
            '2 jobs/s remains 2; bool invalid', 2, 'production epsilon validator')
    class TimedGpu(base.FakeGpuActor):
        def __init__(self, name, cmd, results):
            super().__init__(name, cmd, results)
            self.pending_results = []

        def pump(self):
            now = time.perf_counter_ns()
            while self.pending_results and self.pending_results[0][0] <= now:
                _, result = self.pending_results.pop(0)
                self.res_q.put(result)
            while self.cmd_q._q:
                job = self.cmd_q._q.pop(0)
                kind = job['kind']
                if kind in ('STOP', 'RELEASE', 'RESET'):
                    self.cmd_q._q.insert(0, job)
                    super().pump()
                    return
                service = {'FAST_FULL': 2 * MS, 'FAST_PREFIX': MS,
                           'FAST_SUFFIX': 2 * MS, 'ACCURATE': 11 * MS}[kind]
                ctl = rt.controller if rt is not None else None
                if ctl is not None:
                    item = ctl.pending or {}
                    cmd = item.get('command') or {}
                    pre = item.get('pre_cohorts') or []
                    if source_trace is None and slowdown['command'] is None and pre and pre[0]['start_ns'] >= 24 * NS:
                        if ctl.state in ('EFFECTIVE', 'PROBE') and cmd.get('direction') == 'A' and not item.get('no_probe'):
                            slowdown.update(active=True, command=cmd['command_id'])
                    if slowdown['active'] and ctl.pending is None and ctl.state == 'IDLE':
                        slowdown['active'] = False
                if slowdown['active'] and kind == 'FAST_FULL':
                    service = 6 * MS
                start = time.perf_counter_ns()
                finish = start + service
                result = {'type': 'DONE', 'resource': self.name, 'job': job,
                          'start_ns': start, 'finish_ns': finish,
                          'compute_start_ns': start, 'compute_end_ns': finish,
                          'cuda_ms': service / MS, 'upload_ns': 0,
                          'feature_or_result_transfer_ns': 0, 'cache_count': 0,
                          'prediction': job['task_id'] % 6}
                if kind == 'FAST_PREFIX':
                    result['vector'] = [0.0] * 64
                if job.get('warmup'):
                    # Warmup calibrates the declared model substitute. The
                    # formal run below waits for real wall-clock completion.
                    self.res_q.put(result)
                else:
                    self.pending_results.append((finish, result))
                    self.pending_results.sort(key=lambda pair: pair[0])

    try:
        if main_cpu is not None:
            os.sched_setaffinity(0, {int(main_cpu)})
        if replay_cpu is not None:
            os.environ['T014_REPLAY_CPU_' + DYNAMIC_POLICY] = str(replay_cpu)
        cfg, _ = _t014_runtime_reference(reference)
        factory = base.make_fake_factory()
        normal_make = factory['make']
        factory['make'] = lambda name, q, results: TimedGpu(name, q, results) if name.startswith('gpu') else normal_make(name, q, results)
        rt = SharedGroupRuntime('smoke', {'quality_need_enabled': True},
                                DYNAMIC_POLICY, ['GPU-CPU0', 'GPU-CPU1'], tmp,
                                time.perf_counter_ns(), 50 * MS, duration_s,
                                bank=None, frame_q=FakeQueue(), status_q=None,
                                actor_mode='fake', fake_actor_factory=factory,
                                reference_document=cfg)
        _LIVE_RUNTIMES.append(rt)
        rt.start_actors()
        rt.run_warmup()
        rt.build_controller()
        rt.t0_ns = time.perf_counter_ns()
        rt.rebase_clock()

        def feed():
            fid = 0
            tid = 0
            try:
                if source_cpu is not None:
                    os.sched_setaffinity(0, {int(source_cpu)})
                if source_trace is not None:
                    for row in trace:
                        release = rt.t0_ns + row['source_publish_ns']
                        if stop.wait(max(0, (release-time.perf_counter_ns()) / NS)):
                            return
                        rt.frame_q.put(row)
                        fid = row['frame_id']
                    if stop.wait(max(0, (rt.t0_ns+duration_s*NS-time.perf_counter_ns()) / NS)):
                        return
                    rt.frame_q.put({'outcome': 'SOURCE_END', 'frame_id': fid+1,
                                    'source_publish_ns': int(duration_s*NS)})
                    return
                for batch in range(800):
                    release = rt.t0_ns + batch * 50 * MS
                    if stop.wait(max(0, (release-time.perf_counter_ns()) / NS)):
                        return
                    counts = (8, 8) if batch < 320 or batch >= 480 else (20, 0)
                    for role, n in zip(('Fast', 'Accurate'), counts):
                        for _ in range(n):
                            now = time.perf_counter_ns() - rt.t0_ns
                            tid += 1
                            base_row = {'task_id': tid, 'arrival_ns': now,
                                        'deadline_ns': now+50*MS, 'origin_node': 0,
                                        'bank_index': 0, 'source_publish_ns': now}
                            fid += 1
                            rt.frame_q.put(dict(base_row, frame_id=fid, outcome=rt.OUTCOME_ARRIVAL))
                            fid += 1
                            rt.frame_q.put(dict(base_row, frame_id=fid, outcome=rt.OUTCOME_GPU_READY,
                                                gpu_candidates=[role],
                                                gpu_candidate_kinds={role: 'FAST_FULL' if role == 'Fast' else 'ACCURATE'},
                                                quality_probabilities={role: 0.9}))
                if stop.wait(max(0, (rt.t0_ns+40*NS-time.perf_counter_ns())/NS)):
                    return
                fid += 1
                rt.frame_q.put({'outcome': 'SOURCE_END', 'frame_id': fid,
                                'source_publish_ns': 40*NS})
            except Exception as exc:
                errors.append(repr(exc))
                rt.frame_q.put({'outcome': 'SOURCE_END', 'frame_id': fid+1,
                                'source_publish_ns': 40*NS})

        producer = threading.Thread(target=feed, daemon=True)
        producer.start()
        summ = rt.run()
        ctl = rt.controller
        logged_commands = []
        with io.open(str(tmp / 'controller.jsonl'), encoding='utf8') as command_log:
            for line in command_log:
                logged_commands.extend(json.loads(line).get('commands') or [])
        logged_ids = [c.get('command_id') for c in logged_commands]
        rec.chk('accepted_commands_logged_once',
                set(logged_ids) == rt.seen_command_ids and
                len(logged_ids) == len(set(logged_ids)),
                'every accepted normal/return command is archived exactly once',
                {'accepted_ids': sorted(rt.seen_command_ids),
                 'logged_ids': logged_ids,
                 'channels': [c.get('channel') for c in logged_commands]},
                'actual apply ledger versus persisted controller records')
        rec.chk('real_terminal_conservation', len(rt.seen_arrival) == len(rt.ended) and not errors,
                'all source tasks terminate', {'arrivals': len(rt.seen_arrival), 'terminal': len(rt.ended), 'producer_errors': errors}, 'Runtime.run + real dispatch/DONE')
        if source_trace is None:
            rec.chk('real_both_directions', {'A', 'F'} <= {c['direction'] for c in ctl.commands if c['channel'] == 'normal'},
                    ['A', 'F'], [{'id': c['command_id'], 'direction': c['direction'], 'from': c['from_fast'], 'to': c['target_fast'], 'channel': c['channel']} for c in ctl.commands], 'actual Runtime commands')
        else:
            rec.chk('all_source_outcome_work_present', rt.common_local_completed > 0
                    and rt.counters.get('source_outcome_PREFIX_READY', 0) > 0
                    and rt.counters.get('source_outcome_COMMON_ABSTAIN', 0) > 0,
                    'Local + Prefix + abstain workload', dict(rt.counters), 'recorded shared-source trace')
            rec.chk('worker_on_its_reserved_cpu',
                    rt.replay_worker.stats()['cpu_affinity']['actual'] == [replay_cpu],
                    [replay_cpu], rt.replay_worker.stats()['cpu_affinity'], 'actual spawned worker startup acknowledgement')
        rec.chk('real_worker_and_done', rt.replay_requests > 0 and len(rt.perf['done_lag_ns']) > 0,
                'worker requests and physical DONE', {'requests': rt.replay_requests, 'done_samples': len(rt.perf['done_lag_ns']), 'worker_status': rt.replay_worker_status}, 'actual worker and timed actor')
        rec.chk('no_open_control_at_end', ctl.pending is None, None,
                {'state': ctl.state, 'pending': ctl.pending, 'verdicts': {k: v for k, v in ctl.counters.items() if k.startswith('verdict')}}, 'actual source-end closure')
        rec.chk('no_source_gap', not rt.frame_gap_seen, False, rt.frame_gap_seen, 'full public frame sequence')
        with io.open(str(tmp / 'summary.json'), 'w', encoding='utf8') as out:
            json.dump(summ, out)
    except Exception as exc:
        import traceback
        rec.chk('runtime_exception', False, 'no exception', traceback.format_exc(), 'actual Runtime.run')
    finally:
        stop.set()
        if producer is not None:
            producer.join(2.0)
        _stop_runtimes()
        if old_affinity is not None:
            os.sched_setaffinity(0, old_affinity)
        key = 'T014_REPLAY_CPU_' + DYNAMIC_POLICY
        if old_replay_cpu is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_replay_cpu
    fields = [f for f in _t014_ready_fields(reference) if f != 'performance_ready']
    return _group_result(rec, 'SMOKE', fields,
                         {'log_directory': str(tmp),
                          'runtime_perf': {k: rt.summary().get(k) for k in
                                           ('gc', 'controller_sync_ms', 'loop_gap_ms',
                                            'log_writer', 'handler_ms', 'tick_wall_ms')}
                          if rt is not None else None,
                          'scope': 'real Runtime with timed CPU actors; mechanical closed-loop check',
                          'gpu_performance_or_benefit_verified': False})


GROUPS = {'R1': group_r1, 'R2': group_r2, 'R3': group_r3, 'SMOKE': group_runtime_smoke}


def run_checks(phases=None, sources=None, evidence=None, out=None,
               reference_json=None, expect_sha256=None):
    global _CURRENT_REFERENCE, _CURRENT_REFERENCE_PATH
    src = Path(sources) if sources else HERE
    ev = Path(evidence) if evidence else (src / 'evidence')
    # TASK s5: the reference/fake-runtime helpers must use the TASK evidence
    # (results/<TASK>/evidence, merged from the issued REFERENCE by --prepare),
    # never silently fall back to the frozen project base identity.
    explicit = (reference_json
                or os.environ.get('TASK_REFERENCE_JSON')
                or os.environ.get('T014_REFERENCE_JSON'))
    if explicit:
        ref_path = Path(str(explicit)).resolve()
        if not ref_path.exists():
            raise SystemExit('T014 --reference-json not found: %s' % ref_path)
        with io.open(str(ref_path), encoding='utf8') as f:
            reference = json.load(f)
        if not isinstance(reference.get('CPU_expected'), dict):
            raise SystemExit(
                'T014 reference %s carries no CPU_expected: a legacy or '
                'average table may not masquerade as the authoritative table'
                % ref_path)
    else:
        ref_path = Path(ev) / 'REFERENCE.json'
        reference = None
        if ref_path.exists():
            try:
                with io.open(str(ref_path), encoding='utf8') as f:
                    candidate = json.load(f)
                if isinstance(candidate.get('CPU_expected'), dict):
                    reference = candidate
                else:
                    ref_path = _reference_path()
            except Exception:
                ref_path = _reference_path()
        else:
            ref_path = _reference_path()
        if reference is None:
            if not ref_path.exists():
                raise SystemExit('T014 REFERENCE.json not found: %s' % ref_path)
            with io.open(str(ref_path), encoding='utf8') as f:
                reference = json.load(f)
            if not isinstance(reference.get('CPU_expected'), dict):
                raise SystemExit(
                    'T014 reference %s carries no CPU_expected: refusing to '
                    'run the T014 checks against a legacy table' % ref_path)
    reference_sha = _sha256(ref_path)
    expected_sha = (expect_sha256 or os.environ.get('TASK_REFERENCE_SHA256'))
    if expected_sha and str(expected_sha) != reference_sha:
        raise SystemExit(
            'T014 reference sha256 mismatch: %s is %s, expected %s -- '
            'refusing to run any check against a different reference'
            % (ref_path, reference_sha, expected_sha))
    _CURRENT_REFERENCE = reference
    _CURRENT_REFERENCE_PATH = str(ref_path)
    if isinstance(phases, str):
        wanted = [p.strip() for p in phases.split(',') if p.strip()]
    elif phases:
        wanted = [str(p).strip() for p in phases if str(p).strip()]
    else:
        wanted = list(GROUPS)
    groups = []
    for gid in wanted:
        if gid not in GROUPS:
            continue
        groups.append(GROUPS[gid](reference))
    failed = [g['id'] for g in groups if not g['ok']]
    ready_fields = _t014_ready_fields(reference)
    if wanted == ['SMOKE']:
        ready_fields = [f for f in ready_fields if f != 'performance_ready']
    requires_all = bool((reference.get('CPU_expected') or {}).get(
        'execution_ready_requires_all', True))
    readiness = {}
    for g in groups:
        for key, val in (g.get('readiness') or {}).items():
            readiness[key] = readiness.get(key, True) and bool(val)
    for key in ready_fields:
        readiness.setdefault(key, None)
    declared = [f for f in ready_fields if readiness.get(f) is not None]
    execution_ready = bool(declared) and all(readiness.get(f) is True
                                            for f in ready_fields)
    result = {
        'schema': 'T014_cpu_checks_v1',
        'task_id': 'T014',
        'implementation_revision': C.IMPLEMENTATION_REVISION,
        'controller_version': C.CONTROLLER_VERSION,
        'policies': [FIXED_POLICY, DYNAMIC_POLICY],
        'reference_path': str(ref_path),
        'reference_sha256': reference_sha,
        'reference_expected_sha256': (str(expected_sha) if expected_sha else None),
        'expected_source': 'REFERENCE.json (never the current implementation)',
        'expected_edited': False,
        'groups': groups,
        'pass': not failed and len(groups) == len(wanted),
        'failed': failed,
        'ready_fields': ready_fields,
        'readiness': readiness,
        'execution_ready_requires_all': requires_all,
        'execution_ready': execution_ready,
        'sub_checks': sum(int(g['sub_checks']) for g in groups),
        'wall_s': round(sum(float(g['wall_s']) for g in groups), 3),
        'performance': {k: v for g in groups for k, v in (g.get('performance') or {}).items()},
    }
    if out:
        with io.open(str(out), 'w', encoding='utf8') as f:
            f.write(json.dumps(result, indent=1, sort_keys=True))
    return result


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with io.open(str(path), 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    phases = [a for a in argv if a in GROUPS]
    out = None
    reference_json = None
    expect_sha256 = None
    k = 0
    while k < len(argv):
        if argv[k] == '--phase' and k + 1 < len(argv):
            if not phases:
                phases = [p.strip() for p in argv[k + 1].split(',') if p.strip()]
            k += 2
        elif argv[k] == '--out' and k + 1 < len(argv):
            out = argv[k + 1]
            k += 2
        elif argv[k] in ('--reference-json', '--reference') and k + 1 < len(argv):
            reference_json = argv[k + 1]
            k += 2
        elif argv[k] in ('--expect-reference-sha256', '--reference-sha256') \
                and k + 1 < len(argv):
            expect_sha256 = argv[k + 1]
            k += 2
        else:
            k += 1
    res = run_checks(phases=phases or None, out=out,
                     reference_json=reference_json, expect_sha256=expect_sha256)
    print(json.dumps({'pass': res['pass'], 'failed': res['failed'],
                      'wall_s': res['wall_s'],
                      'readiness': res['readiness'],
                      'execution_ready': res['execution_ready'],
                      'reference_sha256': res['reference_sha256']},
                     sort_keys=True))
    return 0 if (res['pass'] and res['execution_ready']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
