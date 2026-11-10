"""Clockwork-, Symphony- and Proteus-style edge adaptations on the common executor.

Conditional edge-side mechanism adaptations, NOT complete paper reproductions.
Both CNNs are resident on both GPUs (no LOAD actions), the Local stage is the
common frozen trace, and the shared executor is nonpreemptive stop-and-wait
(so Clockwork's 5 ms outstanding-work lookahead cannot be reproduced). Only
public queue state, past execution times and past arrival counts are read.
Each rule cites its source; values the source does not give are ADAPTATION.

Clockwork (Gujarati et al., OSDI 2020, arXiv 2006.02464, Sec. 4.5 and 5):
  * action profile = last 10 action durations per (model, worker, batch size);
    prediction = rolling 99th percentile (Sec. 6.5, prediction error).
  * one batch queue per (model, batch size): requests still satisfiable at
    that size; a request not satisfiable even alone is cancelled.
  * strategy per non-empty batch queue with latest = head deadline - predicted
    duration; the executor takes the valid strategy (latest not elapsed, queue
    holds >= b) with the earliest latest, then speculatively increases the
    batch size while extra requests are available.
  * ADAPTATION: sizes 1..10 (eager CNNs; TVM-precompiled sizes are a
    deployment artifact); duration = controller dispatch -> result received,
    plus the configured return delay; cold cells use the offline profile.
Symphony (Chen et al., arXiv 2308.07470v2, Algorithm 1):
  * GetBatch adds requests from the head while the batch meets the earliest
    deadline; exec = max(now, d - l(|B|+1)) (frontrun), latest = d - l(|B|);
    the model timer at exec dispatches to the lowest-id free GPU; a freed GPU
    takes the schedulable, still-valid candidate with the smallest latest.
    No dispatch before frontrun, so GPUs may idle.
  * l(b): offline profile executor_profile_20261019.json (end-to-end).
  * ADAPTATION: a head that cannot meet its deadline alone is dropped.
Proteus (Ahmad et al., ASPLOS 2024; author code UMass-LIDS/Proteus 6967c93):
  * algorithms/ilp.py: one variant per device; routing z sums to 1; peak
    throughput p = floor(b* * 1000 / l(b*)) for the largest allowed b* with
    l(b*) < SLO; y_i <= p_i, y_i <= beta*s*z_i (beta > 1), sum y >= beta*s;
    maximize routed accuracy sum s*z_i*A(v_i); beta starts 1.05 and steps down
    by 0.05 (0.1 above 1.5) until feasible, stopping at 0.2.
  * run.py / simulator.py: 1 s windows, EWMA(com=1.6, adjust=True) over the
    last 20 windows, solve at windows 5, 20 and every 10th; routing by
    random.choices on z with seed 10; allowed batch sizes [1, 2, 4, 8].
  * core/predictor.py 'accscale': drop heads that would expire at the batch
    size the queue fills; run the maximum size when the queue exceeds it;
    otherwise wait until head deadline - l(fill size), then run.
  * ADAPTATION: exact LP per enumerated placement (scipy HiGHS) instead of
    Gurobi; SLO = median observed edge budget; A(v) = DEVELOPMENT accuracy;
    initial placement Fast/Accurate with equal routing (the author starting
    solution is trace-specific); trace-specific extra solve steps omitted.
  * The plan is solved in a separate planner process, as Proteus' controller
    is separate from its load balancer and workers; the current plan stays in
    force until a new one arrives. Its CPU is accounted as a seventh process.
SuperServe-SlackFit (Khare et al., NSDI 2025, Sec. 4.2 and App. A.5):
  * global EDF queue; whenever a worker is free, take the most urgent query's
    remaining slack, choose the evenly sized latency bucket closest to and
    below the slack (every control in it meets the deadline), and inside it the
    control with the largest batch; queries whose slack is below the smallest
    profiled latency cannot be served and are dropped (App. A.5).
  * controls = Pareto (latency, accuracy) models x power-of-two batch sizes.
  * ADAPTATION: the two resident CNNs replace weight-shared subnets (actuation
    is instantaneous either way); batch sizes 1, 2, 4, 8; latency = end-to-end
    executor profile; bucket width 1 ms (not given in the paper, fixed before
    any run); a chosen batch is capped by the queue length; if no whole bucket
    lies below the slack, the feasible controls of the lowest bucket are used.
PARD (Zhao et al., EuroSys 2026, arXiv 2602.08747v1, Sec. 4.2-4.3, 5.1, 5.4) and RAMSIS
(Mendoza et al., EuroSys 2024; author code dmmendo/RAMSIS 9b74967): see
NEW_BASELINES_DERIVATION_20261021.md for every mapped value and the forced adaptations.
"""
from bisect import bisect_left, bisect_right
from collections import deque
import hashlib
import itertools
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue as queue_module
import random
import statistics
import time

from literature_baselines.current_literature import EdgePolicy, load_calibration
from mixed_fast_batch import batch_kind

PROFILE = Path(__file__).resolve().parent / 'executor_profile_20261019.json'
ROLE_KIND = {'Fast': 'FAST_FULL', 'Accurate': 'ACCURATE'}


def load_profile(path=PROFILE):
    data = json.loads(Path(path).read_text())
    body = {k: v for k, v in data.items() if k != 'content_sha256'}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != data['content_sha256']:
        raise ValueError('Executor profile identity changed')
    return data


class StaticLatency:
    """Offline end-to-end l(kind, b); uncovered cells: calibration mean + return + median overhead."""
    def __init__(self, runtime, data=None):
        self.rt = runtime
        data = data or load_profile()
        self.cells = {k: v['median_ns'] for k, v in data['profile'].items()}
        gaps = []
        for key, value in self.cells.items():
            kind, b = key.rsplit('/', 1)
            gaps.append(value - self.base([dict(kind=kind, task_id=-1)] * int(b)))
        self.overhead = statistics.median(gaps) if gaps else 0.
        self.sha = data['content_sha256']

    def base(self, jobs):
        return self.rt.batch_mean(jobs) + self.rt.cfg['network_return_ns']

    def __call__(self, jobs):
        """l(batch) for the actual member composition."""
        value = self.cells.get('%s/%d' % (batch_kind(jobs), len(jobs)))
        return value if value is not None else self.base(jobs) + self.overhead

    def pure(self, kind, b):
        return self([dict(kind=kind, task_id=-1)] * b)


class ServingPolicy(EdgePolicy):
    def __init__(self, runtime):
        super().__init__(runtime)
        self.counts = {}

    def bump(self, key, n=1):
        self.counts[key] = self.counts.get(key, 0) + n

    def free_gpus(self):
        return [g for g, c in enumerate(self.rt.cards) if not c['busy']]

    def owner(self, job):
        return None

    def queue(self, role, owner=None):
        jobs = [j for j in self.rt.queued.values() if j['role'] == role and
                (owner is None or self.owner(j) == owner)]
        return sorted(jobs, key=self.rt.batch_order_key)

    def drop(self, job, now):
        self.rt.remove(job, loss=True)
        self.rt.finish(job['task_id'], 'Abstain', None, now, 'literature_early_drop')
        self.bump('early_drop')

    def evidence(self, **extra):
        return dict(literature_method=self.rt.arm, geometry='two physical GPUs, either CNN, nonpreemptive',
                    source='current public queues and past execution times only', **extra)

    def close(self):
        self.rt.log('literature', dict(event='FINAL', method=self.rt.arm, calls=self.calls,
                    counts=dict(self.counts), complete_original_paper=False))


class ClockworkEdgePolicy(ServingPolicy):
    WINDOW = 10
    QUANTILE = .99

    def __init__(self, runtime):
        super().__init__(runtime)
        self.static = StaticLatency(runtime)
        self.profiles = {}

    def predict(self, jobs, g):
        cell = self.profiles.get((batch_kind(jobs), len(jobs), g))
        if cell:
            ordered = sorted(cell)
            return ordered[min(len(ordered), math.ceil(self.QUANTILE * len(ordered))) - 1] + \
                self.rt.cfg['network_return_ns']
        return self.static(jobs)

    def book(self, g, batch, evidence):
        return dict(kind=batch_kind(batch), size=len(batch), gpu=g, dispatch_ns=self.rt.now())

    def complete(self, g, booking, actual_ns):
        if booking:
            key = (booking['kind'], booking['size'], booking['gpu'])
            self.profiles.setdefault(key, deque(maxlen=self.WINDOW)).append(
                self.rt.now() - booking['dispatch_ns'] - self.rt.cfg['network_return_ns'])

    def choose(self, now):
        free = self.free_gpus()
        if not free:
            return None
        g = free[0]
        best = None
        for role in ('Fast', 'Accurate'):
            jobs = self.queue(role)
            while jobs and now + self.predict(jobs[:1], g) > jobs[0]['deadline_ns']:
                self.drop(jobs.pop(0), now)
            for b in range(1, self.rt.cfg['batch_max'] + 1):
                if len(jobs) < b:
                    break
                duration = self.predict(jobs[:b], g)
                q = [j for j in jobs if now + duration <= j['deadline_ns']]
                if len(q) < b:
                    continue
                latest = q[0]['deadline_ns'] - duration
                if best is None or latest < best[0]:
                    best = (latest, b, q)
        if best is None:
            return None
        _, b, q = best
        # Speculative growth from the chosen batch queue while its head stays satisfiable.
        while b < self.rt.cfg['batch_max'] and len(q) > b and now + self.predict(q[:b + 1], g) <= q[0]['deadline_ns']:
            b += 1
        self.calls += 1; self.bump('batches')
        return g, q[:b], self.evidence(rule='earliest-latest strategy, speculative growth', size=b)


class SymphonyEdgePolicy(ServingPolicy):
    def __init__(self, runtime):
        super().__init__(runtime)
        self.latency = StaticLatency(runtime)

    def candidate(self, role, now):
        jobs = self.queue(role)
        limit = self.rt.cfg['batch_max']
        while jobs:
            batch = []
            for j in jobs[:limit]:
                trial = batch + [j]
                if now + self.latency(trial) <= trial[0]['deadline_ns']:
                    batch = trial
                else:
                    break
            if not batch:
                self.drop(jobs.pop(0), now)
                continue
            d = batch[0]['deadline_ns']
            # l(|B|+1) for the same composition: the next request joins this batch.
            frontrun = d - self.latency(batch + batch[-1:]) if len(batch) < limit else now
            return batch, max(now, frontrun), d - self.latency(batch)
        return None

    def choose(self, now):
        free = self.free_gpus()
        if not free:
            return None
        ready = []
        for role in ('Fast', 'Accurate'):
            c = self.candidate(role, now)
            if c is None:
                continue
            batch, exec_at, latest = c
            if exec_at <= now <= latest:
                ready.append((latest, batch))
            else:
                self.bump('deferred_checks')
        if not ready:
            return None
        latest, batch = min(ready, key=lambda x: x[0])
        self.calls += 1; self.bump('batches')
        return min(free), batch, self.evidence(rule='frontrun-to-latest window (Algorithm 1)', size=len(batch))


PROTEUS_SIZES = (1, 2, 4, 8)


def proteus_peak(latency, variant, slo):
    """latency: {(variant, b): end-to-end ns}; author peak = floor(b* * 1e9 / l(b*))."""
    fit = [b for b in PROTEUS_SIZES if latency[(variant, b)] < slo]
    return math.floor(fit[-1] * 1e9 / latency[(variant, fit[-1])]) if fit else 0


def proteus_plan(s, slo, accuracy, latency, beta0=1.05, min_beta=.2):
    """algorithms/ilp.py with one variant per device; returns (beta, best or None)."""
    from scipy.optimize import linprog
    beta = beta0
    while beta > min_beta:
        best = None
        for variants in itertools.product(('Fast', 'Accurate'), repeat=2):
            p = [proteus_peak(latency, v, slo) for v in variants]
            cap = beta if beta > 1 else 1.
            # variables [z0, z1, y0, y1]
            r = linprog([-s * accuracy[variants[0]], -s * accuracy[variants[1]], 0, 0],
                        A_ub=[[0, 0, 1, 0], [0, 0, 0, 1], [-cap * s, 0, 1, 0], [0, -cap * s, 0, 1],
                              [0, 0, -1, -1]],
                        b_ub=[p[0], p[1], 0, 0, -beta * s], A_eq=[[1, 1, 0, 0]], b_eq=[1],
                        bounds=[(0, 1), (0, 1), (0, None), (0, None)], method='highs')
            if r.status == 0:
                key = (round(-r.fun, 9), sum(accuracy[v] for v in variants))
                if best is None or key > best[0]:
                    best = (key, list(variants), [max(0., float(x)) for x in r.x[:2]], p)
        if best is not None:
            return beta, best
        beta -= .1 if beta > 1.5 else .05
    return beta, None


def proteus_worker(requests, replies, epoch, folder, cpu):
    """Separate Proteus controller process; its CPU is accounted like Jellyfish's planner."""
    import os
    import traceback
    from live_batch_runtime import cpu_snapshot, save_process_cpu
    os.sched_setaffinity(0, {cpu})
    try:
        import scipy.optimize  # noqa: F401  (import cost before the epoch)
        replies.put(dict(type='READY'))
        while not epoch.value:
            time.sleep(.001)
        while time.monotonic_ns() < epoch.value:
            time.sleep(.0005)
        baseline = cpu_snapshot()
        while True:
            request = requests.get()
            if request is None:
                break
            before = time.process_time_ns()
            latency = {(v, b): ns for v, b, ns in request['latency']}
            beta, best = proteus_plan(request['s'], request['slo'], request['accuracy'], latency)
            replies.put(dict(type='PLAN', beta=beta, best=best, window=request['window'], s=request['s'],
                             slo=request['slo'], submitted_ns=request['submitted_ns'],
                             cpu_ns=time.process_time_ns() - before))
    except BaseException:
        replies.put(dict(type='ERROR', error=traceback.format_exc())); raise
    finally:
        replies.close(); replies.join_thread()
        if 'baseline' in locals():
            save_process_cpu(folder, 'literature_planner', baseline)


class ProteusEdgePolicy(ServingPolicy):
    WINDOW_NS = 1_000_000_000
    SOLVE_EVERY = 10
    EXTRA_SOLVES = (5, 20)
    EWMA_COM = 1.6
    EWMA_WINDOW = 20
    BETA = 1.05
    MIN_BETA = .2
    SIZES = (1, 2, 4, 8)

    def __init__(self, runtime):
        super().__init__(runtime)
        self.calibration = load_calibration(runtime.cfg['literature_calibration'])
        self.accuracy = dict(zip(('Fast', 'Accurate'), self.calibration['accuracy_fast_accurate']))
        self.latency = StaticLatency(runtime)
        self.variants = ['Fast', 'Accurate']
        self.routing = [.5, .5]
        self.rng = random.Random(10)
        self.assigned = {}
        self.window_arrivals = 0; self.window_index = 0; self.next_window = self.WINDOW_NS
        self.demand = deque(maxlen=self.EWMA_WINDOW); self.budgets = deque(maxlen=4096)
        self.table = {(v, b): self.latency.pure(ROLE_KIND[v], b) for v in ('Fast', 'Accurate')
                      for b in self.SIZES}
        self.inflight = False; self.worker = None
        if hasattr(runtime, 'epoch'):  # the live executor; unit tests solve synchronously
            ctx = mp.get_context('spawn'); self.requests = ctx.Queue(); self.replies = ctx.Queue()
            self.worker = ctx.Process(target=proteus_worker, args=(self.requests, self.replies, runtime.epoch,
                                      str(runtime.output), runtime.cfg['literature_planner_cpu']))
            self.worker.start(); runtime.processes.append(self.worker)
            ready = self.replies.get(timeout=60)
            if ready['type'] != 'READY':
                raise RuntimeError(ready)

    def owner(self, job):
        return self.assigned.get(job['task_id'])

    def input_kind(self, frame, now):
        tid = frame['task_id']
        self.window_arrivals += 1
        self.budgets.append(self.rt.tasks[tid]['deadline_ns'] - now)
        g = self.rng.choices((0, 1), weights=self.routing, k=1)[0]
        self.assigned[tid] = g
        return ROLE_KIND[self.variants[g]]

    def ewma(self):
        if not self.demand:
            return None
        alpha = 1 / (1 + self.EWMA_COM)
        weights = [(1 - alpha) ** i for i in range(len(self.demand) - 1, -1, -1)]
        return sum(w * d for w, d in zip(weights, self.demand)) / sum(weights)

    def peak(self, variant, slo):
        return proteus_peak(self.table, variant, slo)

    def solve(self, s, slo):
        return proteus_plan(s, slo, self.accuracy, self.table, self.BETA, self.MIN_BETA)

    def apply(self, beta, best, window, s, slo, **extra):
        if best is None:
            self.bump('infeasible_solves'); return
        _, variants, z, p = best
        total = sum(z)
        self.variants = list(variants)
        self.routing = [x / total for x in z] if total > 0 else [.5, .5]
        self.bump('plans')
        self.rt.log('literature', dict(event='PROTEUS_PLAN', now_ns=self.rt.now(), window=window, demand_rps=s,
                    slo_ms=slo / 1e6, beta=beta, variants=self.variants, routing=self.routing, peak_rps=p,
                    accuracy=self.accuracy, profile_sha256=self.latency.sha,
                    calibration_sha256=self.calibration['content_sha256'], **extra))

    def tick(self, now):
        if self.worker is not None:
            try:
                reply = self.replies.get_nowait()
            except queue_module.Empty:
                reply = None
            if reply is not None:
                if reply['type'] != 'PLAN':
                    raise RuntimeError(reply)
                self.inflight = False
                self.apply(reply['beta'], reply['best'], reply['window'], reply['s'], reply['slo'],
                           submitted_ns=reply['submitted_ns'], planner_cpu_ns=reply['cpu_ns'])
        while now >= self.next_window:
            self.demand.append(self.window_arrivals / (self.WINDOW_NS / 1e9))
            self.window_arrivals = 0; self.window_index += 1; self.next_window += self.WINDOW_NS
            i = self.window_index
            if not (i % self.SOLVE_EVERY == 0 or i in self.EXTRA_SOLVES) or not self.budgets:
                continue
            s = self.ewma()
            if not s:
                continue
            slo = sorted(self.budgets)[len(self.budgets) // 2]
            if self.worker is None:
                beta, best = self.solve(s, slo)
                self.apply(beta, best, i, s, slo)
            elif self.inflight:
                self.bump('solves_skipped_inflight')
            else:
                self.requests.put(dict(s=s, slo=slo, accuracy=self.accuracy, window=i, submitted_ns=now,
                                       latency=[(v, b, ns) for (v, b), ns in self.table.items()]))
                self.inflight = True

    def close(self):
        if self.worker is not None:
            self.requests.put(None); self.requests.close(); self.requests.join_thread()
            self.worker.join(timeout=60)
            if self.worker.exitcode != 0:
                raise RuntimeError('Proteus planner did not stop cleanly')
            self.replies.close(); self.replies.join_thread()
        super().close()

    def fill_size(self, n):
        fit = [b for b in self.SIZES if b <= n]
        return fit[-1] if fit else 0

    def choose(self, now):
        free = self.free_gpus()
        if not free:
            return None
        options = []
        for g in free:
            for role in ('Fast', 'Accurate'):
                q = self.queue(role, g)
                while q:
                    b = self.fill_size(len(q))
                    if now + self.latency(q[:b]) > q[0]['deadline_ns']:
                        self.drop(q.pop(0), now)
                    else:
                        break
                if not q:
                    continue
                b = self.SIZES[-1] if len(q) > self.SIZES[-1] else self.fill_size(len(q))
                if len(q) <= self.SIZES[-1]:
                    wait_until = q[0]['deadline_ns'] - self.latency(q[:b])
                    if now < wait_until - self.rt.cfg['main_slice_ns']:
                        self.bump('deferred_checks'); continue
                options.append(((q[0]['deadline_ns'], q[0]['task_id'], g), g, q[:b]))
        if not options:
            return None
        _, g, batch = min(options, key=lambda x: x[0])
        self.calls += 1; self.bump('batches')
        return g, batch, self.evidence(rule='accscale adaptive batching on planned variant',
                                       variants=list(self.variants))


class SlackFitEdgePolicy(ServingPolicy):
    """SuperServe SlackFit on the two resident CNNs (see module docstring)."""
    SIZES = (1, 2, 4, 8)
    BUCKET_NS = 1_000_000

    def __init__(self, runtime):
        super().__init__(runtime)
        self.calibration = load_calibration(runtime.cfg['literature_calibration'])
        self.accuracy = dict(zip(('Fast', 'Accurate'), self.calibration['accuracy_fast_accurate']))
        self.latency = StaticLatency(runtime)
        models = [m for m in ('Fast', 'Accurate')
                  if not any(self.dominates(o, m) for o in ('Fast', 'Accurate') if o != m)]
        controls = [(self.latency.pure(ROLE_KIND[m], b), m, b) for m in models for b in self.SIZES]
        self.floor = min(c[0] for c in controls)
        # Evenly sized buckets over the profiled latencies: index -> controls.
        self.buckets = {}
        for c in controls:
            self.buckets.setdefault(int((c[0] - self.floor) // self.BUCKET_NS), []).append(c)
        self.models = models

    def dominates(self, a, b):
        """Pareto w.r.t. (latency at batch 1, accuracy)."""
        la, lb = (self.latency.pure(ROLE_KIND[m], 1) for m in (a, b))
        return la <= lb and self.accuracy[a] >= self.accuracy[b] and (la, self.accuracy[a]) != (lb, self.accuracy[b])

    def input_kind(self, frame, now):
        return 'FAST_FULL'  # one global EDF queue; the model is chosen at dispatch

    def control(self, slack):
        """Bucket closest to and below the slack; largest batch inside it."""
        whole = [k for k, cs in self.buckets.items() if max(c[0] for c in cs) <= slack]
        if whole:
            pool = self.buckets[max(whole)]
        else:
            pool = [c for cs in self.buckets.values() for c in cs if c[0] <= slack]
            if not pool:
                return None
            self.bump('partial_bucket')
        return max(pool, key=lambda c: (c[2], self.accuracy[c[1]]))

    def choose(self, now):
        free = self.free_gpus()
        if not free:
            return None
        while True:
            front = self.rt._kind_jobs('FAST_FULL', limit=self.SIZES[-1])
            if not front:
                return None
            choice = self.control(front[0]['deadline_ns'] - now)
            if choice is not None:
                break
            self.drop(front[0], now)
        _, model, b = choice
        batch = front[:b]
        if model == 'Accurate':
            for job in batch:
                self.rt.retarget_public_job(job, 'ACCURATE', now)
            batch = [self.rt.queued[j['task_id']] for j in batch]
            self.bump('accurate_batches')
        self.calls += 1; self.bump('batches')
        return min(free), batch, self.evidence(rule='SlackFit bucket closest below head slack, max batch',
                                               model=model, control_batch=b, size=len(batch))


class PardEdgePolicy(ServingPolicy):
    """PARD on the Fast -> (Accurate) pipeline: proactive drop at batch entry, adaptive HBF/LBF DEPQ.

    One worker per module (two GPUs, two loaded modules); t_e = dispatch (stop-and-wait executor);
    Prefix requests carry the Accurate path L_sub = q_A + d_A(b_A) + F^-1(0.1) (maximum over DAG paths).
    """
    LAMBDA = .1
    WINDOW_NS = 5_000_000_000
    SYNC_NS = 1_000_000_000
    SAMPLES = 10_000
    KINDS = dict(Fast=('FAST_FULL', 'FAST_PREFIX'), Accurate=('ACCURATE',))

    def __init__(self, runtime):
        super().__init__(runtime)
        self.latency = StaticLatency(runtime)
        self.ret = runtime.cfg['network_return_ns']
        self.priority = dict(Fast='LBF', Accurate='LBF')
        self.served = {0: 'Fast', 1: 'Accurate'}
        self.history = {m: deque() for m in self.KINDS}      # [sync_ns, T_in, T_s]
        self.entered = {m: 0 for m in self.KINDS}
        self.next_sync = self.SYNC_NS
        self.queueing = deque()                              # Accurate (t_b, Q = t_b - t_r)
        self.batch_wait = deque(maxlen=self.SAMPLES)         # Accurate W = t_e - t_b, one per request
        self.period = {m: [] for m in self.KINDS}            # (size, execution ns) dispatched this period
        self.size = dict(Fast=1, Accurate=1)
        self.l_sub = self.subsequent(0., 0., 1)
        # PARD's only rule for several subsequent paths is their maximum (DAG fork). A Prefix request's
        # branch is conditional; 'own' is the alternative extrapolation (Prefix treated as its last module).
        self.prefix_path = runtime.cfg.get('pard_prefix_path', 'max')
        if self.prefix_path not in ('max', 'own'):
            raise ValueError('Unknown PARD Prefix path rule')

    @staticmethod
    def module(job):
        return 'Accurate' if job['kind'] == 'ACCURATE' else 'Fast'

    def subsequent(self, q, w, b):
        """L_sub of a Prefix request on its Accurate path; d_A carries the single return."""
        return q + self.latency.pure('ACCURATE', b) + w

    def tick(self, now):
        while now >= self.next_sync:
            self.sync(self.next_sync)
            self.next_sync += self.SYNC_NS

    def sync(self, at):
        """State planner, once per second: T_in, T_s, eps, mu -> HBF/LBF; workers; q_A, b_A, w_A."""
        row = dict(event='PARD_SYNC', now_ns=at)
        for m, kinds in self.KINDS.items():
            total = sum(self.rt.counts.get('enqueued_' + k, 0) for k in kinds)
            t_in = (total - self.entered[m]) / (self.SYNC_NS / 1e9)
            self.entered[m] = total
            h = self.history[m]
            h.append([at, t_in, None])
            while h[0][0] <= at - self.WINDOW_NS:
                h.popleft()
            h[-1][2] = sum(x[1] for x in h) / len(h)
            rows, self.period[m] = self.period[m], []
            if rows:
                self.size[m] = max(1, round(sum(b for b, _ in rows) / len(rows)))
                execution = sum(d for _, d in rows) / len(rows)
            else:
                execution = self.latency.pure(self.KINDS[m][0], self.size[m]) - self.ret
            workers = sum(v == m for v in self.served.values())
            capacity = workers * self.size[m] / (execution / 1e9)
            mu = t_in / capacity if capacity > 0 else math.inf
            total_in = sum(x[1] for x in h)
            eps = sum(abs(x[1] - x[2]) for x in h) / total_in if total_in > 0 else 0.
            before = self.priority[m]
            if mu > 1 + eps:
                self.priority[m] = 'HBF'
            elif mu < 1 - eps:
                self.priority[m] = 'LBF'
            if self.priority[m] != before:
                self.bump('priority_switches')
            row[m] = dict(T_in=t_in, T_s=h[-1][2], mu=mu, eps=eps, priority=self.priority[m],
                          batch_size=self.size[m], execution_ms=execution / 1e6)
        active = [m for m in self.KINDS if sum(x[1] for x in self.history[m]) > 0]
        served = {0: active[0], 1: active[0]} if len(active) == 1 else {0: 'Fast', 1: 'Accurate'}
        if served != self.served:
            self.bump('worker_changes')
        self.served = served
        while self.queueing and self.queueing[0][0] <= at - self.WINDOW_NS:
            self.queueing.popleft()
        weights = [t - (at - self.WINDOW_NS) for t, _ in self.queueing]
        q = (sum(w * x for w, (_, x) in zip(weights, self.queueing)) / sum(weights)
             if self.queueing else 0.)
        ordered = sorted(self.batch_wait)
        w = ordered[max(0, math.ceil(self.LAMBDA * len(ordered)) - 1)] if ordered else 0.
        self.l_sub = self.subsequent(q, w, self.size['Accurate'])
        self.bump('syncs')
        self.rt.log('literature', dict(row, workers=self.served, q_A_ms=q / 1e6, w_A_ms=w / 1e6,
                                       b_A=self.size['Accurate'], l_sub_ms=self.l_sub / 1e6))

    def estimate(self, job, duration, now):
        """Eq. 3 with t_e = now; FAST_PREFIX takes the Accurate DAG path (maximum over paths)."""
        if job['kind'] == 'FAST_PREFIX' and self.prefix_path == 'max':
            self.bump('prefix_path_checks')
            return now + duration - self.ret + self.l_sub
        return now + duration

    def form(self, m, now):
        jobs = self.queue(m)
        if self.priority[m] == 'HBF':
            jobs.reverse()
        size = min(len(jobs), self.rt.cfg['batch_max'])
        batch, rest = jobs[:size], jobs[size:]
        while batch:
            duration = self.latency(batch)
            keep = []
            for job in batch:
                if self.estimate(job, duration, now) > job['deadline_ns']:
                    if job['kind'] == 'FAST_PREFIX':
                        self.bump('prefix_path_drops')
                    self.drop(job, now)
                else:
                    keep.append(job)
            if len(keep) == len(batch):
                return batch
            batch, rest = keep + rest[:size - len(keep)], rest[size - len(keep):]
        return []

    def book(self, g, batch, evidence):
        now = self.rt.now()
        m = self.module(batch[0])
        self.period[m].append((len(batch), self.latency(batch) - self.ret))
        if m == 'Accurate':
            self.queueing.extend((now, now - job['enqueue_ns']) for job in batch)
        return dict(module=m, dispatch_ns=now)

    def complete(self, g, booking, actual_ns):
        message = getattr(self.rt, 'last_gpu_message', None)
        if booking and booking['module'] == 'Accurate' and message and message['actual_batch']:
            self.batch_wait.extend([message['started_ns'] - booking['dispatch_ns']] * message['actual_batch'])

    def choose(self, now):
        for g in self.free_gpus():
            m = self.served[g]
            batch = self.form(m, now)
            if batch:
                self.calls += 1; self.bump('batches')
                # HBF selects the members by largest budget; a batch is one set on the GPU, listed in
                # deadline order like every executor batch (the statistics invariant).
                batch = sorted(batch, key=self.rt.batch_order_key)
                return g, batch, self.evidence(rule='PARD DEPQ with proactive drop at batch entry', module=m,
                                               priority=self.priority[m], size=len(batch),
                                               l_sub_ms=self.l_sub / 1e6, prefix_path=self.prefix_path)
        return None


class RamsisEdgePolicy(ServingPolicy):
    """RAMSIS online phase: round robin, per-worker author MDP policy chosen by load (500 ms) and edge SLO."""
    WINDOW_NS = 500_000_000
    MODELS = ('FAST_FULL', 'ACCURATE')

    def __init__(self, runtime):
        super().__init__(runtime)
        from literature_baselines import ramsis_policies
        path = Path(runtime.cfg['ramsis_policies'])
        self.table = ramsis_policies.load(path if path.is_absolute() else Path(__file__).resolve().parents[1] / path)
        self.size = self.table['arguments']['max_batch']
        if self.size != runtime.cfg['batch_max']:
            raise ValueError('RAMSIS policies were generated for another maximum batch')
        self.slos = sorted(int(s) for s in self.table['policies'])
        self.rates = {s: [p['rate_qps'] for p in self.table['policies'][str(s)]] for s in self.slos}
        self.owners = {}
        self.turn = 0
        self.recent = deque()                                 # (edge arrival ns, budget ns)
        self.slo = self.slos[-1]

    def input_kind(self, frame, now):
        tid = frame['task_id']
        self.owners[tid] = self.turn
        self.turn = (self.turn + 1) % len(self.rt.cards)
        self.recent.append((now, self.rt.tasks[tid]['deadline_ns'] - now))
        return self.MODELS[0]

    def owner(self, job):
        return self.owners.get(job['task_id'])

    def policy(self, now):
        """Load monitor (arrivals in the last 500 ms) and SLO = floor(median edge budget) -> policy row."""
        while self.recent and self.recent[0][0] <= now - self.WINDOW_NS:
            self.recent.popleft()
        load = len(self.recent) / (self.WINDOW_NS / 1e9)
        if self.recent:
            budgets = sorted(b for _, b in self.recent)
            # Largest generated SLO not above the median budget (floor), else the smallest.
            self.slo = self.slos[max(0, bisect_right(self.slos, budgets[len(budgets) // 2] / 1e6) - 1)]
        rates = self.rates[self.slo]
        i = min(bisect_left(rates, load), len(rates) - 1)     # smallest generated rate >= load, else largest
        return self.slo, load, self.table['policies'][str(self.slo)][i]

    def action(self, jobs, now):
        """PlainInstPi.get_action: state (n, quantized earliest slack); n > N -> default (model 1)."""
        slo, load, row = self.policy(now)
        n = len(jobs)
        if n > self.size:
            self.bump('overflow_default')
            return self.MODELS[0], dict(slo_ms=slo, load_qps=load, policy_rate_qps=row['rate_qps'], n=n)
        grid = self.table['grid_us'][str(slo)]
        idx = max(bisect_right(grid, (jobs[0]['deadline_ns'] - now) / 1e3) - 1, 0)
        model = self.MODELS[int(row['actions'][n - 1][idx]) - 1]
        return model, dict(slo_ms=slo, load_qps=load, policy_rate_qps=row['rate_qps'], n=n, slack_index=idx)

    def choose(self, now):
        for g in self.free_gpus():
            jobs = sorted((j for j in self.rt.queued.values() if self.owner(j) == g), key=self.rt.batch_order_key)
            if not jobs:
                continue
            model, state = self.action(jobs, now)
            batch = jobs[:self.size]
            if model == 'ACCURATE':
                # The executor's own realizability trim, applied before retargeting, so no
                # retargeted query is left behind in the Accurate heap.
                trial = [dict(j, kind='ACCURATE') for j in batch]
                while len(trial) > 1 and self.rt.batch_unrealizable(trial, g):
                    trial.pop()
                batch = batch[:len(trial)]
            for job in batch:
                self.rt.retarget_public_job(job, model, now)
            batch = [self.rt.queued[j['task_id']] for j in batch]
            self.calls += 1; self.bump('batches'); self.bump('lookups')
            self.bump('accurate_batches' if model == 'ACCURATE' else 'fast_batches')
            return g, batch, self.evidence(rule='RAMSIS per-worker MDP policy (author code)', model=model,
                                           size=len(batch), **state)
        return None
