"""Literature mechanisms on CLRA's common frozen-Local real-GPU executor.

These are conditional edge-side adaptations, NOT complete paper reproductions.
No adapter sees images, labels, cached predictions or future source frames.
"""
from collections import Counter, deque
import contextlib
import importlib
import io
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def confidence_margin(logits,temperature):
    if not 0<temperature<1e6:raise ValueError('Invalid calibrated temperature')
    top=(logits/temperature).softmax(1).topk(2,dim=1).values
    return top[:,0]-top[:,1]


def load_calibration(path):
    data=json.loads(Path(path).read_text())
    content={k:v for k,v in data.items() if k!='content_sha256'}
    if hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()!=data['content_sha256']:
        raise ValueError('Calibration identity changed')
    if not data['evaluation_disjoint'] or data['source_overlap']!=0:
        raise ValueError('Calibration overlaps evaluation')
    fit,tune=set(data['temperature_fit_sources']),set(data['threshold_sources'])
    if not fit or not tune or fit&tune:raise ValueError('Calibration source groups must be distinct')
    if not 0<data['temperature_fast']<1e6 or not 0<=data['threshold_margin']<=1:
        raise ValueError('Invalid calibrated values')
    if len(data['accuracy_fast_accurate'])!=2 or any(not 0<=x<=1 for x in data['accuracy_fast_accurate']):
        raise ValueError('Invalid independent model accuracy')
    return data


def wma_migrate(work, assignment, nodes=2):
    """Algorithm 2 on equal physical compute nodes, never F/A model roles.

    Equal C_n/Cbar_n makes its multiplier one. Stream work follows the stream
    when it migrates; task quality/model choice is not changed. The common GPU
    actor enforces actual memory capacity; no invented per-stream memory cost.
    """
    assigned = dict(assignment)
    if nodes != 2 or any(v not in (0, 1) for v in assigned.values()):
        raise ValueError('Expected two actual compute nodes')
    if any(v < 0 or not math.isfinite(v) for v in work.values()):
        raise ValueError('Invalid stream work')
    moves = []
    while True:
        load = [sum(w for s, w in work.items() if assigned[s] == n) for n in range(nodes)]
        donor = max(range(nodes), key=lambda n: (load[n], -n))
        receiver = 1 - donor
        streams = [s for s in work if assigned[s] == donor and work[s] > 0]
        if not streams:
            break
        stream = min(streams, key=lambda s: (work[s], s))
        if load[donor] - load[receiver] <= 2 * work[stream]:
            break
        assigned[stream] = receiver
        moves.append((stream, donor, receiver))
    return assigned, moves


class EdgePolicy:
    """Only current/past public tasks. Both physical GPUs may run either CNN."""
    def __init__(self, runtime):
        self.rt = runtime
        self.calls = 0
        self.assignments = {n: n % 2 for n in range(33)}

    def observe(self, frame, now):
        pass

    def tick(self, now):
        pass

    def effective(self, now):
        pass

    def book(self, g, batch, evidence):
        return None

    def complete(self, g, booking, actual_ns):
        pass

    def candidates(self, g):
        return [j for j in self.rt.queued.values()
                if self.assignments[self.rt.tasks[j['task_id']]['origin_node']] == g]

    def choose(self, now):
        free = [g for g, c in enumerate(self.rt.cards) if not c['busy']]
        if not free:
            return None
        choices = []
        for g in free:
            jobs = sorted(self.candidates(g), key=self.rt.batch_order_key)
            if not jobs:
                continue
            role = jobs[0]['role']
            batch = [j for j in jobs if j['role'] == role][:self.rt.cfg['batch_max']]
            choices.append((self.rt.batch_order_key(batch[0]), g, batch))
        if not choices:
            return None
        self.calls += 1
        _, g, batch = min(choices, key=lambda x: (x[0], x[1]))
        return g, batch, dict(literature_method=self.rt.arm,
                             geometry='two physical GPUs, either CNN, nonpreemptive',
                             source='current public queues only')

    def close(self):
        self.rt.log('literature', dict(event='FINAL', method=self.rt.arm,
                    calls=self.calls, complete_original_paper=False))


class WmaNodePolicy(EdgePolicy):
    """Retain CLRA quality routes; Algorithm2 migrates physical stream placement."""
    def __init__(self, runtime):
        super().__init__(runtime)
        self.pending = Counter()
        self.work = Counter()
        self.previous = 0

    def observe(self, frame, now):
        outcome = frame['outcome']
        kinds = []
        if outcome == 'GPU_READY':
            kinds = [frame['gpu_candidate_kinds'][frame['gpu_candidates'][0]]]
        elif outcome == 'PREFIX_READY':
            kinds = ['FAST_PREFIX']
        elif outcome == 'PREFIX_QUALITY_REVEALED' and frame['quality_role'] == 'Accurate':
            kinds = ['ACCURATE']
        if kinds:
            stream = self.rt.tasks[frame['task_id']]['origin_node']
            self.pending[stream] += sum(self.rt.control_unit_ns(k) for k in kinds) / 1e6

    def tick(self, now):
        dt = max(1, now - self.previous)
        a = -math.expm1(-dt / 1e9)
        for stream in self.assignments:
            rate = self.pending[stream] / (dt / 1e9)
            self.work[stream] += a * (rate - self.work[stream])
        self.pending.clear()
        self.previous = now
        before = dict(self.assignments)
        self.assignments, moves = wma_migrate(self.work, self.assignments)
        self.rt.log('literature', dict(event='WMA_MAPPING', now_ns=self.rt.now(),observed_cutoff_ns=now,
                    work_ms_s=dict(self.work), before=before,
                    assignment=self.assignments, moves=moves,
                    online_sources='past offered stage work only'))


def jellyfish_modules():
    """Import pinned author source and its unmodified real numpy/pandas helpers."""
    root = ROOT / 'literature_baselines/vendor/jellyfish'
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sim = root / 'src/simulation'
    if str(sim) not in sys.path:
        sys.path.insert(0, str(sim))
    return tuple(importlib.import_module(n) for n in ('env', 'mapping_algo', 'selection_algo'))


def jellyfish_plan(fps, budgets_ms, accuracy, latencies_ms, initial=(0, 1)):
    """Execute author SA+DP and retain actual GPU/model/batch assignments.

    Network has already happened in this edge-conditional experiment; 1 bit at
    1e12 bits/s avoids division by zero in author helpers (negligible 1e-9 ms).
    Input rates are observed, never read from a future arrival schedule.
    """
    import numpy as np
    env, mapping, selection = jellyfish_modules()
    data = SimpleNamespace(num_clients=len(fps), num_models=2, max_batch_size=10,
        batch_size=list(range(1, 11)), num_gpus=2, fps_lcd=5,
        effectiveness_threshold=1., fps=np.asarray(fps, dtype=int),
        slo=np.asarray(budgets_ms), bw=np.full(len(fps), 1e12),
        frame_s=np.ones(2), acc_m=np.asarray(accuracy), lat_m=np.asarray(latencies_ms),
        throughput_m=np.asarray([[1000 * (b + 1) / t for b, t in enumerate(row)]
                                 for row in latencies_ms]))
    clients = {'client_' + str(i): env.Client(data, i) for i in range(len(fps))}
    models = {}
    for gpu, number in enumerate(initial):
        m = env.DNNModel(data, model_number=number, gpu_number=gpu)
        models[m.id] = m
    with contextlib.redirect_stdout(io.StringIO()):
        result, _ = selection.ModelSelectionSimulatedAnnealing(
            data, models, clients, mapping.DP_on_AggregateRate, debug=False)
    targets = {int(c.split('_')[1]): (None if m is None else
               dict(gpu=int(m.gpu_number), model=int(m.model_number),
                    batch=max(1, int(m.batch_count)))) for c, m in result.clients_map.items()}
    model_numbers = list(initial)
    for m in result.models_info.values():
        model_numbers[int(m.gpu_number)] = int(m.model_number)
    return targets, model_numbers


def jellyfish_worker(requests, replies, epoch, folder, cpu):
    """Separate CPU planner as in the author serving architecture; count it."""
    import os
    import traceback
    from live_batch_runtime import cpu_snapshot,save_process_cpu
    os.sched_setaffinity(0,{cpu})
    try:
        jellyfish_modules()
        import numpy as np
        random.seed(700101);np.random.seed(700101)
        replies.put(dict(type='READY'))
        while not epoch.value:time.sleep(.001)
        while time.monotonic_ns()<epoch.value:time.sleep(.0005)
        baseline=cpu_snapshot()
        while True:
            request=requests.get()
            if request is None:break
            before=time.process_time_ns()
            targets,models=jellyfish_plan(**request['arguments'])
            replies.put(dict(type='PLAN',targets=targets,models=models,
                submitted_ns=request['submitted_ns'],cpu_ns=time.process_time_ns()-before))
    except BaseException:
        replies.put(dict(type='ERROR',error=traceback.format_exc()));raise
    finally:
        replies.close();replies.join_thread()
        if 'baseline' in locals():save_process_cpu(folder,'literature_planner',baseline)


class JellyfishEdgePolicy(EdgePolicy):
    def __init__(self, runtime):
        super().__init__(runtime)
        self.calibration = load_calibration(runtime.cfg['literature_calibration'])
        if not self.calibration['evaluation_disjoint']:
            raise ValueError('Independent source calibration required')
        self.counts = Counter()
        self.budgets = {n: deque(maxlen=256) for n in range(33)}
        self.previous = 0
        self.models = [0, 1]
        self.mapping = {n: dict(gpu=n % 2, model=n % 2, batch=10) for n in range(33)}
        ctx=mp.get_context('spawn');self.requests=ctx.Queue();self.replies=ctx.Queue()
        self.worker=ctx.Process(target=jellyfish_worker,args=(self.requests,self.replies,
            runtime.epoch,str(runtime.output),runtime.cfg['literature_planner_cpu']))
        self.worker.start();runtime.processes.append(self.worker)
        ready=self.replies.get(timeout=60)
        if ready['type']!='READY':raise RuntimeError(ready)
        self.inflight=False

    def input_kind(self, frame, now):
        stream = self.rt.tasks[frame['task_id']]['origin_node']
        self.counts[stream] += 1
        self.budgets[stream].append(max(0, self.rt.tasks[frame['task_id']]['deadline_ns'] -
                                       now - self.rt.cfg['network_return_ns']) / 1e6)
        mapping = self.mapping.get(stream)
        return None if mapping is None else ('FAST_FULL' if mapping['model'] == 0 else 'ACCURATE')

    def tick(self, now):
        try:reply=self.replies.get_nowait()
        except queue.Empty:reply=None
        if reply is not None:
            if reply['type']!='PLAN':raise RuntimeError(reply)
            self.inflight=False;self.mapping=reply['targets'];self.models=reply['models']
            # Pending, unstarted requests follow the completed plan. Running
            # batches keep their model, card and nonpreemptive execution.
            for job in list(self.rt.queued.values()):
                stream=self.rt.tasks[job['task_id']]['origin_node'];target=self.mapping.get(stream)
                self.rt.retarget_public_job(job,None if target is None else
                    ('FAST_FULL' if target['model']==0 else 'ACCURATE'),now)
            self.rt.log('literature',dict(event='JELLYFISH_EFFECTIVE',now_ns=self.rt.now(),
                submitted_ns=reply['submitted_ns'],planner_cpu_ns=reply['cpu_ns'],
                mapping=self.mapping,gpu_models=self.models,
                calibration_sha256=self.calibration['content_sha256']))
        if self.inflight or now - self.previous < 500_000_000:
            return
        if not any(self.counts.values()):
            self.previous=now
            return
        dt = (now - self.previous) / 1e9
        fps = [int(max(5, round(self.counts[n] / dt / 5) * 5)) if self.counts[n] else 0
               for n in range(33)]
        budgets = [sorted(self.budgets[n])[len(self.budgets[n]) // 2]
                   if self.budgets[n] else 0. for n in range(33)]
        latencies = [[self.rt.profile.mean(kind, b) / 1e6 for b in range(1, 11)]
                     for kind in ('FAST_FULL', 'ACCURATE')]
        self.requests.put(dict(submitted_ns=now,arguments=dict(fps=fps,budgets_ms=budgets,
            accuracy=self.calibration['accuracy_fast_accurate'],latencies_ms=latencies,initial=self.models)))
        self.inflight=True
        self.counts.clear(); self.previous = now
        self.rt.log('literature', dict(event='JELLYFISH_SUBMITTED', now_ns=now,
                    observed_fps=fps, observed_budgets_ms=budgets,
                    calibration_sha256=self.calibration['content_sha256']))

    def close(self):
        self.requests.put(None);self.requests.close();self.requests.join_thread()
        self.worker.join(timeout=60)
        if self.worker.exitcode!=0:raise RuntimeError('Jellyfish planner did not stop cleanly')
        self.replies.close();self.replies.join_thread()
        super().close()

    def candidates(self, g):
        jobs = []
        for j in self.rt.queued.values():
            target = self.mapping.get(self.rt.tasks[j['task_id']]['origin_node'])
            if target is not None and target['gpu'] == g:
                jobs.append(j)
        return jobs

    def choose(self, now):
        choice = super().choose(now)
        if choice is None:
            return None
        g, batch, evidence = choice
        caps = [self.mapping[self.rt.tasks[j['task_id']]['origin_node']]['batch'] for j in batch]
        return g, batch[:min(caps)], evidence


class EdgeBoostEdgePolicy(EdgePolicy):
    """Actual calibrated Fast top1-top2 margin -> same-card answer or Accurate.

    This substitutes Fast/Accurate for the paper's edge/cloud pair and retains
    the common upstream Local output. It is not Local-side EdgeBoost.
    """
    def __init__(self, runtime):
        super().__init__(runtime)
        self.assignments = {n: 0 for n in range(33)}

    def input_kind(self, frame, now):
        return 'FAST_FULL'

    def candidates(self, g):
        return list(self.rt.queued.values())


def create_policy(runtime):
    from literature_baselines.serving_baselines import (ClockworkEdgePolicy, SymphonyEdgePolicy,
                                                        ProteusEdgePolicy, SlackFitEdgePolicy,
                                                        PardEdgePolicy, RamsisEdgePolicy)
    classes = dict(wma_node=WmaNodePolicy, jellyfish_edge=JellyfishEdgePolicy,
                   edgeboost_edge=EdgeBoostEdgePolicy, clockwork_edge=ClockworkEdgePolicy,
                   symphony_edge=SymphonyEdgePolicy, proteus_edge=ProteusEdgePolicy,
                   slackfit_edge=SlackFitEdgePolicy, pard_edge=PardEdgePolicy, ramsis_edge=RamsisEdgePolicy)
    return classes[runtime.arm](runtime)
