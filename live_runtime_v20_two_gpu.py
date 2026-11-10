"""Physical CUDA/CPU actors for the T011 joint run (batch1 FP32).

This is the N10 production actor file reduced to the parts the T011 call chain
actually uses (TASK.md section 2 item 5: interface/lifecycle fixes only, both
arms share the same implementation).  The single-arm v20 experiment loop and
its historical controller imports (v22/v23/v24/v25/v26, growth, calendar
kernels, demand_ledger) are deliberately NOT loaded here: the T011 controller
must not load the old demand calendar worker, the full-history replayer or
private labels.  Everything else - the actor protocol (READY/DONE/RELEASE/
RESET/STOP), the bank boundary, the Fast prefix/suffix split, the Local
stages and the public quality heads - is byte-compatible with N10.

Env names move to T011_*: each group pins its own two GPU-actor cores, its
GPU post-Head core and its scheduler core on mutually disjoint spare CPUs.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from physical_probe import (HERE, PHASE, ASSETS, UUIDS, Heads, model, configure,
                            actor_cpu, fast_prefix, fast_suffix, local_suffix,
                            sha, write, apply_source_space_intervention,
                            build_post_intervention_preprocess, load_rows,
                            CLASSES, image_quality_features)
from quality_need_v24 import NeedHeads, quality_order, mid2_candidate

PERIOD = 50_000_000  # quota period; independent of the task deadline
DEADLINE_DEFAULT_MS = 25.  # compatibility default for frozen 25ms evidence only


def deadline_ns_of(ms):
    if not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms <= 0:
        raise ValueError('deadline must be a positive finite ms value')
    return int(round(float(ms) * 1e6))


from experiment_blur_adapter import apply_source_space_intervention  # noqa: E402,F811

CONDITIONS = ['clean', 'blur_0.55', 'blur_0.6']


def bank_prepare(n=270):
    """Preprocessed-ready input boundary; label data are separate from runtime."""
    evaluation_rows = load_rows(PHASE / 'data/splits/clra_split_manifest.csv', 'EVALUATION')[:n]
    development_rows = load_rows(PHASE / 'data/splits/clra_split_manifest.csv', 'DEVELOPMENT')[:20]
    rows = [('EVALUATION', r) for r in evaluation_rows] + [('DEVELOPMENT', r) for r in development_rows]
    hs = Heads()
    assert not ({r['source_image_id'] for r in evaluation_rows} & hs.fit_sources), \
        'Head train/evaluation identity overlap'
    prep = build_post_intervention_preprocess('resnet18')
    tensors = []
    raw = []
    meta = []
    labels = {}
    for condition in CONDITIONS:
        kind, level = ('clean', 0.) if condition == 'clean' else \
            (condition.rsplit('_', 1)[0], float(condition.rsplit('_', 1)[1]))
        for split, r in rows:
            with Image.open(ASSETS / 'datasets/NEU-DET' / r['original_path']) as f:
                im = apply_source_space_intervention(f.convert('RGB'), kind, level)
                tensors.append(prep(im))
                raw.append(np.asarray(im).copy())
                idx = len(tensors) - 1
                meta.append({'bank_index': idx, 'source': r['source_image_id'],
                             'condition': condition, 'split': split})
                labels[str(idx)] = CLASSES.index(r['class_label'])
    if len({x.shape for x in raw}) != 1:
        raise RuntimeError('Raw image dimensions differ; no silent padding')
    bank = torch.stack(tensors).share_memory_()
    rawbank = torch.from_numpy(np.stack(raw)).share_memory_()
    write(HERE / 'evidence/evaluation_private_labels.json', labels)
    write(HERE / 'evidence/input_bank_metadata.json',
          {'evaluation_source_count': len(evaluation_rows),
           'calibration_development_sources': len(development_rows), 'rows': meta,
           'head_fit_overlap': 0, 'preprocessing_in_timed_scope': False,
           'runtime_label_reads': False})
    return bank, rawbank, meta, labels


class GpuEarlyHeads:
    """Original EARLY quality/Need coefficients, evaluated on resident features."""
    def __init__(self, device):
        heads, needs = Heads(), NeedHeads()
        self.device = device
        self.roles = tuple(heads.roles['EARLY'])
        def coefficients(d):
            if d.get('model_kind') == 'constant_model':
                return torch.tensor(d['constant_probability'], dtype=torch.float64, device=device)
            return {k: torch.tensor(d[k], dtype=torch.float64, device=device)
                    for k in ('mean', 'std', 'beta')}
        self.coefficients = {(role, kind): coefficients(heads.models['EARLY', role, kind])
                             for role in self.roles for kind in ('base', 'calibrator')}
        self.need = coefficients(needs.data['models']['EARLY'])
        self.one = torch.ones(1, dtype=torch.float64, device=device)

    def logistic(self, coefficients, x):
        if not isinstance(coefficients, dict):
            return coefficients
        xx = torch.cat((self.one, (x - coefficients['mean']) / coefficients['std']))
        return torch.sigmoid(torch.clamp(xx @ coefficients['beta'], -35, 35))

    def predict(self, pooled):
        vector = pooled.flatten().to(torch.float64)
        probabilities = []
        for role in self.roles:
            p = self.logistic(self.coefficients[role, 'base'], vector)
            p = torch.clamp(p, 1e-12, 1 - 1e-12)
            probabilities.append(self.logistic(self.coefficients[role, 'calibrator'],
                                               torch.log(p / (1 - p)).reshape(1)))
        probabilities = torch.stack(probabilities)
        # EARLY Need uses Fast/Accurate only; Rescue is not a Need feature.
        clipped = torch.clamp(probabilities[:2], 1e-6, 1 - 1e-6)
        need_features = torch.cat((vector, torch.log(clipped / (1 - clipped))))
        need = self.logistic(self.need, need_features)
        return torch.cat((probabilities, need.reshape(1)))


def gpu_actor(gpu, bank, commands, results, feature_queue=None, expected_device_count=None,
              gpu_quality_head=False):
    try:
        configure(actor_cpu(('T012_GPU_CPU_%d' % gpu, 'T011_GPU_CPU_%d' % gpu),
                            1 if gpu == 0 else 0))
        torch.cuda.set_device(gpu)
        from resource_axis import CARD_COUNT
        expected = CARD_COUNT if expected_device_count is None else int(expected_device_count)
        if expected != CARD_COUNT and not (expected == 1 and gpu == 0):
            raise ValueError('Only an explicit single-actor validation may change visible count')
        assert torch.cuda.device_count() == expected
        device = torch.device('cuda', gpu)
        f = model('Fast', device)
        a = model('Accurate', device)
        early_heads = GpuEarlyHeads(device) if gpu_quality_head else None
        cache = {}
        with torch.inference_mode():
            x = bank[0:1].to(device)
            for _ in range(60):
                f(x)
                a(x)
            torch.cuda.synchronize(device)
        results.put({'type': 'READY', 'resource': f'gpu:{gpu}', 'pid': os.getpid(),
                     'uuid': UUIDS[gpu],
                     'memory_bytes': torch.cuda.memory_allocated(gpu),
                     'models_resident': ['Fast', 'Accurate'],
                     'cpu_affinity': list(os.sched_getaffinity(0)),
                     'gpu_numa_node': int(os.environ.get('EDGE_GPU_NUMA_%d' % gpu,
                                                         1 if gpu == 0 else 0))})
        with torch.inference_mode():
            while True:
                j = commands.get()
                if j['kind'] == 'STOP':
                    break
                if j['kind'] == 'RELEASE':
                    cache.pop(j['task_id'], None)
                    continue
                if j['kind'] == 'RESET':
                    if cache:
                        raise RuntimeError('Reset with pending GPU activation')
                    results.put({'type': 'RESET', 'resource': f'gpu:{gpu}'})
                    continue
                before = time.perf_counter_ns()
                kind = j['kind']
                i = j['task_id']
                upload = 0
                if kind != 'FAST_SUFFIX':
                    t = time.perf_counter_ns()
                    x = bank[j['bank_index']:j['bank_index'] + 1].to(device)
                    torch.cuda.synchronize(device)
                    upload = time.perf_counter_ns() - t
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                compute_start = time.perf_counter_ns()
                head_events = None
                if kind == 'FAST_PREFIX':
                    z = fast_prefix(f, x)
                    cache[i] = z
                    value = torch.nn.functional.adaptive_avg_pool2d(z, 1).flatten(1)
                    if early_heads is not None:
                        hs, he = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        hs.record()
                        head_start = time.perf_counter_ns()
                        value = early_heads.predict(value)
                        he.record()
                        head_events = (hs, he)
                elif kind == 'FAST_SUFFIX':
                    value = fast_suffix(f, cache.pop(i))
                elif kind == 'FAST_FULL':
                    value = f(x)
                elif kind == 'ACCURATE':
                    value = a(x)
                else:
                    raise ValueError(kind)
                e.record()
                e.synchronize()
                compute_end = time.perf_counter_ns()
                cuda_ms = float(s.elapsed_time(e))
                if kind == 'FAST_PREFIX':
                    if early_heads is not None:
                        values = value.cpu().tolist()
                        payload = {'gpu_quality_head': True,
                            'probabilities': dict(zip(early_heads.roles, values[:-1])),
                            'accurate_need_probability': values[-1],
                            'head_start_ns': head_start, 'head_finish_ns': compute_end,
                            'head_cuda_ms': float(head_events[0].elapsed_time(head_events[1]))}
                    else:
                        payload = {'vector': value[0].cpu().numpy().tolist()}
                else:
                    payload = {'prediction': int(value.argmax(dim=1).cpu().item())}
                end = time.perf_counter_ns()
                if kind == 'FAST_PREFIX' and j.get('inline_fast_continuation'):
                    if early_heads is None:
                        raise RuntimeError('Inline continuation requires the GPU quality head')
                    order = quality_order(payload['probabilities'], 'B',
                        payload['accurate_need_probability'], bool(j.get('quality_need_enabled', True)))
                    role = order[0] if order else 'Abstain'
                    payload.update(inline_role=role, inline_suffix_done=False,
                        prefix_finish_ns=end, prefix_compute_end_ns=compute_end,
                        prefix_cuda_ms=cuda_ms, prefix_transfer_ns=end-compute_end)
                    if role == 'Fast':
                        tail_budget = int(j['inline_suffix_gate_ns']) + int(j['inline_return_budget_ns'])
                        if time.perf_counter_ns() + tail_budget <= int(j['deadline_abs_ns']):
                            suffix_start = time.perf_counter_ns()
                            payload.update(prefix_finish_ns=suffix_start,
                                           prefix_transfer_ns=suffix_start-compute_end)
                            ss, se = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            ss.record()
                            suffix_compute_start = time.perf_counter_ns()
                            value = fast_suffix(f, cache.pop(i))
                            se.record()
                            se.synchronize()
                            suffix_compute_end = time.perf_counter_ns()
                            prediction = int(value.argmax(dim=1).cpu().item())
                            end = time.perf_counter_ns()
                            payload.update(inline_suffix_done=True, prediction=prediction,
                                suffix_start_ns=suffix_start, suffix_compute_start_ns=suffix_compute_start,
                                suffix_compute_end_ns=suffix_compute_end, suffix_finish_ns=end,
                                suffix_cuda_ms=float(ss.elapsed_time(se)))
                            compute_end = suffix_compute_end
                        else:
                            cache.pop(i, None)
                            payload['inline_suffix_skip'] = 'remaining_deadline_insufficient'
                    else:
                        cache.pop(i, None)
                    if not payload['inline_suffix_done']:
                        end = time.perf_counter_ns()
                        payload.update(prefix_finish_ns=end, prefix_transfer_ns=end-compute_end)
                assert value.is_cuda
                if (kind == 'FAST_PREFIX' and feature_queue is not None
                        and early_heads is None
                        and j.get('async_prefix_pipeline') and not j.get('warmup')):
                    # Only the pooled vector crosses processes. The full activation
                    # stays on this GPU for the eventual same-card Suffix.
                    hj = {'job_id': j.get('decision_job_id', j['job_id']),
                          'prefix_job_id': j['job_id'], 'kind': 'FAST_DECISION',
                          'task_id': i, 'bank_index': j['bank_index'], 'gpu': gpu,
                          'vector': payload.pop('vector'), 'vector_available_ns': end,
                          'enqueue_ns': time.perf_counter_ns(), 'async_prefix_pipeline': True}
                    hj['dispatch_ns'] = hj['enqueue_ns']
                    feature_queue.put_nowait(hj)
                    payload['pipeline_head_job'] = {k: v for k, v in hj.items() if k != 'vector'}
                results.put({'type': 'DONE', 'resource': f'gpu:{gpu}', 'job': j,
                             'start_ns': before, 'finish_ns': end,
                             'compute_start_ns': compute_start,
                             'compute_end_ns': compute_end, 'cuda_ms': cuda_ms,
                             'upload_ns': upload,
                             'feature_or_result_transfer_ns': end - compute_end,
                             'cache_count': len(cache), **payload})
    except Exception as e:
        import traceback
        results.put({'type': 'ERROR', 'resource': f'gpu:{gpu}', 'error': repr(e),
                     'traceback': traceback.format_exc()})


class LocalCpuStages:
    """Same FP32 Local network split, frozen once for CPU inference.

    Freeze folds evaluation BatchNorm and removes Python module dispatch.
    The four graphs retain the original MID1/MID2 boundaries and logits.
    No prediction or quality result is cached.
    """
    def __init__(self, m):
        m = m.to(memory_format=torch.channels_last)
        self.prefix = torch.jit.freeze(torch.jit.script(m.features[:5].eval()))
        self.mid2 = torch.jit.freeze(torch.jit.script(m.features[5:9].eval()))
        def suffix(start):
            block = torch.nn.Sequential(m.features[start:], m.avgpool,
                                        torch.nn.Flatten(1), m.classifier).eval()
            return torch.jit.freeze(torch.jit.script(block))
        self.suffix5 = suffix(5)
        self.suffix9 = suffix(9)

    def suffix(self, z, mid2=False):
        return (self.suffix9 if mid2 else self.suffix5)(z)


def local_actor(node, bank, rawbank, commands, results, cpu_graph=False):
    try:
        configure(actor_cpu(('T012_LOCAL_CPU_%d' % node, 'T011_LOCAL_CPU_%d' % node),
                            node + 2))
        m = model('Local', 'cpu')
        stages_model = LocalCpuStages(m) if cpu_graph else None
        if cpu_graph:
            # Immutable tensor values are unchanged; materialize only layout,
            # before arrivals, on the actor's CPU. GPU arms keep the same bank.
            bank = bank.contiguous(memory_format=torch.channels_last)
        heads = Heads()
        need_heads = NeedHeads()
        held = None
        with torch.inference_mode():
            for _ in range(20):
                if stages_model:
                    z = stages_model.prefix(bank[0:1])
                    stages_model.suffix(z, False)
                    stages_model.suffix(stages_model.mid2(z), True)
                else:
                    m(bank[0:1])
        results.put({'type': 'READY', 'resource': f'local:{node}', 'pid': os.getpid(),
                     'cpu_affinity': list(os.sched_getaffinity(0)),
                     'cpu_graph': bool(cpu_graph),
                     'input_layout': 'channels_last' if cpu_graph else 'contiguous'})
        with torch.inference_mode():
            while True:
                j = commands.get()
                kind = j['kind']
                if kind == 'STOP':
                    break
                if kind == 'RELEASE':
                    held = None
                    continue
                if kind == 'RESET':
                    if held is not None:
                        raise RuntimeError('Reset with held Local continuation')
                    results.put({'type': 'RESET', 'resource': f'local:{node}'})
                    continue
                start = time.perf_counter_ns()
                if kind == 'LOCAL_PROBE':
                    if held is not None:
                        raise RuntimeError('Local continuation lost')
                    x = bank[j['bank_index']:j['bank_index'] + 1]
                    stages = []
                    t = time.perf_counter_ns()
                    z = stages_model.prefix(x) if stages_model else m.features[:5](x)
                    u = time.perf_counter_ns()
                    stages.append({'kind': 'LOCAL_PREFIX', 'start_ns': t, 'finish_ns': u})
                    t = time.perf_counter_ns()
                    quality = image_quality_features(Image.fromarray(rawbank[j['bank_index']].numpy()))
                    v = torch.nn.functional.adaptive_avg_pool2d(z, 1).flatten().numpy()
                    p1 = heads.predict('MID1', v, quality)
                    u = time.perf_counter_ns()
                    stages.append({'kind': 'MID1_FEATURE_HEAD', 'start_ns': t,
                                   'finish_ns': u, 'probabilities': p1})
                    t = time.perf_counter_ns()
                    need = need_heads.predict('MID1', v, quality, p1)
                    u = time.perf_counter_ns()
                    stages.append({'kind': 'ACCURATE_NEED_HEAD', 'start_ns': t, 'finish_ns': u})
                    trigger = mid2_candidate(p1)
                    held = (j['task_id'], z, False, quality)
                    payload = {'probabilities': p1, 'accurate_need_probability': need,
                               'MID2': False, 'MID2_candidate': trigger, 'stages': stages}
                elif kind == 'LOCAL_MID2':
                    if held is None or held[0] != j['task_id'] or held[2]:
                        raise RuntimeError('Invalid MID1-to-MID2 continuation')
                    stages = []
                    t = time.perf_counter_ns()
                    z = stages_model.mid2(held[1]) if stages_model else m.features[5:9](held[1])
                    u = time.perf_counter_ns()
                    stages.append({'kind': 'LOCAL_MID2', 'start_ns': t, 'finish_ns': u})
                    t = time.perf_counter_ns()
                    v = torch.nn.functional.adaptive_avg_pool2d(z, 1).flatten().numpy()
                    p = heads.predict('MID2', v, held[3])
                    u = time.perf_counter_ns()
                    stages.append({'kind': 'MID2_FEATURE_HEAD', 'start_ns': t,
                                   'finish_ns': u, 'probabilities': p})
                    t = time.perf_counter_ns()
                    need = need_heads.predict('MID2', v, held[3], p)
                    u = time.perf_counter_ns()
                    stages.append({'kind': 'ACCURATE_NEED_HEAD', 'start_ns': t, 'finish_ns': u})
                    held = (j['task_id'], z, True, held[3])
                    payload = {'probabilities': p, 'accurate_need_probability': need,
                               'MID2': True, 'MID2_candidate': False, 'stages': stages}
                elif kind == 'LOCAL_SUFFIX':
                    if held is None or held[0] != j['task_id']:
                        raise RuntimeError('Local suffix identity/affinity mismatch')
                    used_mid2 = held[2]
                    v = (stages_model.suffix(held[1], held[2]) if stages_model
                         else local_suffix(m, held[1], held[2]))
                    held = None
                    payload = {'prediction': int(v.argmax(dim=1).item()), 'MID2': used_mid2}
                else:
                    raise ValueError(kind)
                end = time.perf_counter_ns()
                results.put({'type': 'DONE', 'resource': f'local:{node}', 'job': j,
                             'start_ns': start, 'finish_ns': end, **payload})
    except Exception as e:
        import traceback
        results.put({'type': 'ERROR', 'resource': f'local:{node}', 'error': repr(e),
                     'traceback': traceback.format_exc()})


def decision_actor(commands, results, continuations=None, edge_model_only=None,
                   quality_need_enabled=True):
    """Actual CPU EARLY-head execution after the GPU feature is available."""
    try:
        configure(actor_cpu(('T012_DECISION_CPU', 'T011_DECISION_CPU'), 23))
        heads = Heads()
        need_heads = NeedHeads()
        results.put({'type': 'READY', 'resource': 'decision:0', 'pid': os.getpid(),
                     'cpu_affinity': list(os.sched_getaffinity(0))})
        while True:
            j = commands.get()
            if j['kind'] == 'STOP':
                break
            if j['kind'] == 'RESET':
                results.put({'type': 'RESET', 'resource': 'decision:0'})
                continue
            start = time.perf_counter_ns()
            p = heads.predict('EARLY', j['vector'])
            need = need_heads.predict('EARLY', j['vector'], None, p)
            end = time.perf_counter_ns()
            result = {'type': 'DECISION_DONE', 'resource': 'decision:0',
                      'job': ({k: v for k, v in j.items() if k != 'vector'}
                              if j.get('async_prefix_pipeline') else j),
                      'start_ns': start, 'finish_ns': end, 'probabilities': p,
                      'accurate_need_probability': need}
            if continuations is not None and j.get('async_prefix_pipeline') and not j.get('warmup'):
                from quality_need_v24 import quality_order
                order = quality_order(p, 'B', need, quality_need_enabled,
                                      allowed_roles=[edge_model_only] if edge_model_only else None)
                role = order[0] if order else 'Abstain'
                result['route_role'] = role
                # Publish readiness, not a GPU command: the single scheduler
                # retains deadline admission, score ownership and GPU affinity.
                continuations[role].put_nowait(result)
            else:
                results.put(result)
    except Exception as e:
        import traceback
        results.put({'type': 'ERROR', 'resource': 'decision:0', 'error': repr(e),
                     'traceback': traceback.format_exc()})


def targets(gf):
    from resource_axis import card_shares
    return list(card_shares(gf))
