"""Single idle GPU validation of the production Prefix/Head/Suffix pipeline.

This is a bounded saturated execution check, not the formal two-GPU workload
replay, allocation comparison or a 50ms deadline experiment. Both passes use
the same ordered edge tasks selected from the frozen high-load source.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import time
from collections import Counter, deque

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, row):
    Path(path).write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def telemetry(uuid):
    return subprocess.check_output(['nvidia-smi', '--id=' + uuid,
        '--query-gpu=uuid,name,memory.used,utilization.gpu,temperature.gpu,power.draw',
        '--format=csv,noheader'], text=True).strip()


def check_idle(uuid):
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    if any(line.split(',')[0].strip() == uuid for line in rows.splitlines()):
        raise RuntimeError('Selected GPU has a foreign compute context: ' + uuid)
    # Querying a specified UUID also establishes that the device exists.
    return telemetry(uuid)


def prepare_inputs(manifest_path, count):
    import torch
    from PIL import Image
    from physical_probe import ASSETS, PHASE, CHECKPOINTS, TASK17, load_rows
    from experiment_blur_adapter import apply_source_space_intervention
    from physical_probe import build_post_intervention_preprocess
    manifest = json.loads(manifest_path.read_text())
    for key in ('task_plan', 'source_trace', 'bank_manifest', 'environment'):
        item = manifest[key]
        if sha(item['path']) != item['sha256']:
            raise RuntimeError('Frozen input changed: ' + key)
    for role, path in CHECKPOINTS.items():
        if sha(path) != manifest['models_sha256'][role]:
            raise RuntimeError('Model checkpoint changed: ' + role)
    for name, digest in manifest['heads_sha256'].items():
        if sha(TASK17 / 'models' / name) != digest:
            raise RuntimeError('Quality Head changed: ' + name)
    tasks = []
    if count is not None:
        with Path(manifest['source_trace']['path']).open() as stream:
            for line in stream:
                frame = json.loads(line)
                if (frame['outcome'] in ('PREFIX_READY', 'GPU_READY')
                        and 50_000_000_000 <= frame['source_publish_ns'] < 200_000_000_000):
                    tasks.append(frame)
                    if len(tasks) == count:
                        break
        if len(tasks) != count or len({f['task_id'] for f in tasks}) != count:
            raise RuntimeError('Invalid selected workload')
    bank_metadata = {r['bank_index']: r for r in json.loads(
        Path(manifest['bank_manifest']['path']).read_text())}
    source_rows = (load_rows(PHASE / 'data/splits/clra_split_manifest.csv', 'EVALUATION')[:270]
                   + load_rows(PHASE / 'data/splits/clra_split_manifest.csv', 'DEVELOPMENT')[:20])
    sources = {r['source_image_id']: r for r in source_rows}
    indices = sorted(bank_metadata) if count is None else sorted({f['bank_index'] for f in tasks})
    bank = torch.zeros(max(indices) + 1, 3, 224, 224).share_memory_()
    preprocess = build_post_intervention_preprocess('resnet18')
    for index in indices:
        row = bank_metadata[index]
        kind, level = ('clean', 0.) if row['condition'] == 'clean' else (
            row['condition'].rsplit('_', 1)[0], float(row['condition'].rsplit('_', 1)[1]))
        with Image.open(ASSETS / 'datasets/NEU-DET' / sources[row['source']]['original_path']) as im:
            image = apply_source_space_intervention(im.convert('RGB'), kind, level).copy()
        tensor = preprocess(image)
        if hashlib.sha256(tensor.numpy().tobytes()).hexdigest() != row['tensor_sha256']:
            raise RuntimeError('Input tensor changed: ' + str(index))
        bank[index].copy_(tensor)
    return manifest, tasks, bank


def execute(tasks, gpu_commands, head_commands, results, continuations, window, direct=False):
    from quality_need_v24 import quality_order
    pending, completed, heads, components = {}, {}, {}, []
    ready_backend = deque()
    cursor, busy, sequence, peak = 0, False, 0, 0
    started = time.perf_counter_ns()

    def send(kind, frame):
        nonlocal busy, sequence
        sequence += 1
        job = {'job_id': sequence, 'task_id': frame['task_id'], 'kind': kind,
               'bank_index': frame['bank_index'], 'dispatch_ns': time.perf_counter_ns(),
               'async_prefix_pipeline': direct, 'decision_job_id': 1_000_000 + sequence}
        gpu_commands.put(job)
        busy = True

    def decision(row):
        frame = pending[row['job']['task_id']]['frame']
        i = frame['task_id']
        heads[i] = row
        if direct:
            role = row['route_role']
        else:
            order = quality_order(row['probabilities'], 'B',
                                  row['accurate_need_probability'], True)
            role = order[0] if order else 'Abstain'
        if role != 'Fast':
            gpu_commands.put({'kind': 'RELEASE', 'task_id': i})
        if role == 'Abstain':
            completed[i] = {'role': role, 'prediction': None, 'finish_ns': time.perf_counter_ns()}
            del pending[i]
        else:
            ready_backend.append((i, 'FAST_SUFFIX' if role == 'Fast' else 'ACCURATE'))

    while len(completed) < len(tasks) or busy:
        if time.perf_counter_ns() - started > 120_000_000_000:
            raise TimeoutError('Bounded pipeline pass exceeded 120s')
        progressed = False
        while True:
            try:
                row = results.get_nowait()
            except queue.Empty:
                break
            progressed = True
            if row['type'] == 'ERROR':
                raise RuntimeError(row)
            if row['type'] == 'DECISION_DONE':
                decision(row)
                continue
            if row['type'] != 'DONE':
                raise RuntimeError('Unexpected result: ' + str(row))
            busy = False
            job, i = row['job'], row['job']['task_id']
            components.append(row)
            if job['kind'] == 'FAST_PREFIX':
                if not direct:
                    at = time.perf_counter_ns()
                    head_commands.put({'kind': 'FAST_DECISION', 'job_id': 1_000_000 + job['job_id'],
                        'task_id': i, 'gpu': 0, 'bank_index': job['bank_index'],
                        'vector': row['vector'], 'vector_available_ns': row['finish_ns'],
                        'enqueue_ns': at, 'dispatch_ns': at})
            else:
                completed[i] = {'role': 'Accurate' if job['kind'] == 'ACCURATE' else 'Fast',
                                'prediction': row['prediction'], 'finish_ns': time.perf_counter_ns()}
                del pending[i]
        for output_queue in continuations.values():
            while True:
                try:
                    row = output_queue.get_nowait()
                except queue.Empty:
                    break
                progressed = True
                decision(row)
        if not busy:
            if ready_backend:
                i, kind = ready_backend.popleft()
                send(kind, pending[i]['frame'])
            elif cursor < len(tasks) and len(pending) < window:
                frame = tasks[cursor]
                cursor += 1
                pending[frame['task_id']] = {'frame': frame, 'admitted_ns': time.perf_counter_ns()}
                peak = max(peak, len(pending))
                if frame['outcome'] == 'PREFIX_READY':
                    kind = 'FAST_PREFIX'
                else:
                    role = frame['gpu_candidates'][0]
                    kind = frame['gpu_candidate_kinds'][role]
                send(kind, frame)
        if not progressed:
            time.sleep(.00005)
    elapsed = time.perf_counter_ns() - started
    overlap_heads, overlap_ns = 0, 0
    for i, head in heads.items():
        amount = sum(max(0, min(head['finish_ns'], c['compute_end_ns'])
                         - max(head['start_ns'], c['compute_start_ns']))
                     for c in components if c['job']['task_id'] != i)
        overlap_heads += amount > 0
        overlap_ns += amount
    suffix_ids = {c['job']['task_id'] for c in components if c['job']['kind'] == 'FAST_SUFFIX'}
    prefix_ids = {c['job']['task_id'] for c in components if c['job']['kind'] == 'FAST_PREFIX'}
    if not suffix_ids <= prefix_ids or pending or busy:
        raise RuntimeError('Pipeline did not close its tasks')
    return {'window': window, 'direct_GPU_CPU_queue': direct,
            'tasks': len(tasks), 'wall_ms': elapsed / 1e6,
            'tasks_per_s': len(tasks) * 1e9 / elapsed, 'peak_active_roots': peak,
            'roles': dict(Counter(r['role'] for r in completed.values())),
            'heads': len(heads), 'heads_overlapping_other_GPU_work': overlap_heads,
            'head_GPU_overlap_ms': overlap_ns / 1e6,
            'completed': completed, 'head_records': heads, 'gpu_records': components}


def reset(gpu_commands, head_commands, results):
    gpu_commands.put({'kind': 'RESET'})
    head_commands.put({'kind': 'RESET'})
    expected = {'gpu:0', 'decision:0'}
    while expected:
        row = results.get(timeout=30)
        if row['type'] != 'RESET':
            raise RuntimeError('Activation/queue reset failed: ' + str(row))
        expected.remove(row['resource'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tasks', type=int, default=256)
    args = parser.parse_args(argv)
    if not 64 <= args.tasks <= 1024:
        raise SystemExit('Validation tasks must be bounded to 64..1024')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    before = check_idle(args.gpu_uuid)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['T011_GPU_CPU_0'] = '21'
    os.environ['T011_DECISION_CPU'] = '23'
    os.sched_setaffinity(0, {19})
    import multiprocessing as mp
    import torch
    from live_runtime_v20_two_gpu import gpu_actor, decision_actor
    from quality_need_v24 import QUALITY_THRESHOLDS
    torch.set_num_threads(1)
    manifest, tasks, bank = prepare_inputs(Path(args.manifest), args.tasks)
    if QUALITY_THRESHOLDS != manifest['quality_policy']['thresholds']:
        raise RuntimeError('Quality policy differs from frozen input contract')
    save(output / 'tasks.json', tasks)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
    save(output / 'identity.json', {'git_commit': commit, 'gpu_uuid': args.gpu_uuid,
        'gpu_actors_launched': 1, 'cuda_visible_devices': args.gpu_uuid,
        'tasks_sha256': sha(output / 'tasks.json'), 'manifest_sha256': sha(args.manifest),
        'source_trace_sha256': manifest['source_trace']['sha256'],
        'NeedHead_sha256': sha(ROOT / 'evidence/QUALITY_NEED_DEVELOPMENT_MODEL.json'),
        'scope': __doc__, 'before': before, 'cpu': {'main': 19, 'gpu': 21, 'Head': 23}})
    ctx = mp.get_context('spawn')
    results, gpu_commands = ctx.Queue(), ctx.Queue()
    features = ctx.Queue(maxsize=2)
    continuations = {role: ctx.Queue(maxsize=2) for role in ('Fast', 'Accurate', 'Abstain')}
    processes = [ctx.Process(target=gpu_actor, args=(0, bank, gpu_commands, results, features, 1)),
                 ctx.Process(target=decision_actor, args=(features, results, continuations, None, True))]
    from run_t011 import acquire_shared_two_gpu_lock, _SHARED_LOCK_HANDLE
    acquire_shared_two_gpu_lock()
    try:
        check_idle(args.gpu_uuid)
        for process in processes:
            process.start()
        inventory = [results.get(timeout=300) for _ in processes]
        if any(r['type'] != 'READY' for r in inventory):
            raise RuntimeError('Actor startup failed: ' + str(inventory))
        save(output / 'actor_inventory.json', inventory)
        execute(tasks[:8], gpu_commands, features, results, continuations, 1)
        reset(gpu_commands, features, results)
        passes = {}
        for name, window, direct in [('serial', 1, False), ('legacy_async', 2, False),
                                     ('pipeline', 2, True)]:
            passes[name] = execute(tasks, gpu_commands, features, results, continuations, window, direct)
            reset(gpu_commands, features, results)
            save(output / (name + '.json'), passes[name])
            print(json.dumps({'stage': name, **{k: v for k, v in passes[name].items()
                 if k not in ('completed', 'head_records', 'gpu_records')}}, ensure_ascii=False), flush=True)
        mismatches = sum((passes['serial']['completed'][i]['role'],
                          passes['serial']['completed'][i]['prediction']) !=
                         (outcome['completed'][i]['role'], outcome['completed'][i]['prediction'])
                         for outcome in passes.values() for i in passes['serial']['completed'])
        env = json.loads(Path(manifest['environment']['path']).read_text())
        predicted_mismatches = 0
        for outcome in passes.values():
            for frame in tasks:
                done = outcome['completed'][frame['task_id']]
                if done['prediction'] is not None:
                    frozen = env['bank'][str(frame['bank_index'])][done['role'] + '_prediction']
                    predicted_mismatches += frozen != done['prediction']
        report = {'ok': mismatches == 0 and predicted_mismatches == 0,
                  'tasks': len(tasks), 'role_prediction_mismatches': mismatches,
                  'frozen_prediction_mismatches': predicted_mismatches,
                  'cache_empty_after_each_pass': True,
                  'pipeline_heads_overlapping_other_GPU_work': passes['pipeline']['heads_overlapping_other_GPU_work'],
                  'serial_wall_ms': passes['serial']['wall_ms'], 'pipeline_wall_ms': passes['pipeline']['wall_ms'],
                  'legacy_async_wall_ms': passes['legacy_async']['wall_ms'],
                  'direct_queue_wall_reduction_vs_legacy_async_percent': 100 *
                    (1 - passes['pipeline']['wall_ms'] / passes['legacy_async']['wall_ms']),
                  'wall_reduction_percent': 100 * (1 - passes['pipeline']['wall_ms'] / passes['serial']['wall_ms']),
                  'throughput_gain_percent': 100 * (passes['pipeline']['tasks_per_s'] / passes['serial']['tasks_per_s'] - 1),
                  'Suffix_input_upload_zero': all(c['upload_ns'] == 0
                      for outcome in passes.values() for c in outcome['gpu_records']
                      if c['job']['kind'] == 'FAST_SUFFIX'),
                  'after': telemetry(args.gpu_uuid),
                  'formal_high_load_four_way_run': False, 'scope': __doc__}
        save(output / 'REPORT.json', report)
        if not report['ok']:
            raise RuntimeError('Pipeline validation found output mismatches')
        print(json.dumps(report, ensure_ascii=False), flush=True)
    except BaseException as exc:
        import traceback
        save(output / 'FAILED.json', {'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        gpu_commands.put({'kind': 'STOP'})
        try:
            features.put({'kind': 'STOP'}, timeout=2)
        except queue.Full:
            # A failed CPU actor may leave a full queue; cleanup must not hang.
            pass
        for process in processes:
            if process.pid:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
        for handle in _SHARED_LOCK_HANDLE:
            handle.close()
        _SHARED_LOCK_HANDLE.clear()


if __name__ == '__main__':
    main()
