"""RAMSIS offline policy generation with the unmodified author code (CPU only, before any run).

Author code: github.com/dmmendo/RAMSIS @9b74967, vendored in vendor/ramsis (SHA-checked). numba is not
installed; vendor/ramsis_numba_shim provides a jit that returns the Python function (same arithmetic).
Per (SLO, total query rate): VarBatchSMDP + solver.valueIteration_vectorized(max_iter=100) and the
expected accuracy / violation of solver.valueIteration_worker, with the RAMSIS_gen.py arguments
(maximal batching, prob_thresh 0, fixed-interval time grid with 100 intervals, round robin over 2
workers). See NEW_BASELINES_DERIVATION_20261021.md section 2 for every mapped value.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'literature_baselines/vendor/ramsis'
SHIM = ROOT / 'literature_baselines/vendor/ramsis_numba_shim'
AUTHOR = dict(repository='https://github.com/dmmendo/RAMSIS', commit='9b74967ff3e4f159f850d8484c252a8cff9941e3',
              files={'VarBatchSMDP.py': 'f5e4c362e9ade01e38e80a6a045c3db89c24fac90b9435521a1e9460151c859f',
                     'distribution.py': '05130c628488803d0b14c273f6140adb339fd204d8ceb4e5aa9a23b05789bae4',
                     'simulator.py': 'c8ad0c245628c189f32dfc068501f73d1affeb86ad11726be2ff95492dbc5799',
                     'solver.py': 'fc2d96bda59228e68ea94e87edcd93c4d12a7d24809dde6726c18cb506e0312e'})
PROFILE = ROOT / 'literature_baselines/executor_profile_p95_20261021.json'
MODELS = ('FAST_FULL', 'ACCURATE')    # author order: model 1 = lowest latency = default action
MAX_BATCH = 10                         # executor maximum = author N (max_batch_size)
WORKERS = 2
INTERVALS = 100                        # RAMSIS_gen.py num_interval
VALUE_ITERATIONS = 100                 # RAMSIS_gen.py valueIteration_worker num_iters
EVAL_ITERATIONS = 1000                 # solver.valueIteration_worker compute_total_value / get_F_vec
SLO_MS = tuple(range(4, 36))           # largest edge budget 50 - 8.26 - 6.5 = 35.24 ms
RATE_START = (50., 100., 200., 400., 800., 1600., 3200., 6400., 8000.)
ACCURACY_STEP = .01                    # paper Sec. 6: adjacent expected accuracies differ by < 1%
MIN_RATE_GAP = 1.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def author_modules():
    for name, digest in AUTHOR['files'].items():
        if sha(VENDOR / name) != digest:
            raise ValueError('Vendored RAMSIS file changed: ' + name)
    for path in (str(SHIM), str(VENDOR)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import VarBatchSMDP
    import solver
    return VarBatchSMDP.VarBatchSMDP, solver


def latency_us(profile):
    """[[[l(m, b)]] for b in 1..10] for m in MODELS, author units (microseconds)."""
    return [[[ns / 1e3] for ns in profile['ramsis_sizes_1_10_ns'][m]] for m in MODELS]


def solve(job):
    """One author MDP: returns the policy rows and the author expected accuracy / violation."""
    slo_ms, rate_qps, lat, accuracy = job
    import numpy as np
    VarBatchSMDP, solver = author_modules()
    lat = np.array(lat)
    smdp = VarBatchSMDP(rate=rate_qps / 1e6, SLO=slo_ms * 1e3, model_inf_time=lat,
                        model_inf_time_probmass=np.ones_like(lat), model_accuracy=np.array(accuracy),
                        resp_time=None, arrival_dist='erlang', num_inst=WORKERS, max_batch_size=MAX_BATCH,
                        max_action_batch=None, prob_thresh=0., fixed_interval_time=True, num_interval=INTERVALS)
    if smdp.N != MAX_BATCH or len(smdp.resp_time) != INTERVALS + 1:
        raise ValueError('Unexpected author state space')
    _, pi = solver.valueIteration_vectorized(smdp, max_iter=VALUE_ITERATIONS)
    expected_accuracy = float(solver.compute_total_value(smdp, pi, num_iters=EVAL_ITERATIONS))
    expected_violation = float(np.sum(solver.get_F_vec(pi, smdp, max_iter=EVAL_ITERATIONS)))
    rows = []
    for n in range(1, MAX_BATCH + 1):
        row = ''
        for idx in range(len(smdp.resp_time)):
            model, served = pi[(n, idx)]
            if served != n or model not in (1, 2):
                raise ValueError('Expected maximal batching over two models')
            row += str(model)
        rows.append(row)
    if pi[(MAX_BATCH + 1, 0)] != (1, MAX_BATCH + 1):
        raise ValueError('Overflow state must take the author default action')
    return dict(slo_ms=slo_ms, rate_qps=rate_qps, expected_accuracy=expected_accuracy,
                expected_violation=expected_violation, actions=rows,
                grid_us=[float(x) for x in smdp.resp_time])


def refine(done):
    """Midpoints where adjacent expected accuracies differ by >= 1% (per SLO)."""
    new = []
    for slo in SLO_MS:
        rows = sorted((r for r in done if r['slo_ms'] == slo), key=lambda r: r['rate_qps'])
        for a, b in zip(rows, rows[1:]):
            if (abs(a['expected_accuracy'] - b['expected_accuracy']) >= ACCURACY_STEP and
                    b['rate_qps'] - a['rate_qps'] > MIN_RATE_GAP):
                new.append((slo, (a['rate_qps'] + b['rate_qps']) / 2))
    return new


def build(calibration_path, processes):
    from multiprocessing import Pool
    from literature_baselines.current_literature import load_calibration
    calibration = load_calibration(calibration_path)
    profile = json.loads(PROFILE.read_text())
    body = {k: v for k, v in profile.items() if k != 'content_sha256'}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != profile['content_sha256']:
        raise ValueError('p95 profile identity changed')
    lat = latency_us(profile)
    accuracy = list(calibration['accuracy_fast_accurate'])
    author_modules()
    todo = [(slo, rate) for slo in SLO_MS for rate in RATE_START]
    done, rounds = [], 0
    with Pool(processes) as pool:
        while todo:
            rounds += 1
            done += pool.map(solve, [(slo, rate, lat, accuracy) for slo, rate in todo], chunksize=1)
            todo = refine(done)
    policies = {}
    grids = {}
    for row in sorted(done, key=lambda r: (r['slo_ms'], r['rate_qps'])):
        grids[str(row['slo_ms'])] = row.pop('grid_us')
        policies.setdefault(str(row.pop('slo_ms')), []).append(row)
    data = dict(schema='ramsis_policies_v1', author=AUTHOR, numba='pure-Python jit stand-in (numba not installed)',
                arguments=dict(models=list(MODELS), max_batch=MAX_BATCH, workers=WORKERS, intervals=INTERVALS,
                               value_iterations=VALUE_ITERATIONS, eval_iterations=EVAL_ITERATIONS,
                               maximal_batching=True, prob_thresh=0., fixed_interval_time=True,
                               slo_ms=list(SLO_MS), rate_start_qps=list(RATE_START),
                               accuracy_step=ACCURACY_STEP, refinement_rounds=rounds),
                accuracy_fast_accurate=accuracy, calibration_sha256=calibration['content_sha256'],
                latency_profile_sha256=profile['content_sha256'], latency_us=[[x[0] for x in m] for m in lat],
                grid_us=grids, policies=policies)
    data['content_sha256'] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return data


def load(path):
    data = json.loads(Path(path).read_text())
    body = {k: v for k, v in data.items() if k != 'content_sha256'}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != data['content_sha256']:
        raise ValueError('RAMSIS policy identity changed')
    if data['schema'] != 'ramsis_policies_v1' or data['arguments']['max_batch'] != MAX_BATCH:
        raise ValueError('Unexpected RAMSIS policy table')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration', required=True, help='frozen literature calibration.json')
    p.add_argument('--output', default=str(ROOT / 'literature_baselines/ramsis_policies_20261021.json'))
    p.add_argument('--processes', type=int, default=os.cpu_count())
    a = p.parse_args()
    data = build(a.calibration, a.processes)
    Path(a.output).write_text(json.dumps(data, sort_keys=True) + '\n')
    counts = {slo: len(rows) for slo, rows in data['policies'].items()}
    print(json.dumps(dict(output=a.output, content_sha256=data['content_sha256'], policies=sum(counts.values()),
                          rounds=data['arguments']['refinement_rounds'], per_slo=counts)))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    main()
