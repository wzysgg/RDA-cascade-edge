"""Run one method of the paper on one frozen scene with the real two-GPU executor.

    python run_experiment.py --method RDA --manifest <scene>/manifest.json --output <dir>
    python run_experiment.py --method PARD --calibration <calibration.json> --manifest ... --output ...

Methods
  RDA                           realizable-density allocation (all seven components)
  static                        one card per role, fixed (static allocation of Section 3.8)
  initial_dynamic               demand-proportional shares, EDF batches, drop by offline time
  RA, RA_FKA, RA_FKA_RT_FPS,    cumulative ladder of Table 5 (each adds components)
  RA_FKA_RT_FPS_DBP_LFD (= RDA without SON)
  RDA_without_FPS / _RT / _LFD  leave-one-out runs
  RDA_q025 ... RDA_f256         one-factor sensitivity runs of Fig. 11
  EdgeServing, PARD, RAMSIS, Proteus, SlackFit   edge adaptations of the baselines (Section 3.3)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Paper component -> executor switches.
COMPONENTS = dict(
    RA=dict(realizable_admission=True, per_card_need=True),   # realizable admission
    FKA=dict(overload_unit_cost_allocation=True),             # fluid knapsack allocation
    FPS=dict(urgency_swap=True),                              # feasibility-preserving swap
    RT=dict(need_from_terminal=True),                         # return-side timing
    DBP=dict(density_packing=True),                           # density batch packing
    LFD=dict(priority_deferral=True),                         # latest feasible dispatch
    SON=dict(safe_optimistic_need=True))                      # safe-optimistic required time
RDA = ('RA', 'FKA', 'FPS', 'RT', 'DBP', 'LFD', 'SON')

LADDER = {'initial_dynamic': (),
          'RA': ('RA',),
          'RA_FKA': ('RA', 'FKA'),
          'RA_FKA_RT_FPS': ('RA', 'FKA', 'FPS', 'RT'),
          'RA_FKA_RT_FPS_DBP': ('RA', 'FKA', 'FPS', 'RT', 'DBP'),
          'RA_FKA_RT_FPS_DBP_LFD': ('RA', 'FKA', 'FPS', 'RT', 'DBP', 'LFD'),
          'RDA': RDA}
LADDER.update({'RDA_without_' + c: tuple(x for x in RDA if x != c) for c in ('FPS', 'RT', 'LFD')})

# Defaults: median (quantile 0.5), windows 256 / 256 / 512, DBP candidate cap M = 64.
KNOBS = {'q025': dict(need_quantile=.25), 'q075': dict(need_quantile=.75),
         'w025': dict(estimation_window_scale=.25), 'w4': dict(estimation_window_scale=4.),
         'f16': dict(density_packing_frontier=16), 'f256': dict(density_packing_frontier=256)}

BASELINES = dict(EdgeServing='edge_serving', PARD='pard_edge', RAMSIS='ramsis_edge',
                 Proteus='proteus_edge', SlackFit='slackfit_edge')
RAMSIS_POLICIES = 'literature_baselines/ramsis_policies_20261021.json'


def rda_config(base, components, extra=None):
    """Shared executor configuration plus the switches of the given components."""
    cfg = dict(base, clra_variant='C0', capacity_confirmed_window=False, marginal_max_roots=512,
               marginal_public_upload_ns=6500000, realizable_admission=False, per_card_need=False,
               overload_unit_cost_allocation=False, balanced_card_shares=False)
    for c in components:
        cfg.update(COMPONENTS[c])
    cfg.update(extra or {})
    return cfg


def baseline_config(base, method, calibration):
    """Shared executor configuration plus the switches of one baseline (or the static allocation)."""
    cfg = dict(base, clra_variant='C0', capacity_confirmed_window=False,
               fixed_resource_shares=method == 'static', literature_calibration=str(calibration),
               literature_planner_cpu=5)
    arm = BASELINES.get(method)
    if arm not in (None, 'edge_serving'):
        cfg['literature_method'] = arm
    if arm == 'pard_edge':
        cfg['pard_prefix_path'] = 'max'      # prefix requests take the larger downstream estimate
    if arm == 'ramsis_edge':
        cfg['ramsis_policies'] = RAMSIS_POLICIES
    return cfg


def method_config(base, method, calibration=None):
    """(executor arm, configuration) of a method name."""
    if method in LADDER:
        return 'ours', rda_config(base, LADDER[method])
    if method.startswith('RDA_') and method[4:] in KNOBS:
        return 'ours', rda_config(base, RDA, KNOBS[method[4:]])
    if method == 'static' or method in BASELINES:
        if calibration is None:
            raise ValueError('baselines need --calibration (offline service-time profile)')
        return ('ours' if method == 'static' else BASELINES[method]), baseline_config(base, method, calibration)
    raise ValueError('unknown method: ' + method)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--method', required=True)
    p.add_argument('--manifest', required=True, help='frozen scene (prepared Local trace and arrivals)')
    p.add_argument('--output', required=True)
    p.add_argument('--calibration', help='offline service-time profile used by the baselines')
    p.add_argument('--deadline-ms', type=float, help='deadline of the scene (default 50 ms)')
    a = p.parse_args()
    base = json.loads((ROOT / 'config/live_gpu_batch.json').read_text())
    arm, cfg = method_config(base, a.method, a.calibration)
    if a.deadline_ms is not None:
        cfg['deadline_ns'] = round(a.deadline_ms * 1e6)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    path = out / (a.method + '_config.json')
    path.write_text(json.dumps(cfg, indent=1))
    subprocess.run([sys.executable, '-u', '-B', str(ROOT / 'tools/run_live_batch.py'), '--arm', arm,
                    '--manifest', a.manifest, '--output', str(out / a.method), '--config', str(path)],
                   cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
