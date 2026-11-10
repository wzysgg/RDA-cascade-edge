"""T014 implementation entry point: shared Local front end + one joint concurrent
two-arm run (closed-loop repair revision QF-4R).

This module keeps the N11 file name; the T014 thin entry is ``run_t012.py``.

Modes
  --prepare-only : validate the COPIED frozen 250 s T004_mixed_v1 plan under
                   <HERE>/inputs (never regenerates it), migrate the scenario
                   identity/paths to N12 and to controller_version qf4r, write the
                   N12 input manifest recording the original input identity.
  --preflight    : read-only four-GPU + CPU-topology + public-lock + own-process
                   preflight (nvidia-smi / lscpu / sysfs); keeps the N10 card
                   split when those four UUIDs are free (no compute process, idle
                   memory/utilisation), proves model/memory match, resolves the 11
                   key physical cores and the shared Local pool.  Never
                   initialises CUDA and never acquires the public lock.
  --check        : run the ONE T014 CPU check suite (checks_t012.py) ->
                   evidence/cpu_checks.json.
  --freeze       : after the CPU gate has passed, freeze the online dependency
                   manifest (sources, models, evidence, inputs) to
                   evidence/manifest.json.
  --deploy-audit : TASK s8 deployment gate: every per-file hash frozen in
                   evidence/manifest.json must still match on disk, the freeze
                   manifest must be the T014 schema, evidence/REFERENCE.json must
                   be the T014 identity document, evidence/cpu_checks.json must
                   pass, the four T014 entry files must exist and no child-side
                   source may acquire the shared flock.  Refuses on any mismatch
                   (it lists them all, never stops at the first).
  --joint-run    : ONE joint run: the shared Local front end injects the frozen
                   plan once and both GPU groups (fixed_1a1f, qf4r_feedback) run
                   250 s CONCURRENTLY against one common t0/barrier.  Each arm
                   runs its ONE-START warmup calibration before the barrier
                   release; the launch registration happens BEFORE any actor is
                   created.
  --joint-facts  : read the newest joint run and write the drain/wall/budget facts
                   the detached driver copies into RUN_COMPLETE.json (no GPU).
  --verify       : offline re-verification of a finished joint run (no GPU).
  --summarize    : offline one-pass analysis of the joint run -> evidence/metrics.json.

Both arms receive the same --input-plan file; the shared source re-checks its byte
SHA256, its logical plan SHA256 and the arrival count before touching any GPU.
implementation_revision is T014_qf4r_3; the arm names carry T014
(TRIAL_T014_mixed_d50_n20_blur0.55_<policy>).
The dynamic arm is qf4r_feedback; the fixed arm is fixed_1a1f (T014 TASK / REFERENCE.json).
The joint parent never creates a CUDA context: it pops CUDA_VISIBLE_DEVICES and each
GPU group sets its own mutually disjoint mask inside its own spawned process.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUTS = HERE / 'inputs'
EVIDENCE = HERE / 'evidence'
LOGS = HERE / 'logs'
RUNS = HERE / 'runs'

# The single shared two-GPU lock used by every v20-era run.  For T014 the joint
# start parent holds it EXCLUSIVELY for the whole joint run (four GPUs plus the
# shared Local front end) and for the wrap-up; the two child arms and the shared
# Local actors must never re-acquire it (TASK s8 item 1).
SHARED_TWO_GPU_LOCK = os.environ.get(
    'T014_SHARED_LOCK',
    os.environ.get(
        'T011_SHARED_LOCK',
        '/home/wangzengyu/edge_paper/experiments/v20_two_gpu_validation_20261004/.live_two_gpu.lock'))
_SHARED_LOCK_HANDLE = []


def _fcntl():
    """Linux flock support.  Windows has no fcntl: say so instead of pretending."""
    try:
        import fcntl
    except ImportError:
        return None
    return fcntl


def acquire_shared_two_gpu_lock():
    fcntl = _fcntl()
    if fcntl is None:
        raise SystemExit('fcntl is unavailable on this platform: the shared two-GPU lock '
                         'can only be held by the T014 joint parent on the Linux server')
    path = Path(SHARED_TWO_GPU_LOCK)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    handle = open(str(path), 'a+')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise SystemExit('shared two-GPU lock is held by another run: %s' % path)
    _SHARED_LOCK_HANDLE.append(handle)
    return {'path': str(path), 'held_by': 'joint_start_parent', 'pid': os.getpid(),
            'scope': 'four GPUs and the shared Local front end',
            'held_through': 'joint run and wrap-up',
            'children_must_not_reacquire': True}


IMPLEMENTATION_REVISION = 'T016_qf4r_continuous_resource_36'
TASK_ID = 'T016'
ITERATIVE_RESEARCH = True
ALGORITHM_REVISION = 'QF-4R'
EXPERIMENT_BOUNDARY = 'shared_local_frontend_gpu_control'
CONTROLLER_VERSION = 'qf4r'
POLICIES = ('fixed_1a1f', 'qf4r_feedback')
FIXED_POLICY = 'fixed_1a1f'
DYNAMIC_POLICY = 'qf4r_feedback'

PLANNED_COMPLETE_JOINT_RUNS = None
ARMS_PER_JOINT = 2
MAX_JOINT_ATTEMPTS = None
MAX_EARLY_MECHANICAL_REPAIRS = None
MAX_GPU_ACTOR_STARTS = None
GPU_ACTORS_PER_ARM = 2
SHARED_LOCAL_NODES = 20
#: T016 explicitly authorizes diagnosis, repair and another complete attempt
#: after a failed or low-benefit run.  Each attempt preserves both arms and its
#: frozen identity; total starts are recorded without an invented upper limit.
MECHANICAL_REPAIR_SOURCE_TIME_LIMIT_S = None
#: legacy name kept for the frozen manifest fields only
EARLY_MECHANICAL_SOURCE_TIME_LT_S = MECHANICAL_REPAIR_SOURCE_TIME_LIMIT_S
#: TASK s2: both arms plus the shared Local front end must be fully drained
#: within this wall budget after the joint source ends.
DRAIN_MAX_S = 60.0
DRIVER_WALL_MAX_S = 900.0
CPU_WALL_MAX_S = 180.0
CPU_GROUPS = ('R1', 'R2', 'R3')

REQUIRED_TOTAL_IDLE_GPUS = 4
GPU_ACTORS_PER_GROUP = 2
IDLE_MEMORY_MAX_MIB = 1024
IDLE_UTILIZATION_MAX = 20
#: a busy card has a live compute process; TASK s1 forbids occupying one
ALLOWED_COMPUTE_PROCESSES_ON_SELECTED = 0

BARRIER_LEAD_NS = 2_000_000_000

# TASK s1: preferred N10 split (fixed / dynamic pairs), kept when free.
PREFERRED_FIXED_UUIDS = ['GPU-d1b82bfe-73db-7d85-16fa-2e38208c0064',
                         'GPU-7e400790-5792-8095-35cd-42d68b8eeffd']
PREFERRED_DYNAMIC_UUIDS = ['GPU-93b2459d-3feb-d3b1-2322-4228aa649091',
                           'GPU-47aa43ef-6eb8-86a5-7f64-41c009bda4c9']
DEFAULT_GPU_PAIRS = (tuple(PREFERRED_FIXED_UUIDS), tuple(PREFERRED_DYNAMIC_UUIDS))
GPU_PAIR_ORDER = 'standard'
GPU_MODEL_EXPECTED = 'NVIDIA GeForce RTX 2080 Ti'

# TASK s8 CPU topology: the server is 48 logical = 24 physical cores.  The 11 key
# processes get 11 INDEPENDENT physical cores (never shared, never an SMT sibling
# of another key core); the shared 20 Local actors live on the remaining 13
# physical cores and their SMT siblings as one common pool.  The mapping is
# resolved from the REAL topology (lscpu / sysfs / affinity) at joint-run time;
# there is no hardcoded core list any more.  If the 24-physical-core premise does
# not hold (or the GPUs cannot be honoured on their own NUMA node) the run is
# refused with the measured topology conflict instead of pretending.
EXPECTED_LOGICAL_CPUS = 48
EXPECTED_PHYSICAL_CORES = 24
EXPECTED_SMT_THREADS_PER_CORE = 2
KEY_PHYSICAL_CORES = 11
LOCAL_POOL_PHYSICAL_CORES = 13
ACTIVE_CPU_MODE = 'local_first16'
ACTIVE_KEY_PHYSICAL_CORES = 8
ACTIVE_LOCAL_PHYSICAL_CORES = 16
#: per arm: 2 GPU actors + 1 main scheduler + 1 post-Prefix Head + 1 candidate worker
ARM_KEY_ROLES = ('gpu0', 'gpu1', 'main', 'decision', 'replay')
CPU_AFFINITY_RULE = (
    'resolved from the real topology: 11 independent physical cores for the 4 GPU actors, '
    'the 2 arm main schedulers, the 2 post-Prefix Heads, the 1 common source (joint parent) '
    'and the 2 candidate replay workers; each GPU actor on the NUMA node of its own card; '
    'the two arms structurally peer; the shared 20 Local actors on the remaining 13 physical '
    'cores and their SMT siblings as one common pool that never contains a key core')

INPUT_IDENTITY = {
    'baseline_task_id': 'T004',
    'schema_version': 'T004_mixed_v1',
    'seed': 7904001,
    'plan_file_sha256': '0b0fd1237c807f3bc2b43fcdc381327326ccf25678f64edd1070b69000ebfde9',
    'plan_logical_sha256': 'ebd575740a29eb9c3923c51eeb47a7b0654f4e29e7bd2fe049d6fe10d37a9eaa',
    'total_arrivals': 226323,
    'phase_arrivals': [63313, 108395, 54615],
    'phase_blurred': [6309, 48534, 5446],
    'affected_node_arrivals': [31729, 53917, 27253],
}

FROZEN_EVIDENCE_FILES = [
    'cpu_probe.json', 'FROZEN_DEVELOPMENT_CALIBRATION.json',
    'PAST_GPU_TIMING_PRIOR.json', 'NEW_BANK_REFERENCE_ACCEPTANCE.json',
    'input_bank_metadata.json', 'QUALITY_NEED_DEVELOPMENT_MODEL.json',
    'QUALITY_NEED_BANK_REFERENCE.json', 'resource_independent_quality_demand.json',
    'evaluation_private_labels.json', 'provenance.json',
]

QF4_ALGORITHM_SOURCES = (
    'resource_axis.py',
    'queue_feedback_stats.py', 'queue_feedback_controller.py',
    'candidate_replay_t012.py',
    'shared_local_t011.py', 'group_runtime_t011.py',
    'live_runtime_v20_two_gpu.py', 'timeline_protocol.py', 'quota_forecast_v8.py',
    'quality_need_v24.py', 'execution_guards.py', 'physical_probe.py',
    'experiment_blur_adapter.py', 'partial_scenario.py', 'past_timing_budgets.py',
    'mixed_scenario.py',
)

OFFLINE_TOOLS = ('analysis_t012.py', 'checks_t012.py')

SOURCE_POLICY_READ_FORBIDDEN = (
    'gpus[', 'ready[', 'self.controller', 'control_step', 'command_id',
    'effective_fast', 'target_fast', 'quota',
)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


from mixed_scenario import (  # noqa: E402
    T004_SCHEMA,
    T004_SCENARIO,
    logical_plan_sha,
    run_name,
)
import mixed_scenario as _scenario  # noqa: E402
_scenario.RUN_LABEL = TASK_ID
from timeline_protocol import (  # noqa: E402
    T004_EXPECTED_ARRIVALS,
    T004_EXPECTED_PLAN_SHA256,
)
try:  # T014 s1 identity validator (timeline_protocol was extended by the Lead)
    from timeline_protocol import validate_t012_mixed  # noqa: E402
    MIXED_VALIDATOR = 'validate_t012_mixed'
except ImportError:  # pragma: no cover - older protocol copy
    from timeline_protocol import validate_t011_mixed as validate_t012_mixed  # noqa: E402
    MIXED_VALIDATOR = 'validate_t011_mixed'


def utcnow():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
                    encoding='utf8')


def frozen_sources():
    online = {}
    offline = {}
    for pattern in ('*.py', '*.sh'):
        for p in sorted(HERE.glob(pattern)):
            record = {'sha256': sha(p), 'size': p.stat().st_size}
            (offline if p.name in OFFLINE_TOOLS else online)[p.name] = record
    ref_path = Path(os.environ.get('TASK_REFERENCE_JSON')
                    or os.environ.get('T014_REFERENCE_JSON')
                    or (EVIDENCE / 'REFERENCE.json'))
    reference = {'path': str(ref_path), 'override_env': 'TASK_REFERENCE_JSON',
                 'sha256': sha(ref_path) if ref_path.exists() else None,
                 'size': ref_path.stat().st_size if ref_path.exists() else None}
    return {
        'schema_version': T004_SCHEMA,
        'task_id': TASK_ID,
        'implementation_revision': IMPLEMENTATION_REVISION,
        'frozen_before_first_formal_joint_run': True,
        'note': 'SHA256 of every T014 python/shell source that runs online; the offline-only analysis tools are listed separately and may be fixed and rerun offline without touching these hashes',
        'files': online,
        'online_sources': online,
        'offline_tools': offline,
        'effective_reference': reference,
    }


def plan_stats_from_summary():
    return load_json(INPUTS / 'plan_summary.json')


# --------------------------------------------------------------------------
# --prepare-only
# --------------------------------------------------------------------------
def install_reference_if_requested():
    """TASK s2: the T014 REFERENCE.json is issued in the task mailbox, not in this
    repository.  The deploy step may point T014_REFERENCE_JSON at that file.

    The task document carries the T014 expected values and identity but NOT the
    frozen N10 cost tables (``c_ref_ms`` / ``common_source_intrinsic_ms``) nor the
    structural ``configuration`` / ``experiment_input`` that ``SharedGroupRuntime``
    and the qf4r factory read from this same file.  Overwriting the frozen
    document would delete them, so the install MERGES: the frozen document already
    on disk is the base and the task document wins on every key it defines.
    With the env var unset nothing on disk is touched."""
    target = EVIDENCE / 'REFERENCE.json'
    src = str(os.environ.get('TASK_REFERENCE_JSON')
              or os.environ.get('T014_REFERENCE_JSON') or '').strip()
    if not src:
        return {'installed': False, 'env': 'TASK_REFERENCE_JSON',
                'path': str(target),
                'sha256': sha(target) if target.exists() else None}
    src_path = Path(src)
    if not src_path.exists():
        raise SystemExit('TASK_REFERENCE_JSON does not exist: %s' % src_path)
    task_doc = load_json(src_path)
    src_sha = sha(src_path)
    frozen = load_json(target) if target.exists() else {}
    merged = dict(frozen)
    merged.update(task_doc)
    alg = task_doc.get('algorithm') if isinstance(task_doc.get('algorithm'), dict) else {}
    cfg = dict(frozen.get('configuration') or {})
    cfg.update(task_doc.get('configuration') or {})
    cfg.update(alg)  # the current authority for every knob it names (qf4r identity)
    merged['configuration'] = cfg
    if 'experiment_input' not in merged and 'experiment_input' in frozen:
        merged['experiment_input'] = frozen['experiment_input']
    dump_json(target, merged)
    return {'installed': True, 'mode': 'merge_with_frozen',
            'source': str(src_path.resolve()), 'path': str(target),
            'sha256': src_sha, 'merged_sha256': sha(target),
            'frozen_only_keys': sorted(set(frozen) - set(task_doc)),
            'configuration_controller_version': cfg.get('controller_version')}


def cmd_prepare(args):
    """Validate the copied plan; migrate identity/paths to N12/qf4r; write the N12 input manifest."""
    plan_path = INPUTS / 'arrival_plan.json'
    if not plan_path.exists():
        raise SystemExit('inputs/arrival_plan.json missing; copy it from the N11 baseline, do not regenerate')
    file_sha = sha(plan_path)
    plan = load_json(plan_path)
    logical = logical_plan_sha(plan)
    if file_sha != INPUT_IDENTITY['plan_file_sha256']:
        raise SystemExit('arrival_plan file SHA mismatch: %s' % file_sha)
    if logical != INPUT_IDENTITY['plan_logical_sha256']:
        raise SystemExit('arrival_plan logical SHA mismatch: %s' % logical)
    if len(plan) != INPUT_IDENTITY['total_arrivals']:
        raise SystemExit('arrival count mismatch: %d' % len(plan))
    reference_action = install_reference_if_requested()
    summary = plan_stats_from_summary()
    if int(summary.get('realized_total_tasks', -1)) != INPUT_IDENTITY['total_arrivals']:
        raise SystemExit('plan_summary realized_total_tasks mismatch')
    phases = {p['phase']: p for p in summary.get('phase_stats', [])}
    for name, key in (('background', 0), ('degraded', 1), ('recover', 2)):
        got = phases.get(name, {})
        if got.get('arrivals') != INPUT_IDENTITY['phase_arrivals'][key]:
            raise SystemExit('phase %s arrivals mismatch: %r' % (name, got.get('arrivals')))
    spec = load_json(INPUTS / 'scenario.json')
    spec.pop('rate_schedule', None)
    spec.update({
        'task_id': TASK_ID,
        'implementation_revision': IMPLEMENTATION_REVISION,
        'controller_version': CONTROLLER_VERSION,
        'algorithm_revision': ALGORITHM_REVISION,
        'experiment_boundary': EXPERIMENT_BOUNDARY,
        'policy': DYNAMIC_POLICY,
        'record_records': False,
        'local_cpu_graph': True,
        'local_quality_route_mode': 'quality_to_accurate',
        'cpu_affinity_mode': ACTIVE_CPU_MODE,
        'input_plan_path': str(plan_path.resolve()),
        'input_plan_sha256': file_sha,
        'expected_plan_sha256': logical,
        'expected_arrivals': len(plan),
        'input_plan_source': 'the frozen T011 inputs/arrival_plan.json reused byte-identical in N12; original generation (T004_mixed_v1 seed 7904001) unchanged',
        'path_migration': {'baseline_task_id': 'T011', 'executed_task_id': TASK_ID,
                           'n11_dir': '/home/wangzengyu/edge_paper/experiments/T011_demand_feedback_20261007',
                           'n12_dir': str(HERE.resolve())},
        'identity_migration': {'controller_version': {'from': 'qf4', 'to': CONTROLLER_VERSION},
                               'policy': {'from': 'qf4_feedback', 'to': DYNAMIC_POLICY},
                               'experiment_boundary': {'from': 'shared_local_frontend_gpu_control',
                                                       'to': EXPERIMENT_BOUNDARY},
                               'note': 'controller migrated qf4 -> qf4r; revision 15 changes CPU affinity and Local CPU graph/layout execution; frozen plan, phase boundaries, deadlines, weights and quality routing remain unchanged'},
        'reference_install': reference_action,
    })
    validate_t012_mixed(spec)
    dump_json(INPUTS / 'scenario.json', spec)
    dump_json(INPUTS / 'FROZEN_SOURCES.json', frozen_sources())
    manifest = {
        'schema_version': T004_SCHEMA,
        'task_id': TASK_ID,
        'implementation_revision': IMPLEMENTATION_REVISION,
        'experiment_boundary': EXPERIMENT_BOUNDARY,
        'generated_utc': utcnow(),
        'generated_by': 'run_t012.py --prepare-only',
        'input_identity': INPUT_IDENTITY,
        'mixed_validator': MIXED_VALIDATOR,
        'note': 'the arrival plan is the frozen T004 input reused byte-identically; this manifest records the N12 paths and the original identity',
        'seed': spec['seed'],
        'policy_at_prepare_time': spec['policy'],
        'controller_version': spec['controller_version'],
        'duration_s': spec['duration_s'],
        'controller_start_s': spec['controller_start_s'],
        'change_s': spec['change_s'],
        'recover_s': spec['recover_s'],
        'deadline_ms': spec['deadline_ms'],
        'quota_period_ms': 50,
        'observation_windows_s': [[0, 10], [60, 70], [70, 75], [180, 190], [190, 200], [240, 250]],
        'hashes': {
            'arrival_plan_file_sha256': file_sha,
            'logical_plan_sha256': logical,
            'scenario_file_sha256': sha(INPUTS / 'scenario.json'),
            'rate_schedule_file_sha256': sha(INPUTS / 'rate_schedule.json'),
            'plan_summary_file_sha256': sha(INPUTS / 'plan_summary.json'),
            'calibration_file_sha256': sha(EVIDENCE / 'FROZEN_DEVELOPMENT_CALIBRATION.json'),
            'input_bank_metadata_sha256': sha(EVIDENCE / 'input_bank_metadata.json'),
            'frozen_sources_sha256': sha(INPUTS / 'FROZEN_SOURCES.json'),
            'reference_file_sha256': sha(EVIDENCE / 'REFERENCE.json'),
        },
        'input_files': {p.name: {'sha256': sha(p), 'size': p.stat().st_size}
                        for p in sorted(INPUTS.glob('*.json')) if p.name != 'INPUT_MANIFEST.json'},
    }
    try:
        manifest['reference_identity'] = {
            'ok': True, 'probe': _check_reference_identity(load_json(EVIDENCE / 'REFERENCE.json'))}
    except SystemExit as exc:
        manifest['reference_identity'] = {'ok': False, 'error': str(exc)}
        print(json.dumps({'stage': 'prepare_reference_warning', 'ok': False,
                          'detail': str(exc)}, ensure_ascii=False), flush=True)
    dump_json(INPUTS / 'INPUT_MANIFEST.json', manifest)
    print(json.dumps({'stage': 'prepare_only_complete', 'utc': utcnow(), 'cwd': os.getcwd(),
                      'arrival_plan_file_sha256': file_sha, 'logical_plan_sha256': logical,
                      'expected_arrivals': len(plan), 'controller_version': spec['controller_version'],
                      'experiment_boundary': EXPERIMENT_BOUNDARY,
                      'mixed_validator': MIXED_VALIDATOR,
                      'reference_install': reference_action,
                      'reference_identity_ok': bool(manifest['reference_identity'].get('ok')),
                      'implementation_revision': IMPLEMENTATION_REVISION},
                     ensure_ascii=False), flush=True)


# --------------------------------------------------------------------------
# --check / --freeze
# --------------------------------------------------------------------------
def cmd_check(args):
    import checks_t012
    out = Path(args.out).resolve() if args.out else (EVIDENCE / 'cpu_checks.json')
    phases = [p.strip() for p in str(args.phase).split(',') if p.strip()]
    explicit = getattr(args, 'reference_json', None) or os.environ.get('TASK_REFERENCE_JSON')
    expect = os.environ.get('TASK_REFERENCE_SHA256')
    result = checks_t012.run_checks(phases=phases, sources=HERE, evidence=EVIDENCE,
                                    out=out, reference_json=explicit,
                                    expect_sha256=expect)
    summary = {
        'all_pass': bool(result.get('pass')),
        'failed': list(result.get('failed') or []),
        'groups': [(g.get('id'), bool(g.get('ok'))) for g in (result.get('groups') or [])],
        'ready_fields': result.get('ready_fields'),
        'readiness': result.get('readiness'),
        'execution_ready': result.get('execution_ready'),
        'execution_ready_requires_all': result.get('execution_ready_requires_all'),
        'reference_path': result.get('reference_path'),
        'reference_sha256': result.get('reference_sha256'),
        'reference_expected_sha256': result.get('reference_expected_sha256'),
    }
    print(json.dumps({'stage': 'cpu_check_complete', 'utc': utcnow(), 'out': str(out),
                      'summary': summary, 'performance': result.get('performance')},
                     ensure_ascii=False), flush=True)
    if not summary['all_pass']:
        raise SystemExit('T014 CPU checks did not pass: %r' % (summary,))
    if not summary['execution_ready']:
        raise SystemExit(
            'T014 execution_ready is false (%r): a GPU joint run is forbidden '
            'until correctness/performance/runtime-chain readiness all hold'
            % (summary,))


def cmd_freeze(args):
    sources = frozen_sources()
    dump_json(INPUTS / 'FROZEN_SOURCES.json', sources)
    manifest_path = INPUTS / 'INPUT_MANIFEST.json'
    if manifest_path.exists():
        doc = load_json(manifest_path)
        fs_path = INPUTS / 'FROZEN_SOURCES.json'
        fs_sha = sha(fs_path)
        doc.setdefault('hashes', {})['frozen_sources_sha256'] = fs_sha
        doc.setdefault('input_files', {})['FROZEN_SOURCES.json'] = {
            'sha256': fs_sha, 'size': fs_path.stat().st_size}
        dump_json(manifest_path, doc)
    models = {
        'local': '/home/wangzengyu/edge_paper/assets/models/local.pt',
        'fast': '/home/wangzengyu/edge_paper/assets/models/fast.pt',
        'accurate': '/home/wangzengyu/edge_paper/assets/models/accurate.pt',
    }
    manifest = {
        'schema': 'T014_freeze_manifest_v1',
        'task_id': TASK_ID,
        'implementation_revision': IMPLEMENTATION_REVISION,
        'algorithm_revision': ALGORITHM_REVISION,
        'controller_version': CONTROLLER_VERSION,
        'experiment_boundary': EXPERIMENT_BOUNDARY,
        'policies': list(POLICIES),
        'frozen_utc': utcnow(),
        'planned_complete_joint_runs': PLANNED_COMPLETE_JOINT_RUNS,
        'arms_per_joint': ARMS_PER_JOINT,
        'max_joint_attempts': MAX_JOINT_ATTEMPTS,
        'max_early_mechanical_repairs': MAX_EARLY_MECHANICAL_REPAIRS,
        'early_mechanical_source_time_lt_s': EARLY_MECHANICAL_SOURCE_TIME_LT_S,
        'max_gpu_actor_starts': MAX_GPU_ACTOR_STARTS,
        'gpu_actors_per_arm': GPU_ACTORS_PER_ARM,
        'shared_local_nodes': SHARED_LOCAL_NODES,
        'drain_max_s': DRAIN_MAX_S,
        'key_physical_cores': ACTIVE_KEY_PHYSICAL_CORES,
        'local_pool_physical_cores': ACTIVE_LOCAL_PHYSICAL_CORES,
        'cpu_affinity_mode': ACTIVE_CPU_MODE,
        'online_sources': sources['files'],
        'online_sources_sha256': sha(INPUTS / 'FROZEN_SOURCES.json'),
        'effective_reference': sources.get('effective_reference'),
        'offline_tools': sources['offline_tools'],
        'evidence': {(name if (EVIDENCE / name).exists() else name):
                     ({'sha256': sha(EVIDENCE / name),
                       'size': (EVIDENCE / name).stat().st_size}
                      if (EVIDENCE / name).exists()
                      else {'present': False,
                            'reason': 'not part of the migrated T014 evidence set; '
                                      'the frozen base document is untouched'})
                     for name in FROZEN_EVIDENCE_FILES},
        'inputs': {p.name: {'sha256': sha(p), 'size': p.stat().st_size}
                   for p in sorted(INPUTS.glob('*.json')) if p.name != 'INPUT_MANIFEST.json'},
        'input_identity': INPUT_IDENTITY,
        'models': models,
        'quality_heads_dir': '/home/wangzengyu/edge_paper/experiments/task17_four_quality_isolated_rebuild_20261002/models',
        'note': 'one frozen online version for both arms of the one joint run; offline tools (analysis_t012.py, checks_t012.py) may be fixed and rerun offline without touching these hashes; the T014 deployment gate re-hashes every entry from this manifest',
    }
    dump_json(EVIDENCE / 'manifest.json', manifest)
    print(json.dumps({'stage': 'freeze_complete', 'utc': utcnow(),
                      'schema': manifest['schema'],
                      'online_sources': len(sources['files']),
                      'manifest': str(EVIDENCE / 'manifest.json')}, ensure_ascii=False), flush=True)


# --------------------------------------------------------------------------
# execution budget registry (atomic, flock protected)
# --------------------------------------------------------------------------
def _budget_doc():
    return {'task_id': TASK_ID, 'iterative_until_success': ITERATIVE_RESEARCH,
            'restart_allowed': ITERATIVE_RESEARCH,
            'planned_complete_joint_runs': PLANNED_COMPLETE_JOINT_RUNS,
            'arms_per_joint': ARMS_PER_JOINT,
            'max_joint_attempts': MAX_JOINT_ATTEMPTS,
            'max_early_mechanical_repairs': MAX_EARLY_MECHANICAL_REPAIRS,
            'max_gpu_actor_starts': MAX_GPU_ACTOR_STARTS,
            'gpu_actors_per_joint': ARMS_PER_JOINT * GPU_ACTORS_PER_ARM,
            'drain_max_s': DRAIN_MAX_S,
            'early_mechanical_source_time_lt_s': EARLY_MECHANICAL_SOURCE_TIME_LT_S,
            'complete_joint_runs_used': 0,
            'joint_attempts_used': 0, 'gpu_actor_starts_used': 0,
            'early_mechanical_repairs_used': 0,
            'pending_early_mechanical_repair': False,
            'gpu_starts_closed': False, 'gpu_starts_closed_reason': None,
            'entries': []}


def read_budget():
    """The on-disk budget ledger (never creates it)."""
    path = LOGS / 'joint_budget.json'
    doc = _budget_doc()
    if path.exists():
        try:
            doc.update(load_json(path))
        except Exception:
            pass
    return doc


def _write_budget(doc, path):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + '\n',
                   encoding='utf8')
    os.replace(tmp, path)


def _budget_refusal(doc):
    """Why another joint start must be refused right now (or None).  TASK s2: one
    complete joint run, at most two joint starts, and the ONE extra start is only
    released by a deterministic mechanical defect caught while both groups were
    still inside source-experiment t < 5 s."""
    if doc.get('gpu_starts_closed'):
        return ('GPU starts are closed for %s (%s)'
                % (TASK_ID, doc.get('gpu_starts_closed_reason') or 'no reason recorded'))
    if ITERATIVE_RESEARCH:
        if doc.get('iterative_until_success') is not True or doc.get('restart_allowed') is not True:
            return 'the ledger does not carry the explicit iterative research authorization'
        return None
    used = int(doc.get('joint_attempts_used', 0))
    max_attempts = int(doc.get('max_joint_attempts', MAX_JOINT_ATTEMPTS))
    if used >= max_attempts:
        return 'joint attempt budget exhausted (used=%d, max=%d)' % (used, max_attempts)
    starts = int(doc.get('gpu_actor_starts_used', 0))
    need = int(doc.get('gpu_actors_per_joint', ARMS_PER_JOINT * GPU_ACTORS_PER_ARM))
    max_starts = int(doc.get('max_gpu_actor_starts', MAX_GPU_ACTOR_STARTS))
    if starts + need > max_starts:
        return ('GPU actor start budget exhausted (used=%d, need=%d, max=%d)'
                % (starts, need, max_starts))
    if used > 0 and not doc.get('pending_early_mechanical_repair'):
        recorded = [e for e in (doc.get('entries') or []) if e.get('outcome')]
        if recorded:
            return ('the one allowed whole-group restart was not released: the last joint attempt '
                    'ended as %r (T014 s2 releases it only for a deterministic mechanical defect '
                    'recorded as pending after every own process exited and the logs were kept, '
                    'fixed at the affected edge, CPU-passed and re-frozen; never for low benefit, '
                    'zero commands or a bad prediction, and never with mixed-revision arms)'
                    % (recorded[-1].get('outcome'),))
    return None


def reserve_joint_attempt(joint_attempt_id=None):
    """Register one joint start BEFORE any actor exists (TASK s2).  One joint run
    costs one joint attempt and ARMS_PER_JOINT * GPU_ACTORS_PER_ARM GPU actor
    starts; the ledger is flock-protected and replaced atomically."""
    fcntl = _fcntl()
    if fcntl is None:
        raise SystemExit('fcntl is unavailable on this platform: the T014 joint budget registry '
                         'must be updated atomically on the Linux server')
    LOGS.mkdir(parents=True, exist_ok=True)
    path = LOGS / 'joint_budget.json'
    lock_path = LOGS / 'joint_budget.lock'
    code_hash = sha(Path(__file__).resolve())
    with lock_path.open('a+') as lk:
        fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
        try:
            doc = read_budget()
            reason = _budget_refusal(doc)
            if reason:
                raise SystemExit('T014 joint run refused: %s' % reason)
            used = int(doc.get('joint_attempts_used', 0))
            if joint_attempt_id is None:
                stamp = utcnow().replace('-', '').replace(':', '')
                joint_attempt_id = 'joint_attempt_%d_%s' % (used + 1, stamp)
            starts = int(doc.get('gpu_actor_starts_used', 0))
            need = ARMS_PER_JOINT * GPU_ACTORS_PER_ARM
            released_repair = bool(doc.get('pending_early_mechanical_repair'))
            doc['joint_attempts_used'] = used + 1
            doc['gpu_actor_starts_used'] = starts + need
            doc['pending_early_mechanical_repair'] = False
            doc.setdefault('entries', []).append({
                'joint_attempt': used + 1, 'joint_attempt_id': joint_attempt_id,
                'gpu_actor_starts': need, 'code_hash': code_hash, 'utc': utcnow(),
                'pid': os.getpid(), 'implementation_revision': IMPLEMENTATION_REVISION,
                'early_mechanical_repair_release': released_repair, 'outcome': None})
            _write_budget(doc, path)
        finally:
            fcntl.flock(lk.fileno(), fcntl.LOCK_UN)
    return {'joint_attempts_used': doc['joint_attempts_used'],
            'joint_attempt_id': joint_attempt_id,
            'gpu_actor_starts_used': doc['gpu_actor_starts_used'],
            'early_mechanical_repair_release': released_repair,
            'remaining_joint_attempts': (None if doc.get('max_joint_attempts') is None else
                int(doc['max_joint_attempts']) - doc['joint_attempts_used'])}


def close_joint_attempt(joint_attempt_id, ok, source_elapsed_s=None, detail=None):
    """Record how a joint attempt ended and decide whether the ONE early mechanical
    whole-group restart is released (both groups still inside source t < 5 s) or
    whether GPU starts are closed for T014 (complete run, or a defect at or after
    t = 5 s).  Never raises for a missing ledger: the caller already ran."""
    fcntl = _fcntl()
    path = LOGS / 'joint_budget.json'
    if fcntl is None or not path.exists():
        return None
    lock_path = LOGS / 'joint_budget.lock'
    with lock_path.open('a+') as lk:
        fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
        try:
            doc = read_budget()
            entries = doc.setdefault('entries', [])
            entry = None
            for candidate in reversed(entries):
                if candidate.get('joint_attempt_id') == joint_attempt_id:
                    entry = candidate
                    break
            if entry is None:
                entry = {'joint_attempt': int(doc.get('joint_attempts_used', 0)),
                         'joint_attempt_id': joint_attempt_id, 'utc': utcnow(),
                         'implementation_revision': IMPLEMENTATION_REVISION}
                entries.append(entry)
            if ITERATIVE_RESEARCH:
                outcome = 'complete' if ok else 'failed'
                entry.update({'outcome': outcome, 'ok': bool(ok),
                              'source_elapsed_s': source_elapsed_s,
                              'closed_utc': utcnow(), 'detail': detail})
                if ok:
                    doc['complete_joint_runs_used'] = int(doc.get('complete_joint_runs_used', 0)) + 1
                doc['pending_early_mechanical_repair'] = False
                # Execution completion is distinct from scientific success.
                # The next attempt is launched after diagnosis, CPU checks and
                # re-freezing by the authorized researcher, not by this ledger.
                _write_budget(doc, path)
                return {'joint_attempt_id': joint_attempt_id, 'outcome': outcome,
                        'ok': bool(ok), 'source_elapsed_s': source_elapsed_s,
                        'gpu_starts_closed': bool(doc.get('gpu_starts_closed')),
                        'iterative_until_success': True, 'restart_allowed': True,
                        'early_mechanical': False, 'pending_early_mechanical_repair': False,
                        'early_mechanical_repairs_used': doc.get('early_mechanical_repairs_used', 0)}
            # T014 s2: ``mechanical_repair_source_time_limit_s`` is null, so a
            # mechanical defect is no longer time-boxed; the explicit
            # ``pending_early_mechanical_repair`` release still gates it and it
            # stays usable at most once.
            if MECHANICAL_REPAIR_SOURCE_TIME_LIMIT_S is None:
                early = (not ok) and bool(doc.get('pending_early_mechanical_repair'))
            else:
                early = (not ok) and source_elapsed_s is not None and \
                    float(source_elapsed_s) < float(MECHANICAL_REPAIR_SOURCE_TIME_LIMIT_S)
            repairs_used = int(doc.get('early_mechanical_repairs_used', 0))
            max_repairs = int(doc.get('max_early_mechanical_repairs', MAX_EARLY_MECHANICAL_REPAIRS))
            if ok:
                outcome = 'complete'
                closed, reason = True, ('one complete joint run finished: TASK s2 allows no further '
                                        'GPU start in T014')
                doc['complete_joint_runs_used'] = int(doc.get('complete_joint_runs_used', 0)) + 1
            elif early and repairs_used < max_repairs:
                outcome = 'early_mechanical_defect'
                repairs_used += 1
                closed, reason = False, None
            elif early:
                outcome = 'early_mechanical_repair_already_used'
                closed, reason = True, ('the one allowed early mechanical whole-group restart was '
                                        'already used')
            elif source_elapsed_s is None:
                outcome = 'failed_without_source_time'
                closed, reason = True, ('joint attempt failed without a recorded source time: TASK s2 '
                                        'only releases a restart for a defect proven inside source t < %ss'
                                        % EARLY_MECHANICAL_SOURCE_TIME_LT_S)
            else:
                outcome = 'defect_at_or_after_5s'
                closed, reason = True, ('source-experiment defect at t = %.3fs >= %ss: TASK s2 forbids '
                                        'another GPU start'
                                        % (float(source_elapsed_s), EARLY_MECHANICAL_SOURCE_TIME_LT_S))
            entry.update({'outcome': outcome, 'ok': bool(ok),
                          'source_elapsed_s': source_elapsed_s,
                          'closed_utc': utcnow(), 'detail': detail})
            doc['early_mechanical_repairs_used'] = repairs_used
            doc['pending_early_mechanical_repair'] = bool(
                (not closed) and outcome == 'early_mechanical_defect')
            if closed:
                doc['gpu_starts_closed'] = True
                doc['gpu_starts_closed_reason'] = reason
            _write_budget(doc, path)
        finally:
            fcntl.flock(lk.fileno(), fcntl.LOCK_UN)
    return {'joint_attempt_id': joint_attempt_id, 'outcome': outcome, 'ok': bool(ok),
            'source_elapsed_s': source_elapsed_s, 'early_mechanical': bool(early),
            'pending_early_mechanical_repair': doc['pending_early_mechanical_repair'],
            'gpu_starts_closed': bool(doc.get('gpu_starts_closed')),
            'gpu_starts_closed_reason': doc.get('gpu_starts_closed_reason'),
            'early_mechanical_repairs_used': repairs_used}


# --------------------------------------------------------------------------
# read-only four-GPU preflight
# --------------------------------------------------------------------------
def gpu_inventory():
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 6:
            continue
        rows.append({'index': int(float(parts[0])), 'uuid': parts[1], 'name': parts[2],
                     'memory_total_mib': int(float(parts[3])),
                     'memory_used_mib': int(float(parts[4])),
                     'utilization_gpu': int(float(parts[5]))})
    return rows


def gpu_compute_processes():
    """Live compute processes per GPU uuid, enriched with ps age/state.

    Classification rule (recorded with full evidence in the preflight): a
    compute process older than 24 h on a card whose utilisation and memory sit
    inside the idle thresholds is a PERSISTENT IDLE CONTEXT (a leaked CUDA
    context of a finished foreign job), not a busy card.  Everything else is
    an active process and blocks the card.
    """
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
             '--format=csv,noheader,nounits'], text=True)
    except Exception:
        out = ''
    procs = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2 and parts[0] and parts[0] != '[Not Supported]':
            rec = {'pid': parts[0], 'gpu_uuid': parts[1],
                   'used_memory_mib': parts[2] if len(parts) > 2 else None}
            try:
                ps = subprocess.check_output(['ps', '-o', 'etimes=,stat=', '-p', rec['pid']],
                                             text=True).split()
                rec['age_s'] = int(ps[0])
                rec['stat'] = ps[1] if len(ps) > 1 else ''
            except Exception:
                rec['age_s'] = None
                rec['stat'] = ''
            procs.append(rec)
    return procs


PERSISTENT_IDLE_MIN_AGE_S = 24 * 3600


def classify_compute_process(proc, util_by_uuid, mem_by_uuid):
    age = proc.get('age_s')
    if age is not None and age >= PERSISTENT_IDLE_MIN_AGE_S:
        util = util_by_uuid.get(proc['gpu_uuid'], 100)
        mem = mem_by_uuid.get(proc['gpu_uuid'], 1 << 30)
        if util <= IDLE_UTILIZATION_MAX and mem <= IDLE_MEMORY_MAX_MIB:
            return 'persistent_idle_context'
    return 'active'


def public_lock_state():
    """Read-only state of the public two-GPU lock (never acquired here)."""
    path = Path(SHARED_TWO_GPU_LOCK)
    state = {'path': str(path), 'exists': path.exists(), 'acquired': False,
             'acquired_by': None, 'fcntl_available': _fcntl() is not None,
             'note': 'preflight only reads it; the joint parent acquires it exclusively'}
    if path.exists():
        st = path.stat()
        state['size'] = st.st_size
        state['mtime_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(st.st_mtime))
    return state


def process_inventory():
    """Read-only inventory of T011/T014 drivers and launchers (never signalled:
    TASK s8 forbids touching another run's processes)."""
    rec = {'this_process': {'pid': os.getpid(), 'cwd': os.getcwd(), 'argv': list(sys.argv)},
           'matching': [], 'foreign': []}
    try:
        txt = subprocess.check_output(
            ['pgrep', '-a', '-f', 'run_t01[12]\\.py|launch_t01[12]_joint\\.sh'],
            text=True, stderr=subprocess.DEVNULL)
    except Exception:
        txt = ''
    for line in txt.strip().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            rec['matching'].append({'pid': int(parts[0]), 'cmd': parts[1]})
    rec['foreign'] = [r for r in rec['matching'] if r['pid'] != rec['this_process']['pid']]
    return rec


def preflight_payload(reference=None):
    """Read-only four-GPU + CPU-topology + public-lock + own-process preflight.
    No CUDA context is created here and the public lock is never acquired."""
    inventory = gpu_inventory()
    compute = gpu_compute_processes()
    util_by_uuid = {r['uuid']: r['utilization_gpu'] for r in inventory}
    mem_by_uuid = {r['uuid']: r['memory_used_mib'] for r in inventory}
    for p in compute:
        p['classification'] = classify_compute_process(p, util_by_uuid, mem_by_uuid)
    active_uuids = {p['gpu_uuid'] for p in compute if p['classification'] == 'active'}
    idle = [r for r in inventory
            if r['memory_used_mib'] <= IDLE_MEMORY_MAX_MIB
            and r['utilization_gpu'] <= IDLE_UTILIZATION_MAX
            and r['uuid'] not in active_uuids]
    by_uuid = {r['uuid']: r for r in inventory}
    chosen, preferred_kept = [], True
    for u in list(PREFERRED_FIXED_UUIDS) + list(PREFERRED_DYNAMIC_UUIDS):
        r = by_uuid.get(u)
        if r is None or r not in idle or r['uuid'] in [c['uuid'] for c in chosen]:
            preferred_kept = False
            continue
        chosen.append(r)
    if len(chosen) < REQUIRED_TOTAL_IDLE_GPUS:
        for r in idle:
            if len(chosen) >= REQUIRED_TOTAL_IDLE_GPUS:
                break
            if r['uuid'] not in [c['uuid'] for c in chosen]:
                chosen.append(r)
        preferred_kept = False
    chosen = chosen[:REQUIRED_TOTAL_IDLE_GPUS]
    errors = []
    if len(idle) < REQUIRED_TOTAL_IDLE_GPUS:
        errors.append('fewer than %d idle GPUs (idle=%d, active_process_uuids=%d)'
                      % (REQUIRED_TOTAL_IDLE_GPUS, len(idle), len(active_uuids)))
    if len(chosen) < REQUIRED_TOTAL_IDLE_GPUS:
        errors.append('could not select %d idle GPUs' % REQUIRED_TOTAL_IDLE_GPUS)
    names = sorted({r['name'] for r in chosen})
    mems = sorted({r['memory_total_mib'] for r in chosen})
    if len(names) != 1 or len(mems) != 1:
        errors.append('selected cards do not share one model/memory type: %r %r' % (names, mems))
    if len({r['uuid'] for r in chosen}) != len(chosen):
        errors.append('selected cards are not mutually exclusive')
    for r in chosen:
        n_active = sum(1 for p in compute
                       if p['gpu_uuid'] == r['uuid'] and p['classification'] == 'active')
        if n_active > ALLOWED_COMPUTE_PROCESSES_ON_SELECTED:
            errors.append('selected card %s has %d active compute process(es)' % (r['uuid'], n_active))
    fixed = chosen[:2]
    dynamic = chosen[2:4]
    cpu_plan = resolve_cpu_affinity([r['uuid'] for r in fixed] + [r['uuid'] for r in dynamic],
                                    mode=ACTIVE_CPU_MODE)
    errors += ['CPU topology: %s' % c for c in cpu_plan.get('conflicts', [])]
    return {
        'schema': 'T014_gpu_preflight_v1',
        'utc': utcnow(),
        'task_id': TASK_ID,
        'implementation_revision': IMPLEMENTATION_REVISION,
        'host': subprocess.check_output(['hostname'], text=True).strip(),
        'nvidia_smi_inventory': inventory,
        'compute_processes': compute,
        'compute_process_classification_rule':
            'compute process older than 24h on a card inside the idle memory/'
            'utilisation thresholds = persistent_idle_context (leaked CUDA '
            'context of a finished foreign job, not a busy card); any other '
            'compute process = active and blocks the card.  Full per-process '
            'evidence (pid, age, stat, memory, classification) recorded here.',
        'persistent_idle_contexts': [p for p in compute
                                     if p.get('classification') == 'persistent_idle_context'],
        'preferred_fixed_uuids': PREFERRED_FIXED_UUIDS,
        'preferred_dynamic_uuids': PREFERRED_DYNAMIC_UUIDS,
        'preferred_split_kept': preferred_kept,
        'gpu_pair_order': GPU_PAIR_ORDER,
        'idle_memory_max_mib': IDLE_MEMORY_MAX_MIB,
        'idle_utilization_max': IDLE_UTILIZATION_MAX,
        'idle_uuids': [r['uuid'] for r in idle],
        'required_total_idle_GPUs': REQUIRED_TOTAL_IDLE_GPUS,
        'fixed_uuids': [r['uuid'] for r in fixed],
        'dynamic_uuids': [r['uuid'] for r in dynamic],
        'fixed_pair': [{'uuid': r['uuid'], 'name': r['name'], 'memory_total_mib': r['memory_total_mib']}
                       for r in fixed],
        'dynamic_pair': [{'uuid': r['uuid'], 'name': r['name'], 'memory_total_mib': r['memory_total_mib']}
                         for r in dynamic],
        'pair_model_and_memory_match': bool(len(names) == 1 and len(mems) == 1),
        'shared_clock_and_barrier': True,
        'GPU_result_cross_reuse_allowed': False,
        'cuda_context_created_in_preflight': False,
        'cpu_topology': cpu_plan,
        'cpu_topology_ok': bool(cpu_plan.get('ok')),
        'public_lock': public_lock_state(),
        'processes': process_inventory(),
        'foreign_compute_policy': ('never preempt or kill foreign work; only the T011-authorized '
                                   '>24h idle 224MiB 0% leaked CUDA context exception (recorded per '
                                   'process above); if resources are short, wait and write STATUS - '
                                   'never fall back to a sequential two-card run'),
        'pass': not errors,
        'errors': errors,
    }


def cmd_preflight(args):
    payload = preflight_payload()
    out = Path(args.out).resolve() if args.out else (EVIDENCE / 'gpu_preflight.json')
    dump_json(out, payload)
    print(json.dumps({'stage': 'gpu_preflight_complete', 'utc': utcnow(), 'out': str(out),
                      'pass': payload['pass'], 'errors': payload['errors'],
                      'preferred_split_kept': payload['preferred_split_kept'],
                      'fixed_uuids': payload['fixed_uuids'],
                      'dynamic_uuids': payload['dynamic_uuids'],
                      'pair_model_and_memory_match': payload['pair_model_and_memory_match']},
                     ensure_ascii=False), flush=True)
    if not payload['pass']:
        raise SystemExit('T014 four-GPU preflight failed: %r' % payload['errors'])


# --------------------------------------------------------------------------
# CPU topology (TASK s8): one INDEPENDENT physical core per key process and one
# shared pool for the 20 Local actors.  Read the real machine; on any measured
# contradiction of the TASK s8 premise, submit the conflict - never pretend.
# --------------------------------------------------------------------------
def _read_text(path):
    try:
        return Path(path).read_text(encoding='utf8').strip()
    except Exception:
        return ''


def _expand_cpu_list(text):
    """'0-3,8' -> [0, 1, 2, 3, 8]; tolerant of empty/garbage."""
    out = []
    for chunk in str(text or '').replace(' ', '').split(','):
        if not chunk:
            continue
        if '-' in chunk:
            lo, _, hi = chunk.partition('-')
            try:
                out.extend(range(int(lo), int(hi) + 1))
            except ValueError:
                continue
        else:
            try:
                out.append(int(chunk))
            except ValueError:
                continue
    return out


def _sysfs_cpu_topology():
    """{cpu: {'core_id','package_id','thread_siblings'}} from sysfs ({} if absent)."""
    base = Path('/sys/devices/system/cpu')
    if not base.is_dir():
        return {}
    topo = {}
    for entry in sorted(base.glob('cpu[0-9]*')):
        if not entry.name[3:].isdigit():
            continue
        cpu = int(entry.name[3:])
        core = _read_text(entry / 'topology/core_id')
        pkg = _read_text(entry / 'topology/physical_package_id')
        if core == '' or pkg == '':
            continue
        topo[cpu] = {'core_id': int(core), 'package_id': int(pkg),
                     'thread_siblings': _expand_cpu_list(
                         _read_text(entry / 'topology/thread_siblings_list'))}
    return topo


def _lscpu_topology():
    """Fallback topology: lscpu -p=CPU,CORE,SOCKET,NODE ({} if unavailable)."""
    try:
        out = subprocess.check_output(['lscpu', '-p=CPU,CORE,SOCKET,NODE'],
                                      text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return {}
    topo = {}
    for line in out.splitlines():
        if not line or line.startswith('#'):
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 4 or not all(p.lstrip('-').isdigit() for p in parts[:4]):
            continue
        cpu, core, socket, node = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
        topo[cpu] = {'core_id': core, 'package_id': socket, 'node': node, 'thread_siblings': []}
    groups = {}
    for cpu, rec in topo.items():
        groups.setdefault((rec['package_id'], rec['core_id']), []).append(cpu)
    for rec in topo.values():
        rec['thread_siblings'] = sorted(groups[(rec['package_id'], rec['core_id'])])
    return topo


def _numa_node_cpus():
    """{node_id: [cpu, ...]} from /sys/devices/system/node/node*/cpulist."""
    out = {}
    base = Path('/sys/devices/system/node')
    if base.is_dir():
        for entry in sorted(base.glob('node[0-9]*')):
            if not entry.name[4:].isdigit():
                continue
            cpus = _expand_cpu_list(_read_text(entry / 'cpulist'))
            if cpus:
                out[int(entry.name[4:])] = cpus
    return out


def _gpu_pci_numa(uuids):
    """{uuid: numa_node} from nvidia-smi pci.bus_id + sysfs ({} if unreadable)."""
    out = {}
    if not uuids:
        return out
    try:
        txt = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader'], text=True)
    except Exception:
        return out
    for line in txt.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 2 or parts[0] not in uuids or not parts[1]:
            continue
        # nvidia-smi prints an 8-hex-digit UPPERCASE domain ("00000000:3B:00.0")
        # while sysfs names the device with a 4-hex-digit lowercase one
        # ("0000:3b:00.0"); without this normalisation every lookup misses and the
        # NUMA node looks unreadable on a machine where it is perfectly readable.
        fields = parts[1].strip().lower().split(':')
        if len(fields) == 3:
            bus = '%04x:%s:%s' % (int(fields[0], 16), fields[1], fields[2])
        elif len(fields) == 2:
            bus = '0000:%s:%s' % (fields[0], fields[1])
        else:
            bus = parts[1].strip().lower()
        node = _read_text('/sys/bus/pci/devices/%s/numa_node' % bus)
        if node.lstrip('-').isdigit():
            out[parts[0]] = int(node)
    return out


def resolve_cpu_affinity(gpu_uuids=None, mode='legacy'):
    """Resolve the TASK s8 CPU plan from the REAL machine.

    TASK s8 premise: 48 logical CPUs = 24 physical cores, SMT 2; 11 INDEPENDENT
    physical cores for the 11 key processes (4 GPU actors, 2 arm main schedulers,
    2 post-Prefix Heads, 1 common source = this joint parent, 2 candidate replay
    workers); the shared 20 Local actors on the remaining 13 physical cores and
    their SMT siblings as one common pool that never contains a key core; each GPU
    actor on the NUMA node of its own card.  Any measured contradiction is
    reported in ``conflicts`` with ok=False: the caller refuses instead of running
    a fake mapping."""
    if mode == 'local_first16':
        return _resolve_local_first16(gpu_uuids)
    if mode != 'legacy':
        raise ValueError('unknown CPU layout: %s' % mode)
    info = {'rule': CPU_AFFINITY_RULE, 'conflicts': [], 'ok': False,
            'gpu_uuids': list(gpu_uuids or [])}
    topo = _sysfs_cpu_topology()
    source = 'sysfs' if topo else 'none'
    if not topo:
        topo = _lscpu_topology()
        source = 'lscpu' if topo else 'none'
    info['topology_source'] = source
    try:
        allowed = set(os.sched_getaffinity(0))
        info['allowed_core_count'] = len(allowed)
        info['allowed_sample'] = sorted(allowed)[:48]
    except (AttributeError, OSError) as exc:
        allowed = None
        info['allowed_read_error'] = repr(exc)
    if not topo:
        info['conflicts'].append('CPU topology is unreadable (no sysfs and no lscpu): the TASK s8 '
                                 '11-key-core plan cannot be proven')
        return info
    logical = sorted(topo)
    core_threads = {}
    for cpu in logical:
        core_threads.setdefault((topo[cpu]['package_id'], topo[cpu]['core_id']), []).append(cpu)
    for key in list(core_threads):
        core_threads[key] = sorted(core_threads[key])
    info['topology'] = {str(cpu): {'package_id': topo[cpu]['package_id'],
                                   'core_id': topo[cpu]['core_id'],
                                   'thread_siblings': core_threads[(topo[cpu]['package_id'],
                                                                    topo[cpu]['core_id'])]}
                        for cpu in logical}
    info['logical_count'] = len(logical)
    info['physical_core_count'] = len(core_threads)
    info['smt_threads_per_core'] = sorted({len(v) for v in core_threads.values()})
    if len(logical) != EXPECTED_LOGICAL_CPUS:
        info['conflicts'].append('logical CPU count is %d, TASK s8 expects %d'
                                 % (len(logical), EXPECTED_LOGICAL_CPUS))
    if len(core_threads) != EXPECTED_PHYSICAL_CORES:
        info['conflicts'].append('physical core count is %d, TASK s8 expects %d: the 11-key-core '
                                 'plan is not valid on this machine'
                                 % (len(core_threads), EXPECTED_PHYSICAL_CORES))
    if info['smt_threads_per_core'] != [EXPECTED_SMT_THREADS_PER_CORE]:
        info['conflicts'].append('SMT threads per core is %r, TASK s8 expects %d per core'
                                 % (info['smt_threads_per_core'], EXPECTED_SMT_THREADS_PER_CORE))
    node_cpus = _numa_node_cpus()
    if node_cpus:
        info['numa_source'] = 'sysfs'
    else:
        node_cpus = {0: list(logical)}
        info['numa_source'] = 'single_node_assumed'
    info['numa_node_cpu_counts'] = {str(n): len(c) for n, c in sorted(node_cpus.items())}
    gpu_numa = _gpu_pci_numa(set(gpu_uuids or []))
    info['gpu_numa_node'] = dict(gpu_numa)

    def usable(core):
        threads = core_threads[core]
        return allowed is None or set(threads) <= allowed

    def node_of(core):
        threads = core_threads[core]
        for node in sorted(node_cpus):
            if threads[0] in node_cpus[node]:
                return node
        return -1

    usable_cores = [c for c in sorted(core_threads) if usable(c)]
    info['usable_physical_cores'] = len(usable_cores)
    used = {}

    def take(core, role):
        if core is None or core in used:
            return None
        used[core] = role
        return core

    def take_on_node(node, role, avoid_nodes=()):
        if node is None:
            return None
        for core in usable_cores:
            if core in used or node_of(core) != node:
                continue
            if node_of(core) in avoid_nodes:
                continue
            return take(core, role)
        return None

    def take_any(role, prefer_nodes=()):
        for node in list(prefer_nodes) + sorted(node_cpus):
            core = take_on_node(node, role)
            if core is not None:
                return core
        return take(usable_cores[0], role) if usable_cores else None

    # four GPU actors: each on the NUMA node of its own card
    gpu_roles = {}
    for policy in POLICIES:
        gpu_roles[policy] = []
    per_arm_uuids = {
        FIXED_POLICY: list(PREFERRED_FIXED_UUIDS),
        DYNAMIC_POLICY: list(PREFERRED_DYNAMIC_UUIDS),
    }
    if gpu_uuids:
        picked = list(gpu_uuids)
        per_arm_uuids = {FIXED_POLICY: picked[:2], DYNAMIC_POLICY: picked[2:4]}
    for policy in POLICIES:
        for slot, uuid in enumerate(per_arm_uuids.get(policy) or []):
            node = gpu_numa.get(uuid)
            core = take_on_node(node, '%s.gpu%d' % (policy, slot)) if node is not None else None
            if core is None:
                core = take_any('%s.gpu%d' % (policy, slot))
                if node is not None:
                    info['conflicts'].append(
                        'no free physical core on NUMA node %s for GPU actor %s (%s); the GPU actor '
                        'cannot be placed on its own card NUMA node' % (node, policy, uuid))
                else:
                    info['conflicts'].append(
                        'NUMA node of GPU %s (%s) is unreadable: cannot prove the GPU actor sits on '
                        'its own card NUMA node' % (policy, uuid))
            gpu_roles[policy].append(core)
    # per arm: main scheduler, post-Prefix Head, candidate replay worker
    arm_nodes = {}
    for policy in POLICIES:
        nodes = [gpu_numa.get(u) for u in (per_arm_uuids.get(policy) or [])]
        nodes = [n for n in nodes if n is not None]
        arm_nodes[policy] = sorted(set(nodes))[0] if nodes else None
    arm_cores = {}
    for policy in POLICIES:
        prefer = [arm_nodes[policy]] if arm_nodes[policy] is not None else []
        arm_cores[policy] = {
            'main': take_any('%s.main' % policy, prefer),
            'decision': take_any('%s.decision' % policy, prefer),
            'replay': take_any('%s.replay' % policy, prefer),
        }
    joint_parent_core = take_any('common_source')
    free_cores = [c for c in sorted(core_threads) if c not in used]
    pool_cores = list(free_cores)
    info['key_cores'] = {role: core for core, role in sorted(used.items(), key=lambda kv: kv[1])}
    info['key_core_count'] = len(used)
    if len(used) != KEY_PHYSICAL_CORES:
        info['conflicts'].append('could not reserve %d independent key physical cores (reserved %d)'
                                 % (KEY_PHYSICAL_CORES, len(used)))
    if len(pool_cores) != LOCAL_POOL_PHYSICAL_CORES:
        info['conflicts'].append('Local pool would span %d physical cores, TASK s8 expects %d'
                                 % (len(pool_cores), LOCAL_POOL_PHYSICAL_CORES))
    pool_cpus = []
    for core in pool_cores:
        pool_cpus.extend(core_threads[core])
    pool_cpus = sorted(pool_cpus)
    amap = {}
    for policy in POLICIES:
        entry = {}
        gpus = [core for core in gpu_roles.get(policy, []) if core is not None]
        entry['gpu'] = [core_threads[c][0] for c in gpus]
        entry['gpu_cores'] = gpus
        for role in ('main', 'decision', 'replay'):
            core = arm_cores.get(policy, {}).get(role)
            entry[role] = core_threads[core][0] if core is not None else None
            entry[role + '_core'] = core
        amap[policy] = entry
    local_cpu_by_node = {}
    for node in range(SHARED_LOCAL_NODES):
        local_cpu_by_node[node] = pool_cpus[node % len(pool_cpus)] if pool_cpus else None
    if len(set(local_cpu_by_node.values())) != len(
            [v for v in local_cpu_by_node.values() if v is not None]):
        info['conflicts'].append('Local actor CPU assignment is not distinct')
    key_cpu_set = set()
    for entry in amap.values():
        for value in entry.values():
            if isinstance(value, int):
                key_cpu_set.add(value)
    if joint_parent_core is not None:
        key_cpu_set.add(core_threads[joint_parent_core][0])
    if key_cpu_set & set(pool_cpus):
        info['conflicts'].append('a key CPU is also inside the shared Local pool')
    info.update({
        'mode': 'topology_resolved' if not info['conflicts'] else 'topology_conflict',
        'map': amap,
        'joint_parent_cpu': core_threads[joint_parent_core][0] if joint_parent_core is not None else None,
        'joint_parent_core': joint_parent_core,
        'key_cpu_set': sorted(key_cpu_set),
        'key_core_threads': {role: core_threads[core]
                             for core, role in sorted(used.items(), key=lambda kv: kv[1])},
        'local_pool_physical_cores': len(pool_cores),
        'local_pool_cores': pool_cores,
        'local_pool_cpus': pool_cpus,
        'local_cpu_by_node': local_cpu_by_node,
        'local_pool_rule': ('each of the 20 Local actors is pinned to one single CPU drawn '
                            'round-robin from the common pool (the pinning API accepts one CPU per '
                            'actor); no key core and no key sibling belongs to the pool'),
        'key_physical_cores': KEY_PHYSICAL_CORES,
    })
    info['ok'] = not info['conflicts']
    info['mode'] = 'topology_resolved' if info['ok'] else 'topology_conflict'
    return info


def _resolve_local_first16(gpu_uuids=None):
    """Reserve 16 Local cores and 8 explicitly grouped control cores.

    Source, arm schedulers and replay workers remain physically independent.
    Two GPU managers on the same NUMA socket share one physical core's SMT
    threads; the two light post-Prefix Heads share another core. No Local
    actor shares a core with these processes. Legacy topology checks remain
    available, but the previous 11-core reservation is not the active layout.
    """
    gpu_uuids = list(gpu_uuids or (list(PREFERRED_FIXED_UUIDS) + list(PREFERRED_DYNAMIC_UUIDS)))
    info = resolve_cpu_affinity(gpu_uuids, mode='legacy')
    if not info.get('ok'):
        return info
    topo = info['topology']
    cores = {}
    for cpu, item in topo.items():
        key = (item['package_id'], item['core_id'])
        cores.setdefault(key, []).append(int(cpu))
    cores = {k: sorted(v) for k, v in cores.items()}
    nodes = _numa_node_cpus()
    core_node = {k: next((n for n, cpus in nodes.items() if v[0] in cpus), None)
                 for k, v in cores.items()}
    by_node = {n: sorted([k for k in cores if core_node[k] == n],
                         key=lambda k: cores[k][0]) for n in sorted(nodes)}
    if len(by_node) != 2 or any(len(v) < 4 for v in by_node.values()):
        info.update(ok=False, mode='topology_conflict')
        info['conflicts'].append('local_first16 requires two NUMA nodes with four control cores each')
        return info
    node0, node1 = sorted(by_node)
    reserved0, reserved1 = by_node[node0][-4:], by_node[node1][-4:]
    source_core, fixed_main, dynamic_main, gpu0 = reserved0
    fixed_replay, dynamic_replay, head_core, gpu1 = reserved1
    gpu_cores = {node0: gpu0, node1: gpu1}
    uuids = list(gpu_uuids or (list(PREFERRED_FIXED_UUIDS) + list(PREFERRED_DYNAMIC_UUIDS)))
    gpu_cursor = {node0: 0, node1: 0}
    mapping = {}
    roles = {'common_source': source_core}
    for i, policy in enumerate(POLICIES):
        main = fixed_main if i == 0 else dynamic_main
        replay = fixed_replay if i == 0 else dynamic_replay
        entry = {'main': cores[main][0], 'main_core': main,
                 'replay': cores[replay][0], 'replay_core': replay,
                 'decision': cores[head_core][i], 'decision_core': head_core,
                 'gpu': [], 'gpu_cores': []}
        roles[policy + '.main'] = main
        roles[policy + '.replay'] = replay
        roles[policy + '.decision'] = head_core
        for slot, uuid in enumerate(uuids[2*i:2*i+2]):
            n = info['gpu_numa_node'].get(uuid)
            if n not in gpu_cores or gpu_cursor[n] >= len(cores[gpu_cores[n]]):
                info['conflicts'].append('GPU NUMA/SMT grouping unavailable for %s' % uuid)
                continue
            core = gpu_cores[n]
            entry['gpu'].append(cores[core][gpu_cursor[n]])
            entry['gpu_cores'].append(core)
            gpu_cursor[n] += 1
            roles[policy + '.gpu%d' % slot] = core
        mapping[policy] = entry
    reserved = set(roles.values())
    local_cores = sorted([k for k in cores if k not in reserved], key=lambda k: cores[k][0])
    primaries = [cores[k][0] for k in local_cores]
    # Use every physical Local core before assigning any SMT sibling.
    extras = [cores[k][1] for k in local_cores if len(cores[k]) > 1]
    local_cpus = primaries + extras[:SHARED_LOCAL_NODES-len(primaries)]
    key_cpus = [cores[source_core][0]]
    for e in mapping.values():
        key_cpus.extend(e['gpu'] + [e['main'], e['decision'], e['replay']])
    if len(reserved) != 8 or len(local_cores) != 16 or len(local_cpus) != 20:
        info['conflicts'].append('local_first16 requires 8 control cores and 16 Local cores')
    if len(set(key_cpus)) != 11 or set(key_cpus) & set(local_cpus):
        info['conflicts'].append('logical worker CPU overlap in local_first16')
    info.update({'map': mapping, 'joint_parent_cpu': cores[source_core][0],
                 'joint_parent_core': source_core, 'key_cpu_set': sorted(key_cpus),
                 'key_core_count': len(reserved), 'key_physical_cores': len(reserved),
                 'key_cores': roles, 'key_core_threads': {r: cores[k] for r, k in roles.items()},
                 'local_pool_physical_cores': len(local_cores), 'local_pool_cores': local_cores,
                 'local_pool_cpus': local_cpus,
                 'local_cpu_by_node': dict(enumerate(local_cpus)),
                 'rule': 'local_first16: 16 Local physical cores; 5 independent source/main/replay cores; 2 GPU-manager SMT cores on their own NUMA nodes; 1 post-Head SMT core',
                 'local_pool_rule': 'one primary CPU on all 16 physical cores, then four SMT siblings; no control-core overlap',
                 'layout': 'local_first16'})
    info['ok'] = not info['conflicts']
    info['mode'] = 'topology_resolved' if info['ok'] else 'topology_conflict'
    return info


# --------------------------------------------------------------------------
# identity checks shared by --joint-run
# --------------------------------------------------------------------------
def _reference_identity_blob(reference):
    """The T014 REFERENCE.json carries the identity under top-level ``algorithm``
    and the launch budget under ``execution_budget``.  Older documents kept it in
    ``configuration``: merge top level first so a T014 document wins and an older
    one cannot masquerade as T014."""
    alg = reference.get('algorithm')
    blob = dict(reference)
    if isinstance(alg, dict):
        blob.update(alg)
    return blob


def _check_reference_identity(reference, reference_sha256=None, expected_sha256=None):
    blob = _reference_identity_blob(reference)
    budget = reference.get('execution_budget') or {}
    errors = []
    if expected_sha256:
        # T014 s3.1.8: exactly ONE explicit effective reference is injected by
        # this task; a document that does not hash to the recorded value must be
        # refused LOUDLY before any hindsight is available.
        if not reference_sha256:
            errors.append('reference sha256 missing (expected %s)' % expected_sha256)
        elif str(reference_sha256) != str(expected_sha256):
            errors.append('reference sha256 is %s, expected %s'
                          % (reference_sha256, expected_sha256))
    if not isinstance(reference.get('algorithm'), dict):
        errors.append('reference.algorithm missing: this is not the T014 REFERENCE.json')
    if not isinstance(reference.get('execution_budget'), dict):
        errors.append('reference.execution_budget missing: this is not the T014 REFERENCE.json')
    if reference.get('task_id') is not None and reference.get('task_id') != TASK_ID:
        errors.append('reference task_id %r' % reference.get('task_id'))
    if blob.get('controller_version') != CONTROLLER_VERSION:
        errors.append('reference controller_version %r' % blob.get('controller_version'))
    if blob.get('implementation_revision') != IMPLEMENTATION_REVISION:
        errors.append('reference implementation_revision %r' % blob.get('implementation_revision'))
    if list(blob.get('policies') or []) != list(POLICIES):
        errors.append('reference policies %r' % blob.get('policies'))
    expected_budget = (
        ('complete_joint_runs', PLANNED_COMPLETE_JOINT_RUNS),
        ('max_joint_attempts', MAX_JOINT_ATTEMPTS),
        ('max_gpu_actor_starts', MAX_GPU_ACTOR_STARTS),
        ('max_mechanical_repairs', MAX_EARLY_MECHANICAL_REPAIRS),
        ('drain_max_s', DRAIN_MAX_S),
        ('driver_wall_max_s', DRIVER_WALL_MAX_S),
        ('cpu_wall_max_s', CPU_WALL_MAX_S),
        ('cpu_gpu_starts', 0),
    )
    if ITERATIVE_RESEARCH:
        expected_budget += (('iterative_until_success', True), ('restart_allowed', True))
    else:
        expected_budget += (('new_gpu_runs', 0),)
    for key, expected in expected_budget:
        got = budget.get(key)
        if key not in budget:
            errors.append('reference execution_budget.%s missing' % key)
        elif expected is None:
            if got is not None:
                errors.append('reference execution_budget.%s must be null for iterative research' % key)
        elif isinstance(expected, bool):
            if got is not expected:
                errors.append('reference execution_budget.%s must be %r' % (key, expected))
        elif isinstance(got, bool) or not isinstance(got, (int, float)) or float(got) != float(expected):
            errors.append('reference execution_budget.%s is %r, TASK s2 fixes %r'
                          % (key, got, expected))
    if 'mechanical_repair_source_time_limit_s' not in budget:
        errors.append('reference execution_budget.mechanical_repair_source_time_limit_s missing')
    elif budget.get('mechanical_repair_source_time_limit_s') is not None:
        errors.append('reference execution_budget.mechanical_repair_source_time_limit_s must be '
                      'null (T014: the repair is no longer limited to an early source time)')
    if budget.get('successful_or_low_benefit_run_restart_allowed') is not ITERATIVE_RESEARCH:
        errors.append('reference execution_budget.successful_or_low_benefit_run_restart_allowed '
                      'must be %r' % ITERATIVE_RESEARCH)
    if budget.get('mixed_revision_arms_allowed') is not False:
        errors.append('reference execution_budget.mixed_revision_arms_allowed must be false')
    if list(budget.get('cpu_groups') or []) != list(CPU_GROUPS):
        errors.append('reference execution_budget.cpu_groups is %r, TASK s2 fixes %r'
                      % (budget.get('cpu_groups'), list(CPU_GROUPS)))
    if errors:
        raise SystemExit('%s reference identity checks failed: %r' % (TASK_ID, errors))
    return {'task_id': reference.get('task_id'),
            'controller_version': blob.get('controller_version'),
            'implementation_revision': blob.get('implementation_revision'),
            'policies': list(blob.get('policies') or []),
            'reference_sha256': reference_sha256,
            'execution_budget': {k: budget.get(k) for k, _ in expected_budget},
            'mechanical_repair_source_time_limit_s':
                budget.get('mechanical_repair_source_time_limit_s'),
            'cpu_groups': list(budget.get('cpu_groups') or [])}


def _code_only(text):
    import io as _io
    import tokenize as _tokenize
    parts = []
    prev = None
    keep = (_tokenize.NAME, _tokenize.OP, _tokenize.NUMBER)
    try:
        for tok in _tokenize.generate_tokens(_io.StringIO(text).readline):
            if tok.type not in keep:
                continue
            if prev in (_tokenize.NAME, _tokenize.NUMBER) \
                    and tok.type in (_tokenize.NAME, _tokenize.NUMBER):
                parts.append(' ')
            parts.append(tok.string)
            prev = tok.type
    except (_tokenize.TokenError, IndentationError):
        return text
    return ''.join(parts)


def _joint_input_checks(spec, plan, plan_path, file_sha, logical, reference):
    checks = []

    def chk(name, ok, detail=None):
        checks.append({'name': name, 'ok': bool(ok), 'detail': detail})
        if not ok:
            raise SystemExit('T014 pre-run check failed: %s (%s)' % (name, detail))

    chk('frozen_input_identity_file_sha256', file_sha == INPUT_IDENTITY['plan_file_sha256'], file_sha)
    chk('frozen_input_identity_logical_sha256', logical == INPUT_IDENTITY['plan_logical_sha256'], logical)
    chk('frozen_input_identity_expected_plan_sha256', logical == T004_EXPECTED_PLAN_SHA256, logical)
    chk('arrival_count_matches_identity', len(plan) == INPUT_IDENTITY['total_arrivals'], len(plan))
    chk('arrival_count_matches_expected', len(plan) == T004_EXPECTED_ARRIVALS, len(plan))
    chk('arrival_count_matches_scenario', len(plan) == int(spec.get('expected_arrivals', -1)),
        [len(plan), spec.get('expected_arrivals')])
    chk('scenario_plan_file_sha256', file_sha == spec.get('input_plan_sha256'),
        [file_sha, spec.get('input_plan_sha256')])
    chk('scenario_logical_sha256', logical == spec.get('expected_plan_sha256'),
        [logical, spec.get('expected_plan_sha256')])
    chk('schema_version_unchanged', spec.get('schema_version') == T004_SCHEMA, spec.get('schema_version'))
    chk('scenario_name', spec.get('scenario') == T004_SCENARIO, spec.get('scenario'))
    chk('controller_version', spec.get('controller_version') == CONTROLLER_VERSION, spec.get('controller_version'))
    chk('implementation_revision', spec.get('implementation_revision') == IMPLEMENTATION_REVISION,
        spec.get('implementation_revision'))
    chk('local_cpu_graph_enabled', spec.get('local_cpu_graph') is True,
        spec.get('local_cpu_graph'))
    chk('local_quality_route_mode', spec.get('local_quality_route_mode') == 'quality_to_accurate',
        spec.get('local_quality_route_mode'))
    chk('cpu_affinity_mode', spec.get('cpu_affinity_mode') == ACTIVE_CPU_MODE,
        spec.get('cpu_affinity_mode'))
    chk('experiment_boundary', spec.get('experiment_boundary') == EXPERIMENT_BOUNDARY,
        spec.get('experiment_boundary'))
    validate_t012_mixed(spec)
    chk('scenario_identity_validated', True, TASK_ID)
    # TASK s8/s2: the arm directory names must be exactly
    # TRIAL_T014_mixed_d50_n20_blur0.55_<policy>; mixed_scenario.run_name only
    # produces that for an empty tag when its label whitelist contains TASK_ID.
    for policy in POLICIES:
        expected = 'TRIAL_%s_mixed_d50_n20_blur0.55_%s' % (TASK_ID, policy)
        got = run_name(policy, '', T004_SCHEMA)
        chk('arm_name_%s' % policy, got == expected,
            'run_name(%r, %r, %r) -> %r but TASK requires %r; add %r to the label whitelist in '
            'mixed_scenario.run_name' % (policy, '', T004_SCHEMA, got, expected, TASK_ID))

    acceptance = load_json(EVIDENCE / 'NEW_BANK_REFERENCE_ACCEPTANCE.json')
    from physical_probe import CHECKPOINTS, Heads
    chk('model_sha256_matches_frozen_acceptance',
        {role: sha(path) for role, path in CHECKPOINTS.items()} == acceptance['model_sha256'],
        acceptance['model_sha256'])
    chk('quality_head_files_sha256_matches_frozen_acceptance',
        Heads().files == acceptance['head_files_sha256'], acceptance['head_files_sha256'])

    # The shared source must not read one group's private GPU policy/queue state.
    src = _code_only((HERE / 'shared_local_t011.py').read_text(encoding='utf8'))
    hits = [t for t in SOURCE_POLICY_READ_FORBIDDEN if t in src]
    chk('shared_source_has_no_group_policy_reads', not hits, hits)
    return checks


def _source_static_audit():
    raw = (HERE / 'shared_local_t011.py').read_text(encoding='utf8')
    code = _code_only(raw)
    return {'file': 'shared_local_t011.py', 'sha256': sha(HERE / 'shared_local_t011.py'),
            'forbidden_token_hits': [t for t in SOURCE_POLICY_READ_FORBIDDEN if t in code],
            'raw_text_hits': [t for t in SOURCE_POLICY_READ_FORBIDDEN if t in raw],
            'scan': 'executable tokens only, comments and string literals removed',
            'note': 'the shared source publishes frames only; it never commands a group Local actor and never reads a group GPU queue'}


#: TASK s8 item 1: the joint parent holds the public lock; no child-side source
#: may re-acquire the same lock.  These are the files that run inside a spawned
#: child (or the shared Local actors) and are therefore audited for flock use.
CHILD_SIDE_SOURCES = (
    'resource_axis.py',
    'group_runtime_t011.py', 'shared_local_t011.py', 'live_runtime_v20_two_gpu.py',
    'candidate_replay_t012.py', 'queue_feedback_controller.py', 'queue_feedback_stats.py',
)


def _child_lock_audit():
    """Static proof that no child-side source acquires the shared public lock."""
    report = {'scheme': 'token scan of executable text (comments/strings removed)',
              'forbidden_tokens': ['import fcntl', 'flock('],
              'files': {}, 'violations': []}
    for name in CHILD_SIDE_SOURCES:
        path = HERE / name
        if not path.exists():
            report['files'][name] = {'present': False}
            continue
        raw = path.read_text(encoding='utf8')
        code = _code_only(raw)
        hits = [t for t in report['forbidden_tokens'] if t in code]
        report['files'][name] = {'present': True, 'sha256': sha(path),
                                 'forbidden_token_hits': hits}
        if hits:
            report['violations'].append({'file': name, 'hits': hits})
    report['ok'] = not report['violations']
    return report


def export_cpu_env(affinity_info, cpu_map):
    """Export the TASK s8 CPU plan to the joint parent's environment BEFORE any
    actor process is created (spawned children inherit it).

    - ``T014_LOCAL_CPU_<node>`` (+ the ``T011_LOCAL_CPU_<node>`` alias the
      live_runtime already knows) pins each of the 20 common Local actors to one
      CPU of the shared pool.
    - ``T014_REPLAY_CPU_<policy>`` is the candidate-replay worker CPU; the
      consumer is WS-D's candidate_replay_t012.py (group_runtime_t011.py builds
      ReplayWorker() without a cpu argument, so the environment is the channel).
    """
    exported = {'local': {}, 'replay': {}, 'aliases': {}, 'note': None}
    for node, cpu in sorted((affinity_info.get('local_cpu_by_node') or {}).items()):
        if cpu is None:
            continue
        key = 'T014_LOCAL_CPU_%d' % int(node)
        alias = 'T011_LOCAL_CPU_%d' % int(node)
        os.environ[key] = str(int(cpu))
        os.environ[alias] = str(int(cpu))
        exported['local'][key] = int(cpu)
        exported['aliases'][alias] = int(cpu)
    for policy in POLICIES:
        cpu = (cpu_map.get(policy) or {}).get('replay')
        if cpu is None:
            continue
        key = 'T014_REPLAY_CPU_%s' % policy
        os.environ[key] = str(int(cpu))
        exported['replay'][key] = int(cpu)
    if not exported['local']:
        exported['note'] = 'no Local pool CPU was resolved: the Local actors keep their defaults'
    return exported


def _actor_inventory_summary(entry):
    """Per-arm child-actor readiness as announced by group_worker.

    group_runtime_t011.start_actors() spawns exactly three child processes per
    arm - gpu:0, gpu:1 and decision:0 - and only returns after each of them has
    announced READY (group_runtime_t011.py:374-398)."""
    inventory = (entry or {}).get('actor_inventory')
    if not isinstance(inventory, (list, tuple)):
        return {'known': False, 'count': None, 'gpu_actors': None, 'heads': None, 'ready': None}
    gpu = [a for a in inventory
           if isinstance(a, dict) and str(a.get('resource', '')).startswith('gpu:')]
    heads = [a for a in inventory
             if isinstance(a, dict) and str(a.get('resource', '')).startswith('decision')]
    return {'known': True, 'count': len(inventory), 'gpu_actors': len(gpu),
            'heads': len(heads),
            'ready': all(isinstance(a, dict) and a.get('type') == 'READY' for a in inventory),
            'resources': [a.get('resource') for a in inventory if isinstance(a, dict)],
            'pids': [a.get('pid') for a in inventory if isinstance(a, dict)]}


def _warmup_bank_digest(bank):
    """Digest of the fixed warm-up bank handed to BOTH arms (TASK s8 item 6).

    Records the per-array shape/dtype and a prefix hash, and states explicitly
    that the warm-up bank is not counted as a live run."""
    digest = {'type': type(bank).__name__, 'passed_identically_to_both_arms': True,
              'warmup_used_as_live': False, 'hashed_bytes_per_array': 4096}
    try:
        import hashlib
        import numpy as _np
        keys = sorted(bank.keys()) if hasattr(bank, 'keys') else []
        handle = hashlib.sha256()
        arrays = {}
        for key in keys:
            arr = _np.asarray(bank[key])
            arrays[str(key)] = {'shape': list(arr.shape), 'dtype': str(arr.dtype)}
            handle.update(str(key).encode('utf8'))
            handle.update(_np.ascontiguousarray(arr).tobytes()[:4096])
        digest.update({'keys': [str(k) for k in keys], 'arrays': arrays,
                       'sha256_prefix': handle.hexdigest()})
    except Exception as exc:  # noqa: BLE001
        digest.update({'keys': None, 'error': repr(exc)})
    return digest


def pin_cpu(cpu):
    info = {'requested': int(cpu)}
    if not hasattr(os, 'sched_setaffinity'):
        info.update({'applied': False, 'reason': 'sched_setaffinity unavailable'})
        return info
    try:
        os.sched_setaffinity(0, {int(cpu)})
    except OSError as exc:
        info.update({'applied': False, 'reason': repr(exc)})
        return info
    info.update({'applied': True, 'actual': sorted(os.sched_getaffinity(0))})
    return info


# --------------------------------------------------------------------------
# --joint-run
# --------------------------------------------------------------------------
def _snapshot_joint_identity(folder, reference_path):
    """Preserve this attempt's small inputs and source fingerprints, not code copies."""
    files = ((EVIDENCE / 'manifest.json', 'freeze_manifest.json'),
             (INPUTS / 'FROZEN_SOURCES.json', 'FROZEN_SOURCES.json'),
             (reference_path, 'REFERENCE.json'))
    snapshots = {}
    for source, name in files:
        if not source.is_file():
            raise SystemExit('attempt identity input is missing: %s' % source)
        target = folder / name
        shutil.copyfile(str(source), str(target))
        snapshots[name] = {'source': str(source), 'sha256': sha(target),
                           'size': target.stat().st_size}
    commit = subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', 'HEAD'],
                                     universal_newlines=True).strip()
    status = subprocess.check_output(['git', '-C', str(HERE), 'status', '--porcelain',
                                      '--untracked-files=no'], universal_newlines=True).strip()
    record = {'task_id': TASK_ID, 'implementation_revision': IMPLEMENTATION_REVISION,
              'git_commit': commit, 'tracked_worktree_status': status,
              'snapshots': snapshots, 'utc': utcnow()}
    dump_json(folder / 'git_commit.json', record)
    return record


def cmd_joint_run(args):
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    _scenario.RUN_LABEL = TASK_ID
    # T014 s3.1.8: ONE explicit effective reference.  TASK_REFERENCE_JSON is the
    # single override entry; it defaults to the frozen project/evidence copy and
    # its sha256 is recorded, injected into every arm and verified there.
    reference_path = Path(os.environ.get('TASK_REFERENCE_JSON')
                          or os.environ.get('T014_REFERENCE_JSON')
                          or (EVIDENCE / 'REFERENCE.json')).resolve()
    if not reference_path.exists():
        raise SystemExit('effective reference does not exist: %s' % reference_path)
    reference_sha256 = sha(reference_path)
    expected_ref_sha = str(os.environ.get('TASK_REFERENCE_SHA256') or '').strip() or None
    reference = load_json(reference_path)
    reference_identity = _check_reference_identity(reference, reference_sha256,
                                                   expected_ref_sha)

    plan_path = Path(args.input_plan).resolve() if args.input_plan else (INPUTS / 'arrival_plan.json')
    scenario_path = Path(args.scenario_config).resolve() if args.scenario_config else (INPUTS / 'scenario.json')
    spec = load_json(scenario_path)
    spec.pop('rate_schedule', None)
    plan = load_json(plan_path)
    logical = logical_plan_sha(plan)
    file_sha = sha(plan_path)
    pre_run_checks = _joint_input_checks(spec, plan, plan_path, file_sha, logical, reference)

    pre = preflight_payload(reference)
    gpu_uuid_order = list(pre.get('fixed_uuids') or []) + list(pre.get('dynamic_uuids') or [])
    affinity_info = pre.get('cpu_topology') or resolve_cpu_affinity(gpu_uuid_order, mode=ACTIVE_CPU_MODE)
    if not affinity_info.get('ok'):
        raise SystemExit('CPU topology conflict: the declared control-core grouping, '
                         '16-physical-core shared Local pool or GPU-on-its-own-NUMA '
                         'placement cannot be proven on this machine: %r'
                         % (affinity_info.get('conflicts'),))
    cpu_map = affinity_info['map']
    joint_parent_cpu = pin_cpu(affinity_info['joint_parent_cpu'])
    cpu_env = export_cpu_env(affinity_info, cpu_map)
    if not pre['pass']:
        raise SystemExit('T014 four-GPU preflight failed: %r' % pre['errors'])

    reservation = reserve_joint_attempt()
    attempt_id = reservation['joint_attempt_id']
    joint_folder = RUNS / attempt_id
    if joint_folder.exists():
        raise SystemExit('refuse overwrite: %s' % joint_folder)
    joint_folder.mkdir(parents=True)
    attempt_identity = _snapshot_joint_identity(joint_folder, reference_path)
    reference_path = joint_folder / 'REFERENCE.json'
    dump_json(joint_folder / 'gpu_preflight.json', pre)
    dump_json(joint_folder / 'pre_run_checks.json', {'checks': pre_run_checks,
                                                     'cpu_affinity': affinity_info})
    dump_json(LOGS / 'LATEST_JOINT.json',
              {'joint_attempt_id': attempt_id, 'folder': str(joint_folder), 'utc': utcnow()})
    stale = LOGS / 'RUN_COMPLETE.json'
    if stale.exists():
        stale.unlink()

    shared_lock = acquire_shared_two_gpu_lock()

    import multiprocessing as mp
    from shared_local_t011 import FrameBus, FrozenSourceCosts, SharedLocalSource
    from group_runtime_t011 import group_worker

    ctx = mp.get_context('spawn')
    costs = FrozenSourceCosts(reference=reference, evidence_dir=EVIDENCE)
    source_spec = dict(spec)
    source_spec['policy'] = FIXED_POLICY   # identity only: the source carries no policy
    source_spec['name'] = attempt_id
    source = SharedLocalSource(source_spec, plan, costs, folder=joint_folder, ctx=ctx)
    source.prepare_bank()
    local_ready = source.start_actors()
    if any(r.get('cpu_graph') is not True or r.get('input_layout') != 'channels_last'
           for r in local_ready):
        source.stop_actors()
        raise SystemExit('Local CPU execution does not match revision 15 graph/layout')
    bus = FrameBus(ctx, [FIXED_POLICY, DYNAMIC_POLICY])
    status_q = ctx.Queue()
    ready_q = ctx.Queue()
    deadline_ns = int(round(float(spec['deadline_ms']) * 1e6))
    duration_s = float(spec['duration_s'])

    arms = {}
    go_queues = {}
    procs = []
    fixed_uuids = list(pre['fixed_uuids'])
    dynamic_uuids = list(pre['dynamic_uuids'])
    warmup_bases = {FIXED_POLICY: -1_000_000, DYNAMIC_POLICY: -2_000_000}
    for policy, uuids in ((FIXED_POLICY, fixed_uuids), (DYNAMIC_POLICY, dynamic_uuids)):
        arm_name = run_name(policy, '', T004_SCHEMA)
        folder = joint_folder / arm_name
        if folder.exists():
            source.stop_actors()
            raise SystemExit('refuse overwrite: %s' % folder)
        folder.mkdir(parents=True)
        arm_spec = dict(spec)
        arm_spec['policy'] = policy
        arm_spec['name'] = arm_name
        arm_spec['task_id'] = TASK_ID
        if policy == FIXED_POLICY:
            # the fixed arm keeps 1.0/1.0 and emits no command, while running
            # the SAME qf4 code and the same shadow math
            arm_spec['initial_fast'] = 1.0
            arm_spec['requests'] = []
        go_q = ctx.Queue()
        payload = {'name': arm_name, 'spec': arm_spec, 'policy': policy,
                   'gpu_uuids': list(uuids), 'folder': str(folder),
                   'deadline_ns': deadline_ns, 'duration_s': duration_s,
                   't0_ns': None, 'bank': source.bank,
                   'frame_q': bus.queues[policy], 'status_q': status_q,
                   'ready_q': ready_q, 'go_q': go_q,
                   'cpu': cpu_map[policy],
                   'warmup_id_base': warmup_bases[policy],
                   'reference_path': str(reference_path),
                   'reference_sha256': reference_sha256,
                   'joint_attempt_id': attempt_id}
        p = ctx.Process(target=group_worker, args=(payload,), name='arm_' + policy)
        p.start()
        procs.append(p)
        go_queues[policy] = go_q
        arms[policy] = {'name': arm_name, 'folder': str(folder), 'gpu_uuids': list(uuids),
                        'pid': p.pid}

    banner = {'stage': 'joint_start', 'joint_attempt_id': attempt_id, 'task_id': TASK_ID,
              'attempt_identity': attempt_identity,
              'implementation_revision': IMPLEMENTATION_REVISION,
              'algorithm_revision': ALGORITHM_REVISION,
              'experiment_boundary': EXPERIMENT_BOUNDARY,
              'controller_version': CONTROLLER_VERSION, 'policies': list(POLICIES),
              'utc': utcnow(), 'cwd': os.getcwd(), 'pid': os.getpid(), 'argv': sys.argv,
              'input_plan_path': str(plan_path), 'input_plan_file_sha256': file_sha,
              'expected_plan_sha256': logical, 'expected_arrivals': len(plan),
              'duration_s': duration_s, 'deadline_ns': deadline_ns,
              'shared_local_nodes': SHARED_LOCAL_NODES, 'arms': arms,
              'reference_path': str(reference_path),
              'reference_sha256': reference_sha256,
              'reference_expected_sha256': expected_ref_sha,
              'reference_identity': reference_identity,
              'gpu_pair_assignment': {FIXED_POLICY: fixed_uuids, DYNAMIC_POLICY: dynamic_uuids},
              'cuda_visible_devices_in_joint_parent': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'cpu_affinity': affinity_info,
              'cpu_env': cpu_env,
              'child_lock_audit': _child_lock_audit(),
              'joint_parent_cpu': joint_parent_cpu,
              'shared_lock': shared_lock,
              'pre_run_checks': pre_run_checks}
    print(json.dumps(banner, ensure_ascii=False), flush=True)

    def abort(reason):
        for q in go_queues.values():
            try:
                q.put({'t0_ns': 0, 'abort': True})
            except Exception:
                pass
        try:
            bus.close()
        except Exception:
            pass
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=30)
        try:
            source.stop_actors()
        except Exception:
            pass
        detail = {'schema': 'T014_joint_failure_v1', 'joint_attempt_id': attempt_id,
                  'utc': utcnow(), 'reason': reason, 'arms': arms,
                  'implementation_revision': IMPLEMENTATION_REVISION}
        try:
            detail['budget_close'] = close_joint_attempt(
                attempt_id, False, source_elapsed_s=None,
                detail={'reason': reason, 'phase': 'abort'})
        except Exception as exc:  # noqa: BLE001
            detail['budget_close_error'] = repr(exc)
        dump_json(joint_folder / 'joint_failure.json', detail)
        print(json.dumps({'stage': 'joint_failed', 'joint_attempt_id': attempt_id,
                          'reason': reason}, ensure_ascii=False), flush=True)

    barrier = []
    try:
        for _ in range(ARMS_PER_JOINT):
            r = ready_q.get(timeout=1800)
            if not r.get('ready'):
                abort('arm %r did not announce READY: %r' % (r.get('group'), r))
                raise SystemExit('T014 joint barrier failed: %r' % r)
            barrier.append(r)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        abort('joint barrier did not complete: %r' % exc)
        raise SystemExit('T014 joint barrier did not complete: %r' % exc)

    # T014 s3.1.8: every arm must confirm the SAME injected effective reference
    # before the one common t0; a mismatch refuses the run before any hindsight.
    bad_refs = {}
    for r in barrier:
        got = r.get('reference_sha256')
        if not got:
            bad_refs[r.get('policy')] = {'reference_sha256': None,
                                         'expected': reference_sha256}
        elif str(got) != str(reference_sha256):
            bad_refs[r.get('policy')] = {'reference_sha256': str(got),
                                         'expected': reference_sha256}
    if bad_refs:
        abort('arm effective-reference sha mismatch: %r' % bad_refs)
        raise SystemExit('T014 joint barrier failed: reference sha mismatch %r' % bad_refs)

    # TASK s8 item 7: both arms' four GPU actors, both post-Prefix Heads and the
    # common shared Local source must all be ready before the single common t0.
    arm_actors = {r['policy']: _actor_inventory_summary(r) for r in barrier}
    expected_arm_actors = GPU_ACTORS_PER_ARM + 1   # gpu:0 + gpu:1 + decision:0
    bad_actors = {}
    for policy, info in arm_actors.items():
        if not info.get('known'):
            continue
        if (info['count'] != expected_arm_actors or info['gpu_actors'] != GPU_ACTORS_PER_ARM
                or info['heads'] != 1 or not info['ready']):
            bad_actors[policy] = info
    if bad_actors:
        abort('GPU actor inventory incomplete at the barrier: %r' % bad_actors)
        raise SystemExit('T014 joint barrier failed: GPU actor inventory incomplete: %r'
                         % bad_actors)
    gpu_actors_ready = sum((info.get('gpu_actors') or 0) for info in arm_actors.values())
    if gpu_actors_ready != ARMS_PER_JOINT * GPU_ACTORS_PER_ARM:
        abort('only %d of %d GPU actors announced READY'
              % (gpu_actors_ready, ARMS_PER_JOINT * GPU_ACTORS_PER_ARM))
        raise SystemExit('T014 joint barrier failed: only %d of %d GPU actors announced READY'
                         % (gpu_actors_ready, ARMS_PER_JOINT * GPU_ACTORS_PER_ARM))
    if len(local_ready) != SHARED_LOCAL_NODES:
        abort('shared Local source announced %d/%d actors' % (len(local_ready), SHARED_LOCAL_NODES))
        raise SystemExit('T014 joint barrier failed: shared Local source announced %d/%d actors'
                         % (len(local_ready), SHARED_LOCAL_NODES))

    t0_ns = time.perf_counter_ns() + BARRIER_LEAD_NS
    source.set_t0(t0_ns)
    for q in go_queues.values():
        q.put({'t0_ns': t0_ns})
    warmup_bank_digest = _warmup_bank_digest(source.bank)
    dump_json(joint_folder / 'barrier.json',
              {'joint_attempt_id': attempt_id, 't0_ns': t0_ns, 'utc': utcnow(),
               'barrier_lead_ns': BARRIER_LEAD_NS,
               'shared_local_ready': len(local_ready),
               'shared_local_actor_affinity': [
                   {'resource': r.get('resource'), 'pid': r.get('pid'),
                    'cpu_affinity': r.get('cpu_affinity')} for r in local_ready],
               'gpu_groups_ready': [{'group': r['group'], 'policy': r['policy'],
                                     'pid': r['pid'], 'gpu_uuids': r['gpu_uuids'],
                                     'cpu': r.get('cpu'), 'cpu_pin': r.get('cpu_pin'),
                                     'warmup': r.get('warmup')}
                                    for r in barrier],
               'arm_actor_inventory': arm_actors,
               'reference_path': str(reference_path),
               'reference_sha256': reference_sha256,
               'gpu_actors_ready': gpu_actors_ready,
               'gpu_heads_ready': sum((info.get('heads') or 0) for info in arm_actors.values()),
               'gpu_models_loaded': ARMS_PER_JOINT * GPU_ACTORS_PER_ARM,
               'gpu_models_ready': len(barrier) == ARMS_PER_JOINT,
               'gpu_actor_count': ARMS_PER_JOINT * GPU_ACTORS_PER_ARM,
               'gpu_post_heads_loaded': ARMS_PER_JOINT,
               'warmup_component_calls_total': ARMS_PER_JOINT * 320,
               'warmup_by_policy': {r['policy']: r.get('warmup') for r in barrier},
               'warmup_bank_digest': warmup_bank_digest,
               'cpu_env': cpu_env})
    print(json.dumps({'stage': 'joint_barrier_released', 'joint_attempt_id': attempt_id,
                      't0_ns': t0_ns, 'shared_local_ready': len(local_ready),
                      'gpu_groups_ready': len(barrier), 'utc': utcnow()},
                     ensure_ascii=False), flush=True)

    source_summary = None
    source_error = None
    source_holder = {}

    def _run_source():
        try:
            source_holder['summary'] = source.run(bus)
        except Exception as exc:  # noqa: BLE001
            import traceback
            source_holder['error'] = {'error': repr(exc), 'traceback': traceback.format_exc()}

    import threading
    source_thread = threading.Thread(target=_run_source, name='common_source', daemon=True)
    source_thread.start()
    source_join_timeout = duration_s + DRAIN_MAX_S + 30.0
    source_thread.join(timeout=source_join_timeout)
    source_done_ns = time.perf_counter_ns()
    source_run_s = max(0.0, (source_done_ns - t0_ns) / 1e9)
    if source_thread.is_alive():
        source_error = {'error': 'common source did not finish within %.1f s'
                                 % source_join_timeout,
                        'source_run_s': source_run_s}
    else:
        source_summary = source_holder.get('summary')
        source_error = source_holder.get('error')
        if source_summary is None and source_error is None:
            source_error = {'error': 'common source returned no summary'}

    # drain: ONE shared absolute wall deadline for every queue wait, actor stop,
    # worker join and writer flush (T014 s3.5.8; never a serial 900 s + N x 60 s)
    drain_start_ns = time.perf_counter_ns()
    drain_deadline_ns = drain_start_ns + int(DRAIN_MAX_S * 1e9)

    def drain_remaining_s():
        return max(0.0, (drain_deadline_ns - time.perf_counter_ns()) / 1e9)

    _mark = time.perf_counter()
    try:
        bus.close()
    except Exception:  # noqa: BLE001
        pass
    bus_close_s = time.perf_counter() - _mark
    _mark = time.perf_counter()
    try:
        source.stop_actors()
        shared_local_stopped = True
    except Exception as exc:  # noqa: BLE001
        shared_local_stopped = repr(exc)
    stop_actors_s = time.perf_counter() - _mark

    _mark = time.perf_counter()
    statuses = {}
    for _ in range(ARMS_PER_JOINT):
        left = drain_remaining_s()
        if left <= 0:
            r = {'group': None, 'ok': False, 'error': 'drain deadline exhausted'}
        else:
            try:
                r = status_q.get(timeout=left)
            except Exception:
                r = {'group': None, 'ok': False, 'error': 'arm status timeout'}
        statuses[str(r.get('group'))] = r
    status_wait_s = time.perf_counter() - _mark
    _mark = time.perf_counter()
    for p in procs:
        p.join(timeout=drain_remaining_s())
    proc_join_s = time.perf_counter() - _mark
    drain_s = (time.perf_counter_ns() - drain_start_ns) / 1e9
    drain = {'drain_s': drain_s, 'drain_budget_s': DRAIN_MAX_S,
             'drain_budget_ok': drain_s <= DRAIN_MAX_S,
             'drain_deadline_ns': drain_deadline_ns,
             'bus_close_s': bus_close_s, 'stop_actors_s': stop_actors_s,
             'status_wait_s': status_wait_s, 'proc_join_s': proc_join_s,
             'shared_local_stopped': shared_local_stopped,
             'arms_alive_after_join': [p.name for p in procs if p.is_alive()]}

    run_ok = bool(source_summary is not None and not source_error
                  and len(statuses) == ARMS_PER_JOINT
                  and all(v.get('ok') for v in statuses.values()))
    ok = bool(run_ok and drain['drain_budget_ok'])
    budget_close = None
    budget_close_error = None
    try:
        budget_close = close_joint_attempt(
            attempt_id, ok, source_elapsed_s=source_run_s,
            detail={'run_ok': run_ok, 'drain_budget_ok': drain['drain_budget_ok'],
                    'drain_s': drain_s,
                    'arm_ok': {k: bool(v.get('ok')) for k, v in statuses.items()}})
    except Exception as exc:  # noqa: BLE001
        budget_close_error = repr(exc)
    joint = {
        'schema': 'T014_joint_summary_v1',
        'joint_attempt_id': attempt_id,
        'task_id': TASK_ID,
        'implementation_revision': IMPLEMENTATION_REVISION,
        'algorithm_revision': ALGORITHM_REVISION,
        'controller_version': CONTROLLER_VERSION,
        'experiment_boundary': EXPERIMENT_BOUNDARY,
        'policies': list(POLICIES),
        'utc': utcnow(),
        't0_ns': t0_ns,
        'duration_s': duration_s,
        'deadline_ns': deadline_ns,
        'shared_local_nodes': SHARED_LOCAL_NODES,
        'shared_lock': shared_lock,
        'cpu_affinity': affinity_info,
        'joint_parent_cpu': joint_parent_cpu,
        'cpu_env': cpu_env,
        'shared_local_stopped': shared_local_stopped,
        'shared_local_actor_affinity': [
            {'resource': r.get('resource'), 'pid': r.get('pid'),
             'cpu_affinity': r.get('cpu_affinity')} for r in local_ready],
        'gpu_pair_assignment': {FIXED_POLICY: fixed_uuids, DYNAMIC_POLICY: dynamic_uuids},
        'preflight_pass': pre['pass'],
        'preferred_split_kept': pre['preferred_split_kept'],
        'pair_model_and_memory_match': pre['pair_model_and_memory_match'],
        'barrier': {'shared_local_ready': len(local_ready), 'gpu_groups_ready': len(barrier),
                    'gpu_models_loaded': ARMS_PER_JOINT * GPU_ACTORS_PER_ARM,
                    'gpu_actors_ready': gpu_actors_ready,
                    'gpu_post_heads_loaded': ARMS_PER_JOINT,
                    'arm_actor_inventory': arm_actors,
                    'reference_sha256': reference_sha256,
                    'barrier_lead_ns': BARRIER_LEAD_NS,
                    'warmup_by_policy': {r['policy']: r.get('warmup') for r in barrier},
                    'warmup_bank_digest': warmup_bank_digest},
        'source_static_audit': _source_static_audit(),
        'child_lock_audit': _child_lock_audit(),
        'source_cost_provenance': costs.provenance(),
        'source': source_summary,
        'source_error': source_error,
        'source_run_s': source_run_s,
        'drain': drain,
        'drain_s': drain_s,
        'drain_budget_ok': drain['drain_budget_ok'],
        'budget_reservation': reservation,
        'budget_close': budget_close,
        'budget_close_error': budget_close_error,
        'arms': arms,
        'arm_folders': {k: v.get('folder') for k, v in arms.items()},
        'arm_statuses': {k: {kk: vv for kk, vv in v.items() if kk != 'summary'} for k, v in statuses.items()},
        'arm_summaries': {k: v.get('summary') for k, v in statuses.items()},
        'joint_ok': ok,
        'run_ok': run_ok,
    }
    dump_json(joint_folder / 'joint_summary.json', joint)
    print(json.dumps({'stage': 'joint_run_complete', 'joint_attempt_id': attempt_id,
                      'joint_ok': ok, 'run_ok': run_ok, 'joint_folder': str(joint_folder),
                      'drain_s': drain_s, 'drain_budget_ok': drain['drain_budget_ok'],
                      'source_frames': (source_summary or {}).get('frames'),
                      'source_counters': (source_summary or {}).get('counters'),
                      'arm_ended': {k: (v.get('summary') or {}).get('ended') for k, v in statuses.items()},
                      'arm_common_local_completed': {k: (v.get('summary') or {}).get('common_local_completed')
                                                     for k, v in statuses.items()},
                      'utc': utcnow()}, ensure_ascii=False), flush=True)
    if not run_ok:
        raise SystemExit('T014 joint run did not complete: source_error=%r statuses=%r'
                         % (source_error, joint['arm_statuses']))
    if not drain['drain_budget_ok']:
        raise SystemExit('T014 joint drain exceeded the TASK s2 budget of %ss: %r'
                         % (DRAIN_MAX_S, drain))


# --------------------------------------------------------------------------
# --verify / --summarize
# --------------------------------------------------------------------------
def latest_joint_folder():
    doc = LOGS / 'LATEST_JOINT.json'
    if doc.exists():
        folder = Path(load_json(doc).get('folder', ''))
        if folder.exists():
            return folder
    candidates = sorted(RUNS.glob('joint_attempt_*'), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit('no joint attempt folder under %s' % RUNS)
    return candidates[-1]


def joint_folder_from_args(args):
    if getattr(args, 'joint', None):
        p = Path(args.joint)
        if not p.is_absolute():
            p = RUNS / args.joint
        if not p.exists():
            raise SystemExit('joint folder missing: %s' % p)
        return p
    return latest_joint_folder()


def cmd_verify(args):
    folder = joint_folder_from_args(args)
    import analysis_t012
    payload = analysis_t012.verify_joint(folder, evidence=EVIDENCE, revision=IMPLEMENTATION_REVISION,
                                         task_id=TASK_ID)
    written = []
    dump_json(folder / 'verification.json', payload)
    written.append(str(folder / 'verification.json'))
    evidence_copy = EVIDENCE / 'verification.json'
    if evidence_copy.resolve() != (folder / 'verification.json').resolve():
        dump_json(evidence_copy, payload)
        written.append(str(evidence_copy))
    if getattr(args, 'out', None):
        extra = Path(args.out).resolve()
        dump_json(extra, payload)
        written.append(str(extra))
    print(json.dumps({'stage': 'verify_complete', 'joint_folder': str(folder),
                      'written': written,
                      'pass_check': payload.get('pass'),
                      'errors': (payload.get('errors') or [])[:10],
                      'commands': payload.get('commands')}, ensure_ascii=False), flush=True)


def cmd_summarize(args):
    folder = joint_folder_from_args(args)
    import analysis_t012
    out = Path(args.out).resolve() if args.out else (EVIDENCE / 'metrics.json')
    payload = analysis_t012.summarize(joint_folder=folder, evidence=EVIDENCE, inputs=INPUTS,
                                      out=out, task_id=TASK_ID, revision=IMPLEMENTATION_REVISION)
    print(json.dumps({'stage': 'summarize_complete', 'utc': utcnow(), 'out': str(out),
                      'joint_folder': str(folder), 'task_id': TASK_ID,
                      'arms': {k: v.get('status') for k, v in (payload.get('arms') or {}).items()}},
                     ensure_ascii=False), flush=True)


#: TASK s8 item 3: every one of these must be listed in the frozen manifest and
#: hash-match on the deployed server before the joint run may start.
REQUIRED_ENTRY_POINTS = ('run_t011.py', 'run_t012.py', 'start_t012.sh', 'launch_t012_joint.sh')


def deploy_audit(out=None):
    """Per-file SHA256 deployment gate (TASK s8 item 3).

    Re-hashes every file the frozen manifest lists, refuses any unlisted
    python/shell file on disk, re-checks the T014 reference identity, proves no
    child-side source acquires the public lock, and requires a passing T014 CPU
    check suite.  Every mismatch is printed; the gate then exits non-zero."""
    problems = []
    report = {'schema': 'T014_deploy_audit_v1', 'task_id': TASK_ID,
              'implementation_revision': IMPLEMENTATION_REVISION,
              'algorithm_revision': ALGORITHM_REVISION,
              'utc': utcnow(), 'here': str(HERE),
              'required_entry_points': list(REQUIRED_ENTRY_POINTS)}
    manifest_path = EVIDENCE / 'manifest.json'
    manifest = None
    if not manifest_path.exists():
        problems.append('evidence/manifest.json missing: run --freeze before deploying')
    else:
        try:
            manifest = load_json(manifest_path)
        except Exception as exc:  # noqa: BLE001
            problems.append('evidence/manifest.json unreadable: %r' % (exc,))
    if manifest is not None:
        if manifest.get('schema') != 'T014_freeze_manifest_v1':
            problems.append('manifest schema is %r, expected T014_freeze_manifest_v1'
                            % manifest.get('schema'))
        if manifest.get('implementation_revision') != IMPLEMENTATION_REVISION:
            problems.append('manifest implementation_revision is %r'
                            % manifest.get('implementation_revision'))
        recorded = {}
        for section in ('online_sources', 'offline_tools'):
            for name, record in (manifest.get(section) or {}).items():
                recorded[name] = (section, record or {})
        report['manifest_files'] = len(recorded)
        files = {}
        for name in sorted(recorded):
            section, record = recorded[name]
            path = HERE / name
            entry = {'section': section, 'listed_sha256': record.get('sha256'),
                     'listed_size': record.get('size'), 'present': path.exists()}
            if not path.exists():
                entry['status'] = 'missing'
                problems.append('deployed file missing: %s (%s)' % (name, section))
            else:
                entry['actual_sha256'] = sha(path)
                entry['actual_size'] = path.stat().st_size
                entry['status'] = ('match' if entry['actual_sha256'] == record.get('sha256')
                                   else 'mismatch')
                if entry['status'] == 'mismatch':
                    problems.append('deployed hash mismatch: %s listed=%s actual=%s'
                                    % (name, record.get('sha256'), entry['actual_sha256']))
            files[name] = entry
        report['files'] = files
        on_disk = sorted({p.name for pattern in ('*.py', '*.sh') for p in HERE.glob(pattern)})
        unlisted = [name for name in on_disk if name not in recorded]
        report['unlisted_on_disk'] = unlisted
        if unlisted:
            problems.append('python/shell files on disk but not in the frozen manifest: %r'
                            % unlisted)
        missing_entry = [name for name in REQUIRED_ENTRY_POINTS if name not in recorded]
        report['missing_entry_points'] = missing_entry
        if missing_entry:
            problems.append('required entry points not frozen in the manifest: %r' % missing_entry)
        for name in REQUIRED_ENTRY_POINTS:
            if name in recorded and files.get(name, {}).get('status') != 'match':
                problems.append('entry point %s does not match its frozen hash' % name)
        evidence_files = {}
        for name, record in (manifest.get('evidence') or {}).items():
            path = EVIDENCE / name
            if not path.exists():
                if (record or {}).get('present') is False:
                    # explicitly frozen as absent (not part of the migrated set)
                    evidence_files[name] = {'status': 'absent_by_manifest'}
                else:
                    evidence_files[name] = {'status': 'missing'}
                    problems.append('frozen evidence file missing: %s' % name)
                continue
            actual = sha(path)
            ok = actual == (record or {}).get('sha256')
            evidence_files[name] = {'status': 'match' if ok else 'mismatch',
                                    'listed_sha256': (record or {}).get('sha256'),
                                    'actual_sha256': actual}
            if not ok:
                problems.append('frozen evidence hash mismatch: %s listed=%s actual=%s'
                                % (name, (record or {}).get('sha256'), actual))
        report['evidence'] = evidence_files
    try:
        reference = load_json(EVIDENCE / 'REFERENCE.json')
        report['reference_identity'] = _check_reference_identity(reference)
    except SystemExit as exc:
        report['reference_identity'] = {'ok': False, 'error': str(exc)}
        problems.append(str(exc))
    except Exception as exc:  # noqa: BLE001
        report['reference_identity'] = {'ok': False, 'error': repr(exc)}
        problems.append('evidence/REFERENCE.json unreadable: %r' % (exc,))
    lock_audit = _child_lock_audit()
    report['child_lock_audit'] = lock_audit
    if not lock_audit['ok']:
        problems.append('a child-side source acquires the public two-GPU lock: %r'
                        % lock_audit['violations'])
    checks_path = EVIDENCE / 'cpu_checks.json'
    if not checks_path.exists():
        report['cpu_checks'] = {'status': 'missing'}
        problems.append('evidence/cpu_checks.json missing: run --check before the joint run')
    else:
        try:
            checks = load_json(checks_path)
        except Exception as exc:  # noqa: BLE001
            checks = {}
            problems.append('evidence/cpu_checks.json unreadable: %r' % (exc,))
        passed = bool(checks.get('pass'))
        report['cpu_checks'] = {'status': 'pass' if passed else 'fail',
                                'schema': checks.get('schema'), 'sha256': sha(checks_path)}
        if not passed:
            problems.append('evidence/cpu_checks.json is not a passing T014 CPU check suite')
    report['problems'] = problems
    report['ok'] = not problems
    target = Path(out).resolve() if out else (EVIDENCE / 'deploy_audit.json')
    dump_json(target, report)
    print(json.dumps({'stage': 'deploy_audit', 'ok': report['ok'], 'out': str(target),
                      'problems': problems, 'checked_files': report.get('manifest_files'),
                      'unlisted_on_disk': report.get('unlisted_on_disk')},
                     ensure_ascii=False), flush=True)
    if problems:
        raise SystemExit('T014 deployment audit failed with %d problem(s): %r'
                         % (len(problems), problems))
    return report


def cmd_deploy_audit(args):
    deploy_audit(out=getattr(args, 'out', None))


def cmd_joint_facts(args):
    """Compact facts for the detached driver's RUN_COMPLETE.json."""
    folder = joint_folder_from_args(args)
    joint_path = folder / 'joint_summary.json'
    summary = load_json(joint_path) if joint_path.exists() else {}
    try:
        budget = read_budget()
    except Exception as exc:  # noqa: BLE001
        budget = {'error': repr(exc)}
    facts = {'schema': 'T014_joint_facts_v1', 'task_id': TASK_ID,
             'implementation_revision': IMPLEMENTATION_REVISION,
             'algorithm_revision': ALGORITHM_REVISION, 'utc': utcnow(),
             'joint_folder': str(folder),
             'joint_attempt_id': summary.get('joint_attempt_id') or folder.name,
             'joint_ok': bool(summary.get('joint_ok')),
             'run_ok': bool(summary.get('run_ok')),
             'summary_schema': summary.get('schema'),
             'joint_summary_present': joint_path.exists(),
             'drain_s': summary.get('drain_s'),
             'drain_budget_ok': summary.get('drain_budget_ok'),
             'source_run_s': summary.get('source_run_s'),
             't0_ns': summary.get('t0_ns'),
             'duration_s': summary.get('duration_s'),
             'arm_folders': summary.get('arm_folders') or {},
             'budget_close': summary.get('budget_close'),
             'budget': {k: budget.get(k) for k in (
                 'joint_attempts_used', 'gpu_actor_starts_used', 'complete_joint_runs_used',
                 'early_mechanical_repairs_used', 'pending_early_mechanical_repair',
                 'gpu_starts_closed', 'gpu_starts_closed_reason')},
             'verification_present': ((folder / 'verification.json').exists()
                                      or (EVIDENCE / 'verification.json').exists())}
    out = Path(args.out).resolve() if args.out else (LOGS / 'T014_joint_facts.json')
    dump_json(out, facts)
    print(json.dumps({'stage': 'joint_facts', 'out': str(out), 'joint_ok': facts['joint_ok'],
                      'drain_budget_ok': facts['drain_budget_ok'],
                      'joint_attempt_id': facts['joint_attempt_id']},
                     ensure_ascii=False), flush=True)
    return facts


def apply_trace_input_delay(frames, delay_ns):
    """Delay image-ready requests; Local computation and control metadata stay live.

    This is a controlled, constant input-transport abstraction, not a measured
    wireless link. Original arrivals, deadlines and Local results are retained.
    """
    delay_ns = int(delay_ns)
    if not 0 <= delay_ns <= 50_000_000:
        raise ValueError('input-transport delay must be between 0 and 50ms')
    if not delay_ns:
        return frames, 0
    delayed = 0
    transformed = []
    for original in frames:
        frame = dict(original)
        frame['source_frame_id'] = original['frame_id']
        frame['source_frame_publish_ns'] = original['source_publish_ns']
        if original['outcome'] in ('GPU_READY', 'PREFIX_READY'):
            frame['source_publish_ns'] += delay_ns
            frame['simulated_network_input_delay_ns'] = delay_ns
            delayed += 1
        transformed.append(frame)
    transformed.sort(key=lambda f: (f['source_publish_ns'], f['source_frame_id']))
    for sequence, frame in enumerate(transformed, 1):
        frame['frame_id'] = sequence
    return transformed, delayed


def trace_replay_worker(payload):
    """Real-time immutable Local events; only GPU inference is executed again."""
    import gc
    import queue
    try:
        if hasattr(os,'sched_setaffinity'):
            os.sched_setaffinity(0, {int(payload['cpu'])})
        path = Path(payload['trace'])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != payload['trace_sha256']:
            raise RuntimeError('frozen source trace changed')
        # Parsing and I/O finish BEFORE the joint epoch. No future event is
        # exposed to the runtime: publication still obeys its recorded clock.
        frames = [json.loads(line) for line in raw.splitlines()]
        del raw
        deadline_ns = int(payload.get('deadline_ns', 50_000_000))
        if deadline_ns not in (25_000_000, 50_000_000):
            raise RuntimeError('unsupported source trace deadline')
        last = -1
        for seq, f in enumerate(frames, 1):
            if f['frame_id'] != seq or f['source_publish_ns'] < last:
                raise RuntimeError('trace sequence or clock invalid')
            if f.get('outcome') == 'SOURCE_END':
                if seq != len(frames):
                    raise RuntimeError('Source end must be the last original frame')
            elif f['deadline_ns'] != f['arrival_ns'] + deadline_ns:
                raise RuntimeError('trace deadline changed')
            last = f['source_publish_ns']
        input_delay_ns = int(payload.get('network_input_delay_ns', 0))
        frames, delayed_frames = apply_trace_input_delay(frames, input_delay_ns)
        raw_sequence = hashlib.sha256()
        if payload.get('frozen_input_audit'):
            from shared_local_t011 import source_frame_digest
            for f in frames:
                digest = source_frame_digest(f)
                raw_sequence.update(bytes.fromhex(digest))
                f['_replay_raw_digest'] = digest
        gc.collect()
        gc.disable()
        payload['ready_q'].put({'ready': True, 'frames': len(frames)})
        epoch = int(payload['go_q'].get(timeout=900)['t0_ns'])
        max_lag, late5, released = 0, 0, 0
        next_report = 0
        for f in frames:
            due = epoch + int(f['source_publish_ns'])
            while True:
                remaining = due - time.perf_counter_ns()
                if remaining <= 0:
                    break
                time.sleep(min(remaining/1e9, .0002))
            lag = time.perf_counter_ns() - due
            max_lag = max(max_lag, lag)
            late5 += lag > 5_000_000
            if payload.get('frozen_input_audit'):
                f['_replay_actual_publish_ns'] = time.perf_counter_ns() - epoch
            # Full queue is a failed run, not permission to drop source data.
            try:
                payload['frame_q'].put_nowait(f)
            except queue.Full:
                raise RuntimeError('trace publication queue full; no frame dropped silently')
            released += 1
            if f['source_publish_ns'] >= next_report:
                print(json.dumps({'stage':'frozen_source_progress', 'policy':payload['policy'],
                    'source_s':f['source_publish_ns']/1e9, 'frames':released}), flush=True)
                next_report += 10_000_000_000
        if not frames or frames[-1].get('outcome') != 'SOURCE_END':
            payload['frame_q'].put({'schema':'T011_shared_frame_v1','outcome':'SOURCE_END',
                                   'frame_id':-1,'task_id':-1}, timeout=5)
        report = {'ok':True,'frames':released,'dropped':0,'early_publications':0,
                  'max_publication_lag_ms':max_lag/1e6,'publications_lag_over5ms':late5,
                  'trace_sha256':payload['trace_sha256'],'t0_ns':epoch,'deadline_ns':deadline_ns,
                  'network_input_delay_ns':input_delay_ns,'delayed_input_frames':delayed_frames,
                  'raw_sequence_sha256':raw_sequence.hexdigest() if payload.get('frozen_input_audit') else None}
        dump_json(payload['report_path'],report)
        payload['status_q'].put(report)
    except BaseException as exc:
        import traceback
        error={'ok':False,'error':repr(exc),'traceback':traceback.format_exc()}
        dump_json(payload['report_path'],error)
        payload['ready_q'].put(error)
        payload['status_q'].put(error)
        raise


def ingress_worker(payload):
    """Independent edge reception/audit on the former CPU Head core."""
    import gzip
    import queue
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.sched_setaffinity(0, {int(payload['cpu'])})
        from shared_local_t011 import source_frame_digest
        plan = load_json(Path(payload['plan']))
        expected = {r['task_id']: (r['bank_index'], r['origin_node'],
            r['scheduled_offset_ns'], r['scheduled_offset_ns'] + payload['deadline_ns']) for r in plan}
        payload['ready_q'].put({'ready': True, 'pid': os.getpid(),
                                'cpu_affinity': list(os.sched_getaffinity(0))})
        epoch = int(payload['go_q'].get(timeout=900)['t0_ns'])
        count, arrivals, maximum = 0, 0, 0
        sequence = hashlib.sha256()
        buffered = []
        with gzip.open(payload['receipts_path'], 'wt', encoding='utf8', compresslevel=1) as receipts:
            while True:
                frame = payload['source_q'].get(timeout=330)
                received = time.perf_counter_ns() - epoch
                digest = source_frame_digest(frame)
                if digest != frame.get('_replay_raw_digest') or frame['frame_id'] != count + 1:
                    raise RuntimeError('Ingress raw digest or sequence differs from frozen source')
                if frame['outcome'] == 'SHARED_ARRIVAL':
                    actual = (frame['bank_index'], frame['origin_node'], frame['arrival_ns'], frame['deadline_ns'])
                    if actual != expected.get(frame['task_id']):
                        raise RuntimeError('Ingress task image/node/arrival/deadline changed')
                    arrivals += 1
                count += 1
                sequence.update(bytes.fromhex(digest))
                maximum = max(maximum, received - int(frame['source_publish_ns']))
                frame['_replay_ingress_received_ns'] = received
                frame['_replay_ingress_digest'] = digest
                try:
                    payload['scheduler_q'].put_nowait(frame)
                except queue.Full:
                    raise RuntimeError('Ingress-to-scheduler transport full; no input dropped')
                buffered.append(json.dumps({k: frame.get(k) for k in (
                    'frame_id', 'task_id', 'source_publish_ns', '_replay_raw_digest',
                    '_replay_actual_publish_ns', '_replay_ingress_received_ns')}) + '\n')
                if len(buffered) >= 64 or frame['outcome'] == 'SOURCE_END':
                    receipts.write(''.join(buffered))
                    buffered.clear()
                if frame['outcome'] == 'SOURCE_END':
                    break
        report = {'ok': True, 'frames': count, 'arrivals': arrivals,
            'raw_sequence_sha256': sequence.hexdigest(), 'dropped': 0,
            'max_ingress_lag_ms': maximum / 1e6, 'cpu': payload['cpu'],
            'boundary': 'source queue read before audit and scheduler forwarding'}
        dump_json(payload['report_path'], report)
        payload['status_q'].put(report)
    except BaseException as exc:
        import traceback
        report = {'ok': False, 'error': repr(exc), 'traceback': traceback.format_exc()}
        dump_json(payload['report_path'], report)
        payload['ready_q'].put(report)
        payload['status_q'].put(report)
        raise


def real_score_audit(rows):
    """Conservative forward-error bound for accumulated floating ns credits.

    Dispatch/empty deduction/undo/settlement use fewer than 64 ledger float
    operations per settled stage. A picosecond fixed tolerance ignores the
    scale and count of accumulated additions; gamma_n accounts for both.
    This changes verification only, never the score formula or outcome.
    """
    records = []
    for row in rows:
        n = 64 * (int(row['settled']) + 1)
        neps = n * sys.float_info.epsilon
        gamma = neps / (1. - neps)
        roles = {}
        for role, error in row['invariant_error_ns'].items():
            scale = sum(abs(float(row.get(key, {}).get(role, 0)))
                        for key in ('origin', 'earned', 'used', 'idle_repaid', 'balance'))
            tolerance = max(.001, gamma * max(1., scale))
            roles[role] = {'error_ns': error, 'roundoff_bound_ns': tolerance}
            if not (abs(error) <= tolerance):
                raise RuntimeError('Score invariant exceeds accumulated roundoff bound')
        if row['inflight_count'] or sum(row['endpoint_loans'].values()):
            raise RuntimeError('Unclosed or illegal GPU service ledger')
        records.append(roles)
    return records


def cmd_two_card_run(args):
    """Four sequential real-GPU arms on the same frozen source and two cards."""
    import copy
    import multiprocessing as mp
    import signal
    from resource_axis import CARD_COUNT
    from queue_feedback_controller import IMPLEMENTATION_REVISION as revision
    if CARD_COUNT != 2 or not args.workload_manifest or not args.gpu_uuids:
        raise SystemExit('two-card run requires the frozen manifest and two explicit UUIDs')
    uuids = args.gpu_uuids.split(',')
    if len(uuids) != 2 or len(set(uuids)) != 2:
        raise SystemExit('Exactly two distinct GPUs are required')
    manifest_path = Path(args.workload_manifest).resolve()
    manifest = load_json(manifest_path)
    if manifest['scene'] != 'high_load_only' or manifest['duration_s'] != 250:
        raise SystemExit('This authorization is the full frozen high-load scene')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    from tools.validate_prefix_pipeline_gpu import prepare_inputs
    from quality_need_v24 import QUALITY_THRESHOLDS
    manifest, _, bank = prepare_inputs(manifest_path, None)
    if QUALITY_THRESHOLDS != manifest['quality_policy']['thresholds']:
        raise RuntimeError('Quality gates differ from frozen contract')
    environment = load_json(Path(manifest['environment']['path']))
    snapshot = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.free,utilization.gpu',
                                       '--format=csv,noheader'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                    '--format=csv,noheader'], text=True)
    devices = {line.split(',')[0].strip(): [s.strip() for s in line.split(',')[1:]]
               for line in snapshot.splitlines()}
    for uuid in uuids:
        if uuid not in devices or devices[uuid][-1] != '0 %':
            raise RuntimeError('Selected GPU is executing work at preflight: ' + uuid)
        if not args.allow_resident_peers and any(line.split(',')[0].strip() == uuid for line in apps.splitlines()):
            raise RuntimeError('Resident peers require explicit shared-use authorization')
    numa = _gpu_pci_numa(uuids)
    topology = _sysfs_cpu_topology()
    allowed, used = os.sched_getaffinity(0), set()
    def choose(node=None):
        for c in sorted(allowed, reverse=True):
            rec = topology.get(c)
            if rec is None or (node is not None and rec.get('numa_node') != node):
                continue
            key = (rec['package_id'], rec['core_id'])
            if key not in used:
                used.add(key)
                return c
        raise RuntimeError('No independent physical CPU core')
    # CPU topology records identify packages/cores; NUMA pools come separately.
    node_cpus = _numa_node_cpus()
    for rec_cpu, rec in topology.items():
        rec['numa_node'] = next((n for n, cs in node_cpus.items() if rec_cpu in cs), None)
    cpu = {'gpu': [choose(numa.get(u)) for u in uuids], 'main': choose(),
           'decision': choose(), 'source': choose(), 'replay': choose()}
    # GPU Head replaces the CPU Head; reuse its already allocated core.
    cpu['ingress'] = cpu['decision']
    for g, uuid in enumerate(uuids):
        os.environ['EDGE_GPU_NUMA_%d' % g] = str(numa.get(uuid, -1))
    order = ('fixed_1a1f', 'qf4r_feedback', 'all_a', 'all_f')
    folder = (Path(args.resume_real_run).resolve() if args.resume_real_run else
              RUNS / ('two_card_real_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    if args.resume_real_run:
        if folder.parent != RUNS.resolve() or (folder / 'RUN_COMPLETE.json').exists():
            raise RuntimeError('Resume requires an incomplete canonical run folder')
    else:
        folder.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(HERE), text=True).strip()
    identity = {'git_commit': commit, 'implementation_revision': revision,
        'pipeline': 'resident_GPU_inline_F_ingress_v3', 'gpu_uuids': uuids, 'cpu': cpu,
        'cpu_budget_note': 'five active physical cores; ingress reuses former CPU Head core',
        'single_model_unprobed_execution': 'direct FAST_FULL / ACCURATE',
        'execution_order': list(order), 'duration_s': 250, 'deadline_ns': manifest['deadline_ns'],
        'manifest_path': str(manifest_path), 'manifest_sha256': sha(manifest_path),
        'input_plan_sha256': manifest['task_plan']['sha256'],
        'trace_sha256': manifest['source_trace']['sha256'],
        'arrivals': manifest['arrivals'], 'allow_resident_peers': args.allow_resident_peers,
        'preflight_gpu': snapshot, 'preflight_contexts': apps,
        'source_provenance': manifest['source_trace']['provenance'],
        'network_return_ns': manifest['network_return_ns'], 'maximum_wall_s': 3600,
        'online_predictions': 'real CNN and FP64 EARLY Head/Need on GPU; Local inputs frozen'}
    if args.resume_real_run:
        previous = load_json(folder / 'identity.json')
        if any(previous[k] != identity[k] for k in ('gpu_uuids', 'manifest_sha256',
                   'input_plan_sha256', 'trace_sha256', 'arrivals', 'execution_order', 'pipeline')):
            raise RuntimeError('Resume hardware or frozen input differs')
        cpu = previous['cpu']
        identity = previous
        dump_json(folder / ('RESUME_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'),
                  {'git_commit': commit, 'reason': 'continue compatible incomplete ingress/GPU-Head run',
                   'original_git_commit': previous['git_commit'], 'original_identity_unchanged': True})
    else:
        dump_json(folder / 'identity.json', identity)
    print(json.dumps({'stage': 'two_card_registered', 'folder': str(folder), **identity}), flush=True)
    ctx = mp.get_context('spawn')
    reports = {}
    acquire_shared_two_gpu_lock()
    monitor_log = (folder / 'gpu_process_monitor.log').open('a')
    monitor = subprocess.Popen(['nvidia-smi', 'pmon', '-s', 'um', '-d', '1'],
                               stdout=monitor_log, stderr=subprocess.STDOUT)
    def timeout(_sig, _frame):
        raise TimeoutError('Two-card arm exceeded 900s wall limit')
    signal.signal(signal.SIGALRM, timeout)
    started_all = time.monotonic()
    try:
        from group_runtime_t011 import group_worker
        for policy in order:
            if time.monotonic() - started_all > 3600:
                raise TimeoutError('Four-way wall budget exhausted')
            signal.alarm(900)
            arm = folder / policy
            if args.resume_real_run and (arm / 'group_summary.json').exists():
                summary = load_json(arm / 'group_summary.json')
                source = load_json(arm / 'source_replay.json')
                score = real_score_audit(summary['service_debt'])
                if (summary['arrivals'] != manifest['arrivals'] or summary['ended'] != manifest['arrivals']
                        or any(summary[k] for k in ('frame_gap_seen', 'pipeline_pending_decisions',
                            'pipeline_early_decisions', 'pipeline_pending_releases'))
                        or summary['input_audit']['raw_sequence_sha256'] != source['raw_sequence_sha256']):
                    raise RuntimeError('Completed arm is not reusable')
                reports[policy] = {'ok': True, 'reused_completed': True,
                    'original_gpu_execution_commit': identity['git_commit'],
                    'arrivals': summary['arrivals'], 'ended': summary['ended'],
                    'input_audit': summary['input_audit'], 'source': source,
                    'score_audit': score, 'actor_inventory': summary['actor_inventory']}
                dump_json(arm / 'score_roundoff_audit.json', score)
                print(json.dumps({'stage': 'completed_arm_reused', 'policy': policy}), flush=True)
                signal.alarm(0)
                continue
            arm.mkdir()
            ref = copy.deepcopy(load_json(Path(environment['baseline']) / 'REFERENCE.json'))
            initial = 2. if policy == 'all_f' else 0. if policy == 'all_a' else 1.
            ref['configuration'].update(initial_fast=initial, axis_min_fast=0., axis_max_fast=2.,
                GPUs_per_group=2, GPU_group_count=1, all_GPU_actor_count=2,
                shared_local_nodes=33, resource_accounting='service_debt',
                delivery_objective='latency_timeout', allow_idle_endpoint_loans=False,
                fixed_gpu_isolation=True, implementation_revision=revision)
            ref['experiment_input'] = {'controller_start_s': 10.}
            reference = arm / 'REFERENCE.json'
            dump_json(reference, ref)
            ready, go, status = ctx.Queue(), ctx.Queue(), ctx.Queue()
            src_ready, src_go, src_status = ctx.Queue(), ctx.Queue(), ctx.Queue()
            in_ready, in_go, in_status = ctx.Queue(), ctx.Queue(), ctx.Queue()
            frames = ctx.Queue(maxsize=32768)
            scheduler_frames = ctx.Queue(maxsize=32768)
            only = {'all_a': 'Accurate', 'all_f': 'Fast'}.get(policy)
            payload = {'name': 'T016_real_' + policy, 'policy': 'fixed_1a1f' if only else policy,
                'gpu_uuids': uuids, 'folder': str(arm), 't0_ns': None,
                'deadline_ns': manifest['deadline_ns'], 'duration_s': 250., 'bank': bank,
                'frame_q': scheduler_frames, 'status_q': status, 'ready_q': ready, 'go_q': go,
                'cpu': cpu, 'reference_path': str(reference), 'reference_sha256': sha(reference),
                'isolated_session': True, 'spec': {'quality_need_enabled': True,
                    'protect_accurate_deadline': True, 'async_prefix_pipeline': True,
                    'gpu_quality_head': True, 'fair_input_processing': True,
                    'inline_fast_continuation': True,
                    'log_process_cpu': cpu['ingress'],
                    'dedicated_ingress': True, 'single_model_direct': True,
                    'finite_queue_policy': True, 'edge_waiting_roots': 64,
                    'edge_model_only': only, 'frozen_input_audit': True,
                    'frozen_plan_path': manifest['task_plan']['path'],
                    'real_network_return': True, 'network_return_ns': manifest['network_return_ns'],
                    'network_upload_ns': environment['costs']['network_ns']}}
            source_payload = {'cpu': cpu['source'], 'trace': manifest['source_trace']['path'],
                'trace_sha256': manifest['source_trace']['sha256'], 'deadline_ns': manifest['deadline_ns'],
                'network_input_delay_ns': 0, 'frozen_input_audit': True,
                'ready_q': src_ready, 'go_q': src_go, 'status_q': src_status,
                'frame_q': frames, 'report_path': str(arm / 'source_replay.json'), 'policy': policy}
            ingress_payload = {'cpu': cpu['ingress'], 'plan': manifest['task_plan']['path'],
                'deadline_ns': manifest['deadline_ns'], 'source_q': frames,
                'scheduler_q': scheduler_frames, 'ready_q': in_ready, 'go_q': in_go,
                'status_q': in_status, 'report_path': str(arm / 'ingress_summary.json'),
                'receipts_path': str(arm / 'ingress_receipts.jsonl.gz')}
            group = ctx.Process(target=group_worker, args=(payload,))
            publisher = ctx.Process(target=trace_replay_worker, args=(source_payload,))
            ingress = ctx.Process(target=ingress_worker, args=(ingress_payload,))
            started = time.monotonic()
            try:
                group.start()
                publisher.start()
                ingress.start()
                gready, sready = ready.get(timeout=600), src_ready.get(timeout=180)
                iready = in_ready.get(timeout=180)
                if not gready.get('ready') or not sready.get('ready') or not iready.get('ready'):
                    raise RuntimeError('Arm startup failed: %r %r %r' % (gready, sready, iready))
                dump_json(arm / 'ready.json', gready)
                dump_json(arm / 'ingress_ready.json', iready)
                epoch = time.perf_counter_ns() + 2_000_000_000
                go.put({'t0_ns': epoch})
                src_go.put({'t0_ns': epoch})
                in_go.put({'t0_ns': epoch})
                print(json.dumps({'stage': 'real_arm_started', 'policy': policy, 'pid': group.pid,
                                  'publisher_pid': publisher.pid, 'epoch': epoch}), flush=True)
                source_result = src_status.get(timeout=310)
                ingress_result = in_status.get(timeout=120)
                result = status.get(timeout=120)
                group.join(timeout=30)
                publisher.join(timeout=10)
                ingress.join(timeout=10)
                if (not source_result.get('ok') or not result.get('ok') or not ingress_result.get('ok')
                        or group.exitcode != 0 or publisher.exitcode != 0 or ingress.exitcode != 0):
                    raise RuntimeError('Arm failed: %r %r %r' % (source_result, ingress_result, result))
                summary = result['summary']
                audit = summary['input_audit']
                if (summary['arrivals'] != manifest['arrivals'] or summary['ended'] != manifest['arrivals']
                        or summary['frame_gap_seen'] or audit['frames'] != source_result['frames']
                        or audit['arrivals'] != manifest['arrivals']
                        or ingress_result['arrivals'] != audit['arrivals']
                        or ingress_result['frames'] != audit['frames']
                        or ingress_result['raw_sequence_sha256'] != audit['raw_sequence_sha256']
                        or audit['raw_sequence_sha256'] != source_result['raw_sequence_sha256']):
                    raise RuntimeError('Frozen input integrity or task closure failed')
                if reports and audit['raw_sequence_sha256'] != next(iter(reports.values()))['input_audit']['raw_sequence_sha256']:
                    raise RuntimeError('Arms received different raw inputs')
                if any(summary[k] for k in ('pipeline_pending_decisions', 'pipeline_early_decisions',
                                           'pipeline_pending_releases', 'pending_input_outcomes')):
                    raise RuntimeError('Pipeline did not close')
                score = real_score_audit(summary['service_debt'])
                dump_json(arm / 'score_roundoff_audit.json', score)
                checked = 0
                with (arm / 'components.jsonl').open() as stream:
                    for line in stream:
                        c = json.loads(line)
                        if not c.get('resource', '').startswith('gpu:'):
                            continue
                        card = int(c['resource'].split(':')[1])
                        fast = c['kind'].startswith('FAST_')
                        if (policy == 'fixed_1a1f' and fast != (card == 0)) or (
                            only and fast != (only == 'Fast')):
                            raise RuntimeError('Physical role isolation failed')
                        checked += 1
                reports[policy] = {'ok': True, 'wall_s': time.monotonic() - started,
                    't0_ns': epoch, 'observed_stop_ns': time.perf_counter_ns(),
                    'arrivals': summary['arrivals'], 'ended': summary['ended'], 'input_audit': audit,
                    'gpu_components': checked, 'source': source_result, 'ingress': ingress_result,
                    'actor_inventory': gready['actor_inventory']}
                reports[policy]['score_audit'] = score
                dump_json(folder / 'comparison_status.json', reports)
                print(json.dumps({'stage': 'real_arm_complete', 'policy': policy, **reports[policy]}), flush=True)
            finally:
                if group.pid:
                    try:
                        if os.getpgid(group.pid) == group.pid and group.exitcode != 0:
                            os.killpg(group.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    group.join(timeout=5)
                if publisher.pid and publisher.is_alive():
                    publisher.terminate()
                    publisher.join(timeout=5)
                if ingress.pid and ingress.is_alive():
                    ingress.terminate()
                    ingress.join(timeout=5)
                signal.alarm(0)
        for key in ('task_plan', 'source_trace', 'bank_manifest', 'environment'):
            if sha(manifest[key]['path']) != manifest[key]['sha256']:
                raise RuntimeError('Frozen input changed during execution: ' + key)
        dump_json(folder / 'RUN_COMPLETE.json', {'ok': True, 'identity': identity, 'arms': reports})
    except BaseException as exc:
        import traceback
        failure = folder / 'RUN_FAILED.json'
        if failure.exists():
            failure = folder / ('RUN_FAILED_resume_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
        dump_json(failure, {'error': repr(exc), 'traceback': traceback.format_exc(), 'arms': reports})
        raise
    finally:
        signal.alarm(0)
        monitor.terminate()
        monitor.wait(timeout=5)
        monitor_log.close()
        for handle in _SHARED_LOCK_HANDLE:
            handle.close()
        _SHARED_LOCK_HANDLE.clear()
    return folder


def cmd_four_card_run(args):
    """Reuse the production runtime, actors, lock and calibration for four cards."""
    import copy
    import multiprocessing as mp
    import signal
    from resource_axis import CARD_COUNT
    if CARD_COUNT != 4 or not args.source_trace or not args.scenario_config:
        raise SystemExit('four-card run requires explicit four-card geometry, scene and frozen trace')
    execution_order = (POLICIES if args.four_card_order == 'fixed-first'
                       else tuple(reversed(POLICIES)))
    trace = Path(args.source_trace).resolve()
    scene = load_json(Path(args.scenario_config))
    merged = load_json(trace.parent/'merge_report.json')
    plan_path = Path(scene['input_plan_path'])
    if scene['total_nodes'] != 33 or len(set(scene['affected_nodes'])) != 20:
        raise SystemExit('expected authorized 33 Local / 20 degraded scene')
    deadline_ns = int(scene['deadline_ns'])
    if (scene['duration_s'] != 250 or deadline_ns not in (25_000_000, 50_000_000)
            or float(scene['deadline_ms']) * 1e6 != deadline_ns):
        raise SystemExit('expected full 250s scene and coherent original arrival+25ms/50ms')
    if merged.get('deadline_ns', 50_000_000) != deadline_ns:
        raise SystemExit('Local capture and GPU scene deadlines differ')
    network_input_delay_ns = int(round(float(scene.get('network_input_delay_ms', 0))*1e6))
    if not 0 <= network_input_delay_ns <= 50_000_000:
        raise SystemExit('input-transport delay must be between 0 and 50ms')
    if merged['duration_s'] != 250 or merged['arrivals'] != scene['expected_arrivals']:
        raise SystemExit('short/incomplete Local capture cannot launch formal four-card run')
    if sha(plan_path) != scene['input_plan_sha256'] or sha(trace) != merged['trace_sha256']:
        raise SystemExit('frozen plan/trace hash mismatch')
    checks = load_json(HERE/'results/T016/credit_validation_20261009/four_card_cpu/validation.json')
    if not checks.get('passed') or checks.get('cards') != 4:
        raise SystemExit('four-card CPU wiring checks missing/failed')
    if not checks.get('module_hashes') or any(sha(HERE/name)!=digest for name,digest in checks['module_hashes'].items()):
        raise SystemExit('four-card checks belong to a different source version')
    trace_checks = load_json(HERE/'results/T016/credit_validation_20261009/trace_timing_cpu/validation.json')
    if not trace_checks.get('passed'):
        raise SystemExit('real-clock trace replay check missing/failed')
    if network_input_delay_ns and trace_checks.get('network_input_delay_ns') != network_input_delay_ns:
        raise SystemExit('input-transport real-clock check missing for this delay')
    preflight = preflight_payload()
    if not preflight['pass']:
        raise SystemExit('four-card resource preflight failed: %r' % preflight['errors'])
    uuids = preflight['fixed_uuids'] + preflight['dynamic_uuids']
    numa = _gpu_pci_numa(uuids)
    topology, node_cpus = _sysfs_cpu_topology(), _numa_node_cpus()
    allowed, used = os.sched_getaffinity(0), set()
    def physical_cpu(node=None):
        pool = node_cpus.get(node, []) if node is not None else sorted(topology)
        for cpu in sorted(pool):
            if cpu not in allowed or cpu not in topology:
                continue
            rec = topology[cpu]
            key = (rec['package_id'],rec['core_id'])
            if key not in used:
                used.add(key)
                return cpu
        raise RuntimeError('no independent physical control core available')
    gpu_cpus = [physical_cpu(numa[u]) for u in uuids]
    for g,u in enumerate(uuids):
        os.environ['EDGE_GPU_NUMA_%d'%g] = str(numa[u])
    cpu = {'gpu':gpu_cpus,'main':physical_cpu(),'decision':physical_cpu(),
           'source':physical_cpu(),'replay':physical_cpu()}
    run_id = 'four_card_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = RUNS/run_id
    folder.mkdir(parents=True,exist_ok=False)
    ref = copy.deepcopy(load_json(args.reference_json or EVIDENCE/'REFERENCE.json'))
    ref['configuration'].update(initial_fast=2.,axis_min_fast=0.,axis_max_fast=4.,grid=.1,
         resource_accounting='service_debt',delivery_objective='latency_timeout',
         fixed_gpu_isolation=True,
         allow_idle_endpoint_loans=False,
         policy_definitions={'fixed_1a1f':'static physical 2F2A; no cross-role idle borrowing',
                             'qf4r_feedback':'dynamic allocation; zero-share role cannot borrow card'},
         implementation_revision=IMPLEMENTATION_REVISION)
    # The controller has no phase instant/degraded identity/ground-truth input.
    ref['experiment_input'] = {'controller_start_s':scene['controller_start_s']}
    ref_path = folder/'REFERENCE.json'
    dump_json(ref_path,ref)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(HERE),text=True).strip()
    identity = {'run_id':run_id,'git_commit':commit,'implementation_revision':IMPLEMENTATION_REVISION,
        'execution_order':list(execution_order),
        'input_plan_sha256':scene['input_plan_sha256'],'trace_sha256':merged['trace_sha256'],
        'source_manifest':merged,'gpu_uuids':uuids,'cpu':cpu,'duration_s':250,
        'deadline_ms':deadline_ns/1e6,'scenario_config':str(Path(args.scenario_config).resolve()),
        'network_input_delay_ms':network_input_delay_ns/1e6,
        'network_model':'constant image-ready input delay; control metadata and Local results unchanged; existing IPC and GPU upload retained',
        'comparison':'same four GPUs sequential isolated fixed2F2A versus dynamic; neither arm borrows zero-share cards; real Local trace / live GPU',
        'reference_sha256':sha(ref_path),'code_sha256':{p.name:sha(p) for p in HERE.glob('*.py')},
        'preflight':preflight,'CPU_geometry_checks':checks}
    dump_json(folder/'identity.json',identity)
    print(json.dumps({'stage':'four_card_run_registered','folder':str(folder),'cpu':cpu,'gpu_uuids':uuids}),flush=True)
    acquire_shared_two_gpu_lock()
    def timeout(_sig,_frame):
        raise TimeoutError('four-card arm startup/run wall exceeded 900s')
    signal.signal(signal.SIGALRM,timeout)
    ctx = mp.get_context('spawn')
    arm_reports = {}
    bank = None
    try:
        # Reuse the same CPU-preprocessed image bank as the original real actors.
        os.environ['CUDA_VISIBLE_DEVICES']=''
        from shared_local_t011 import SharedLocalSource, FrozenSourceCosts
        source = SharedLocalSource(scene,[],FrozenSourceCosts(ref,EVIDENCE),folder=folder,nodes=33)
        source.prepare_bank(); bank = source.bank
        from group_runtime_t011 import group_worker
        for policy in execution_order:
            signal.alarm(900)
            arm = folder/policy
            arm.mkdir()
            ready, go, status = ctx.Queue(),ctx.Queue(),ctx.Queue()
            src_ready, src_go, src_status = ctx.Queue(),ctx.Queue(),ctx.Queue()
            frame_q = ctx.Queue(maxsize=32768)
            os.environ['T014_REPLAY_CPU_%s'%policy]=str(cpu['replay'])
            payload={'name':'T016_4gpu_'+policy,'spec':{'quality_need_enabled':True,
                      'protect_accurate_deadline':True,'local_quality_route_mode':'quality_to_accurate'},
                     'policy':policy,'gpu_uuids':uuids,'folder':str(arm),'t0_ns':None,
                     'deadline_ns':deadline_ns,'duration_s':250.,'bank':bank,'frame_q':frame_q,
                     'status_q':status,'ready_q':ready,'go_q':go,'cpu':cpu,
                     'reference_path':str(ref_path),'reference_sha256':identity['reference_sha256'],
                     'isolated_session':True}
            src_payload={'cpu':cpu['source'],'trace':str(trace),'trace_sha256':merged['trace_sha256'],
                'deadline_ns':deadline_ns,
                'network_input_delay_ns':network_input_delay_ns,
                'ready_q':src_ready,'go_q':src_go,'status_q':src_status,'frame_q':frame_q,
                'report_path':str(arm/'source_replay.json'),'policy':policy}
            group = ctx.Process(target=group_worker,args=(payload,))
            publisher = ctx.Process(target=trace_replay_worker,args=(src_payload,))
            started = time.monotonic()
            try:
                # Each process owns its descendant tree so emergency cleanup
                # can never signal unrelated user/foreign processes.
                with (folder/'GPU_start_registrations.jsonl').open('a',encoding='utf8') as stream:
                    stream.write(json.dumps({'utc':utcnow(),'policy':policy,'gpu_actors':4,
                        'gpu_uuids':uuids,'git_commit':commit,'wall_limit_s':900,'drain_limit_s':60})+'\n')
                group.start(); publisher.start()
                gready = ready.get(timeout=600)
                sready = src_ready.get(timeout=180)
                if not gready.get('ready') or not sready.get('ready'):
                    raise RuntimeError('four-card startup failed: %r %r'%(gready,sready))
                dump_json(arm/'ready.json',gready)
                epoch = time.perf_counter_ns()+2_000_000_000
                go.put({'t0_ns':epoch}); src_go.put({'t0_ns':epoch})
                print(json.dumps({'stage':'four_card_arm_started','policy':policy,'t0_ns':epoch}),flush=True)
                src_result = src_status.get(timeout=310)
                result = status.get(timeout=60)
                group.join(timeout=60); publisher.join(timeout=10)
                if not src_result.get('ok') or not result.get('ok') or group.exitcode != 0 or publisher.exitcode != 0:
                    raise RuntimeError('four-card execution failed: %r %r'%(src_result,result))
                summary = result['summary']
                if summary['arrivals'] != merged['arrivals'] or summary['ended'] != merged['arrivals'] or summary['frame_gap_seen']:
                    raise RuntimeError('four-card task/frame closure mismatch')
                if any(abs(v)>1e-3 for row in summary['service_debt'] for v in row.get('invariant_error_ns',{}).values()):
                    raise RuntimeError('four-card service debt conservation failed')
                if summary.get('allow_idle_endpoint_loans') is not False or any(
                    sum(row['endpoint_loans'].values()) for row in summary['service_debt']):
                    raise RuntimeError('an arm borrowed a zero-share card')
                if policy == 'fixed_1a1f':
                    checked = 0
                    with (arm/'components.jsonl').open(encoding='utf8') as rows:
                        for line in rows:
                            component = json.loads(line)
                            resource = component.get('resource', '')
                            if not resource.startswith('gpu:'):
                                continue
                            card = int(resource.split(':')[1])
                            fast = component['kind'].startswith('FAST_')
                            if fast != (card < 2):
                                raise RuntimeError('fixed GPU role isolation violated')
                            checked += 1
                    if not checked:
                        raise RuntimeError('fixed arm has no checked GPU work')
                    dump_json(arm/'fixed_isolation_audit.json',
                              {'passed':True,'gpu_components_checked':checked,
                               'Fast_cards':[0,1],'Accurate_cards':[2,3],
                               'cross_role_dispatches':0,'endpoint_loans':0})
                arm_reports[policy]={'ok':True,'wall_s':time.monotonic()-started,'source':src_result,
                                     'arrivals':summary['arrivals'],'ended':summary['ended']}
                dump_json(folder/'comparison_status.json',arm_reports)
                print(json.dumps({'stage':'four_card_arm_complete','policy':policy,**arm_reports[policy]}),flush=True)
            finally:
                if group.pid:
                    # The new worker creates its own session before actors;
                    # its descendants remain in that recorded process group.
                    try:
                        if group.is_alive() and os.getpgid(group.pid) == group.pid:
                            os.killpg(group.pid,signal.SIGTERM)
                        elif group.exitcode not in (None,0):
                            os.killpg(group.pid,signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    group.join(timeout=5)
                if publisher.pid and publisher.is_alive():
                    publisher.terminate(); publisher.join(timeout=5)
                signal.alarm(0)
        dump_json(folder/'RUN_COMPLETE.json',{'ok':True,'identity':identity,'arms':arm_reports})
    except BaseException as exc:
        import traceback
        dump_json(folder/'RUN_FAILED.json',{'ok':False,'error':repr(exc),'traceback':traceback.format_exc(),'arms':arm_reports})
        raise
    finally:
        signal.alarm(0)
        for handle in _SHARED_LOCK_HANDLE:
            handle.close()
        _SHARED_LOCK_HANDLE.clear()
    return folder


def build_parser():
    parser = argparse.ArgumentParser(
        description='T014 shared-Local-frontend GPU-control QF-4R joint entry point')
    parser.add_argument('--prepare-only', action='store_true', help='validate frozen inputs only (pure CPU)')
    parser.add_argument('--preflight', action='store_true', help='read-only four-GPU + CPU topology preflight')
    parser.add_argument('--gpu-pair-order', choices=('standard','crossover'), default='standard',
                        help='swap the two original GPU pairs for a balanced hardware comparison; preserve input and algorithms')
    parser.add_argument('--check', action='store_true', help='run the ONE T014 CPU check suite (pure CPU)')
    parser.add_argument('--phase', default='R1,R2,R3',
                        help='CPU check groups: R1,R2,R3 (T014 TASK s2)')
    parser.add_argument('--reference-json', dest='reference_json', default=None,
                        help='the ONE explicit effective REFERENCE.json (default: env '
                             'TASK_REFERENCE_JSON, else results/<TASK>/evidence/REFERENCE.json)')
    parser.add_argument('--out', default=None,
                        help='output path for --check / --preflight / --deploy-audit / --joint-facts / --summarize')
    parser.add_argument('--freeze', action='store_true', help='freeze the online manifest (pure CPU)')
    parser.add_argument('--deploy-audit', dest='deploy_audit', action='store_true',
                        help='per-file SHA256 deployment gate before the joint run (pure CPU)')
    parser.add_argument('--joint-run', dest='joint_run', action='store_true',
                        help='run the ONE joint concurrent two-arm 250 s experiment')
    parser.add_argument('--four-card-run', action='store_true',
                        help='same four GPUs sequential fixed/dynamic against a frozen real Local trace')
    parser.add_argument('--two-card-run', action='store_true', help='sequential real fixed/dynamic/all-A/all-F')
    parser.add_argument('--workload-manifest')
    parser.add_argument('--gpu-uuids', help='same comma-separated physical UUID pair for all arms')
    parser.add_argument('--allow-resident-peers', action='store_true', help='authorized sharing with memory-resident peers')
    parser.add_argument('--resume-real-run', help='reuse completed arms after verification-only repair')
    parser.add_argument('--four-card-order', choices=('fixed-first','dynamic-first'), default='fixed-first',
                        help='balance sequential execution order; keep the same GPUs, inputs and algorithms')
    parser.add_argument('--source-trace', default=None, help='validated merged real Local frames.jsonl')
    parser.add_argument('--joint-facts', dest='joint_facts', action='store_true',
                        help='write the compact joint facts used to stamp RUN_COMPLETE.json')
    parser.add_argument('--verify', action='store_true', help='offline verification of a finished joint run')
    parser.add_argument('--summarize', action='store_true', help='offline one-pass analysis of the joint run')
    parser.add_argument('--joint', default=None, help='joint attempt folder or id (default: latest)')
    parser.add_argument('--schema-version', dest='schema_version', default=T004_SCHEMA)
    parser.add_argument('--seed', type=int, default=7904001)
    parser.add_argument('--deadline-ms', type=float, default=50.0)
    parser.add_argument('--input-plan', default=None)
    parser.add_argument('--scenario-config', default=None)
    return parser


def main(argv=None):
    global PREFERRED_FIXED_UUIDS, PREFERRED_DYNAMIC_UUIDS, GPU_PAIR_ORDER
    args = build_parser().parse_args(argv)
    GPU_PAIR_ORDER = args.gpu_pair_order
    pairs = DEFAULT_GPU_PAIRS if GPU_PAIR_ORDER == 'standard' else DEFAULT_GPU_PAIRS[::-1]
    PREFERRED_FIXED_UUIDS, PREFERRED_DYNAMIC_UUIDS = map(list,pairs)
    if args.prepare_only:
        cmd_prepare(args)
    elif args.preflight:
        cmd_preflight(args)
    elif args.check:
        cmd_check(args)
    elif args.freeze:
        cmd_freeze(args)
    elif args.deploy_audit:
        cmd_deploy_audit(args)
    elif args.verify:
        cmd_verify(args)
    elif args.summarize:
        cmd_summarize(args)
    elif args.joint_facts:
        cmd_joint_facts(args)
    elif args.two_card_run:
        cmd_two_card_run(args)
    elif args.four_card_run:
        cmd_four_card_run(args)
    elif args.joint_run:
        cmd_joint_run(args)
    else:
        raise SystemExit('choose one of --prepare-only / --preflight / --check / --freeze / '
                         '--deploy-audit / --joint-run / --joint-facts / --verify / --summarize')


if __name__ == '__main__':
    main()
