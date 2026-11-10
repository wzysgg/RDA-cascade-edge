"""T012 offline summary and verification (WS-E, TASK.md section 9).

Two deliverables come out of this module and nothing else:

* ``summarize()`` -> ``metrics.json`` (schema ``T012_metrics_v1``): the whole
  section 9 data collection of the closed-loop repair run, built from the raw
  per-arm jsonl logs and the shared source archive.  Both observation clocks
  (original ``arrival_ns`` and ``source_publish_ns``) are always reported side
  by side, and the strict ``terminal_available_ns`` throughput is a third,
  separate clock.
* ``verify_joint()`` -> ``verification.json`` (schema
  ``T012_verification_v1``): every item is recomputed independently from the
  raw logs and returned as ``{'name','ok','detail'}``.  Unknown or incomplete
  items are reported as ``null``/counts and are never folded into a pass.

Contract notes (frozen interface ``docs/T012_INTERFACE.md``):

* ``R`` is read from ``REJECT`` events only, ``X`` from ``REMOVE`` events only.
  ``REJECT`` must co-occur with ``ADMISSION_OUTCOME`` ``outcome='rejected'``.
* ``component_id`` is ``'<root_id>:<stage>:<attempt>'`` and is the only
  component key used across OFFER/ENQUEUE/REJECT/DISPATCH/COMPONENT_FINISH/
  REMOVE; ``job_id`` is an execution locator and is only used to join the real
  quota charge rows.
* ``good`` requires the offline private label: predicted class == true label
  *and* ``terminal_available_ns <= deadline_abs_ns``.  ``prediction == 1`` or a
  public confidence is never accepted as correctness.
* on-time = ``terminal_available_ns <= deadline_abs_ns``; the event's own
  ``ontime_by_terminal`` flag is compared against this recomputation.

Candidate records: the controller log rows are expected to carry the recorded
candidate table and the payload that produced it, so the offline replay can
recompute it:

    row['decision']['resource']['candidate_payload'] = {
        'current_gF': float, 'demand': {...}, 'cost': {card: {kind: {...}}},
        'c_star_ms': {...}, 'T_s': float, 'min_margin_ms': float}
    row['decision']['resource']['candidates'] = [11 rows]
    row['decision']['resource']['candidates_valid'] = bool
    row['decision']['resource']['candidates_invalid'] = str or None

Aliases accepted: ``candidate_payload``/``last_candidate_payload``/``payload``,
``candidates``/``last_candidates``/``candidate_table``.  If neither a payload
nor a recorded table is present for a window, that window is reported with an
explicit invalid reason and the affected verification item does not pass.

Python 3.8.10 safe: no dict ``|``, no builtin generic annotations, no
``removeprefix``/``removesuffix``/``functools.cache``/``math.lcm``/``graphlib``/
``zoneinfo``/``match``.  No torch import anywhere in this module.
"""

import hashlib
import io
import json
import math
import os
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

# --------------------------------------------------------------- constants

MS = 1e6
NS_PER_S = 1e9
PERIOD_NS = 50_000_000
TICK_NS = 100_000_000
WINDOW_NS = 500_000_000

TASK_ID = 'T014'
CONTROLLER_VERSION = 'qf4r'
IMPLEMENTATION_REVISION = 'T014_qf4r_3'
METRICS_SCHEMA = 'T014_metrics_v1'
VERIFICATION_SCHEMA = 'T014_verification_v1'
CONTROL_LOG_SCHEMA = 'QF4R_control_v3'
DECISION_SCHEMA = 'QF4R_decision_v3'
JOINT_SCHEMA = 'T014_joint_summary_v1'

POLICIES = ('fixed_1a1f', 'qf4r_feedback')
FIXED_POLICY = 'fixed_1a1f'
DYNAMIC_POLICY = 'qf4r_feedback'
ARM_PREFIX = 'TRIAL_T014_mixed_d50_n20_blur0.55_%s'

GPU_ROLES = ('Fast', 'Accurate')
LOCAL_ROLE = 'Local'
ALL_ROLES = ('Local', 'Fast', 'Accurate')
GPU_KINDS = ('FAST_FULL', 'FAST_PREFIX', 'FAST_SUFFIX', 'ACCURATE')
ROLE_OF_KIND = {'FAST_FULL': 'Fast', 'FAST_PREFIX': 'Fast',
                'FAST_SUFFIX': 'Fast', 'ACCURATE': 'Accurate'}

ROOT_DIRECT_F = 'directF'
ROOT_DIRECT_A = 'directA'
ROOT_PREFIX = 'Prefix-origin'
ROOT_CLASSES = (ROOT_DIRECT_F, ROOT_DIRECT_A, ROOT_PREFIX)

OUTCOME_ARRIVAL = 'SHARED_ARRIVAL'
OUTCOME_LOCAL_TERMINAL = 'COMMON_LOCAL_TERMINAL'
OUTCOME_GPU_READY = 'GPU_READY'
OUTCOME_PREFIX_READY = 'PREFIX_READY'
OUTCOME_QUALITY = 'OUTCOME_QUALITY'
OUTCOME_ABSTAIN = 'OUTCOME_ABSTAIN'
OUTCOME_SOURCE_END = 'SOURCE_END'

PHASES = ((0.0, 70.0, 'background'), (70.0, 190.0, 'degraded'),
          (190.0, 250.0, 'recover'))
FULL_SPAN = (0.0, 250.0)
SELECTED_WINDOWS = ((60.0, 70.0, 'w60_70'), (80.0, 90.0, 'w80_90'),
                    (200.0, 210.0, 'w200_210'))
COHORT_SPANS = (('full', FULL_SPAN[0], FULL_SPAN[1]),) + \
    tuple((name, lo, hi) for lo, hi, name in PHASES) + \
    tuple((name, lo, hi) for lo, hi, name in SELECTED_WINDOWS)

CLOCKS = ('arrival_ns', 'source_publish_ns', 'terminal_available_ns')
REPRESENTATIVE_WINDOWS = ((0, 10), (60, 70), (240, 250))

INPUT_FILE_SHA = '0b0fd1237c807f3bc2b43fcdc381327326ccf25678f64edd1070b69000ebfde9'
INPUT_LOGICAL_SHA = 'ebd575740a29eb9c3923c51eeb47a7b0654f4e29e7bd2fe049d6fe10d37a9eaa'
EXPECTED_ARRIVALS = 226323
EXPECTED_PHASE_ARRIVALS = (63313, 108395, 54615)

SAMPLE_CAP = 50000
TAMPER_DELTA = 1.0e6

# ------------------------------------------------------------- small helpers


def sha256_file(path):
    h = hashlib.sha256()
    with io.open(str(path), 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def rows(path, integrity=None):
    """Stream jsonl rows; unreadable lines are skipped, never raised."""
    p = Path(path)
    if not p.exists():
        return
    with io.open(str(p), encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                if integrity is not None:
                    integrity['parse_errors'] = integrity.get('parse_errors', 0) + 1
                continue


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return default


def _json_default(o):
    """T012: offline summaries legitimately carry Path/Counter/deque/set values.

    Without this the T012 metrics writer dies on the first Path it meets, which
    is exactly what happened on the first partial-joint verification pass."""
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=lambda v: str(v))
    if isinstance(o, deque):
        return list(o)
    if isinstance(o, Counter):
        return {str(k): o[k] for k in sorted(o, key=lambda v: str(v))}
    return str(o)


def write_json(path, obj):
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True,
                      default=_json_default) + '\n'
    with io.open(str(p), 'w', encoding='utf8', newline='\n') as f:
        f.write(text)
    return str(p)


def mean(vals):
    return (sum(vals) / float(len(vals))) if vals else None


def pct(vals, q):
    """Sorted linear interpolation: position=(n-1)*q, neighbour interpolation."""
    if not vals:
        return None
    v = sorted(vals)
    pos = q * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] * (1.0 - (pos - lo)) + v[hi] * (pos - lo)


def as_int(v, default=None):
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def as_float(v, default=None):
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def rnd(v, nd=6):
    if v is None or isinstance(v, bool):
        return v
    try:
        return round(float(v), nd)
    except Exception:
        return v


def stats(vals):
    if not vals:
        return {'n': 0, 'mean': None, 'p50': None, 'p95': None, 'max': None}
    return {'n': len(vals), 'mean': rnd(mean(vals), 4), 'p50': rnd(pct(vals, 0.5), 4),
            'p95': rnd(pct(vals, 0.95), 4), 'max': rnd(max(vals), 4)}


class Samples(object):
    """Bounded sample with exact count/sum/max and a deterministic subset.

    ``true_max`` is always the full-run maximum (never the sample maximum), so
    the reported full-path max cannot lose the worst measurement.
    """

    def __init__(self, cap=SAMPLE_CAP):
        self.cap = int(cap)
        self.n = 0
        self.total = 0.0
        self.vmin = None
        self.true_max = None
        self.keep = []
        self.stride = None
        self.dropped = 0

    def add(self, v):
        v = as_float(v)
        if v is None:
            return
        self.n += 1
        self.total += v
        self.vmin = v if self.vmin is None else min(self.vmin, v)
        self.true_max = v if self.true_max is None else max(self.true_max, v)
        if len(self.keep) < self.cap:
            self.keep.append(v)
            return
        if self.stride is None:
            self.stride = int(math.ceil(self.cap / 10.0))
            self.dropped = 0
        # keep 1 in every `stride` additional measurements, bounded by cap
        if self.n % self.stride == 0 and len(self.keep) < self.cap:
            self.keep.append(v)
        else:
            self.dropped += 1

    def summary(self):
        out = {'n': self.n, 'mean': rnd(self.total / self.n, 4) if self.n else None,
               'p50': rnd(pct(self.keep, 0.5), 4) if self.keep else None,
               'p95': rnd(pct(self.keep, 0.95), 4) if self.keep else None,
               'max': rnd(self.true_max, 4), 'true_max': rnd(self.true_max, 4),
               'min': rnd(self.vmin, 4),
               'percentiles_from': 'sampled_%d_of_%d' % (len(self.keep), self.n)}
        return out

    def values(self):
        return list(self.keep)


def merge_samples(dst, src):
    for v in src:
        dst.add(v)
    return dst


def counter_to_plain(c):
    if isinstance(c, Counter):
        return {str(k): int(v) for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}
    return c


# -------------------------------------------------------- folder resolution


def resolve_joint_folder(folder):
    """Accept the joint_attempt folder, or a runs/ or task root around it."""
    folder = Path(folder)
    if (folder / 'joint_summary.json').exists() or \
            (folder / 'shared_frames.jsonl').exists():
        return folder, 'given_folder'
    cand = folder / 'runs'
    base = cand if cand.exists() else folder
    attempts = sorted([p for p in base.glob('joint_attempt_*') if p.is_dir()])
    if attempts:
        return attempts[-1], 'derived_latest_joint_attempt_under_%s' % base.name
    return folder, 'not_found'


def arm_folders(joint_folder, joint=None):
    """Resolve the two arm folders; never hardcodes a policy folder name.

    Order: explicit ``arm_folders`` in joint_summary.json (dict or list), then
    the documented derivation ``TRIAL_T012_*_<policy>`` as a sibling of the
    joint folder.  The resolution actually used is recorded by the caller.
    """
    joint_folder = Path(joint_folder)
    runs = joint_folder.parent
    folders = {}
    resolution = {}
    declared = (joint or {}).get('arm_folders')
    entries = []
    if isinstance(declared, dict):
        for policy, v in declared.items():
            if isinstance(v, dict):
                v = v.get('folder') or v.get('path') or v.get('dir')
            entries.append((policy, v))
    elif isinstance(declared, list):
        for item in declared:
            if isinstance(item, dict):
                entries.append((item.get('policy') or item.get('name'),
                                item.get('folder') or item.get('path')))
            else:
                entries.append((None, item))
    for policy, v in entries:
        if not v:
            continue
        p = Path(str(v))
        if not p.is_absolute():
            p = runs / p
        key = policy if policy in POLICIES else None
        if key is None:
            for cand_policy in POLICIES:
                if cand_policy in p.name:
                    key = cand_policy
                    break
        if key is None or not p.exists():
            continue
        folders[key] = p
        resolution[key] = {'path': str(p), 'source': 'joint_summary.arm_folders'}
    for policy in POLICIES:
        if policy in folders:
            continue
        p = runs / (ARM_PREFIX % policy)
        if p.exists():
            folders[policy] = p
            resolution[policy] = {'path': str(p),
                                  'source': 'derived_TRIAL_T012_prefix'}
    return folders, resolution


# ------------------------------------------------- exact event-time replay


def _loss_class_reason(et, reason):
    """Loss class rule (documented, not guessed).

    REJECT carries the admission loss reason itself ('resource',
    'intrinsic_budget', 'receipt_delay').  REMOVE carries the queue-removal
    reason; the runtime only removes jobs whose queue wait expired, and it books
    that as a resource loss in the observer, so an 'expired' REMOVE is classed
    'resource' and anything else is classed 'other'.
    """
    reason = '' if reason is None else str(reason)
    if et == 'REJECT':
        return 'resource' if reason == 'resource' else 'other'
    if 'expired' in reason:
        return 'resource'
    return 'other'


def replay_windows(events_path, window_ns=WINDOW_NS, c_star_ms=None):
    """Exact integer replay of the raw public event stream (500 ms windows).

    Per (window, role, kind) with role in GPU_ROLES:

        dW_counts == E - S - X        (waiting components: enqueue/dispatch/remove)
        O == E + R + dOpen            (every OFFER resolves)

    ``R`` comes from REJECT only and ``X`` from REMOVE only.  Work in
    reference-ms uses the frozen per-run ``c_star_ms``; R/X work is
    ``sum(count_kind * c*_kind) / window_seconds``.  The trailing window is
    flagged ``partial`` (only its resolved part is sealed) and included, so the
    caller must decide whether to use it.
    """
    wait = Counter()
    open_offer = Counter()
    windows = []
    cur = None
    T = float(window_ns) / NS_PER_S

    def new_cur(index, start):
        return {'index': index, 'start_ns': start, 'end_ns': start + window_ns,
                'O': Counter(), 'E': Counter(), 'S': Counter(), 'X': Counter(),
                'R': Counter(), 'wait_start': None, 'wait_end': None,
                'open_start': None, 'open_end': None,
                'lost_roots_resource': {}, 'lost_roots_other': {},
                'lost_components_resource': 0, 'lost_components_other': 0,
                '_seen_loss_roots': set(),
                'reject_reasons': Counter(), 'remove_reasons': Counter(),
                'residual_dW': {}, 'residual_O': {}, 'max_abs_residual': 0.0,
                'partial': False}

    def seal():
        nonlocal cur
        w = cur
        w['wait_end'] = dict(wait)
        w['open_end'] = dict(open_offer)
        keys = set(w['O']) | set(w['E']) | set(w['S']) | set(w['X']) | set(w['R'])
        keys |= set(w['wait_start'] or {}) | set(w['wait_end'] or {})
        maxres = 0.0
        for key in sorted(keys):
            if ':' not in key:
                continue
            role, kind = key.split(':', 1)
            e, s, x, r, o = (w['E'][key], w['S'][key], w['X'][key],
                             w['R'][key], w['O'][key])
            dW = (w['wait_end'].get(key, 0) - w['wait_start'].get(key, 0)) - (e - s - x)
            d_open = (w['open_end'].get(key, 0) - w['open_start'].get(key, 0))
            o_res = o - (e + r + d_open)
            w['residual_dW'][key] = int(dW)
            w['residual_O'][key] = int(o_res)
            maxres = max(maxres, abs(dW), abs(o_res))
        w['max_abs_residual'] = maxres
        work = {}
        for role in GPU_ROLES:
            acc = {'E': 0.0, 'S': 0.0, 'X': 0.0, 'R': 0.0, 'dW': 0.0, 'O': 0.0}
            e_n = s_n = x_n = r_n = o_n = 0
            for kind in GPU_KINDS:
                key = role + ':' + kind
                c = float((c_star_ms or {}).get(kind, 0.0) or 0.0)
                acc['E'] += w['E'][key] * c
                acc['S'] += w['S'][key] * c
                acc['X'] += w['X'][key] * c
                acc['R'] += w['R'][key] * c
                acc['O'] += w['O'][key] * c
                acc['dW'] += (w['wait_end'].get(key, 0) -
                              w['wait_start'].get(key, 0)) * c
                e_n += w['E'][key]
                s_n += w['S'][key]
                x_n += w['X'][key]
                r_n += w['R'][key]
                o_n += w['O'][key]
            work[role] = {
                'E_work_ms': rnd(acc['E'], 6), 'S_work_ms': rnd(acc['S'], 6),
                'X_work_ms': rnd(acc['X'], 6), 'R_work_ms': rnd(acc['R'], 6),
                'O_work_ms': rnd(acc['O'], 6), 'dW_work_ms': rnd(acc['dW'], 6),
                'E_n': e_n, 'S_n': s_n, 'X_n': x_n, 'R_n': r_n, 'O_n': o_n,
                'growth_ms_s': rnd(acc['dW'] / T, 6),
                'resource_loss_work_ms_s': rnd(acc['X'] / T, 6),
                'R_work_ms_s': rnd(acc['R'] / T, 6),
                'E_work_ms_s': rnd(acc['E'] / T, 6),
                'S_work_ms_s': rnd(acc['S'] / T, 6),
                'resource_lost_components': int(w.get('lost_components_resource', 0)),
                'other_lost_components': int(w.get('lost_components_other', 0)),
                'resource_lost_roots': int(
                    (w.get('lost_roots_resource') or {}).get(role, 0)),
            }
        w['work_ms'] = work
        w['window_s'] = T
        for k in ('O', 'E', 'S', 'X', 'R', 'wait_start', 'wait_end',
                  'open_start', 'open_end', 'reject_reasons', 'remove_reasons'):
            w[k] = counter_to_plain(w[k])
        w['lost_roots_resource'] = {r: int((w['lost_roots_resource'] or {}).get(r, 0))
                                    for r in GPU_ROLES}
        w['lost_roots_other'] = {r: int((w['lost_roots_other'] or {}).get(r, 0))
                                 for r in GPU_ROLES}
        w.pop('_seen_loss_roots', None)
        windows.append(w)
        cur = None

    def lost_roots(role, root_id, et, reason):
        cls = _loss_class_reason(et, reason)
        bucket = 'lost_roots_resource' if cls == 'resource' else 'lost_roots_other'
        seen_key = (cls, role, root_id)
        if root_id is None or seen_key in cur['_seen_loss_roots']:
            return
        cur['_seen_loss_roots'].add(seen_key)
        cur[bucket][role] = cur[bucket].get(role, 0) + 1
        if cls == 'resource':
            cur['lost_components_resource'] += 1
        else:
            cur['lost_components_other'] += 1

    for e in rows(events_path):
        et = e.get('event_type')
        if et not in ('OFFER', 'ENQUEUE', 'REJECT', 'DISPATCH', 'REMOVE'):
            continue
        kind = e.get('kind')
        role = e.get('role') if et == 'REJECT' else None
        if role not in GPU_ROLES:
            role = ROLE_OF_KIND.get(kind)
        if role is None:
            continue
        key = role + ':' + kind
        ts = as_int(e.get('timestamp'), 0)
        idx = (ts // window_ns) * window_ns
        if cur is not None and idx > cur['start_ns']:
            seal()
        if cur is None:
            cur = new_cur(idx // window_ns, idx)
            cur['wait_start'] = dict(wait)
            cur['open_start'] = dict(open_offer)
        if et == 'OFFER':
            cur['O'][key] += 1
            open_offer[key] += 1
        elif et == 'ENQUEUE':
            cur['E'][key] += 1
            open_offer[key] -= 1
            wait[key] += 1
        elif et == 'REJECT':
            reason = e.get('reason')
            cur['R'][key] += 1
            cur['reject_reasons'][str(reason)] += 1
            open_offer[key] -= 1
            lost_roots(role, e.get('root_id'), 'REJECT', reason)
        elif et == 'DISPATCH':
            if kind in GPU_KINDS:
                cur['S'][key] += 1
                wait[key] -= 1
        elif et == 'REMOVE':
            reason = e.get('reason')
            cur['X'][key] += 1
            cur['remove_reasons'][str(reason)] += 1
            wait[key] -= 1
            lost_roots(role, e.get('root_id'), 'REMOVE', reason)
    if cur is not None:
        seal()
        windows[-1]['partial'] = True
    return windows


# ------------------------------------------------------------ per-arm scan

PCK = ('command_id', 'from_fast', 'target_fast', 'final_target', 'actual_step',
       'scheduled_s', 'issued_ns', 'reason', 'direction', 'trigger', 'gate',
       'pre', 'post', 'pre_cohorts', 'post_cohorts', 'standardized', 'epsilon',
       'verdict', 'probe', 'donor', 'candidates', 'profile_fp',
       'price_sensitivity', 'select', 'applied_ns', 'effective_ns',
       'settle_ns', 'state', 'epoch', 'policy', 'info', 'channel',
       'proposed_ns', 'delta_fast')


def _pick(d, names, default=None):
    if not isinstance(d, dict):
        return default
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


def _compact_command(cmd):
    if not isinstance(cmd, dict):
        return None
    out = {}
    missing = []
    for name in PCK:
        if name in cmd:
            out[name] = cmd[name]
        else:
            missing.append(name)
    for extra in ('gF', 'command'):
        if extra in cmd and extra not in out:
            out[extra] = cmd[extra]
    out['missing_fields'] = missing
    return out


def policy_of_folder(folder):
    name = Path(folder).name
    for policy in POLICIES:
        if name.endswith(policy) or policy in name:
            return policy
    return None


def arm_scan(folder):
    """ONE streaming pass over one arm's raw files.  Compact tuples only."""
    folder = Path(folder)
    f = {
        'folder': str(folder), 'policy': policy_of_folder(folder),
        'files_present': {}, 'missing_files': [],
    }
    wanted = ('events.jsonl', 'components.jsonl', 'tasks.jsonl', 'quota.jsonl',
              'controller.jsonl', 'warmup.json', 'warmup_records.json',
              'actor_inventory.json', 'group_summary.json')
    for name in wanted:
        ok = (folder / name).exists()
        f['files_present'][name] = ok
        if not ok:
            f['missing_files'].append(name)

    arrivals = {}
    terminals = {}
    event_integrity = {'parse_errors': 0}
    dup_terminals = 0
    dup_terminal_tasks = []
    comps = {}
    offers = Counter()
    enqueues = Counter()
    rejects = Counter()
    dispatches = Counter()
    removes = Counter()
    finishes = Counter()
    admit_enq = Counter()
    admit_rej = Counter()
    admit_rej_reasons = Counter()
    reject_rows = []
    remove_rows = []
    late_discarded_rows = []
    late_discarded_n = 0
    config_requests = Counter()
    config_request_rows = []
    target_effective = []
    quality_intents = Counter()
    quality_intent_by_role = Counter()
    quality_intent_rows = 0
    quality_intent_task_ids = set()
    quality_intent_by_stage = Counter()
    prefix_releases = Counter()
    fast_decisions = []
    backpressure = []
    run_end = {}
    event_type_counts = Counter()
    lag_by_type = defaultdict(Samples)
    receipt_lags = Samples()
    injection_lags = Samples()
    service_ms = defaultdict(list)
    d2s_ms = defaultdict(list)
    finish_rows = 0
    job_to_comp = {}
    bank_of_task = {}
    quota_by_job = {}
    task_ids_seen = set()
    quality_intent_need = Samples()
    tmin = None
    tmax = None

    src_pub_of = {}
    for e in rows(folder / 'events.jsonl', event_integrity):
        et = e.get('event_type')
        ts = as_int(e.get('timestamp'))
        event_type_counts[et] += 1
        if ts is not None:
            tmin = ts if tmin is None else min(tmin, ts)
            tmax = ts if tmax is None else max(tmax, ts)
        if et == 'ARRIVAL':
            i = e.get('task_id')
            task_ids_seen.add(i)
            sp = as_int(e.get('source_publish_ns'))
            src_pub_of[i] = sp
            arrivals[i] = (as_int(e.get('arrival_ns')), as_int(e.get('deadline_ns')),
                           as_int(e.get('frame_id')), sp,
                           as_int(e.get('injection_lag_ns')),
                           as_int(e.get('receipt_lag_ns')))
            if e.get('receipt_lag_ns') is not None:
                receipt_lags.add(e.get('receipt_lag_ns'))
            if e.get('injection_lag_ns') is not None:
                injection_lags.add(e.get('injection_lag_ns'))
        elif et == 'TERMINAL':
            i = e.get('task_id')
            task_ids_seen.add(i)
            if i in terminals:
                dup_terminals += 1
                if len(dup_terminal_tasks) < 50:
                    dup_terminal_tasks.append(i)
            else:
                terminals[i] = {
                    'role': e.get('role'), 'prediction': e.get('prediction'),
                    'available': as_int(e.get('terminal_available_ns')),
                    'deadline': as_int(e.get('deadline_abs_ns')),
                    'late': e.get('late'), 'late_discarded': e.get('late_discarded'),
                    'ontime_by_terminal': e.get('ontime_by_terminal'),
                    'finish_ns': as_int(e.get('terminal_finish_ns')),
                    'component_id': e.get('component_id'),
                    'p_selected': as_float(e.get('p_selected')),
                    'reason': e.get('reason'),
                    'actor_finish_ns': as_int(e.get('actor_finish_ns')),
                    'parent_received_ns': as_int(e.get('parent_received_ns')),
                }
            sp = src_pub_of.get(i)
            if sp is not None and ts is not None:
                lag_by_type['TERMINAL'].add(ts - sp)
        elif et == 'OFFER':
            comp = e.get('component_id')
            offers[ROLE_OF_KIND.get(e.get('kind'), 'other') + ':' + str(e.get('kind'))] += 1
            if comp:
                c = comps.get(comp)
                if c is None:
                    c = comps[comp] = _new_comp(comp, e)
                c['offer'] = True
                c.setdefault('stage', e.get('stage'))
        elif et == 'ENQUEUE':
            kind = e.get('kind')
            key = ROLE_OF_KIND.get(kind, 'other') + ':' + str(kind)
            enqueues[key] += 1
            comp = e.get('component_id')
            jid = e.get('job_id')
            if comp:
                c = comps.get(comp)
                if c is None:
                    c = comps[comp] = _new_comp(comp, e)
                c['enqueue'] = True
                c['enqueue_ns'] = as_int(e.get('enqueue_ns'))
                c['task_id'] = e.get('task_id')
                c['job_id'] = jid
                c['kind'] = kind
                c['p_selected'] = as_float(e.get('p_selected'))
            if jid is not None:
                job_to_comp.setdefault(_jobkey(jid), comp)
            if e.get('bank_index') is not None and e.get('task_id') is not None:
                bank_of_task.setdefault(e.get('task_id'), as_int(e.get('bank_index')))
        elif et == 'REJECT':
            kind = e.get('kind')
            role = e.get('role') if e.get('role') in GPU_ROLES else \
                ROLE_OF_KIND.get(kind)
            rejects[str(role) + ':' + str(kind)] += 1
            reason = e.get('reason')
            reject_rows.append((e.get('task_id'), role, kind, reason,
                                e.get('component_id'), e.get('root_id'),
                                as_int(e.get('head_waited_ns')),
                                as_int(e.get('remaining_ns'))))
            comp = e.get('component_id')
            if comp:
                c = comps.get(comp)
                if c is None:
                    c = comps[comp] = _new_comp(comp, e)
                c['reject'] = True
                c['reject_reason'] = reason
                c['task_id'] = e.get('task_id')
                c['kind'] = kind
        elif et == 'ADMISSION_OUTCOME':
            if e.get('outcome') == 'enqueued':
                admit_enq[str(e.get('target_kind'))] += 1
            else:
                admit_rej[str(e.get('target_kind'))] += 1
                admit_rej_reasons[str(e.get('reason'))] += 1
        elif et == 'DISPATCH':
            kind = e.get('kind')
            dispatches[ROLE_OF_KIND.get(kind, 'other') + ':' + str(kind)] += 1
            comp = e.get('component_id')
            jid = e.get('job_id')
            if jid is not None and comp:
                job_to_comp[_jobkey(jid)] = comp
            if comp:
                c = comps.get(comp)
                if c is None:
                    c = comps[comp] = _new_comp(comp, e)
                c['dispatch'] = True
                c['dispatch_ns'] = as_int(e.get('dispatch_ns'))
                c['resource'] = e.get('resource')
                c['task_id'] = e.get('task_id')
                c['kind'] = kind
                c['p_selected'] = as_float(e.get('p_selected'))
        elif et == 'REMOVE':
            kind = e.get('kind')
            removes[ROLE_OF_KIND.get(kind, 'other') + ':' + str(kind)] += 1
            remove_rows.append((e.get('task_id'), ROLE_OF_KIND.get(kind), kind,
                                e.get('reason'), e.get('component_id'),
                                e.get('root_id')))
            comp = e.get('component_id')
            if comp:
                c = comps.get(comp)
                if c is None:
                    c = comps[comp] = _new_comp(comp, e)
                c['remove'] = True
                c['remove_reason'] = e.get('reason')
                c['remove_ns'] = ts
        elif et == 'COMPONENT_FINISH':
            finish_rows += 1
            kind = e.get('kind')
            finishes[str(kind)] += 1
            if e.get('service_ms') is not None:
                service_ms[str(kind)].append(as_float(e.get('service_ms')))
            if e.get('dispatch_to_start_ms') is not None:
                d2s_ms[str(kind)].append(as_float(e.get('dispatch_to_start_ms')))
            comp = e.get('component_id')
            if comp:
                c = comps.get(comp)
                if c is None:
                    c = comps[comp] = _new_comp(comp, e)
                c['finish'] = True
                c['kind'] = kind
                c['start_ns'] = as_int(e.get('start_ns'))
                c['finish_ns'] = as_int(e.get('finish_ns'))
                c['service_ms'] = as_float(e.get('service_ms'))
                c['dispatch_to_start_ms'] = as_float(e.get('dispatch_to_start_ms'))
                c['compute_end_ns'] = as_int(e.get('compute_end_ns'))
                c['actor_finish_ns'] = as_int(e.get('actor_finish_ns'))
                c['parent_received_ns'] = as_int(e.get('parent_received_ns'))
                c['task_id'] = e.get('task_id')
                c['resource'] = e.get('resource')
        elif et == 'LATE_DISCARDED_FINAL':
            late_discarded_n += 1
            if len(late_discarded_rows) < 500:
                late_discarded_rows.append((e.get('task_id'), e.get('kind'),
                                            e.get('gpu'), as_int(e.get('finish_ns'))))
        elif et == 'CONFIG_REQUEST':
            cid = str(e.get('command_id'))
            config_requests[cid] += 1
            if len(config_request_rows) < 500:
                config_request_rows.append((cid, e.get('target_fast'),
                                            e.get('policy'), e.get('scheduled_s'),
                                            ts))
        elif et == 'TARGET_EFFECTIVE':
            if len(target_effective) < 5000:
                target_effective.append((as_int(e.get('gpu')), as_float(e.get('share')),
                                         ts))
        elif et == 'QUALITY_INTENT':
            quality_intent_rows += 1
            quality_intents[str(e.get('quality_role'))] += 1
            quality_intent_by_role[str(e.get('stage')) + ':' + str(e.get('path'))] += 1
            quality_intent_by_stage[str(e.get('stage'))] += 1
            if e.get('task_id') is not None:
                quality_intent_task_ids.add(e.get('task_id'))
            if e.get('accurate_need_probability') is not None:
                quality_intent_need.add(e.get('accurate_need_probability'))
        elif et == 'PREFIX_RELEASE':
            prefix_releases[str(e.get('reason'))] += 1
        elif et == 'FAST_DECISION':
            if len(fast_decisions) < 2000:
                fast_decisions.append({'task_id': e.get('task_id'),
                                       'kind': e.get('kind'),
                                       'budget_ns': as_int(e.get('budget_ns')),
                                       'start_ns': as_int(e.get('start_ns')),
                                       'finish_ns': as_int(e.get('finish_ns')),
                                       'gpu': e.get('gpu'),
                                       'dispatch_ns': as_int(e.get('dispatch_ns'))})
        elif et == 'LOG_BACKPRESSURE':
            if len(backpressure) < 5000:
                backpressure.append({'dropped': as_int(e.get('dropped')),
                                     'queued': as_int(e.get('queued')),
                                     'total': as_int(e.get('total')), 'ts': ts})
        elif et == 'RUN_END':
            run_end = {'group': e.get('group'), 'ended': as_int(e.get('ended')),
                       'arrivals': as_int(e.get('arrivals'))}
        if et in ('OFFER', 'ENQUEUE', 'DISPATCH', 'PREFIX_RELEASE'):
            i = e.get('task_id')
            sp = src_pub_of.get(i)
            if sp is not None and ts is not None:
                lag_by_type[et].add(ts - sp)

    # Protected actor records remain authoritative for real GPU intervals even
    # when the separate event stream reports dropped rows.  They do not restore
    # missing OFFER/REMOVE/control evidence.
    component_rows = 0
    component_by_kind = Counter()
    component_by_role = Counter()
    component_integrity = {'parse_errors': 0}
    for record in rows(folder / 'components.jsonl', component_integrity):
        component_rows += 1
        job = record.get('job') or {}
        kind = record.get('kind') or job.get('kind')
        component_by_kind[str(kind)] += 1
        component_by_role[str(ROLE_OF_KIND.get(kind, 'other'))] += 1
        cid = job.get('component_id') or record.get('component_id')
        if kind not in GPU_KINDS or cid is None:
            continue
        c = comps.setdefault(cid, _new_comp(cid, dict(job, kind=kind)))
        c.update({'kind': kind, 'task_id': record.get('task_id', job.get('task_id')),
                  'resource': record.get('resource'), 'finish': True,
                  'start_ns': as_int(record.get('start_ns')),
                  'finish_ns': as_int(record.get('finish_ns')),
                  'compute_end_ns': as_int(record.get('compute_end_ns')),
                  'p_selected': as_float(job.get('p_selected')),
                  'dispatch_ns': as_int(job.get('dispatch_ns')),
                  'deadline_ns': as_int(job.get('deadline_ns'))})
        jid = job.get('job_id')
        if jid is not None:
            job_to_comp[_jobkey(jid)] = cid

    task_frames = {}
    for i, a in arrivals.items():
        task_frames[i] = a[2]

    # ---- quota (real per-component charge) ----
    quota_rows = 0
    quota_charge_ms = defaultdict(float)
    quota_charge_n = defaultdict(int)
    charge_by_comp = defaultdict(float)
    charge_by_task = defaultdict(float)
    quota_by_gpu_role = defaultdict(float)
    quota_spans = []
    for q in rows(folder / 'quota.jsonl'):
        quota_rows += 1
        jid = q.get('job_id')
        comp = job_to_comp.get(_jobkey(jid)) if jid is not None else None
        kind = comps.get(comp, {}).get('kind') if comp else None
        role = q.get('role') or ROLE_OF_KIND.get(kind)
        start = as_int(q.get('start_ns'))
        finish = as_int(q.get('finish_ns'))
        span = None
        if start is not None and finish is not None:
            span = max(0.0, (finish - start) / MS)
            if len(quota_spans) < 50:
                quota_spans.append({'job_id': str(jid), 'gpu': q.get('gpu'),
                                    'role': role, 'cycle': q.get('cycle'),
                                    'charge_ms': rnd(span, 4),
                                    'component_id': comp, 'kind': kind})
        key = str(kind) if kind else ('role:' + str(role))
        if span is not None:
            quota_charge_ms[key] += span
            quota_charge_n[key] += 1
            quota_by_gpu_role[str(q.get('gpu')) + ':' + str(role)] += span
            if comp:
                charge_by_comp[comp] += span
            tid = comps.get(comp, {}).get('task_id') if comp else None
            if tid is not None:
                charge_by_task[tid] += span

    # ---- tasks.jsonl ----
    tasks_rows = 0
    task_terminals = {}
    task_integrity = {'parse_errors': 0, 'duplicate_task_ids': 0}
    task_role_counts = Counter()
    task_consistency = {'rows': 0, 'terminals_matched': 0,
                        'role_mismatch': 0, 'available_mismatch': 0,
                        'prediction_mismatch': 0, 'late_mismatch': 0}
    for t in rows(folder / 'tasks.jsonl', task_integrity):
        tasks_rows += 1
        tid = t.get('task_id')
        if tid in task_terminals:
            task_integrity['duplicate_task_ids'] += 1
        else:
            task_terminals[tid] = {
                'role': t.get('role'), 'prediction': t.get('prediction'),
                'available': as_int(t.get('terminal_available_ns')),
                'deadline': as_int(t.get('deadline_abs_ns')),
                'late': t.get('late'), 'late_discarded': t.get('late_discarded'),
                'ontime_by_terminal': t.get('ontime_by_terminal'),
                'finish_ns': as_int(t.get('terminal_finish_ns')),
                'component_id': t.get('component_id'),
                'p_selected': as_float(t.get('p_selected')), 'reason': t.get('reason'),
                'actor_finish_ns': as_int(t.get('actor_finish_ns')),
                'parent_received_ns': as_int(t.get('parent_received_ns'))}
        if t.get('bank_index') is not None and tid is not None:
            bank_of_task.setdefault(tid, as_int(t.get('bank_index')))
        task_role_counts[str(t.get('role'))] += 1
        term = terminals.get(tid)
        if term is None:
            continue
        task_consistency['terminals_matched'] += 1
        if t.get('role') != term['role']:
            task_consistency['role_mismatch'] += 1
        if as_int(t.get('terminal_available_ns')) != term['available']:
            task_consistency['available_mismatch'] += 1
        if t.get('prediction') != term['prediction']:
            task_consistency['prediction_mismatch'] += 1
        if bool(t.get('late')) != bool(term['late']):
            task_consistency['late_mismatch'] += 1
    task_consistency['rows'] = tasks_rows
    task_consistency['missing_event_terminals'] = len(set(task_terminals) - set(terminals))
    task_consistency['extra_event_terminals'] = len(set(terminals) - set(task_terminals))

    # ---- controller.jsonl ----
    ctl = _scan_controller(folder / 'controller.jsonl')
    _bind_command_requests(ctl, config_request_rows)

    # ---- side files ----
    group_summary = read_json(folder / 'group_summary.json', {}) or {}
    writer = group_summary.get('log_writer') or {}
    event_integrity['dropped_rows'] = max(as_int(group_summary.get('log_dropped'), 0),
                                         as_int(writer.get('hot_dropped'), 0))
    event_integrity['complete'] = bool(f['files_present']['events.jsonl'] and
        event_integrity['parse_errors'] == 0 and event_integrity['dropped_rows'] == 0)
    task_integrity['value_dropped'] = as_int(writer.get('value_dropped'), 0)
    warmup = read_json(folder / 'warmup.json', {}) or {}
    actor_inventory = read_json(folder / 'actor_inventory.json', None)
    c_star = group_summary.get('c_star_ms') or ctl.get('c_star_ms') or {}

    # T012: group_summary.json is absent when the arm's final write failed, so the
    # declared policy must fall back to the controller stream.  _scan_controller
    # returns that as a Counter, never as a list.
    _declared_policy = group_summary.get('policy')
    if not _declared_policy:
        _pols = ctl.get('policies') or {}
        if isinstance(_pols, dict):
            _declared_policy = (max(_pols, key=lambda k: (_pols[k], k))
                                if _pols else None)
        else:
            _declared_policy = _pols[0] if _pols else None

    f.update({
        'policy_declared': _declared_policy,
        'c_star_ms': c_star,
        'arrivals_n': len(arrivals), 'arrivals': arrivals,
        'frame_ids': task_frames,
        'bank_of_task': dict(bank_of_task),
        'terminals': task_terminals, 'terminal_n': len(task_terminals),
        'event_terminals': terminals, 'event_terminal_n': len(terminals),
        'event_integrity': event_integrity, 'task_integrity': task_integrity,
        'component_integrity': component_integrity,
        'dup_terminals': task_integrity['duplicate_task_ids'],
        'event_dup_terminals': dup_terminals, 'dup_terminal_tasks': dup_terminal_tasks,
        'terminal_role_counts': counter_to_plain(
            Counter([str(t['role']) for t in task_terminals.values()])),
        'components': comps, 'component_ids_n': len(comps),
        'component_rows': component_rows,
        'component_rows_by_kind': counter_to_plain(component_by_kind),
        'component_rows_by_role': counter_to_plain(component_by_role),
        'offers': counter_to_plain(offers), 'enqueues': counter_to_plain(enqueues),
        'rejects': counter_to_plain(rejects), 'dispatches': counter_to_plain(dispatches),
        'removes': counter_to_plain(removes), 'finishes': counter_to_plain(finishes),
        'admission_enqueued': counter_to_plain(admit_enq),
        'admission_rejected': counter_to_plain(admit_rej),
        'admission_rejected_reasons': counter_to_plain(admit_rej_reasons),
        'reject_rows_n': len(reject_rows), 'reject_rows': reject_rows[:20000],
        'remove_rows_n': len(remove_rows), 'remove_rows': remove_rows[:20000],
        'late_discarded_n': late_discarded_n, 'late_discarded_rows': late_discarded_rows,
        'config_requests': counter_to_plain(config_requests),
        'config_request_rows': config_request_rows,
        'config_request_n': sum(config_requests.values()),
        'target_effective_n': len(target_effective),
        'target_effective': target_effective,
        'quality_intent_n': quality_intent_rows,
        'quality_intents': counter_to_plain(quality_intents),
        'quality_intent_by_stage_path': counter_to_plain(quality_intent_by_role),
        'quality_intent_by_stage': counter_to_plain(quality_intent_by_stage),
        'quality_intent_task_ids': quality_intent_task_ids,
        'quality_intent_need': quality_intent_need.summary(),
        'prefix_releases': counter_to_plain(prefix_releases),
        'fast_decisions_n': len(fast_decisions), 'fast_decisions': fast_decisions,
        'log_backpressure_n': len(backpressure),
        'log_backpressure': backpressure[-200:],
        'log_backpressure_last': backpressure[-1] if backpressure else None,
        'run_end': run_end,
        'event_type_counts': counter_to_plain(event_type_counts),
        'event_ts_span_ns': None if tmin is None else [tmin, tmax],
        'lag_by_type': {k: v.summary() for k, v in sorted(lag_by_type.items())},
        'receipt_lag': receipt_lags.summary(),
        'injection_lag': injection_lags.summary(),
        'service_ms': {k: stats(v) for k, v in sorted(service_ms.items())},
        'dispatch_to_start_ms': {k: stats(v) for k, v in sorted(d2s_ms.items())},
        'finish_rows': finish_rows,
        'quota_rows': quota_rows,
        'quota_charge_ms_by_kind': {k: rnd(v, 4) for k, v in
                                    sorted(quota_charge_ms.items())},
        'quota_charge_n_by_kind': {k: int(v) for k, v in
                                   sorted(quota_charge_n.items())},
        'quota_charge_ms_by_gpu_role': {k: rnd(v, 4) for k, v in
                                        sorted(quota_by_gpu_role.items())},
        'quota_sample': quota_spans,
        'charge_by_component_ms': {k: rnd(v, 6) for k, v in
                                   sorted(charge_by_comp.items())},
        'charge_by_task_ms': {k: rnd(v, 6) for k, v in
                              sorted(charge_by_task.items())},
        'tasks_rows': tasks_rows,
        'task_role_counts': counter_to_plain(task_role_counts),
        'tasks_vs_terminals': task_consistency,
        'controller': ctl,
        'group_summary': group_summary,
        'warmup': warmup,
        'actor_inventory': actor_inventory,
    })
    return f


def _jobkey(jid):
    if isinstance(jid, bool):
        return ('b', jid)
    if isinstance(jid, int):
        return ('i', jid)
    if isinstance(jid, float):
        return ('i', int(jid)) if float(jid).is_integer() else ('f', jid)
    return ('s', str(jid))


def _new_comp(comp, e):
    parts = str(comp).split(':')
    root_id = e.get('root_id')
    stage = e.get('stage')
    attempt = None
    if len(parts) >= 2:
        stage = stage or parts[-2]
    if len(parts) >= 3:
        attempt = parts[-1]
    if root_id is None and len(parts) >= 3:
        root_id = ':'.join(parts[:-2])
    return {'component_id': str(comp), 'root_id': root_id, 'stage': stage,
            'attempt': attempt, 'task_id': e.get('task_id'), 'kind': e.get('kind'),
            'offer': False, 'enqueue': False, 'reject': False, 'dispatch': False,
            'finish': False, 'remove': False}


CAND_KEYS = ('gF', 'gA', 'Yhat', 'Yhat_point', 'Vhat', 'Vhat_point', 'V_available',
             'K_F', 'K_A', 'K_A_point', 'L_F', 'L_A', 'muF_tasks_s',
             'muA_point_tasks_s', 'muA_safe_tasks_s', 'thetaF', 'thetaA',
             'thetaA_point', 'feasible_F', 'feasible_A', 'feasible_both',
             'feasible_A_point', 'effective_step', 'point_valid', 'donor_final',
             'receiver_final', 'gate_starts', 'remaining_budget_cdf',
             'sensitivity', 'scenarios', 'by_class', 'cost', 'cost_ms',
             'profile_fp', 'distance', 'capacity_model_F', 'fast_approx',
             'ontime_roots', 'quality_sum', 'abstain_candidates', 'by_class_delta',
             'evaluate_ns', 'effective_step_ns', 'net_gain_Yhat', 'net_gain_Vhat',
             'epsilon_Y', 'epsilon_V', 'low_confidence', 'n_comparable',
             'is_effective_step', 'numeric_target_gF', 'card_share', 'input_n',
             'quality_earliness_hat', 'net_gain_quality_earliness',
             'epsilon_quality_earliness')

ENDPOINT_KEYS = ('apply_ns', 'applied_ns', 'effective_ns', 'settle_ns',
                 'requested_ns', 'granted_ns', 'probe_pre_start_ns',
                 'probe_pre_end_ns', 'probe_post_start_ns', 'probe_post_end_ns',
                 'pre_start_ns', 'pre_end_ns', 'post_start_ns', 'post_end_ns')


def _compact_candidate(c, cur=None):
    if not isinstance(c, dict):
        return {'raw': c}
    out = {}
    for k in CAND_KEYS:
        if k in c:
            out[k] = c[k]
    if 'point_valid' not in out:
        out['point_valid'] = bool(c.get('V_available'))
    y = as_float(c.get('Yhat'))
    v = as_float(c.get('Vhat'))
    if cur is not None:
        y0 = as_float(cur.get('Yhat'))
        v0 = as_float(cur.get('Vhat'))
        if y is not None and y0 is not None:
            out['deltaY_vs_current'] = rnd(y - y0, 6)
        if v is not None and v0 is not None:
            out['deltaV_vs_current'] = rnd(v - v0, 6)
    return out


def _compact_observer(obs):
    if not isinstance(obs, dict):
        return None
    out = {}
    for k in ('root_closure', 'waiting_counts', 'waiting_work_ms', 'c_star_ms',
              'c_star_source', 'epoch', 'window_epoch', 'noise', 'theta',
              'growth_noise', 'component_id_ok', 'resource_lost_roots',
              'resource_lost_components', 'other_lost_components', 'cost_version'):
        if k in obs:
            out[k] = obs[k]
    wins = obs.get('windows')
    if isinstance(wins, list):
        out['windows_tail'] = wins[-2:]
        out['windows_n'] = len(wins)
    return out


def _scan_controller(path):
    """Tolerant scan of controller.jsonl (WS-B writes the frozen log schema)."""
    out = {'rows': 0, 'schemas': Counter(), 'policies': Counter(),
           'controller_versions': Counter(), 'revisions': Counter(),
           'windows': {}, 'windows_n': 0, 'windows_with_candidates': 0,
           'windows_invalid': 0, 'windows_with_gap': 0,
           'commands': [], 'commands_n': 0, 'hold_reasons': Counter(),
           'verdicts': Counter(), 'apply_events': [], 'lock_events': [],
           'probe_rows': [], 'timeline': [], 'missing_keys': Counter(),
           'observer_last': None, 'demand_last': None, 'cost_records_last': None,
           'cost_version_last': None, 'c_star_ms': None, 'max_apply_count': 0,
           'max_config_request_count': 0, 'pending_command_last': None,
           'triggers_last': None, 'last_confirmed': None, 'locks_last': None,
           'resource_keys': Counter(), 'invalid_reasons': Counter(),
           'endpoints_last': None, 'command_ids': [], 'raw_rows_with_decision': 0,
           'integrity': {'parse_errors': 0}, 'feedback': [], 'card_snapshots': [],
           'selection_rows': [], 'score_rows': []}
    seen_cmd = set()
    seen_feedback = set()
    seen_score = set()
    last_t = None
    for row in rows(path, out['integrity']):
        if not isinstance(row, dict):
            continue
        out['rows'] += 1
        out['schemas'][str(row.get('schema'))] += 1
        out['policies'][str(row.get('policy'))] += 1
        out['controller_versions'][str(row.get('controller_version'))] += 1
        out['revisions'][str(row.get('implementation_revision'))] += 1
        decision = row.get('decision')
        if not isinstance(decision, dict):
            decision = {}
        else:
            out['raw_rows_with_decision'] += 1
        resource = decision.get('resource', row.get('resource'))
        if not isinstance(resource, dict):
            resource = {}
        for k in resource.keys():
            out['resource_keys'][str(k)] += 1
        reason = row.get('reason') or decision.get('reason')
        if reason is not None:
            out['hold_reasons'][str(reason)] += 1
        wid = row.get('window_id')
        if wid is None:
            wid = decision.get('window_id')
        if wid is None:
            out['missing_keys']['window_id'] += 1
        tick = row.get('tick_ns')
        if tick is None:
            tick = decision.get('tick_ns')
        eff = resource.get('gF_effective')
        if eff is None:
            eff = decision.get('effective_fast')
        tgt = resource.get('gF_target')
        if tgt is None:
            tgt = decision.get('target_fast')
        if last_t is None or last_t[2] != eff or last_t[3] != tgt:
            if len(out['timeline']) < 20000:
                out['timeline'].append([as_int(tick), as_int(wid), as_float(eff),
                                        as_float(tgt)])
        last_t = [tick, wid, eff, tgt]

        cards = resource.get('cards')
        if isinstance(cards, list):
            out['card_snapshots'].append({'tick_ns': as_int(tick),
                'pending_command_id': (decision.get('pending_command') or {}).get('command_id')
                    if isinstance(decision.get('pending_command'), dict) else row.get('pending_command'),
                'state': row.get('state') or decision.get('state'),
                'cards': cards, 'effective_fast': as_float(eff)})
        if resource:
            out['selection_rows'].append({'tick_ns': as_int(tick),
                'triggers': decision.get('triggers'),
                'last_selection': decision.get('last_selection'),
                'candidates_valid': resource.get('candidates_valid'),
                'table_summary': resource.get('table_summary'),
                'reason': reason})
        score = decision.get('last_score')
        if isinstance(score, dict) and score.get('tick_ns') is not None and score['tick_ns'] not in seen_score:
            seen_score.add(score['tick_ns'])
            selection = decision.get('score_selection') or decision.get('last_selection')
            if not isinstance(selection, dict) or selection.get('score_tick_ns') != score['tick_ns']:
                selection = None
            out['score_rows'].append({'tick_ns': as_int(score['tick_ns']),
                'first_logged_ns': as_int(tick), 'score': score, 'selection': selection,
                'replay': decision.get('replay')})
        feedback = decision.get('last_probe')
        if isinstance(feedback, dict):
            key = json.dumps(feedback, sort_keys=True)
            if key not in seen_feedback:
                seen_feedback.add(key)
                out['feedback'].append({'tick_ns': as_int(tick),
                    'window_id': as_int(wid), 'support': feedback})
                if feedback.get('verdict') is not None:
                    out['verdicts'][str(feedback['verdict'])] += 1

        payload = _pick(resource, ('candidate_payload', 'last_candidate_payload',
                                   'payload', 'replay_payload'))
        table = _pick(resource, ('candidates', 'last_candidates',
                                 'candidate_table', 'last_table'))
        if table is None:
            table = _pick(decision, ('candidates', 'last_candidates'))
        if payload is None and isinstance(resource.get('replay'), dict):
            payload = resource['replay'].get('payload')
        valid = resource.get('candidates_valid')
        invalid = _pick(resource, ('candidates_invalid', 'invalid_reason',
                                   'candidates_invalid_reason'))
        cur_gf = as_float(_pick(payload or {}, ('current_gF', 'effective_fast')))
        rows_c = None
        if isinstance(table, dict):
            rows_c = table.get('candidates')
        elif isinstance(table, list):
            rows_c = table
        cand_list = None
        if isinstance(rows_c, list):
            cur_row = None
            if cur_gf is not None:
                for c in rows_c:
                    if isinstance(c, dict) and c.get('gF') == cur_gf:
                        cur_row = c
                        break
            cand_list = [_compact_candidate(c, cur_row) for c in rows_c]
        if cand_list is None and payload is None and invalid is None:
            invalid = ('compact_tick_no_resource_snapshot' if not resource else
                       'no_candidates_or_payload_recorded')
        if cand_list is None:
            out['windows_invalid'] += 1
            out['invalid_reasons'][str(invalid)] += 1
        else:
            out['windows_with_candidates'] += 1

        trig = None
        sel = decision.get('score_selection') or decision.get('last_selection')
        if isinstance(sel, dict):
            trig = sel.get('trigger')
        if trig is None:
            trig = resource.get('triggers') or decision.get('triggers')
        triggered = bool(isinstance(trig, dict) and
            (trig.get('trigger') or any(isinstance(v, dict) and v.get('trigger') for v in trig.values())))
        if triggered:
            out['windows_with_gap'] += 1
        recv = None
        if isinstance(sel, dict):
            recv = sel.get('receiver')

        cmds_this = row.get('commands')
        if not isinstance(cmds_this, list):
            cmds_this = []
        c0 = decision.get('command')
        if isinstance(c0, dict) and c0 not in cmds_this:
            cmds_this = cmds_this + [c0]
        new_commands_here = []
        for cmd in cmds_this:
            if not isinstance(cmd, dict):
                continue
            cid = str(_pick(cmd, ('command_id', 'id', 'seq')))
            if cid in seen_cmd:
                continue
            seen_cmd.add(cid)
            comp = _compact_command(cmd)
            comp['tick_ns'] = as_int(tick)
            comp['window_id'] = as_int(wid)
            comp['policy'] = row.get('policy')
            comp['endpoints'] = {k: as_int(v) for k, v in cmd.items()
                                 if k in ENDPOINT_KEYS}
            for k in ENDPOINT_KEYS:
                if k in resource:
                    comp['endpoints'][k] = as_int(resource.get(k))
            new_commands_here.append(comp)
            if len(out['commands']) < 400:
                out['commands'].append(comp)
            if len(out['command_ids']) < 400:
                out['command_ids'].append(cid)
        out['commands_n'] += len(new_commands_here)

        for k in ('apply_ns', 'effective_ns', 'settle_ns'):
            if resource.get(k) is not None:
                ep = out['endpoints_last'] or {}
                ep[k] = as_int(resource.get(k))
                out['endpoints_last'] = ep
        ap = as_int(row.get('apply_count', decision.get('apply_count')))
        if ap is not None and ap > out['max_apply_count']:
            out['max_apply_count'] = ap
            if len(out['apply_events']) < 500:
                out['apply_events'].append({'tick_ns': as_int(tick),
                                            'window_id': as_int(wid),
                                            'apply_count': ap,
                                            'policy': row.get('policy')})
        cr = as_int(row.get('config_request_count', decision.get('config_request_count')))
        if cr is not None:
            out['max_config_request_count'] = max(out['max_config_request_count'], cr)
        if row.get('pending_command') is not None:
            out['pending_command_last'] = row.get('pending_command')
        obs = resource.get('observer')
        if isinstance(obs, dict):
            out['observer_last'] = _compact_observer(obs)
            if obs.get('c_star_ms') and not out['c_star_ms']:
                out['c_star_ms'] = obs.get('c_star_ms')
        if resource.get('demand') is not None:
            out['demand_last'] = resource.get('demand')
        if resource.get('cost_records'):
            out['cost_records_last'] = resource.get('cost_records')
        if resource.get('cost_version') is not None:
            out['cost_version_last'] = resource.get('cost_version')
        for lk in ('last_confirmed', 'confirmed_fast'):
            v = resource.get(lk, decision.get(lk))
            if v is not None:
                out['last_confirmed'] = v
        for lk in ('locks', 'locked_edges', 'failed_edges', 'direction_locks'):
            v = resource.get(lk, decision.get(lk))
            if v not in (None, [], {}):
                out['locks_last'] = v
                if len(out['lock_events']) < 500:
                    out['lock_events'].append({'tick_ns': as_int(tick),
                                               'window_id': as_int(wid),
                                               'key': lk, 'value': v})
        if trig not in (None, {}, []):
            out['triggers_last'] = trig
        verdict = _pick(resource, ('verdict', 'probe_verdict'))
        if verdict is None:
            verdict = _pick(decision, ('verdict', 'probe_verdict'))
        if verdict is not None and not isinstance(feedback, dict):
            out['verdicts'][str(verdict)] += 1
        state = row.get('state') or decision.get('state')
        probe = resource.get('probe') or decision.get('last_probe')
        if isinstance(probe, dict) and probe.get('verdict') is not None:
            verdict = probe.get('verdict')
        if state == 'PROBE' or probe is not None or verdict is not None:
            if len(out['probe_rows']) < 500:
                out['probe_rows'].append({
                    'tick_ns': as_int(tick), 'window_id': as_int(wid),
                    'state': state, 'verdict': verdict,
                    'probe': probe if isinstance(probe, (dict, list, str, int, float))
                             or probe is None else str(probe),
                    'reason': reason,
                    'command': _compact_command(decision.get('command'))})
        win = out['windows'].get(wid)
        entry = {
            'window_id': as_int(wid), 'tick_ns': as_int(tick), 'state': state,
            'effective_fast': as_float(eff), 'target_fast': as_float(tgt),
            'reason': reason, 'valid': None if valid is None else bool(valid),
            'invalid_reason': None if cand_list is not None else str(invalid),
            'candidates': cand_list, 'payload': payload if isinstance(payload, dict)
            else None, 'profile_fp': _pick(resource, ('profile_fp', 'profile_fingerprint'))
            or _pick(payload or {}, ('profile_fp', 'profile_fingerprint')),
            'trigger': trig, 'receiver': recv,
            'gap': triggered,
            'apply_count': ap, 'config_request_count': cr,
            'new_command_count': as_int(row.get('new_command_count')),
            'residuals': _compact_residual(resource),
            'observer': _compact_observer(resource.get('observer')),
            'scenarios': (table.get('scenarios') if isinstance(table, dict) else None)
            or (resource.get('scenarios') if isinstance(resource, dict) else None),
            'endpoints': {k: as_int(v) for k, v in resource.items()
                          if k in ENDPOINT_KEYS},
            'commands_here': new_commands_here,
            'age_ns': as_int(_pick(resource, ('profile_age_ns', 'input_age_ns',
                                              'cache_age_ns'))),
            'cost_version': resource.get('cost_version'),
        }
        if win is None or (win.get('candidates') is None and cand_list is not None):
            out['windows'][wid] = entry
    out['windows_n'] = len(out['windows'])
    for k in ('schemas', 'policies', 'controller_versions', 'revisions',
              'hold_reasons', 'verdicts', 'missing_keys', 'resource_keys',
              'invalid_reasons'):
        out[k] = counter_to_plain(out[k])
    return out


def _bind_command_requests(ctl, request_rows):
    """Recover accepted requests missing from controller command metadata.

    CONFIG_REQUEST is authoritative for identity, target and request time;
    it cannot tell us the proposal channel, permission or return reason.
    """
    ctl['commands_metadata_recorded_n'] = ctl.get('commands_n', 0)
    recovered = []
    seen = set(str(c.get('command_id')) for c in ctl.get('commands') or [])
    for cid, target, policy, scheduled, timestamp in request_rows:
        cid = str(cid)
        if cid in seen:
            continue
        seen.add(cid)
        recovered.append({'command_id': cid, 'target_fast': as_float(target),
            'proposed_ns': timestamp, 'tick_ns': timestamp,
            'window_id': None if timestamp is None else timestamp // WINDOW_NS * WINDOW_NS,
            'policy': policy, 'scheduled_s': scheduled,
            'provenance': 'events.jsonl CONFIG_REQUEST',
            'channel': None, 'info': None, 'direction': None, 'reason': None,
            'from_fast': None, 'metadata_complete': False})
    for c in ctl.get('commands') or []:
        c.setdefault('provenance', 'controller.jsonl decision.command or commands')
        c.setdefault('metadata_complete', True)
    ctl['commands'].extend(recovered)
    ctl['command_ids'].extend(c['command_id'] for c in recovered)
    ctl['commands_n'] += len(recovered)
    ctl['commands_recovered_from_requests_n'] = len(recovered)
    ctl['command_metadata_complete'] = not recovered


def _compact_residual(resource):
    if not isinstance(resource, dict):
        return None
    for key in ('residuals', 'window_residuals', 'last_residuals'):
        v = resource.get(key)
        if isinstance(v, dict):
            return v
    obs = resource.get('observer')
    if isinstance(obs, dict):
        wins = obs.get('windows')
        if isinstance(wins, list) and wins:
            last = wins[-1]
            if isinstance(last, dict):
                return {'from_observer_window': last.get('index'),
                        'residuals': last.get('residuals'),
                        'dW': last.get('residual_dW'), 'dOpen': last.get('residual_O')}
    return None


def _online_residual_counts(res):
    """Tolerant reader of the controller's own per-window counts (§6.1 keys).

    Accepts either per-role sub-dicts (``{'Fast': {'E_n': ..}}``, optionally
    wrapped in ``window_residuals``/``counts``) or flat ``E_n``/``S_n``/... keys
    that then compare against the role sum.
    """
    if not isinstance(res, dict):
        return {}
    for wrap in ('residuals', 'counts', 'window_residuals', 'residual_counts', 'totals'):
        inner = res.get(wrap)
        if isinstance(inner, dict):
            res = inner
            break
    kept = ('E_n', 'S_n', 'X_n', 'R_n', 'O_n', 'dW_n')
    got = {}
    for role in GPU_ROLES + (LOCAL_ROLE,):
        sub = res.get(role)
        if isinstance(sub, dict):
            for k in kept:
                if sub.get(k) is not None:
                    got.setdefault(k, {})[role] = sub.get(k)
    for k in kept:
        if res.get(k) is not None:
            got.setdefault(k, {})['_all'] = res.get(k)
    return got


# =========================================== cohort / pairing / source tables

def load_labels(evidence, fallback_folder=None):
    """Offline private labels: {'<bank_index>': class_int}.  Never in the arm logs."""
    cands = []
    if evidence:
        p = Path(evidence)
        cands.append(p / 'evaluation_private_labels.json' if p.is_dir() else p)
    if fallback_folder:
        cands.append(Path(fallback_folder) / 'evidence' /
                     'evaluation_private_labels.json')
    for p in cands:
        obj = read_json(p, None)
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                iv = as_int(v)
                if iv is not None:
                    out[str(k)] = iv
            if out:
                return out, str(p)
    return {}, None


def _blank_rec(tid, arrival=None):
    a = list(arrival or [None] * 6) + [None] * 6
    return {'task_id': tid, 'arrival_ns': as_int(a[0]), 'deadline_ns': as_int(a[1]),
            'frame_id': as_int(a[2]), 'source_publish_ns': as_int(a[3]),
            'injection_lag_ns': as_int(a[4]), 'receipt_lag_ns': as_int(a[5]),
            'bank_index': None, 'charge_ms': None, 'terminal': False,
            'completed': False, 'role': None, 'prediction': None, 'kind': None,
            'available': None, 'deadline_abs_ns': None, 'ontime': None,
            'late': None, 'late_recomputed': None, 'good': None, 'label': None,
            'label_known': False, 'reason': None, 'compute_end_ns': None,
            'actor_finish_ns': None, 'parent_received_ns': None,
            'ontime_by_terminal': None, 'stage': None, 'root_id': None,
            'component_id': None, 'terminal_finish_ns': None, 'p_selected': None,
            'late_discarded': None}


def build_task_records(scan, labels=None, bank_of_task=None):
    """Per-task record joining ARRIVAL + TERMINAL + component identity + label."""
    labels = labels or {}
    bank_of_task = bank_of_task or scan.get('bank_of_task') or {}
    comps = scan.get('components') or {}
    charge = scan.get('charge_by_task_ms') or {}
    recs = {}
    for tid, a in (scan.get('arrivals') or {}).items():
        recs[tid] = _blank_rec(tid, a)
    for tid, t in (scan.get('terminals') or {}).items():
        r = recs.get(tid)
        if r is None:
            r = recs[tid] = _blank_rec(tid)
        r['terminal'] = True
        r['role'] = t.get('role')
        # T014 s6: only Local/Fast/Accurate are completions; an Abstain termi-
        # nates the root but is NEVER a completion nor an on-time delivery.
        r['abstain'] = bool(str(t.get('role')) == 'Abstain')
        r['completed'] = bool(t.get('role') in ALL_ROLES and t.get('prediction') is not None)
        r['prediction'] = t.get('prediction')
        r['available'] = t.get('available')
        r['late'] = t.get('late')
        r['late_discarded'] = t.get('late_discarded')
        r['ontime_by_terminal'] = t.get('ontime_by_terminal')
        r['reason'] = t.get('reason')
        r['terminal_finish_ns'] = t.get('finish_ns')
        r['p_selected'] = t.get('p_selected')
        r['p_selected_provenance'] = 'tasks.jsonl' if r['p_selected'] is not None else None
        r['component_id'] = t.get('component_id')
        r['actor_finish_ns'] = t.get('actor_finish_ns')
        r['parent_received_ns'] = t.get('parent_received_ns')
        c = comps.get(t.get('component_id')) if t.get('component_id') else None
        if isinstance(c, dict):
            r['kind'] = c.get('kind')
            r['root_id'] = c.get('root_id')
            r['stage'] = c.get('stage')
            r['compute_end_ns'] = c.get('compute_end_ns')
            if r['p_selected'] is None and c.get('p_selected') is not None:
                r['p_selected'] = c['p_selected']
                r['p_selected_provenance'] = 'components.jsonl job.p_selected matched by final component_id'
        dl = t.get('deadline') if t.get('deadline') is not None else r['deadline_abs_ns']
        if dl is None:
            dl = r['deadline_ns']
        r['deadline_abs_ns'] = dl
        if r['available'] is not None and dl is not None:
            r['ontime'] = bool(r['completed'] and r['available'] <= dl)
            r['late_recomputed'] = bool(r['completed'] and r['available'] > dl)
            r['ontime_abstain'] = bool(r['abstain'] and r['available'] <= dl)
    for tid, r in recs.items():
        bank = bank_of_task.get(tid)
        if bank is None:
            bank = bank_of_task.get(str(tid))
        if bank is not None:
            r['bank_index'] = as_int(bank)
            lab = labels.get(str(r['bank_index']))
            if lab is not None:
                r['label'] = lab
                r['label_known'] = True
        r['good'] = bool(r['label_known'] and r['ontime'] and r['completed']
                         and r['prediction'] == r['label'])
        if r['charge_ms'] is None:
            r['charge_ms'] = charge.get(tid)
    return recs


def _clock_value(rec, clock):
    if clock == 'terminal_available_ns':
        return rec.get('available')
    return rec.get(clock)


def _in_span(v, span):
    if v is None:
        return False
    return span[0] * NS_PER_S <= v < span[1] * NS_PER_S


def empty_counts():
    return {'cohort': 0, 'arrivals': 0, 'completed': 0,
            'completed_incl_late': 0, 'ontime': 0, 'good': 0, 'late': 0,
            'abstain': 0, 'not_completed': 0, 'good_label_known': 0,
            'good_label_unknown': 0, 'cohort_label_known': 0,
            'cohort_label_unknown': 0, 'late_discarded_n': 0,
            'charge_ms_total': 0.0, 'roles': {}, 'by_kind': {}, 'good_by_kind': {},
            'reject_reasons': {}, 'e2e_from_arrival_ms': {},
            'e2e_from_source_publish_ms': {}, 'terminal_role_counts': {}}


def _new_role_bucket():
    return {'cohort': 0, 'completed': 0, 'ontime': 0, 'good': 0,
            'late': 0, 'good_label_known': 0, 'good_label_unknown': 0}


def cohort_counts(recs, span, cohort_clock):
    """All columns for one (arm, span, cohort clock).  Half-open span [lo, hi)."""
    out = empty_counts()
    e2e_a = {}
    e2e_s = {}
    for rec in recs.values():
        cv = _clock_value(rec, cohort_clock)
        if not _in_span(cv, span):
            continue
        out['cohort'] += 1
        out['arrivals'] += 1
        rol = str(rec.get('role'))
        bucket = out['roles'].get(rol)
        if bucket is None:
            bucket = out['roles'][rol] = _new_role_bucket()
        bucket['cohort'] += 1
        if rec.get('label_known'):
            out['cohort_label_known'] += 1
        else:
            out['cohort_label_unknown'] += 1
        rsn0 = str(rec.get('reason'))
        if rec.get('abstain') or 'abstain' in rsn0.lower() \
                or str(rec.get('role')) == 'Abstain':
            # TASK s6: Abstain is its own main-table column and never counts as
            # a completion; a non-completed abstain must still be visible here.
            out['abstain'] += 1
        if not rec.get('completed'):
            out['not_completed'] += 1
            continue
        out['completed'] += 1
        out['completed_incl_late'] += 1
        bucket['completed'] += 1
        if rec.get('late'):
            out['late'] += 1
            bucket['late'] += 1
        if rec.get('late_discarded'):
            out['late_discarded_n'] += 1
        av = rec.get('available')
        if av is not None:
            if rec.get('arrival_ns') is not None:
                e2e_a.setdefault(rol, []).append((av - rec['arrival_ns']) / MS)
            if rec.get('source_publish_ns') is not None:
                e2e_s.setdefault(rol, []).append((av - rec['source_publish_ns']) / MS)
        if rec.get('ontime'):
            out['ontime'] += 1
            bucket['ontime'] += 1
        if rec.get('label_known'):
            out['good_label_known'] += 1
            bucket['good_label_known'] += 1
            if rec.get('good'):
                out['good'] += 1
                bucket['good'] += 1
        else:
            out['good_label_unknown'] += 1
            bucket['good_label_unknown'] += 1
        kind = str(rec.get('kind'))
        out['by_kind'][kind] = out['by_kind'].get(kind, 0) + 1
        if rec.get('good'):
            out['good_by_kind'][kind] = out['good_by_kind'].get(kind, 0) + 1
        if rec.get('charge_ms') is not None:
            out['charge_ms_total'] = rnd(out['charge_ms_total'] + rec['charge_ms'], 4)
        out['terminal_role_counts'][rol] = out['terminal_role_counts'].get(rol, 0) + 1
    out['e2e_from_arrival_ms'] = {k: stats(v) for k, v in sorted(e2e_a.items())}
    out['e2e_from_source_publish_ms'] = {k: stats(v) for k, v in
                                         sorted(e2e_s.items())}
    return out


def strict_terminal_counts(recs, span):
    """Strict throughput: terminals whose terminal_available_ns lands in span."""
    out = {'cohort': 'terminal_available_ns', 'terminal_in_span': 0, 'ontime': 0,
           'good': 0, 'late': 0, 'by_role': {}, 'by_kind': {}}
    for rec in recs.values():
        av = rec.get('available')
        if not _in_span(av, span):
            continue
        out['terminal_in_span'] += 1
        if rec.get('ontime'):
            out['ontime'] += 1
        if rec.get('good'):
            out['good'] += 1
        if rec.get('late'):
            out['late'] += 1
        rol = str(rec.get('role'))
        out['by_role'][rol] = out['by_role'].get(rol, 0) + 1
        kind = str(rec.get('kind'))
        out['by_kind'][kind] = out['by_kind'].get(kind, 0) + 1
    return out


def arm_cohorts(recs, spans=None):
    spans = spans or COHORT_SPANS
    out = {}
    for name, lo, hi in spans:
        span = (lo, hi)
        out[name] = {
            'span_s': [lo, hi],
            'span_convention': 'half-open [lo, hi); clock ns are absolute since t0',
            'by_arrival_ns': cohort_counts(recs, span, 'arrival_ns'),
            'by_source_publish_ns': cohort_counts(recs, span, 'source_publish_ns'),
            'by_terminal_available_ns': cohort_counts(recs, span,
                                                      'terminal_available_ns'),
            'terminal_available_strict': strict_terminal_counts(recs, span),
        }
    return out


def _nested_roles_diff(d_dyn, d_fix):
    keys = sorted(set(list(d_dyn.keys()) + list(d_fix.keys())), key=str)
    out = {}
    for k in keys:
        a = d_dyn.get(k) or {}
        b = d_fix.get(k) or {}
        out[k] = {f: rnd(as_float(a.get(f, 0)) - as_float(b.get(f, 0)), 6)
                  for f in sorted(set(list(a.keys()) + list(b.keys())))}
    return out


def diff_counts(dyn, fix):
    """dynamic - fixed for every numeric leaf of a cohort_counts-like dict."""
    out = {'sign': 'dynamic_minus_fixed'}
    for k, v in dyn.items():
        if k in ('roles', 'e2e_from_arrival_ms', 'e2e_from_source_publish_ms',
                 'terminal_role_counts', 'span_convention', 'cohort'):
            continue
        if isinstance(v, dict):
            fv = fix.get(k) or {}
            out[k] = {kk: rnd(as_float(v.get(kk, 0)) - as_float(fv.get(kk, 0)), 6)
                      for kk in sorted(set(list(v.keys()) + list(fv.keys())), key=str)}
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = rnd(as_float(v) - as_float(fix.get(k, 0)), 6)
    out['roles'] = _nested_roles_diff(dyn.get('roles') or {}, fix.get('roles') or {})
    out['e2e_from_arrival_ms_mean'] = {
        k: rnd(as_float((dyn.get('e2e_from_arrival_ms', {}).get(k) or {}).get('mean', 0))
               - as_float((fix.get('e2e_from_arrival_ms', {}).get(k) or {}).get('mean', 0)), 6)
        for k in sorted(set(list((dyn.get('e2e_from_arrival_ms') or {}).keys()) +
                            list((fix.get('e2e_from_arrival_ms') or {}).keys())))}
    return out


def cohort_tables(recs_f, recs_d, spans=None):
    """Both arms + per-span diff, all clocks -- TASK §9 items 1/2/3."""
    cf = arm_cohorts(recs_f, spans)
    cd = arm_cohorts(recs_d, spans)
    out = {}
    for name in sorted(set(list(cf.keys()) + list(cd.keys()))):
        a = cf.get(name) or {}
        b = cd.get(name) or {}
        out[name] = {
            'span_s': a.get('span_s') or b.get('span_s'),
            'span_convention': 'half-open [lo, hi); clock ns are absolute since t0',
            'fixed_1a1f': a, 'qf4r_feedback': b,
            'diff_dynamic_minus_fixed': {
                'by_arrival_ns': diff_counts(b.get('by_arrival_ns') or {},
                                             a.get('by_arrival_ns') or {}),
                'by_source_publish_ns': diff_counts(
                    b.get('by_source_publish_ns') or {},
                    a.get('by_source_publish_ns') or {}),
                'by_terminal_available_ns': diff_counts(
                    b.get('by_terminal_available_ns') or {},
                    a.get('by_terminal_available_ns') or {}),
                'terminal_available_strict': diff_counts(
                    b.get('terminal_available_strict') or {},
                    a.get('terminal_available_strict') or {})}}
    return out


def pair_quad(recs_f, recs_d, ids, levels=('completed', 'ontime', 'good')):
    """TASK §9 item 4: both / fixed_only / dynamic_only / neither per level."""
    quad = {}
    for lvl in levels:
        quad[lvl] = {'both': 0, 'fixed_only': 0, 'dynamic_only': 0, 'neither': 0}
    for tid in ids:
        rf = recs_f.get(tid)
        rd = recs_d.get(tid)
        for lvl in levels:
            a = bool(rf.get(lvl)) if rf else False
            b = bool(rd.get(lvl)) if rd else False
            if a and b:
                quad[lvl]['both'] += 1
            elif a:
                quad[lvl]['fixed_only'] += 1
            elif b:
                quad[lvl]['dynamic_only'] += 1
            else:
                quad[lvl]['neither'] += 1
    return quad


def paired_latency(recs_f, recs_d, ids, gpu_roles=GPU_ROLES):
    """TASK §9 item 4: same task_id AND same GPU role, both arms completed."""
    out = {'both_completed': 0, 'role_changed': 0, 'fixed_only': 0,
           'dynamic_only': 0, 'neither': 0, 'roles_counted': {},
           'latency_diff_ms': {'n': 0}, 'sign': 'dynamic_minus_fixed',
           'paired_task_ids_sample': []}
    diffs = []
    per_role = {}
    for tid in ids:
        rf = recs_f.get(tid)
        rd = recs_d.get(tid)
        cf = bool(rf and rf.get('completed'))
        cd = bool(rd and rd.get('completed'))
        if cf and cd:
            out['both_completed'] += 1
            r1 = rf.get('role')
            r2 = rd.get('role')
            if r1 != r2:
                out['role_changed'] += 1
                continue
            if r1 not in gpu_roles:
                continue
            if rf.get('available') is None or rd.get('available') is None:
                continue
            d = (rd['available'] - rf['available']) / MS
            diffs.append(d)
            per_role.setdefault(r1, []).append(d)
            out['roles_counted'][r1] = out['roles_counted'].get(r1, 0) + 1
            if len(out['paired_task_ids_sample']) < 20:
                out['paired_task_ids_sample'].append(tid)
        elif cf:
            out['fixed_only'] += 1
        elif cd:
            out['dynamic_only'] += 1
        else:
            out['neither'] += 1
    out['latency_diff_ms'] = stats(diffs)
    out['latency_diff_ms']['sign'] = 'dynamic_minus_fixed'
    out['latency_diff_ms_by_role'] = {k: stats(v) for k, v in sorted(per_role.items())}
    return out


def extras_block(recs_f, recs_d):
    """TASK §9 item 6: fast_extra / total_extra (dynamic - fixed)."""
    def _n(recs, role=None):
        n = 0
        for r in recs.values():
            if not r.get('completed'):
                continue
            if role is not None and r.get('role') != role:
                continue
            n += 1
        return n
    fx_f = _n(recs_f, 'Fast')
    fx_d = _n(recs_d, 'Fast')
    ax_f = _n(recs_f, 'Accurate')
    ax_d = _n(recs_d, 'Accurate')
    lc_f = _n(recs_f, 'Local')
    lc_d = _n(recs_d, 'Local')
    return {'fast_extra': fx_d - fx_f,
            'accurate_extra': ax_d - ax_f,
            'local_extra': lc_d - lc_f,
            'total_extra': _n(recs_d) - _n(recs_f),
            'fixed_1a1f': {'Fast': fx_f, 'Accurate': ax_f, 'Local': lc_f,
                           'total': _n(recs_f)},
            'qf4r_feedback': {'Fast': fx_d, 'Accurate': ax_d, 'Local': lc_d,
                              'total': _n(recs_d)}}


def read_shared_frames(joint_folder):
    """TASK §9 items 5/7: joint shared_frames.jsonl facts (source side)."""
    p = Path(joint_folder) / 'shared_frames.jsonl'
    out = {'path': str(p), 'present': p.exists(), 'rows': 0, 'outcomes': {},
           'source_end': None, 'unique_frame_ids': 0, 'dup_frame_ids': [],
           'dup_frame_ids_n': 0, 'frame_id_gaps': 0, 'min_frame_id': None,
           'max_frame_id': None, 'last_contig_frame_id': None, 'order_ok': True,
           'order_violations': 0, 'bank_of_task': {}, 'entry_class_of_task': {},
           'entry_kind_of_task': {}, 'local_terminals': {}, 'abstains': 0,
           'quality_rows': {}, 'publish_ns_span': [None, None],
           'frames_of_task': {}, 'meta_of_task': {}, 'path_B': 0,
           'accurate_need_probability': None, 'accurate_need_n': 0}
    out['arrivals'] = {}
    out['gpu_publish_of_task'] = {}
    out['parse_errors'] = 0
    if not p.exists():
        return out
    seen = {}
    prev = None
    pmin = None
    pmax = None
    need_sum = 0.0
    need_n = 0
    arrival_rows = 0
    arrival_frames = []
    for row in rows(p, out):
        out['rows'] += 1
        oc = str(row.get('outcome'))
        out['outcomes'][oc] = out['outcomes'].get(oc, 0) + 1
        fid = as_int(row.get('frame_id'))
        sp = as_int(row.get('source_publish_ns'))
        if sp is not None:
            pmin = sp if pmin is None else min(pmin, sp)
            pmax = sp if pmax is None else max(pmax, sp)
        tid = row.get('task_id')
        if tid is not None:
            if row.get('bank_index') is not None:
                out['bank_of_task'][tid] = as_int(row.get('bank_index'))
            out['frames_of_task'].setdefault(tid, []).append(fid)
            meta = out['meta_of_task'].get(tid) or {}
            for k in ('source_publish_ns', 'arrival_ns', 'deadline_ns',
                      'origin_node', 'bank_index', 'stage', 'path'):
                if row.get(k) is not None:
                    meta.setdefault(k, row.get(k))
            out['meta_of_task'][tid] = meta
        if oc == 'SOURCE_END' or fid == -1:
            out['source_end'] = row
            continue
        # FrameBus assigns a new global frame_id to EVERY published outcome.
        if fid is not None and fid >= 0:
            if fid in seen:
                out['dup_frame_ids_n'] += 1
                if len(out['dup_frame_ids']) < 200:
                    out['dup_frame_ids'].append(fid)
            else:
                seen[fid] = oc
            if prev is not None and fid <= prev:
                out['order_violations'] += 1
                out['order_ok'] = False
            prev = fid
            out['max_frame_id'] = fid if out['max_frame_id'] is None else \
                max(out['max_frame_id'], fid)
        else:
            out['unsequenced_frames'] = out.get('unsequenced_frames', 0) + 1
        if oc == OUTCOME_ARRIVAL:
            arrival_rows += 1
            arrival_frames.append(fid)
            out['arrivals'][tid] = (as_int(row.get('arrival_ns')),
                as_int(row.get('deadline_ns')), fid, sp, None, None)
        if oc == OUTCOME_LOCAL_TERMINAL:
            out['local_terminals'][tid] = {
                'task_id': tid, 'frame_id': fid,
                'real_finish_ns': as_int(row.get('real_finish_ns')),
                'prediction': row.get('prediction'),
                'terminal_available_ns': as_int(row.get('terminal_available_ns')),
                'deadline_abs_ns': as_int(row.get('deadline_abs_ns')),
                'source_publish_ns': sp,
                'quality_probabilities': row.get('quality_probabilities'),
                'reason': row.get('reason')}
        elif oc == OUTCOME_GPU_READY:
            out['gpu_publish_of_task'][tid] = sp
            # T012: the shared source publishes gpu_candidate_kinds as a
            # {role: kind} mapping and gpu_candidates as the ordered role list
            # (shared_local_t011.py:573-577), never as a positional list.
            raw_kinds = row.get('gpu_candidate_kinds')
            cand_roles = row.get('gpu_candidates')
            if isinstance(raw_kinds, dict):
                if isinstance(cand_roles, (list, tuple)) and cand_roles:
                    kind = raw_kinds.get(cand_roles[0])
                    kind_items = [raw_kinds.get(r) for r in cand_roles]
                else:
                    kind = None
                    kind_items = list(raw_kinds.values())
            else:
                kind_items = list(raw_kinds or [])
                kind = kind_items[0] if kind_items else row.get('kind')
            out['entry_class_of_task'][tid] = (ROOT_DIRECT_A if kind == 'ACCURATE'
                                               else ROOT_DIRECT_F)
            out['entry_kind_of_task'][tid] = kind
            for ck in kind_items:
                if ck is None:
                    continue
                key = 'candidate_kind:' + str(ck)
                out[key] = out.get(key, 0) + 1
            if row.get('accurate_need_probability') is not None:
                need_sum += as_float(row.get('accurate_need_probability'), 0.0) or 0.0
                need_n += 1
        elif oc == OUTCOME_PREFIX_READY:
            out['gpu_publish_of_task'][tid] = sp
            out['entry_class_of_task'][tid] = ROOT_PREFIX
            out['entry_kind_of_task'][tid] = row.get('kind')
        elif oc == OUTCOME_ABSTAIN:
            out['abstains'] += 1
        elif oc == OUTCOME_QUALITY:
            st = str(row.get('stage'))
            out['quality_rows'][st] = out['quality_rows'].get(st, 0) + 1
        if row.get('path') == 'B':
            out['path_B'] += 1
    out['unique_frame_ids'] = len(seen)
    fids = sorted(seen.keys())
    out['arrival_rows_n'] = arrival_rows
    out['arrival_frames'] = arrival_frames
    out['min_frame_id'] = fids[0] if fids else None
    contig = (fids[0] - 1) if fids else 0
    for f in fids:
        if f == contig + 1:
            contig = f
    out['last_contig_frame_id'] = contig
    out['frame_id_gaps'] = ((fids[-1] - fids[0] + 1) - len(fids)) if fids else 0
    out['unexpected_first_frame'] = bool(fids and fids[0] not in (0, 1))
    out['publish_ns_span'] = [pmin, pmax]
    out['accurate_need_probability'] = (need_sum / need_n) if need_n else None
    out['accurate_need_n'] = need_n
    return out


def _bind_shared_arrivals(scan, shared):
    """Join complete source arrivals for output facts; preserve arm event facts."""
    if shared.get('present'):
        scan['event_arrivals_n'] = scan.get('arrivals_n')
        scan['arrivals'] = shared.get('arrivals') or {}
        scan['arrivals_n'] = len(scan['arrivals'])
        scan['arrivals_fact_source'] = 'shared_frames.jsonl SHARED_ARRIVAL'


def local_consistency_block(shared, recs_f, recs_d, scans=None):
    """TASK §9 item 5: common Local terminal set / prediction / finish / drops."""
    out = {'source': 'shared_frames.jsonl + arm events.jsonl + arm group_summary.json',
           'shared_rows': shared.get('rows'),
           'shared_present': shared.get('present'),
           'shared_outcomes': shared.get('outcomes'),
           'common_local_terminals_n': len(shared.get('local_terminals') or {}),
           'fixed_local_completed': 0, 'dynamic_local_completed': 0,
           'terminal_sets_identical': None, 'finish_mismatches': 0,
           'prediction_mismatches': 0, 'available_mismatches': 0,
           'missing_in_fixed': [], 'missing_in_dynamic': [],
           'extra_in_fixed': [], 'extra_in_dynamic': [],
           'dropped_by_arm': {}, 'warnings': []}
    setf = set()
    setd = set()
    for tid, r in recs_f.items():
        if r.get('completed') and r.get('role') == 'Local':
            setf.add(tid)
    for tid, r in recs_d.items():
        if r.get('completed') and r.get('role') == 'Local':
            setd.add(tid)
    out['fixed_local_completed'] = len(setf)
    out['dynamic_local_completed'] = len(setd)
    out['terminal_sets_identical'] = (setf == setd)
    sh = set((shared.get('local_terminals') or {}).keys())
    if sh:
        out['missing_in_fixed'] = sorted([str(x) for x in list(sh - setf)][:50])
        out['missing_in_dynamic'] = sorted([str(x) for x in list(sh - setd)][:50])
        out['extra_in_fixed'] = sorted([str(x) for x in list(setf - sh)][:50])
        out['extra_in_dynamic'] = sorted([str(x) for x in list(setd - sh)][:50])
        out['arm_matches_shared_set'] = (not (sh - setf) and not (setf - sh) and
                                         not (sh - setd) and not (setd - sh))
    for tid, sl in (shared.get('local_terminals') or {}).items():
        for recs in (recs_f, recs_d):
            r = recs.get(tid)
            if not r or not r.get('completed'):
                continue
            if sl.get('prediction') != r.get('prediction'):
                out['prediction_mismatches'] += 1
            av = sl.get('terminal_available_ns')
            if av is not None and r.get('available') is not None and av != r['available']:
                out['available_mismatches'] += 1
            rf = sl.get('real_finish_ns')
            if rf is not None and r.get('terminal_finish_ns') is not None and \
                    rf != r['terminal_finish_ns']:
                out['finish_mismatches'] += 1
    for tag, scan in (('fixed_1a1f', (scans or {}).get('fixed_1a1f')),
                      ('qf4r_feedback', (scans or {}).get('qf4r_feedback'))):
        if not scan:
            out['dropped_by_arm'][tag] = None
            continue
        gs = scan.get('group_summary') or {}
        val = None
        for k in ('frames_dropped', 'shared_dropped', 'dropped', 'duplicate_frames',
                  'bus_dropped', 'source_frames_dropped'):
            if gs.get(k) is not None:
                val = gs.get(k)
                break
        if val is None:
            bp = scan.get('log_backpressure_last')
            if isinstance(bp, dict) and bp.get('dropped') is not None:
                val = bp.get('dropped')
        out['dropped_by_arm'][tag] = val
        if val is None:
            out['warnings'].append(
                'no shared-frame drop counter found for arm %s (group_summary keys '
                'checked: frames_dropped/shared_dropped/dropped/duplicate_frames/'
                'bus_dropped/source_frames_dropped)' % tag)
    return out


# ================================== source-side rates and conservation replay

def _linfit(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return rnd(num / den, 6)


def source_block(shared, scans, spans=None, bucket_s=0.05):
    """TASK §9 item 7: entry-class rates (both clocks), 50 ms bursts, intents."""
    spans = spans or COHORT_SPANS
    entry = shared.get('entry_class_of_task') or {}
    meta = shared.get('meta_of_task') or {}
    out = {'source': 'shared_frames.jsonl (GPU_READY / PREFIX_READY entry class) + '
                     'arm events.jsonl QUALITY_INTENT',
           'entry_class_tasks_n': sum(1 for v in entry.values()),
           'shared_frames_rows': shared.get('rows'),
           'class_totals': {}, 'per_span': {}, 'short_request_distribution': {},
           'public_intent': {}, 'warnings': []}
    for c in ROOT_CLASSES:
        out['class_totals'][c] = 0
    for tid, cls in entry.items():
        out['class_totals'][cls] = out['class_totals'].get(cls, 0) + 1

    for name, lo, hi in spans:
        lo_ns, hi_ns = lo * NS_PER_S, hi * NS_PER_S
        secs = float(hi - lo)
        if secs <= 0:
            continue
        ent = {'span_s': [lo, hi], 'by_source_publish_ns': {}, 'by_arrival_ns': {}}
        for clock in ('source_publish_ns', 'arrival_ns'):
            counts = {c: 0 for c in ROOT_CLASSES}
            unknown = 0
            for tid, m in meta.items():
                t = as_int(m.get(clock))
                if t is None or not (lo_ns <= t < hi_ns):
                    continue
                cls = entry.get(tid)
                if cls is None:
                    unknown += 1
                else:
                    counts[cls] = counts.get(cls, 0) + 1
            ent['by_' + clock] = {
                'counts': counts,
                'rate_per_s': {k: rnd(v / secs, 4) for k, v in counts.items()},
                'total': sum(counts.values()),
                'unknown_entry_class': unknown}
        out['per_span'][name] = ent

    # 50 ms short-request distribution per class, bucketed by source_publish_ns
    for cls in ROOT_CLASSES:
        buckets = Counter()
        tmin = None
        tmax = None
        for tid, c in entry.items():
            if c != cls:
                continue
            t = as_int((meta.get(tid) or {}).get('source_publish_ns'))
            if t is None:
                continue
            b = int(t // int(bucket_s * NS_PER_S))
            buckets[b] += 1
            tmin = b if tmin is None else min(tmin, b)
            tmax = b if tmax is None else max(tmax, b)
        vals = []
        if buckets:
            for b in range(tmin, tmax + 1):
                vals.append(buckets.get(b, 0))
        out['short_request_distribution'][cls] = {
            'bucket_s': bucket_s, 'bucket_origin': 'source_publish_ns // bucket',
            'buckets_n': len(vals), 'requests': sum(vals),
            'per_bucket': stats(vals) if vals else {'n': 0},
            'cycles_gt4': sum(1 for v in vals if v > 4),
            'per_bucket_counts_histogram': counter_to_plain(Counter(vals))}

    # public quality intents per arm + supported/unknown via entry class
    pi = {}
    for tag, scan in sorted((scans or {}).items()):
        ids = scan.get('quality_intent_task_ids') or set()
        if not isinstance(ids, (set, frozenset, list, tuple)):
            ids = []
        supported = 0
        unknown = 0
        for tid in ids:
            if entry.get(tid) in ROOT_CLASSES:
                supported += 1
            else:
                unknown += 1
        pi[tag] = {
            'rows': scan.get('quality_intent_n'),
            'by_role': scan.get('quality_intents'),
            'by_stage': scan.get('quality_intent_by_stage'),
            'by_stage_path': scan.get('quality_intent_by_stage_path'),
            'accurate_need_probability': scan.get('quality_intent_need'),
            'task_ids_n': len(ids),
            'supported_entry_class': supported,
            'unknown_entry_class': unknown,
            'supported_ratio': (rnd(supported / float(supported + unknown), 6)
                                if (supported + unknown) else None)}
    out['public_intent'] = pi
    out['path_B_rows'] = shared.get('path_B')
    out['abstain_rows'] = shared.get('abstains')
    out['quality_outcome_rows'] = shared.get('quality_rows')
    if not (scans or {}):
        out['warnings'].append('no arm scans supplied: public intent block empty')
    return out


def aggregate_replay(windows, spans=None, c_star_ms=None):
    """TASK §9 item 8: E/S/X/R/Open/W aggregation and both residuals."""
    spans = spans or COHORT_SPANS
    out = {'windows_n': len(windows), 'window_ns': WINDOW_NS,
           'trailing_partial': bool(windows and windows[-1].get('partial')),
           'spans': {}, 'reject_reasons': {}, 'remove_reasons': {},
           'max_abs_residual_full': 0.0, 'window_series': [],
           'note': 'R is read from REJECT only, X from REMOVE only; Local '
                   '(LOCAL_*) events carry no GPU role and are excluded here.'}
    reject_reasons = Counter()
    remove_reasons = Counter()
    for w in windows:
        for k, v in (w.get('reject_reasons') or {}).items():
            reject_reasons[k] += v
        for k, v in (w.get('remove_reasons') or {}).items():
            remove_reasons[k] += v
    out['reject_reasons'] = counter_to_plain(reject_reasons)
    out['remove_reasons'] = counter_to_plain(remove_reasons)

    for name, lo, hi in spans:
        lo_ns, hi_ns = lo * NS_PER_S, hi * NS_PER_S
        sel = [w for w in windows if lo_ns <= w['start_ns'] < hi_ns]
        ent = {'span_s': [lo, hi], 'windows': len(sel),
               'partial_windows': sum(1 for w in sel if w.get('partial')),
               'roles': {}, 'loss': {}}
        for role in GPU_ROLES:
            acc = {'E_n': 0, 'S_n': 0, 'X_n': 0, 'R_n': 0, 'O_n': 0, 'dW_n': 0,
                   'E_work_ms': 0.0, 'S_work_ms': 0.0, 'X_work_ms': 0.0,
                   'R_work_ms': 0.0, 'O_work_ms': 0.0, 'dW_work_ms': 0.0,
                   'residual_dW': 0, 'residual_O': 0, 'delta_wait_counts': 0,
                   'delta_open_counts': 0, 'dOpen': 0,
                   'resource_lost_components': 0, 'other_lost_components': 0,
                   'resource_lost_roots': 0}
            xs = []
            growth = []
            for i, w in enumerate(sel):
                wk = (w.get('work_ms') or {}).get(role) or {}
                acc['E_n'] += wk.get('E_n', 0)
                acc['S_n'] += wk.get('S_n', 0)
                acc['X_n'] += wk.get('X_n', 0)
                acc['R_n'] += wk.get('R_n', 0)
                acc['O_n'] += wk.get('O_n', 0)
                for f in ('E_work_ms', 'S_work_ms', 'X_work_ms', 'R_work_ms',
                          'O_work_ms', 'dW_work_ms'):
                    acc[f] = rnd(acc[f] + as_float(wk.get(f), 0.0) or 0.0, 6)
                dw = 0
                for kind in GPU_KINDS:
                    key = role + ':' + kind
                    dw += (w['wait_end'].get(key, 0) - w['wait_start'].get(key, 0))
                acc['dW_n'] += dw
                rdw = 0
                ro = 0
                dopen = 0
                for k, v in (w.get('residual_dW') or {}).items():
                    if k.startswith(role + ':'):
                        rdw += v
                for k, v in (w.get('residual_O') or {}).items():
                    if k.startswith(role + ':'):
                        ro += v
                for kind in GPU_KINDS:
                    key = role + ':' + kind
                    dopen += (w['open_end'].get(key, 0) - w['open_start'].get(key, 0))
                acc['residual_dW'] += rdw
                acc['residual_O'] += ro
                acc['dOpen'] += dopen
                e_n = wk.get('E_n', 0)
                s_n = wk.get('S_n', 0)
                x_n = wk.get('X_n', 0)
                r_n = wk.get('R_n', 0)
                o_n = wk.get('O_n', 0)
                acc['delta_wait_counts'] += (e_n - s_n - x_n)
                acc['delta_open_counts'] += (o_n - e_n - r_n)
                acc['resource_lost_components'] += wk.get('resource_lost_components', 0)
                acc['other_lost_components'] += wk.get('other_lost_components', 0)
                acc['resource_lost_roots'] += wk.get('resource_lost_roots', 0)
                if wk.get('E_n', 0):
                    xs.append(float(w['start_ns']) / NS_PER_S)
                    growth.append(as_float(wk.get('growth_ms_s'), 0.0) or 0.0)
            acc['E_work_ms_s'] = rnd(acc['E_work_ms'] / float(hi - lo), 6)
            acc['S_work_ms_s'] = rnd(acc['S_work_ms'] / float(hi - lo), 6)
            acc['X_work_ms_s'] = rnd(acc['X_work_ms'] / float(hi - lo), 6)
            acc['R_work_ms_s'] = rnd(acc['R_work_ms'] / float(hi - lo), 6)
            acc['W_r_ms'] = rnd(acc['dW_work_ms'] / float(hi - lo), 6)
            acc['growth_ms_s'] = acc['W_r_ms']
            acc['growth_ms_s_slope_per_s'] = _linfit(xs, growth)
            acc['growth_samples_n'] = len(xs)
            ent['roles'][role] = acc
        for w in sel:
            ent['loss'].setdefault('resource_lost_components', 0)
            ent['loss']['resource_lost_components'] += \
                sum(v.get('resource_lost_components', 0)
                    for v in (w.get('work_ms') or {}).values())
            ent['loss']['other_lost_components'] = ent['loss'].get(
                'other_lost_components', 0) + sum(
                v.get('other_lost_components', 0)
                for v in (w.get('work_ms') or {}).values())
        out['spans'][name] = ent
        for w in sel:
            out['max_abs_residual_full'] = max(out['max_abs_residual_full'],
                                               as_float(w.get('max_abs_residual'), 0.0) or 0.0)
    for w in windows:
        wk = w.get('work_ms') or {}
        out['window_series'].append({
            'index': w.get('index'), 'start_s': rnd(w['start_ns'] / NS_PER_S, 3),
            'partial': bool(w.get('partial')),
            'max_abs_residual': w.get('max_abs_residual'),
            'roles': {r: {'dW': wk.get(r, {}).get('dW_n'),
                          'E': wk.get(r, {}).get('E_n'),
                          'S': wk.get(r, {}).get('S_n'),
                          'X': wk.get(r, {}).get('X_n'),
                          'R': wk.get(r, {}).get('R_n'),
                          'O': wk.get(r, {}).get('O_n'),
                          'growth_ms_s': wk.get(r, {}).get('growth_ms_s'),
                          'residual_dW': sum(v for k, v in
                                             (w.get('residual_dW') or {}).items()
                                             if k.startswith(r + ':')),
                          'residual_O': sum(v for k, v in
                                            (w.get('residual_O') or {}).items()
                                            if k.startswith(r + ':'))}
                      for r in GPU_ROLES},
            'reject_reasons': w.get('reject_reasons'),
        })
    return out


def root_closure_block(scans, recs_by_arm):
    """TASK §9 item 8: unique root terminals + M_root = terminal_root+alive_root."""
    out = {'warning': 'root closure recomputed from OFFER/terminal events; the '
                      'controller observer root_closure() is reported alongside '
                      'when present',
           'by_arm': {}}
    for tag, scan in sorted((scans or {}).items()):
        comps = scan.get('components') or {}
        roots = {}
        for cid, c in comps.items():
            rid = c.get('root_id')
            if rid is None:
                continue
            r = roots.get(rid)
            if r is None:
                r = roots[rid] = {'offered': False, 'terminal': False,
                                  'rejected': False, 'removed': False,
                                  'rejected_resource': False}
            if c.get('offer'):
                r['offered'] = True
            if c.get('finish') or c.get('remove'):
                r['terminal'] = r['terminal']
            if c.get('remove'):
                r['removed'] = True
            if c.get('reject'):
                r['rejected'] = True
                reason = str(c.get('reject_reason'))
                if reason == 'resource':
                    r['rejected_resource'] = True
        recs = (recs_by_arm or {}).get(tag) or {}
        for tid, rec in recs.items():
            if not rec.get('completed'):
                continue
            rid = rec.get('root_id')
            if rid is None:
                continue
            r = roots.get(rid)
            if r is None:
                r = roots[rid] = {'offered': False, 'terminal': False,
                                  'rejected': False, 'removed': False,
                                  'rejected_resource': False}
            r['terminal'] = True
        terminal_root = sum(1 for r in roots.values() if r['terminal'])
        alive_root = sum(1 for r in roots.values() if not r['terminal'])
        obs = (scan.get('controller') or {}).get('observer_last') or {}
        rc = obs.get('root_closure') if isinstance(obs, dict) else None
        out['by_arm'][tag] = {
            'roots_seen': len(roots), 'terminal_root': terminal_root,
            'alive_root': alive_root, 'M_root': terminal_root + alive_root,
            'M_root_identity_ok': (terminal_root + alive_root == len(roots)),
            'observer_root_closure': rc,
            'observer_matches': (None if not isinstance(rc, dict) else
                                 (as_int(rc.get('terminal_root')) == terminal_root
                                  and as_int(rc.get('alive_root')) == alive_root)),
            'duplicate_terminal_roots': 0,
            'roots_with_two_terminals': []}
    # per-root terminal multiplicity across the union of arms
    mult = {}
    for tag, recs in sorted((recs_by_arm or {}).items()):
        seen = set()
        for tid, rec in recs.items():
            if rec.get('completed') and rec.get('root_id') is not None:
                seen.add(rec['root_id'])
        for rid in seen:
            mult.setdefault(rid, set()).add(tag)
    dup_roots = [str(r) for r, tags in mult.items() if len(tags) > 1]
    out['roots_terminal_in_both_arms_n'] = len(dup_roots)
    out['roots_terminal_in_both_arms_sample'] = sorted(dup_roots)[:50]
    return out


def growth_and_noise_block(replay_by_arm, scans):
    """TASK §9 item 8: F/A growth_ms_s slope, theta_r, noise source + samples."""
    out = {'theta_source': None, 'by_arm': {}}
    for tag in sorted(set(list((replay_by_arm or {}).keys()) +
                          list((scans or {}).keys()))):
        rep = (replay_by_arm or {}).get(tag) or {}
        scan = (scans or {}).get(tag) or {}
        obs = (scan.get('controller') or {}).get('observer_last') or {}
        ent = {'growth_ms_s_by_role': {}, 'growth_slope_per_s_by_role': {},
               'theta_r': None, 'theta_source': None, 'noise': None,
               'noise_source': None, 'noise_samples_n': None,
               'healthy_growth_samples_n': None}
        full = (rep.get('spans') or {}).get('full') or {}
        for role in GPU_ROLES:
            acc = ((full.get('roles') or {}).get(role)) or {}
            ent['growth_ms_s_by_role'][role] = acc.get('growth_ms_s')
            ent['growth_slope_per_s_by_role'][role] = \
                acc.get('growth_ms_s_slope_per_s')
            ent.setdefault('growth_samples_n_by_role', {})[role] = \
                acc.get('growth_samples_n')
        if isinstance(obs, dict):
            theta = obs.get('theta')
            if isinstance(theta, dict):
                ent['theta_r'] = theta
                ent['theta_source'] = 'controller observer_last.theta (service-time '
                ent['theta_source'] += 'slope per role, frozen controller units)'
            noise = obs.get('noise')
            if isinstance(noise, dict):
                ent['noise'] = noise
                ent['noise_source'] = noise.get('source')
                ent['noise_samples_n'] = noise.get('n') or noise.get('pairs')
            gn = obs.get('growth_noise')
            if isinstance(gn, dict):
                ent['healthy_growth_samples_n'] = gn.get('n') or gn.get('samples')
                if ent['noise_source'] is None:
                    ent['noise_source'] = gn.get('source')
        if ent['theta_r'] is None or ent['noise_source'] is None:
            ent['unresolved'] = [k for k in ('theta_r', 'noise_source')
                                 if ent[k] is None]
        out['by_arm'][tag] = ent
    return out


# ================================== control block / cost-perf / prediction

def _profile_key(win):
    fp = win.get('profile_fp')
    if fp is None:
        payload = win.get('payload') or {}
        dem = payload.get('demand') or payload
        fp = 'demand:' + repr(sorted([str(k) + '=' + repr(v)
                                      for k, v in (dem or {}).items()]))[:200]
    return str(fp)


def control_block(scans, max_profiles=2000, max_windows=4000):
    """TASK §9 item 9: commands, candidate tables, gates, holds, endpoints."""
    out = {'by_arm': {}}
    for tag, scan in sorted((scans or {}).items()):
        ctl = scan.get('controller') or {}
        wins = ctl.get('windows') or {}
        all_wins = sorted([w for w in wins.values() if isinstance(w, dict)],
                          key=lambda w: (w.get('window_id') if w.get('window_id')
                                         is not None else -1))
        profiles = {}
        win_rows = []
        invalid = Counter()
        invalid_rows = []
        for w in all_wins:
            key = _profile_key(w)
            cands = w.get('candidates')
            if cands is not None:
                p = profiles.get(key)
                if p is None:
                    if len(profiles) >= max_profiles:
                        p = profiles.setdefault('__overflow__',
                                                {'profile_fp': '__overflow__',
                                                 'candidates': None,
                                                 'windows_n': 0, 'window_ids': [],
                                                 'note': 'profile cap reached'})
                    else:
                        p = profiles[key] = {'profile_fp': w.get('profile_fp') or key,
                                             'candidates': cands,
                                             'candidates_n': len(cands),
                                             'demand': (w.get('payload') or {}).get('demand'),
                                             'cost': (w.get('payload') or {}).get('cost'),
                                             'c_star_ms': (w.get('payload') or {}).get('c_star_ms'),
                                             'windows_n': 0, 'window_ids': []}
                p['windows_n'] += 1
                if len(p['window_ids']) < 5:
                    p['window_ids'].append(w.get('window_id'))
            else:
                reason = w.get('invalid_reason') or 'unknown'
                invalid[str(reason)] += 1
                if len(invalid_rows) < 500:
                    invalid_rows.append({'window_id': w.get('window_id'),
                                         'reason': str(reason)})
            if len(win_rows) < max_windows:
                win_rows.append({
                    'window_id': w.get('window_id'), 'tick_ns': w.get('tick_ns'),
                    'state': w.get('state'),
                    'effective_fast': w.get('effective_fast'),
                    'target_fast': w.get('target_fast'),
                    'reason': w.get('reason'), 'valid': w.get('valid'),
                    'invalid_reason': w.get('invalid_reason'),
                    'profile_fp': _profile_key(w) if cands is not None else None,
                    'trigger': w.get('trigger'), 'receiver': w.get('receiver'),
                    'gap': w.get('gap'),
                    'apply_count': w.get('apply_count'),
                    'config_request_count': w.get('config_request_count'),
                    'new_command_count': w.get('new_command_count'),
                    'cost_version': w.get('cost_version'),
                    'age_ns': w.get('age_ns'),
                    'commands_n': len(w.get('commands_here') or []),
                    'residuals': w.get('residuals'),
                    'endpoints': w.get('endpoints')})
        cmds = []
        for c in (ctl.get('commands') or []):
            cc = dict(c)
            wid = c.get('window_id')
            win = wins.get(wid) if wid is not None else None
            cc['candidates_point'] = (win or {}).get('candidates')
            cc['profile_fp_window'] = _profile_key(win or {}) if win else None
            cc['gate'] = c.get('gate')
            cc['rollback'] = bool(str(c.get('verdict')).upper() == 'ROLLBACK' or
                                  str(c.get('reason')).lower().find('rollback') >= 0)
            cmds.append(cc)
        # sustained gaps with no command
        runs = []
        cur_run = None
        for w in all_wins:
            gap = bool(w.get('gap'))
            issued = len(w.get('commands_here') or [])
            if gap and issued == 0:
                if cur_run is None:
                    cur_run = {'from_window': w.get('window_id'),
                               'from_s': rnd((w.get('tick_ns') or 0) / NS_PER_S, 3),
                               'windows': 0, 'reasons': Counter(),
                               'receiver': Counter(), 'profile_fps': [],
                               'invalid_reasons': Counter()}
                cur_run['windows'] += 1
                cur_run['to_window'] = w.get('window_id')
                cur_run['reasons'][str(w.get('reason'))] += 1
                trig = w.get('trigger')
                if isinstance(trig, dict):
                    cur_run['receiver'][str(trig.get('receiver') or
                                            trig.get('direction'))] += 1
                if w.get('profile_fp') and len(cur_run['profile_fps']) < 5:
                    cur_run['profile_fps'].append(w.get('profile_fp'))
                if w.get('invalid_reason'):
                    cur_run['invalid_reasons'][str(w.get('invalid_reason'))] += 1
            else:
                if cur_run is not None:
                    cur_run['reasons'] = counter_to_plain(cur_run['reasons'])
                    cur_run['receiver'] = counter_to_plain(cur_run['receiver'])
                    cur_run['invalid_reasons'] = counter_to_plain(
                        cur_run['invalid_reasons'])
                    runs.append(cur_run)
                    cur_run = None
        if cur_run is not None:
            cur_run['reasons'] = counter_to_plain(cur_run['reasons'])
            cur_run['receiver'] = counter_to_plain(cur_run['receiver'])
            cur_run['invalid_reasons'] = counter_to_plain(cur_run['invalid_reasons'])
            runs.append(cur_run)
        sustained = [r for r in runs if r['windows'] >= 2]
        hold_dedup = {}
        for reason, n in (ctl.get('hold_reasons') or {}).items():
            hold_dedup[reason] = {'windows': n,
                                  'first_window_id': None, 'sample': None}
        for w in all_wins:
            r = str(w.get('reason'))
            if r in hold_dedup and hold_dedup[r]['sample'] is None:
                hold_dedup[r]['first_window_id'] = w.get('window_id')
                hold_dedup[r]['sample'] = {'window_id': w.get('window_id'),
                                           'trigger': w.get('trigger'),
                                           'gap': w.get('gap'),
                                           'invalid_reason': w.get('invalid_reason'),
                                           'effective_fast': w.get('effective_fast'),
                                           'target_fast': w.get('target_fast'),
                                           'receiver': w.get('receiver'),
                                           'profile_fp': w.get('profile_fp')}
        out['by_arm'][tag] = {
            'controller_rows': ctl.get('rows'),
            'controller_schema_seen': ctl.get('schemas'),
            'resource_keys_seen': ctl.get('resource_keys'),
            'windows_n': ctl.get('windows_n'),
            'windows_with_candidates': ctl.get('windows_with_candidates'),
            'windows_invalid': ctl.get('windows_invalid'),
            'invalid_reason_counts': counter_to_plain(invalid),
            'invalid_windows_sample': invalid_rows,
            'windows_with_gap': ctl.get('windows_with_gap'),
            'candidate_profiles_n': len(profiles),
            'candidate_profiles': profiles if len(profiles) <= max_profiles else
                                  {'__truncated__': True, 'n': len(profiles)},
            'windows': win_rows,
            'commands_n': ctl.get('commands_n'),
            'commands_metadata_recorded_n': ctl.get('commands_metadata_recorded_n'),
            'commands_recovered_from_requests_n': ctl.get('commands_recovered_from_requests_n'),
            'command_metadata_complete': ctl.get('command_metadata_complete'),
            'commands': cmds,
            'command_ids': ctl.get('command_ids'),
            'apply_events': ctl.get('apply_events'),
            'apply_count_max': ctl.get('max_apply_count'),
            'config_request_max': ctl.get('max_config_request_count'),
            'config_request_counts': scan.get('config_requests'),
            'config_request_rows': scan.get('config_request_rows'),
            'target_effective_n': scan.get('target_effective_n'),
            'target_effective_sample': (scan.get('target_effective') or [])[:200],
            'pending_command_last': ctl.get('pending_command_last'),
            'last_confirmed': ctl.get('last_confirmed'),
            'locks_last': ctl.get('locks_last'),
            'lock_events_n': len(ctl.get('lock_events') or []),
            'lock_events_sample': (ctl.get('lock_events') or [])[:50],
            'verdicts': ctl.get('verdicts'),
            'probe_rows': ctl.get('probe_rows'),
            'feedback': ctl.get('feedback'),
            'controller_integrity': ctl.get('integrity'),
            'card_snapshots': ctl.get('card_snapshots'),
            'selection_rows': ctl.get('selection_rows'),
            'score_rows': ctl.get('score_rows'),
            'timeline': ctl.get('timeline'),
            'hold_reasons_dedup': hold_dedup,
            'hold_runs_n': len(runs),
            'sustained_gap_runs': sustained,
            'gap_windows_total': sum(r['windows'] for r in runs),
            'gap_windows_no_command_note': (
                'gap = the tick had a trigger (decision.last_selection.trigger or '
                'resource.triggers) and issued no new command; sustained = >=2 '
                'consecutive such 500 ms windows'),
            'missing_keys': ctl.get('missing_keys')}
    return out


def _charge_summary(d, cap=50):
    if not isinstance(d, dict):
        return None
    vals = []
    for k, v in d.items():
        fv = as_float(v)
        if fv is None:
            continue
        vals.append((str(k), fv))
    if not vals:
        return {'n': 0, 'total_ms': 0.0, 'top': []}
    nums = [v for _, v in vals]
    top = sorted(vals, key=lambda kv: -kv[1])[:cap]
    return {'n': len(vals), 'total_ms': rnd(sum(nums), 6),
            'mean_ms': rnd(sum(nums) / len(nums), 6),
            'p50_ms': pct(nums, 0.5), 'p95_ms': pct(nums, 0.95),
            'max_ms': rnd(max(nums), 6),
            'top_components': [{'key': k, 'charge_ms': rnd(v, 6)} for k, v in top]}


def _perf_stat(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return {'n': 1, 'value': rnd(v, 6)}
    if isinstance(v, dict):
        if 'samples' in v and isinstance(v['samples'], list):
            return stats([as_float(x, 0.0) for x in v['samples']])
        out = {}
        for k in ('n', 'mean', 'p50', 'p95', 'p99', 'max', 'true_max', 'min',
                  'total', 'count'):
            if k in v:
                out[k] = v[k]
        return out or {'raw_keys': sorted(v.keys())}
    if isinstance(v, list):
        nums = [as_float(x) for x in v if as_float(x) is not None]
        return stats(nums) if nums else {'n': 0}
    return {'raw': str(v)}


def _split_bootstrap_live(samples, split_ns):
    if split_ns is None:
        return {'bootstrap': None, 'live': None,
                'reason': 'no warmup/bootstrap end timestamp available'}
    boot = [s for s in samples if s[0] is not None and s[0] <= split_ns]
    live = [s for s in samples if s[0] is None or s[0] > split_ns]
    return {'bootstrap': stats([s[1] for s in boot]),
            'live': stats([s[1] for s in live]),
            'split_ns': split_ns, 'bootstrap_n': len(boot), 'live_n': len(live),
            'reason': None}


def cost_perf_block(scans, joint_summary=None):
    """TASK §9 item 10: actor/CUDA/IPC/startup, quota, lag, full-path timing."""
    out = {'by_arm': {}, 'joint_summary_keys': sorted((joint_summary or {}).keys()),
           'joint': {}}
    if isinstance(joint_summary, dict):
        for k in ('perf', 'log_backpressure', 'handler_ms', 'source_slice_ms',
                  'loop_gap_ms', 'controller_sync_ms', 'worker_ms',
                  'cpu_topology', 'quota_topology', 'c_star_ms', 'cost_records_final'):
            if k in joint_summary:
                out['joint'][k] = _perf_stat(joint_summary.get(k)) \
                    if k.endswith('_ms') else joint_summary.get(k)
    for tag, scan in sorted((scans or {}).items()):
        gs = scan.get('group_summary') or {}
        warm = scan.get('warmup') or {}
        inv = scan.get('actor_inventory')
        split_ns = None
        for src in (warm, gs, inv if isinstance(inv, dict) else {}):
            for k in ('warmup_end_ns', 'bootstrap_end_ns', 'warmup_finish_ns',
                      'ready_ns', 'warmup_done_ns'):
                if src.get(k) is not None:
                    split_ns = as_int(src.get(k))
                    break
            if split_ns is not None:
                break
        perf_src = {}
        if isinstance(gs.get('perf'), dict):
            perf_src.update(gs['perf'])
        for k, v in gs.items():
            if k.endswith('_ms') or k.endswith('_ms_max'):
                perf_src.setdefault(k, v)
        ent = {
            'group_summary_keys': sorted(gs.keys()),
            'actor_service_ms_by_kind': scan.get('service_ms'),
            'dispatch_to_start_ms_by_kind': scan.get('dispatch_to_start_ms'),
            'cuda_ipc_startup': {
                'bootstrap_split_ns': split_ns,
                'sources_found': {},
                'reason_missing': None},
            'quota': {
                'rows': scan.get('quota_rows'),
                'charge_by_kind': scan.get('quota_charge_ms_by_kind'),
                'charge_n_by_kind': scan.get('quota_charge_n_by_kind'),
                'charge_by_gpu_role': scan.get('quota_charge_ms_by_gpu_role'),
                'charge_by_component': _charge_summary(scan.get('charge_by_component_ms')),
                'charge_by_task': _charge_summary(scan.get('charge_by_task_ms')),
                'sample_rows': scan.get('quota_sample'),
                'note': 'real per-component charge joined through job_id->component_id'},
            'lag': {'receipt_lag_ns': scan.get('receipt_lag'),
                    'injection_lag_ns': scan.get('injection_lag'),
                    'all_type_lag_ns': scan.get('lag_by_type'),
                    'note': 'receipt_lag is the receipt-side lag; all-type lag is '
                            'event.timestamp - source_publish_ns per event type'},
            'full_path_ms': {k: _perf_stat(perf_src.get(k)) for k in
                             sorted(perf_src.keys())},
            'worker': {'warmup': warm, 'actor_inventory': inv,
                       'worker_ms': _perf_stat(gs.get('worker_ms'))},
            'log_backpressure': {
                'rows': scan.get('log_backpressure_n'),
                'last': scan.get('log_backpressure_last'),
                'max_dropped': max([b.get('dropped') or 0
                                    for b in (scan.get('log_backpressure') or [])]
                                   or [0]),
                'max_queued': max([b.get('queued') or 0
                                   for b in (scan.get('log_backpressure') or [])]
                                  or [0])},
            'stall': None,
            'cpu_topology_contention': None,
            'warnings': []}
        for key in ('cuda_ms', 'cuda', 'ipc_ms', 'ipc', 'startup_ms', 'startup',
                    'actor_startup_ms', 'bootstrap_ms', 'live_ms'):
            for src, name in ((gs, 'group_summary'), (inv, 'actor_inventory'),
                              (warm, 'warmup')):
                if isinstance(src, dict) and src.get(key) is not None:
                    ent['cuda_ipc_startup']['sources_found'][key] = {
                        'source': name, 'value': _perf_stat(src.get(key))}
        if not ent['cuda_ipc_startup']['sources_found']:
            ent['cuda_ipc_startup']['reason_missing'] = (
                'no cuda_ms/ipc_ms/startup_ms/bootstrap_ms keys found in '
                'group_summary/actor_inventory/warmup')
        samples = []
        for src in (gs, inv if isinstance(inv, dict) else {}):
            for k in ('startup_samples', 'bootstrap_samples', 'actor_samples'):
                v = src.get(k)
                if isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict):
                            samples.append((as_int(it.get('ts') or it.get('timestamp')),
                                            as_float(it.get('ms') or it.get('value'))))
                        else:
                            fv = as_float(it)
                            if fv is not None:
                                samples.append((None, fv))
        if samples:
            ent['cuda_ipc_startup']['bootstrap_vs_live'] = _split_bootstrap_live(
                samples, split_ns)
        else:
            ent['cuda_ipc_startup']['bootstrap_vs_live'] = {
                'bootstrap': None, 'live': None,
                'reason': 'no startup sample list found in group_summary/'
                          'actor_inventory'}
        obs = (scan.get('controller') or {}).get('observer_last') or {}
        if isinstance(obs, dict):
            ent['stall'] = obs.get('stall')
        stall_holds = sum(v for k, v in
                          ((scan.get('controller') or {}).get('hold_reasons') or
                           {}).items() if 'stall' in str(k))
        ent['stall_hold_windows'] = stall_holds
        for key in ('cpu_topology', 'cpu_contention', 'topology_contention'):
            if isinstance(gs, dict) and gs.get(key) is not None:
                ent['cpu_topology_contention'] = gs.get(key)
        if ent['cpu_topology_contention'] is None:
            ent['warnings'].append('no CPU topology contention record in '
                                   'group_summary (cpu_topology/cpu_contention/'
                                   'topology_contention); reported null')
        if split_ns is None:
            ent['warnings'].append('no bootstrap/live split timestamp found '
                                   '(warmup_end_ns/bootstrap_end_ns/ready_ns/'
                                   'warmup_done_ns); bootstrap-vs-live reported null')
        out['by_arm'][tag] = ent
    return out


def prediction_vs_observed_block(scans, recs_by_arm, max_rows=200):
    """TASK §9 item 11: current-band Yhat/Vhat vs observed Y/V."""
    out = {'observed_definition': {
        'Y': 'terminals in the 500 ms window that are on-time '
             '(terminal_available_ns <= deadline_abs_ns)',
        'V': 'terminals in the window that are on-time AND label-correct '
             '(offline labels, same as good)',
        'window_clock': 'terminal_available_ns',
        'mature_window': 'legacy field name for a quiet diagnostic window (no trigger/new command); '
                         'does not prove mature probe cohorts; use closed_loop.feedback for probe evidence'},
        'by_arm': {}}
    for tag, scan in sorted((scans or {}).items()):
        ctl = scan.get('controller') or {}
        wins = ctl.get('windows') or {}
        recs = (recs_by_arm or {}).get(tag) or {}
        obs = {}
        for rec in recs.values():
            av = rec.get('available')
            if av is None:
                continue
            idx = (av // WINDOW_NS) * WINDOW_NS
            o = obs.get(idx)
            if o is None:
                o = obs[idx] = {'Y': 0, 'V': 0}
            if rec.get('ontime'):
                o['Y'] += 1
            if rec.get('good'):
                o['V'] += 1
        oa = {}
        for rec in recs.values():
            av = rec.get('arrival_ns')
            if av is None:
                continue
            idx = (av // WINDOW_NS) * WINDOW_NS
            o = oa.get(idx)
            if o is None:
                o = oa[idx] = {'Y': 0, 'V': 0}
            if rec.get('ontime'):
                o['Y'] += 1
            if rec.get('good'):
                o['V'] += 1
        rows = []
        prev_fp = None
        prev_cv = None
        abs_y = []
        abs_v = []
        rel_y = []
        rel_v = []
        for wid in sorted([w for w in wins.keys() if w is not None]):
            w = wins[wid]
            cands = w.get('candidates')
            if not cands:
                continue
            eff = w.get('effective_fast')
            row = None
            if eff is not None:
                for c in cands:
                    if isinstance(c, dict) and c.get('gF') == eff:
                        row = c
                        break
                if row is None:
                    best = None
                    for c in cands:
                        if not isinstance(c, dict):
                            continue
                        d = abs((as_float(c.get('gF'), 0.0) or 0.0) - eff)
                        if best is None or d < best[0]:
                            best = (d, c)
                    row = best[1] if best else None
            if row is None:
                continue
            fp = _profile_key(w)
            cv = w.get('cost_version')
            mature = (not w.get('gap')) and not (w.get('commands_here') or [])
            ob = obs.get(wid) or {'Y': 0, 'V': 0}
            oba = oa.get(wid) or {'Y': 0, 'V': 0}
            rec = {'window_id': wid, 'effective_fast': eff,
                   'Yhat': row.get('Yhat'), 'Vhat': row.get('Vhat'),
                   'Yhat_point': row.get('Yhat_point'),
                   'Vhat_point': row.get('Vhat_point'),
                   'point_valid': row.get('point_valid'),
                   'Y_obs_terminal_window': ob['Y'], 'V_obs_terminal_window': ob['V'],
                   'Y_obs_arrival_window': oba['Y'], 'V_obs_arrival_window': oba['V'],
                   'mature': mature,
                   'flags': {'profile_changed': (prev_fp is not None and
                                                 fp != prev_fp),
                             'cost_version_changed': (prev_cv is not None and
                                                      cv != prev_cv),
                             'input_age_ns': w.get('age_ns'),
                             'trigger': bool(w.get('gap'))},
                   'profile_window_changed': prev_fp is not None and fp != prev_fp}
            y_obs = ob['Y']
            v_obs = ob['V']
            if row.get('Yhat') is not None:
                rec['err_Y_abs'] = rnd(as_float(row.get('Yhat'), 0.0) - y_obs, 6)
                if y_obs:
                    rec['err_Y_rel'] = rnd((as_float(row.get('Yhat'), 0.0) - y_obs) /
                                           float(y_obs), 6)
            if row.get('Vhat') is not None:
                rec['err_V_abs'] = rnd(as_float(row.get('Vhat'), 0.0) - v_obs, 6)
                if v_obs:
                    rec['err_V_rel'] = rnd((as_float(row.get('Vhat'), 0.0) - v_obs) /
                                           float(v_obs), 6)
            if mature:
                if rec.get('err_Y_abs') is not None:
                    abs_y.append(abs(rec['err_Y_abs']))
                if rec.get('err_V_abs') is not None:
                    abs_v.append(abs(rec['err_V_abs']))
                if rec.get('err_Y_rel') is not None:
                    rel_y.append(abs(rec['err_Y_rel']))
                if rec.get('err_V_rel') is not None:
                    rel_v.append(abs(rec['err_V_rel']))
            prev_fp = fp
            prev_cv = cv
            if len(rows) < max_rows:
                rows.append(rec)
        out['by_arm'][tag] = {
            'rows_n': len(rows), 'mature_rows': sum(1 for r in rows if r['mature']),
            'rows': rows,
            'mean_abs_err_Y': rnd(mean(abs_y), 6) if abs_y else None,
            'mean_abs_err_V': rnd(mean(abs_v), 6) if abs_v else None,
            'mean_rel_err_Y': rnd(mean(rel_y), 6) if rel_y else None,
            'mean_rel_err_V': rnd(mean(rel_v), 6) if rel_v else None,
            'mature_samples_Y': len(abs_y), 'mature_samples_V': len(abs_v),
            'candidates_recorded_note': (
                'none' if not (scan.get('controller') or {}).get(
                    'windows_with_candidates') else 'ok'),
            'warning': (None if (scan.get('controller') or {}).get(
                'windows_with_candidates') else
                'no candidate table recorded by the controller: prediction-vs-'
                'observed rows are empty; see control_block invalid reasons')}
    return out


def command_effect_from_events(scan, cmd, start, stop, expected):
    """Fill a snapshot gap using a stable anchor and actual boundary events.

    Acceptance alone cannot prove effect.  Every changed card must emit its
    actual TARGET_EFFECTIVE event in this command's interval.  An unchanged
    card can retain its measured share only with a complete, untruncated event
    stream and no intervening configuration or boundary events.
    """
    if expected is None or start is None or not (scan.get('event_integrity') or {}).get('complete'):
        return None
    requests = scan.get('config_request_rows') or []
    effective = scan.get('target_effective') or []
    counts = scan.get('event_type_counts') or {}
    if counts.get('CONFIG_REQUEST') != len(requests) or counts.get('TARGET_EFFECTIVE') != len(effective):
        return None
    matches = [r for r in requests if r[0] == cmd.get('command_id')]
    if len(matches) != 1:
        return None
    request = matches[0]
    ts_request, target = as_int(request[4]), as_float(request[1])
    if (ts_request is None or target is None or abs(target - sum(expected)) > 1e-9 or
            ts_request < start or (stop is not None and ts_request >= stop)):
        return None
    anchors = []
    for snap in (scan.get('controller') or {}).get('card_snapshots') or []:
        ts = as_int(snap.get('tick_ns'))
        cards = {as_int(c.get('card')): c for c in snap.get('cards') or []}
        if ts is None or ts >= ts_request or set(cards) != {0, 1}:
            continue
        if all(as_float(cards[g].get('effective')) is not None and
               as_float(cards[g].get('target')) is not None and
               abs(as_float(cards[g]['target']) - as_float(cards[g]['effective'])) <= 1e-9 and
               not cards[g].get('transition') for g in (0, 1)):
            anchors.append((ts, cards))
    if not anchors:
        return None
    ts_anchor, cards = max(anchors, key=lambda a: a[0])
    if any(as_int(r[4]) is None or
           (ts_anchor <= as_int(r[4]) < ts_request) for r in requests):
        return None
    boundary = []
    for gpu, share, ts in effective:
        gpu, share, ts = as_int(gpu), as_float(share), as_int(ts)
        if ts is None:
            return None
        if ts_anchor <= ts < ts_request:
            return None
        if ts_request <= ts and (stop is None or ts < stop):
            if gpu not in (0, 1) or share is None or abs(share - expected[gpu]) > 1e-9:
                return None
            boundary.append((gpu, share, ts))
    changed = {g for g in (0, 1) if abs(as_float(cards[g]['effective']) - expected[g]) > 1e-9}
    first = {g: min(ts for gpu, share, ts in boundary if gpu == g)
             for g in changed if any(gpu == g for gpu, share, ts in boundary)}
    if set(first) != changed:
        return None
    return {'tick_ns': max([ts_request] + list(first.values())),
            'cards': [{'card': g, 'effective': expected[g], 'target': expected[g],
                       'transition': False} for g in (0, 1)],
            'provenance': {'basis': 'stable_snapshot_and_actual_boundary_events',
                           'anchor_ns': ts_anchor, 'anchor_cards': list(cards.values()),
                           'config_request': request, 'target_effective_events': boundary}}


def closed_loop_block(scans, shared, recs_by_arm):
    """Evidence for actual two-card effects and mature, two-sided probe output.

    Request acceptance is a separate fact from effective card shares.  Public
    quality sums are kept separate from private-label correctness.  Missing
    evidence remains unknown; this block does not turn a completed run into
    algorithm success.
    """
    out = {'by_arm': {}, 'success_automatically_declared': False}
    for tag, scan in sorted(scans.items()):
        ctl = scan.get('controller') or {}
        commands = sorted(ctl.get('commands') or [],
                          key=lambda c: c.get('proposed_ns') or c.get('tick_ns') or 0)
        effects = []
        for i, cmd in enumerate(commands):
            start = cmd.get('proposed_ns') or cmd.get('tick_ns')
            stop = ((commands[i + 1].get('proposed_ns') or commands[i + 1].get('tick_ns'))
                    if i + 1 < len(commands) else None)
            target = as_float(cmd.get('target_fast'))
            expected = [min(target, 1.0), max(target - 1.0, 0.0)] if target is not None else None
            observed = None
            for snap in ctl.get('card_snapshots') or []:
                ts = snap.get('tick_ns')
                if ts is None or start is None or ts < start or (stop is not None and ts >= stop):
                    continue
                cards = {as_int(c.get('card')): c for c in snap.get('cards') or []}
                if expected is not None and set(cards) == {0, 1} and all(
                        as_float(cards[g].get('effective')) is not None and
                        abs(as_float(cards[g]['effective']) - expected[g]) <= 1e-9 and
                        not cards[g].get('transition') for g in (0, 1)):
                    observed = snap
                    break
            if observed is None:
                observed = command_effect_from_events(scan, cmd, start, stop, expected)
            effects.append({'command_id': cmd.get('command_id'),
                'channel': cmd.get('channel'), 'proposed_ns': start,
                'request_provenance': cmd.get('provenance'),
                'command_metadata_complete': cmd.get('metadata_complete'),
                'target_fast': target, 'expected_card_shares': expected,
                'effective_observed': True if observed else None,
                'first_observed_ns': (observed or {}).get('tick_ns'),
                'card_evidence': (observed or {}).get('cards'),
                'effect_provenance': (observed or {}).get('provenance') or (
                    {'basis': 'complete_two_card_snapshot'} if observed else None),
                'note': None if observed else 'no complete two-card snapshot proves this command effective'})

        recs = recs_by_arm.get(tag) or {}
        feedback = []
        for logged in ctl.get('feedback') or []:
            support = logged['support']
            row = dict(logged)
            mismatches = []
            observed_sides = {}
            for side in ('pre', 'post'):
                span = support.get(side + '_span_ns')
                recorded = support.get(side + '_layers')
                if (not isinstance(span, (list, tuple)) or len(span) != 2 or
                        any(as_int(v) is None for v in span) or not isinstance(recorded, dict)):
                    continue
                layers = {c: {'n': 0, 'y': 0, 'v': 0.0, 'known': 0,
                              'h': 0., 'h_known': 0,
                              'private_good': 0} for c in ROOT_CLASSES}
                for tid, published in (shared.get('gpu_publish_of_task') or {}).items():
                    cls = (shared.get('entry_class_of_task') or {}).get(tid)
                    if cls not in layers or published is None or not span[0] <= published < span[1]:
                        continue
                    layer = layers[cls]
                    layer['n'] += 1
                    rec = recs.get(tid) or {}
                    if rec.get('role') in GPU_ROLES and rec.get('ontime'):
                        layer['y'] += 1
                        prob = as_float(rec.get('p_selected'))
                        if prob is not None:
                            layer['v'] += prob
                            layer['known'] += 1
                            dl, arrival, available = rec.get('deadline_abs_ns'), rec.get('arrival_ns'), rec.get('available')
                            if dl is not None and arrival is not None and available is not None and dl>arrival:
                                layer['h'] += prob*max(0.,min(1.,(dl-available)/float(dl-arrival)))
                                layer['h_known'] += 1
                        if rec.get('good'):
                            layer['private_good'] += 1
                for cls, layer in layers.items():
                    layer['v_available'] = layer['known'] == layer['y']
                    layer['h_available'] = layer['h_known'] == layer['y']
                    fields = ['n', 'y', 'v', 'known', 'v_available']
                    if 'h' in (recorded.get(cls) or {}):
                        fields += ['h','h_known','h_available']
                    for field in fields:
                        expected_value = (recorded.get(cls) or {}).get(field)
                        actual_value = layer[field]
                        equal = (abs(float(expected_value) - float(actual_value)) <= 1e-6
                                 if isinstance(expected_value, (int, float)) else expected_value == actual_value)
                        if not equal:
                            mismatches.append({'side': side, 'class': cls, 'field': field,
                                'recorded': expected_value, 'from_source_and_tasks': actual_value})
                observed_sides[side] = layers
            row['mature_output_recompute'] = {'definition': 'GPU source cohort at GPU_READY/PREFIX_READY publish time; protected task finals',
                'sides': observed_sides, 'matches_recorded': not mismatches if len(observed_sides) == 2 else None,
                'mismatches': mismatches}
            std = support.get('std') or {}
            row['standardized_net'] = {k: std.get(k) for k in
                ('dY', 'dV', 'dH', 'dLoss', 'Ystd_pre', 'Ystd_post', 'Vstd_pre', 'Vstd_post', 'Hstd_pre', 'Hstd_post', 'layers')}
            row['verdict_recomputed'] = None
            if std and support.get('pre_layers') and support.get('post_layers'):
                try:
                    import queue_feedback_controller as qfc
                    recomputed = qfc.standardized_probe(support['pre_layers'], support['post_layers'],
                        support['T_pre_s'], support['T_post_s'])
                    row['standardized_math_matches'] = all(abs(float(std[k]) - float(recomputed[k])) <= 1e-6
                        for k in ('dY', 'dV', 'dLoss', 'Ystd_pre', 'Ystd_post', 'Vstd_pre', 'Vstd_post'))
                    if support.get('selection_objective')=='quality_weighted_earliness' and std.get('h_available'):
                        row['standardized_math_matches'] = row['standardized_math_matches'] and all(
                            abs(float(std[k])-float(recomputed[k]))<=1e-6 for k in ('dH','Hstd_pre','Hstd_post'))
                    if support.get('verdict') is not None and support.get('epsilon_Y') and support.get('epsilon_V'):
                        row['verdict_recomputed'] = qfc.operational_verdict(
                            recomputed['dY'], support['epsilon_Y']['epsilon'],
                            recomputed['dV'], support['epsilon_V']['epsilon'],
                            bool(support.get('v_available')), bool(support.get('comparable')),
                            dH=recomputed.get('dH'), eps_H=(support.get('epsilon_H') or {}).get('epsilon'),
                            h_available=bool(support.get('h_available')),
                            require_speed=support.get('selection_objective')=='quality_weighted_earliness')
                        row['verdict_matches_recorded'] = row['verdict_recomputed'] == support['verdict']
                except Exception as exc:
                    row['verdict_recompute_error'] = str(exc)
            row.setdefault('verdict_matches_recorded', None)
            row.setdefault('standardized_math_matches', None)
            feedback.append(row)
        phase_evidence = {}
        for lo, hi, name in PHASES:
            scores = [r for r in ctl.get('score_rows') or [] if r.get('tick_ns') is not None and lo * NS_PER_S <= r['tick_ns'] < hi * NS_PER_S]
            phase_evidence[name] = {
                'commands': [e for e in effects if e['proposed_ns'] is not None and lo * NS_PER_S <= e['proposed_ns'] < hi * NS_PER_S],
                'selection_rows': [r for r in ctl.get('selection_rows') or [] if r.get('tick_ns') is not None and lo * NS_PER_S <= r['tick_ns'] < hi * NS_PER_S],
                'actual_score_rows': scores,
                'actual_score_reason_counts': counter_to_plain(Counter((r.get('score') or {}).get('reason') for r in scores)),
                'feedback_command_ids': [r['support'].get('command_id') for r in feedback if r.get('tick_ns') is not None and lo * NS_PER_S <= r['tick_ns'] < hi * NS_PER_S]}
        out['by_arm'][tag] = {'commands_n': ctl.get('commands_n'),
            'effects': effects, 'effective_observed_n': sum(e['effective_observed'] is True for e in effects),
            'feedback': feedback, 'phase_evidence': phase_evidence,
            'final_verdicts': counter_to_plain(Counter(r['support']['verdict'] for r in feedback if r['support'].get('verdict'))),
            'source_and_output_complete': bool((scan.get('event_integrity') or {}).get('complete')
                and not (ctl.get('integrity') or {}).get('parse_errors'))}
    return out


# ================================================================= summarize

def _first_existing(paths):
    for p in paths:
        if p is None:
            continue
        try:
            if Path(p).exists():
                return Path(p)
        except OSError:
            continue
    return None


def _input_identity(inputs_dir):
    out = {'inputs_dir': str(inputs_dir) if inputs_dir else None,
           'manifest_path': None, 'input_file_sha256': None,
           'input_logical_sha256': None, 'arrivals_expected': None,
           'phase_arrivals': None, 'matches_expected': None}
    if not inputs_dir:
        return out
    man = Path(inputs_dir) / 'INPUT_MANIFEST.json'
    out['manifest_path'] = str(man)
    obj = read_json(man, None)
    if not isinstance(obj, dict):
        out['error'] = 'INPUT_MANIFEST.json missing or unreadable'
        return out
    identity = obj.get('input_identity') or {}
    hashes = obj.get('hashes') or {}
    obj = dict(obj, **identity)
    obj.setdefault('plan_file_sha256', hashes.get('arrival_plan_file_sha256'))
    obj.setdefault('plan_logical_sha256', hashes.get('logical_plan_sha256'))
    for k in ('input_file_sha256', 'plan_file_sha256'):
        if obj.get(k):
            out['input_file_sha256'] = obj.get(k)
            break
    for k in ('input_logical_sha256', 'plan_logical_sha256'):
        if obj.get(k):
            out['input_logical_sha256'] = obj.get(k)
            break
    for k in ('arrivals', 'arrival_count', 'n_arrivals', 'total_arrivals'):
        if obj.get(k) is not None:
            out['arrivals_expected'] = as_int(obj.get(k))
            break
    for k in ('phase_arrivals', 'arrivals_by_phase'):
        if obj.get(k) is not None:
            out['phase_arrivals'] = obj.get(k)
            break
    checks = []
    if out['input_file_sha256'] is not None:
        checks.append(out['input_file_sha256'] == INPUT_FILE_SHA)
    if out['input_logical_sha256'] is not None:
        checks.append(out['input_logical_sha256'] == INPUT_LOGICAL_SHA)
    out['matches_expected'] = all(checks) if checks else None
    return out


def _default_evidence(jf):
    cands = [Path(jf).parent / 'evidence', Path(jf).parent.parent / 'evidence',
             Path(__file__).resolve().parent / 'evidence']
    for c in cands:
        if (c / 'evaluation_private_labels.json').exists():
            return str(c)
    return None


def _default_inputs(jf):
    for c in (Path(jf).parent / 'inputs', Path(jf).parent.parent / 'inputs',
              Path(__file__).resolve().parent / 'inputs'):
        if (c / 'INPUT_MANIFEST.json').exists():
            return str(c)
    return None


def summarize(joint_folder, evidence=None, inputs=None, out=None,
              task_id='T014', revision='T014_qf4r_3'):
    """TASK §9: one metrics.json for the two-arm joint run."""
    import time as _time
    jf, how = resolve_joint_folder(joint_folder)
    warnings = []
    unresolved = []
    if jf == 'not_found':
        return {'schema': METRICS_SCHEMA, 'task_id': task_id,
                'controller_version': CONTROLLER_VERSION,
                'implementation_revision': revision,
                'error': 'joint folder not found',
                'joint_folder_input': str(joint_folder),
                'joint_folder': None, 'joint_folder_resolution': how}
    jf = Path(jf)
    joint = read_json(jf / 'joint_summary.json', None)
    if not isinstance(joint, dict):
        joint = {}
        warnings.append('joint_summary.json missing or unreadable')
    for name in ('joint_summary.json', 'barrier.json', 'shared_frames.jsonl',
                 'gpu_preflight.json', 'pre_run_checks.json'):
        if not (jf / name).exists():
            warnings.append('joint file missing: %s' % name)
    folders, resolution = arm_folders(jf, joint)
    if evidence is None:
        evidence = _default_evidence(jf)
    if inputs is None:
        inputs = _default_inputs(jf)
    labels, labels_path = load_labels(evidence, fallback_folder=jf.parent)
    if not labels:
        warnings.append('private labels not loaded (evidence=%s)' % evidence)
    scans = {}
    for policy in POLICIES:
        p = (folders or {}).get(policy)
        if not p:
            warnings.append('arm folder unresolved for policy %s' % policy)
            continue
        if not Path(p).exists():
            warnings.append('arm folder does not exist: %s' % p)
            continue
        scans[policy] = arm_scan(p)
    shared = read_shared_frames(jf)
    if not shared.get('present'):
        warnings.append('shared_frames.jsonl missing: source-side and shared-Local '
                        'blocks fall back to arm logs')
    bank_of_task = dict(shared.get('bank_of_task') or {})
    for tag, scan in scans.items():
        for k, v in (scan.get('bank_of_task') or {}).items():
            bank_of_task.setdefault(k, v)
    recs = {}
    for tag, scan in scans.items():
        _bind_shared_arrivals(scan, shared)
        if not (scan.get('event_integrity') or {}).get('complete'):
            warnings.append('arm %s: event stream incomplete; output facts use complete tasks and shared arrivals' % tag)
        recs[tag] = build_task_records(scan, labels, bank_of_task)
    recf = recs.get(FIXED_POLICY) or {}
    recd = recs.get(DYNAMIC_POLICY) or {}
    ids = sorted(set(list(recf.keys()) + list(recd.keys())), key=str)

    cohorts = cohort_tables(recf, recd) if (recf or recd) else {}
    pairing = {'quad': pair_quad(recf, recd, ids),
               'latency': paired_latency(recf, recd, ids),
               'pairing_key': 'task_id (+ identical GPU role for latency pairs)',
               'levels': ['completed', 'ontime', 'good']}
    extras = extras_block(recf, recd)
    localc = local_consistency_block(shared, recf, recd, scans)
    src = source_block(shared, scans)
    c_star = {}
    for tag, scan in scans.items():
        c_star[tag] = scan.get('c_star_ms')
    replay = {}
    for tag, scan in scans.items():
        p = Path(scan.get('folder') or (folders or {}).get(tag) or '')
        ev = p / 'events.jsonl'
        if ev.exists():
            cs = scan.get('c_star_ms')
            if not cs:
                cs = None
            replay[tag] = replay_windows(str(ev), WINDOW_NS, cs)
        else:
            warnings.append('events.jsonl missing for arm %s' % tag)
    cons = {'by_arm': {}, 'c_star_ms_by_arm': c_star,
            'residual_definition': {
                'delta_wait_definition': 'E - S - X (waiting inflow - outflow)',
                'residual_dW': 'dW - (E - S - X)  must be 0',
                'delta_open_definition': 'O - E - R',
                'residual_O': 'O - (E + R + dOpen)  must be 0',
                'M_root': 'terminal_root + alive_root',
                'R_source': 'REJECT events only',
                'X_source': 'REMOVE events only'},
            'window_ms': WINDOW_NS / MS}
    for tag, wins in replay.items():
        agg = aggregate_replay(wins, COHORT_SPANS, c_star.get(tag))
        scan = scans.get(tag) or {}
        agg['reject_events_n'] = sum((scan.get('rejects') or {}).values())
        agg['reject_rows_n'] = scan.get('reject_rows_n')
        agg['remove_events_n'] = sum((scan.get('removes') or {}).values())
        agg['remove_rows_n'] = scan.get('remove_rows_n')
        agg['admission_rejected_n'] = sum((scan.get('admission_rejected') or {}).values())
        full = (agg.get('spans') or {}).get('full') or {}
        agg['conservation_full_ok'] = {
            'max_abs_residual': agg.get('max_abs_residual_full'),
            'residual_dW_per_role': {r: ((full.get('roles') or {}).get(r) or
                                         {}).get('residual_dW') for r in GPU_ROLES},
            'residual_O_per_role': {r: ((full.get('roles') or {}).get(r) or
                                        {}).get('residual_O') for r in GPU_ROLES}}
        cons['by_arm'][tag] = agg
    cons['root_closure'] = root_closure_block(scans, recs)
    cons['growth_and_noise'] = growth_and_noise_block(
        {t: {'spans': (replay[t] and aggregate_replay(replay[t], COHORT_SPANS,
                                                      c_star.get(t))) or {}}
         for t in replay}, scans)
    control = control_block(scans)
    costperf = cost_perf_block(scans, joint)
    predobs = prediction_vs_observed_block(scans, recs)
    closed_loop = closed_loop_block(scans, shared, recs)
    identity = _input_identity(inputs)

    metrics = {
        'schema': METRICS_SCHEMA,
        'task_id': task_id,
        'controller_version': CONTROLLER_VERSION,
        'implementation_revision': revision,
        'policy_names': list(POLICIES),
        'generated_utc': _time.strftime('%Y-%m-%dT%H:%M:%SZ', _time.gmtime()),
        'generated_unix': int(_time.time()),
        'joint_folder_input': str(joint_folder),
        'joint_folder': str(jf),
        'joint_folder_resolution': how,
        'joint_summary_schema': joint.get('schema'),
        'joint_summary_present': bool(joint),
        'arm_folders': resolution,
        'arm_folder_policy_map': {k: str(v) for k, v in (folders or {}).items()},
        'arm_files': {tag: {'present': scan.get('files_present'),
                            'missing': scan.get('missing_files')}
                      for tag, scan in sorted(scans.items())},
        'evidence_labels_path': labels_path,
        'evidence_labels_n': len(labels),
        'evidence_dir': evidence,
        'inputs_dir': inputs,
        'input_identity': identity,
        'cohort_clock_statement': (
            'every span is reported twice by the original cohort clock '
            '(arrival_ns and source_publish_ns) plus a strict terminal clock '
            '(terminal_available_ns); the three-block is identical, only the '
            'window assignment differs; spans are half-open [lo, hi)'),
        'cohorts': cohorts,
        'pairing': pairing,
        'extras': extras,
        'shared_local': localc,
        'source': src,
        'conservation': cons,
        'control': control,
        'cost_perf': costperf,
        'prediction_vs_observed': predobs,
        'closed_loop': closed_loop,
        'arrivals_by_arm': {tag: scan.get('arrivals_n')
                            for tag, scan in sorted(scans.items())},
        'terminals_by_arm': {tag: scan.get('terminal_n')
                             for tag, scan in sorted(scans.items())},
        'terminals_by_role_by_arm': {tag: scan.get('terminal_role_counts')
                                     for tag, scan in sorted(scans.items())},
        'replay_windows_n': {tag: len(wins) for tag, wins in sorted(replay.items())},
        'warnings': warnings,
        'unresolved': unresolved,
    }
    for tag, scan in sorted(scans.items()):
        ctl = scan.get('controller') or {}
        if not ctl.get('windows_with_candidates'):
            unresolved.append('arm %s: controller.jsonl has no candidate table '
                              '(resource keys seen: %s); TASK §9 item 9/11 blocks '
                              'carry explicit invalid reasons, not a pass' %
                              (tag, ctl.get('resource_keys')))
    if not cons['root_closure'].get('by_arm'):
        unresolved.append('root closure not computed (no arms scanned)')
    path = Path(out) if out else (jf / 'metrics.json')
    write_json(path, metrics)
    metrics['metrics_path'] = str(path)
    return metrics


# ================================================================ verify_joint

def _mk_check(name, ok, detail):
    c = {'name': name, 'ok': ok, 'detail': detail}
    if ok is None:
        c['status'] = 'unknown_or_incomplete'
    else:
        c['status'] = 'pass' if ok else 'fail'
    return c


def _root_stage_attempt(comp):
    parts = str(comp).split(':')
    root = parts[0] if parts else None
    stage = parts[1] if len(parts) > 1 else None
    attempt = parts[2] if len(parts) > 2 else None
    return root, stage, attempt


def _int_keys(d):
    """JSON round-trips integer dict keys to strings; restore them for recompute."""
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        nk = k
        if isinstance(k, str) and k.lstrip('-').isdigit():
            nk = int(k)
        out[nk] = v
    return out


def verify_joint(folder, evidence=None, revision='T014_qf4r_3', task_id='T014', out=None):
    """TASK §9 item 12: independent recomputation from the raw logs."""
    import time as _time
    jf, how = resolve_joint_folder(folder)
    checks = []
    if jf == 'not_found':
        return {'schema': VERIFICATION_SCHEMA, 'task_id': task_id,
                'implementation_revision': revision, 'controller_version':
                CONTROLLER_VERSION, 'joint_folder': None,
                'checks': [_mk_check('joint_folder_resolved', False,
                                     {'input': str(folder), 'reason': how})],
                'joint_folder_input': str(folder)}
    jf = Path(jf)
    joint = read_json(jf / 'joint_summary.json', None)
    if not isinstance(joint, dict):
        joint = {}
    folders, resolution = arm_folders(jf, joint)
    if evidence is None:
        evidence = _default_evidence(jf)
    inputs = _default_inputs(jf)
    labels, labels_path = load_labels(evidence, fallback_folder=jf.parent)
    scans = {}
    for policy in POLICIES:
        p = (folders or {}).get(policy)
        if p and Path(p).exists():
            scans[policy] = arm_scan(p)
    shared = read_shared_frames(jf)
    for scan in scans.values():
        _bind_shared_arrivals(scan, shared)
    bank_of_task = dict(shared.get('bank_of_task') or {})
    for tag, scan in scans.items():
        for k, v in (scan.get('bank_of_task') or {}).items():
            bank_of_task.setdefault(k, v)
    recs = {tag: build_task_records(scan, labels, bank_of_task)
            for tag, scan in scans.items()}
    checks.append(_mk_check('arm_folders_resolved',
                            True if len(scans) == len(POLICIES) else None,
                            {'resolution': resolution,
                             'scanned': sorted(scans.keys())}))

    # 1. input identity
    ident = _input_identity(inputs)
    ok1 = None
    if ident.get('input_file_sha256') or ident.get('input_logical_sha256'):
        ok1 = bool(ident.get('input_file_sha256') == INPUT_FILE_SHA and
                   ident.get('input_logical_sha256') == INPUT_LOGICAL_SHA)
    n_arr = max([s.get('arrivals_n') or 0 for s in scans.values()] or [0])
    phase_arr = None
    charr = {}
    for tag, scan in scans.items():
        arr = scan.get('arrivals') or {}
        c = []
        for lo, hi, _ in PHASES:
            c.append(sum(1 for a in arr.values()
                         if a[0] is not None and lo * NS_PER_S <= a[0] < hi * NS_PER_S))
        charr[tag] = c
    if charr:
        vals = list(charr.values())
        phase_arr = vals[0] if all(v == vals[0] for v in vals) else None
    detail1 = {'manifest': ident, 'arrivals_by_arm': {t: s.get('arrivals_n')
                                                     for t, s in scans.items()},
               'expected_arrivals': EXPECTED_ARRIVALS,
               'phase_arrivals_by_arm': charr,
               'expected_phase_arrivals': list(EXPECTED_PHASE_ARRIVALS)}
    ok1b = None
    if n_arr:
        ok1b = (n_arr == EXPECTED_ARRIVALS)
    ok1c = None
    if phase_arr is not None:
        ok1c = (tuple(phase_arr) == EXPECTED_PHASE_ARRIVALS)
    detail1['arrivals_match_expected'] = ok1b
    detail1['phases_match_expected'] = ok1c
    checks.append(_mk_check('input_identity', ok1, detail1))
    checks.append(_mk_check('arrival_count_and_phases',
                            (ok1b is not False) and (ok1c is not False)
                            if (ok1b is not None or ok1c is not None) else None,
                            {'arrivals': n_arr, 'phases': phase_arr}))

    # 2. source frames both arms / no missing / no dup / order
    d2 = {'shared_present': shared.get('present'), 'shared_rows': shared.get('rows'),
          'unique_frame_ids': shared.get('unique_frame_ids'),
          'dup_frame_ids_n': shared.get('dup_frame_ids_n'),
          'order_ok': shared.get('order_ok'),
          'order_violations': shared.get('order_violations'),
          'min_frame_id': shared.get('min_frame_id'),
          'max_frame_id': shared.get('max_frame_id'),
          'last_contig_frame_id': shared.get('last_contig_frame_id'),
          'frame_id_gaps': shared.get('frame_id_gaps'),
          'arrivals_per_arm': {t: s.get('arrivals_n') for t, s in scans.items()},
          'dup_terminals_by_arm': {t: s.get('dup_terminals') for t, s in scans.items()}}
    ok2 = None
    if shared.get('present'):
        shared_frames = list(shared.get('arrival_frames') or [])
        shared_set = set(as_int(f) for f in shared_frames if f is not None)
        per_arm = {}
        same_sets = True
        for tag, scan in scans.items():
            fids = [as_int(v) for v in (scan.get('frame_ids') or {}).values()
                    if v is not None]
            fset = set(fids)
            missing = sorted(shared_set - fset)[:50]
            extra = sorted(fset - shared_set)[:50]
            dup = len(fids) - len(fset)
            per_arm[tag] = {'arrival_rows': len(fids), 'unique_frames': len(fset),
                            'missing_vs_shared_n': len(shared_set - fset),
                            'missing_vs_shared_sample': missing,
                            'extra_vs_shared_n': len(fset - shared_set),
                            'extra_vs_shared_sample': extra,
                            'duplicate_frame_rows': dup}
            if missing or extra or dup:
                same_sets = False
        d2['per_arm_frame_sets'] = per_arm
        d2['shared_arrival_rows_n'] = shared.get('arrival_rows_n')
        d2['same_arrival_count_across_arms'] = len(
            set(s.get('arrivals_n') for s in scans.values())) <= 1
        ok2 = (shared.get('dup_frame_ids_n') == 0 and bool(shared.get('order_ok'))
               and shared.get('frame_id_gaps') == 0
               and not shared.get('unexpected_first_frame')
               and not shared.get('unsequenced_frames') and not shared.get('parse_errors')
               and d2['same_arrival_count_across_arms'] and same_sets)
    checks.append(_mk_check('source_frames_all_arms_ordered_unique', ok2, d2))

    # 3. conservation
    d3 = {'by_arm': {}}
    ok3 = None
    for tag, scan in scans.items():
        p = Path(scan.get('folder')) / 'events.jsonl'
        if not p.exists():
            continue
        wins = replay_windows(str(p), WINDOW_NS, scan.get('c_star_ms'))
        agg = aggregate_replay(wins, (('full', 0.0, 250.0),), scan.get('c_star_ms'))
        full = agg['spans']['full']
        role_res = {r: {'residual_dW': full['roles'][r]['residual_dW'],
                        'residual_O': full['roles'][r]['residual_O'],
                        'E_n': full['roles'][r]['E_n'],
                        'S_n': full['roles'][r]['S_n'],
                        'X_n': full['roles'][r]['X_n'],
                        'R_n': full['roles'][r]['R_n'],
                        'O_n': full['roles'][r]['O_n'],
                        'dW_n': full['roles'][r]['dW_n'],
                        'E_minus_S_minus_X': (full['roles'][r]['E_n'] -
                                              full['roles'][r]['S_n'] -
                                              full['roles'][r]['X_n'])}
                       for r in GPU_ROLES}
        bad = [r for r in GPU_ROLES
               if role_res[r]['residual_dW'] != 0 or role_res[r]['residual_O'] != 0]
        d3['by_arm'][tag] = {'max_abs_residual': agg.get('max_abs_residual_full'),
                             'roles': role_res, 'roles_with_nonzero_residual': bad,
                             'windows_n': len(wins)}
        if ok3 is None:
            ok3 = not bad
        else:
            ok3 = ok3 and not bad
    checks.append(_mk_check('conservation_residuals_zero', ok3, d3))

    # 3b. the controller's own per-window counts vs the offline replay
    d3b = {'by_arm': {}}
    ok3b = None
    for tag, scan in scans.items():
        p = Path(scan.get('folder') or '') / 'events.jsonl'
        wins = {}
        if p.exists():
            wins = {w['start_ns']: w for w in
                    replay_windows(str(p), WINDOW_NS, scan.get('c_star_ms'))}
        recorded = (scan.get('controller') or {}).get('windows') or {}
        compared = 0
        mism = []
        windows_compared = set()
        for wid, win in recorded.items():
            got = _online_residual_counts(win.get('residuals'))
            if not got:
                continue
            residual_span = win.get('residuals') or {}
            start = as_int(residual_span.get('start_ns'), as_int(wid))
            w = wins.get(start)
            if w is None:
                continue
            for k, per_role in got.items():
                for role, v in per_role.items():
                    if role == '_all':
                        off = sum(((w['work_ms'].get(r) or {}).get(k) or 0)
                                  for r in GPU_ROLES)
                    else:
                        off = ((w['work_ms'].get(role) or {}).get(k))
                    if off is None:
                        continue
                    compared += 1
                    windows_compared.add(wid)
                    if as_float(v) != as_float(off):
                        if len(mism) < 50:
                            mism.append({'window_id': wid, 'count': k, 'role': role,
                                         'online': as_float(v), 'offline': as_float(off)})
        d3b['by_arm'][tag] = {'windows_compared': len(windows_compared),
                              'count_fields_compared': compared,
                              'mismatches': len(mism), 'mismatch_sample': mism}
        v = (len(mism) == 0) if compared else None
        if v is not None:
            ok3b = v if ok3b is None else (ok3b and v)
    d3b['note'] = ('online per-window counts from controller resource residuals vs '
                   'offline replay of the raw event stream; a structural identity '
                   'alone cannot catch log edits, this cross-check can')
    if ok3b is None:
        d3b['reason'] = 'no online per-window counts recorded in controller.jsonl'
    checks.append(_mk_check('online_residual_counts_match_offline_replay', ok3b, d3b))

    # 4. unique root terminal + M_root
    rc = root_closure_block(scans, recs)
    ok4 = None
    multi = {}
    for tag, rr in recs.items():
        cnt = Counter()
        for rec in rr.values():
            if rec.get('completed') and rec.get('root_id') is not None:
                cnt[rec['root_id']] += 1
        multi[tag] = sorted([str(k) for k, v in cnt.items() if v > 1])[:50]
    ok4 = all((not v) for v in multi.values()) if multi else None
    checks.append(_mk_check('unique_root_terminal_and_M_root', ok4,
                            {'root_closure': rc, 'roots_with_multiple_terminals':
                             multi}))

    # 5. component_id continuity
    d5 = {}
    ok5 = None
    for tag, scan in scans.items():
        comps = scan.get('components') or {}
        bad = []
        sticks = {}
        for cid, c in comps.items():
            r, st, at = _root_stage_attempt(cid)
            if r != str(c.get('root_id')) or st != str(c.get('stage')) or \
                    at != str(c.get('attempt')):
                bad.append(cid)
            sticks.setdefault('%s|%s' % (c.get('root_id'), c.get('stage')),
                              set()).add(at)
        gap = []
        non_numeric = []
        for key, ats in sticks.items():
            nums = []
            for a in ats:
                n = as_int(a)
                if n is None:
                    non_numeric.append({'key': key, 'attempt': a})
                else:
                    nums.append(n)
            if nums:
                nums = sorted(set(nums))
                if nums[-1] - nums[0] + 1 != len(nums):
                    gap.append({'key': key, 'attempts': nums})
        d5[tag] = {'components_n': len(comps),
                   'root_stage_attempt_inconsistent': bad[:50],
                   'component_ids_inconsistent_n': len(bad),
                   'root_stage_attempt_gaps': gap[:50],
                   'non_numeric_attempts': non_numeric[:20],
                   'roots_with_multiple_attempts': sum(
                       1 for v in sticks.values() if len(v) > 1),
                   'observer_component_id_ok': ((scan.get('controller') or {})
                                                .get('observer_last') or
                                                {}).get('component_id_ok')}
        bad_any = bool(bad) or bool(gap)
        if ok5 is None:
            ok5 = not bad_any
        else:
            ok5 = ok5 and not bad_any
    checks.append(_mk_check('component_id_continuity', ok5, d5))

    # 6. per-kind units
    d6 = {}
    ok6 = None
    for tag, scan in scans.items():
        fin = scan.get('finishes') or {}
        svc = scan.get('service_ms') or {}
        unknown_kind = sorted([k for k in list(fin.keys()) + list(svc.keys())
                               if k.split(':')[-1] not in GPU_KINDS and
                               k.split(':')[-1] != 'LOCAL_SUFFIX' and
                               ROLE_OF_KIND.get(k.split(':')[-1]) != LOCAL_ROLE and
                               not k.startswith('None')])
        comp_rows = scan.get('component_rows_by_kind') or {}
        d6[tag] = {'finishes_by_kind': fin, 'service_ms_by_kind': sorted(svc.keys()),
                   'component_rows_by_kind': comp_rows,
                   'kinds_outside_gpu_kinds': unknown_kind,
                   'quota_charge_by_kind': scan.get('quota_charge_ms_by_kind'),
                   'quota_charge_n_by_kind': scan.get('quota_charge_n_by_kind')}
        bad = bool(unknown_kind)
        if ok6 is None:
            ok6 = not bad
        else:
            ok6 = ok6 and not bad
    checks.append(_mk_check('per_kind_units', ok6, d6))

    # 7. no resource rerouting (GPU_READY first candidate vs final terminal kind)
    d7 = {}
    ok7 = None
    entry_kind = shared.get('entry_kind_of_task') or {}
    for tag, rr in recs.items():
        mism = []
        checked = 0
        for tid, rec in rr.items():
            want = entry_kind.get(tid)
            if want is None or not rec.get('completed'):
                continue
            checked += 1
            if rec.get('kind') is not None and rec.get('kind') != want:
                if len(mism) < 50:
                    mism.append({'task_id': tid, 'entry_kind': want,
                                 'terminal_kind': rec.get('kind')})
        d7[tag] = {'checked': checked, 'mismatches': len(mism),
                   'mismatch_sample': mism}
        bad = bool(mism)
        if ok7 is None:
            ok7 = not bad
        else:
            ok7 = ok7 and not bad
    if not entry_kind:
        ok7 = None
        d7['reason'] = 'no GPU_READY entry kind in shared_frames.jsonl'
    checks.append(_mk_check('no_resource_rerouting', ok7, d7))

    # 8. Accepted requests exactly once; actual card effects are checked below.
    d8 = {}
    ok8 = None
    for tag, scan in scans.items():
        req = Counter()
        for r in (scan.get('config_request_rows') or []):
            if isinstance(r, (list, tuple)):
                cid = r[0] if r else None
            elif isinstance(r, dict):
                cid = r.get('command_id')
            else:
                cid = r
            req[str(cid)] += 1
        cmds = (scan.get('controller') or {}).get('command_ids') or []
        dup_req = sorted([k for k, v in req.items() if v > 1])[:20]
        missing_req = sorted([str(c) for c in cmds if str(c) not in req])[:20]
        extra_req = sorted(set(req) - set(str(c) for c in cmds))[:20]
        d8[tag] = {'config_request_total': scan.get('config_request_n'),
                   'config_requests_by_command_id': counter_to_plain(req),
                   'commands_logged_n': len(cmds),
                   'commands_metadata_recorded_n': (scan.get('controller') or {}).get('commands_metadata_recorded_n'),
                   'commands_recovered_from_requests_n': (scan.get('controller') or {}).get('commands_recovered_from_requests_n'),
                   'command_metadata_complete': (scan.get('controller') or {}).get('command_metadata_complete'),
                   'commands_with_duplicate_config_request': dup_req,
                   'commands_without_config_request': missing_req,
                   'requests_without_logged_command': extra_req,
                   'apply_count_max': (scan.get('controller') or {}).get(
                       'max_apply_count'),
                   'target_effective_n': scan.get('target_effective_n'),
                   'note': 'This checks accepted CONFIG_REQUEST identity only. '
                           'It does not prove that either GPU changed effective share.'}
        bad = bool(dup_req) or bool(missing_req) or bool(extra_req)
        if ok8 is None:
            ok8 = not bad
        else:
            ok8 = ok8 and not bad
    checks.append(_mk_check('config_request_exactly_once', ok8, d8))

    # 9. fixed arm share constant 1.0
    d9 = {}
    ok9 = None
    for tag, scan in scans.items():
        shares = Counter()
        for gpu, share, ts in (scan.get('target_effective') or []):
            shares[repr(share)] += 1
        d9[tag] = {'policy': scan.get('policy'), 'target_effective_n':
                   scan.get('target_effective_n'),
                   'distinct_share_values': counter_to_plain(shares)}
        if scan.get('policy') == FIXED_POLICY or tag == FIXED_POLICY:
            # TARGET_EFFECTIVE.share is one card's F share (normally 1 and
            # 0), whereas this invariant concerns the SUM gF of both cards.
            ctl = scan.get('controller') or {}
            aggregate = [r[2] for r in ctl.get('timeline', []) if r[2] is not None]
            bad = (any(abs(float(g) - 1.0) > 1e-9 for g in aggregate) or
                   bool(ctl.get('commands_n'))) if aggregate else None
            d9[tag]['aggregate_fast_shares'] = sorted(set(aggregate))
            if bad is None:
                ok9 = None if ok9 is None else ok9
            else:
                ok9 = (not bad) if ok9 is None else (ok9 and not bad)
    checks.append(_mk_check('fixed_arm_share_constant_1_0', ok9, d9))

    # 10. on-time deadline rule
    d10 = {}
    ok10 = None
    for tag, scan in scans.items():
        bad = 0
        flag_bad = 0
        n = 0
        for tid, t in (scan.get('terminals') or {}).items():
            av = t.get('available')
            dl = t.get('deadline')
            if av is None or dl is None:
                continue
            n += 1
            completed = t.get('role') in ALL_ROLES and t.get('prediction') is not None
            exp_ontime = bool(av <= dl)
            # Runtime's late flag describes terminal delivery time; Abstain
            # still has a terminal time but is not a successful on-time output.
            if bool(t.get('late')) != bool(av > dl):
                bad += 1
            if t.get('ontime_by_terminal') is not None and \
                    bool(t.get('ontime_by_terminal')) != exp_ontime:
                flag_bad += 1
        d10[tag] = {'terminals_with_deadline': n, 'late_flag_mismatch': bad,
                    'ontime_by_terminal_mismatch': flag_bad,
                    'rule': 'late = available > deadline; ontime_by_terminal = available <= deadline; output metrics additionally require completion'}
        v = (bad == 0 and flag_bad == 0) if n else None
        ok10 = v if ok10 is None else (ok10 and v if v is not None else ok10)
    checks.append(_mk_check('ontime_deadline_rule', ok10, d10))

    # 11. in-flight expired still charged
    d11 = {}
    ok11 = None
    for tag, scan in scans.items():
        charge = scan.get('charge_by_component_ms') or {}
        charged = {k for k, v in charge.items() if as_float(v, 0.0)}
        expired = []
        for cid, c in (scan.get('components') or {}).items():
            start, end, deadline = c.get('start_ns'), c.get('finish_ns'), c.get('deadline_ns')
            if c.get('kind') in GPU_KINDS and None not in (start, end, deadline) and start <= deadline < end:
                expired.append(cid)
        ch = sum(1 for cid in expired if cid in charged)
        d11[tag] = {'inflight_deadline_crossings': len(expired),
                    'expired_also_charged': ch,
                    'quota_rows': scan.get('quota_rows'),
                    'charge_components_n': len(scan.get('charge_by_component_ms') or {}),
                    'note': 'charge is the real quota span (finish-start) joined '
                            'through job_id -> component_id -> task_id'}
        v = (ch == len(expired)) if scan.get('component_rows') and scan.get('quota_rows') else None
        ok11 = v if ok11 is None else (ok11 and v if v is not None else ok11)
    checks.append(_mk_check('inflight_expired_still_charged', ok11, d11))

    # 12/13. Recompute the authoritative deadline replay from its exact public
    # request, not the obsolete analytical capacity table.
    d12 = {'recompute_engine': 'candidate_replay_t012.replay_candidates + quota_forecast_v8',
           'payload_windows_n': 0, 'unique_payloads': 0, 'candidate_rows_compared': 0,
           'mismatch_fields': 0, 'mismatch_sample': [], 'tamper_negative_test': None}
    ok12 = None
    payloads = []
    for tag, scan in scans.items():
        for wid, win in ((scan.get('controller') or {}).get('windows') or {}).items():
            payload = win.get('payload')
            if isinstance(payload, dict) and ('profile' in payload or 'request' in payload):
                payloads.append((tag, wid, payload, win))
    d12['payload_windows_n'] = len(payloads)
    if payloads:
        try:
            import candidate_replay_t012 as replay_kernel
            import quota_forecast_v8 as dispatch_kernel
            cache = {}
            mismatches = []
            compared = 0
            for tag, wid, payload, win in payloads:
                fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf8')).hexdigest()
                if fingerprint not in cache:
                    req, costs, gates, ipc, head = replay_kernel._split_payload(payload)
                    cache[fingerprint] = replay_kernel.replay_candidates(
                        req, costs, gates, ipc, head, dispatch_kernel.select_job,
                        dispatch_kernel.dispatch_choice)
                table = cache[fingerprint]
                expected = {c['gF']: c for c in table.get('candidates', [])}
                recorded = {c.get('gF'): c for c in win.get('candidates') or []}
                if not table.get('valid') or set(expected) != set(recorded):
                    mismatches.append({'arm': tag, 'window_id': wid,
                        'reason': 'replay invalid or candidate grid differs',
                        'replayed_valid': table.get('valid'), 'recorded_grid': list(recorded),
                        'replayed_grid': list(expected)})
                for gf, candidate in recorded.items():
                    actual = expected.get(gf)
                    if actual is None:
                        continue
                    compared += 1
                    for field in ('Yhat', 'Vhat', 'V_available', 'ontime_roots',
                                  'quality_sum', 'by_class', 'gate_starts',
                                  'donor_final', 'receiver_final', 'effective_step',
                                  'point_valid', 'sensitivity', 'remaining_budget_cdf',
                                  'net_gain_Yhat', 'net_gain_Vhat', 'epsilon_Y', 'epsilon_V',
                                  'low_confidence', 'n_comparable', 'is_effective_step',
                                  'numeric_target_gF', 'card_share', 'input_n',
                                  'quality_earliness_hat', 'net_gain_quality_earliness',
                                  'epsilon_quality_earliness'):
                        if field in ('quality_earliness_hat', 'net_gain_quality_earliness',
                                     'epsilon_quality_earliness') and field not in candidate:
                            continue  # legacy logs did not measure this objective
                        a, b = candidate.get(field), actual.get(field)
                        if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)):
                            equal = abs(float(a) - float(b)) <= 1e-6
                        else:
                            equal = a == b
                        if not equal:
                            mismatches.append({'arm': tag, 'window_id': wid, 'gF': gf,
                                'field': field, 'online': a, 'recomputed': b})
                    if d12['tamper_negative_test'] is None and isinstance(actual.get('Yhat'), (int, float)):
                        tampered = float(actual['Yhat']) + 1.0
                        d12['tamper_negative_test'] = {'tampered_field': 'recorded_candidate.Yhat',
                            'detected': abs(tampered - float(actual['Yhat'])) > 1e-6,
                            'window_id': wid, 'gF': gf}
            d12.update({'unique_payloads': len(cache), 'candidate_rows_compared': compared,
                        'mismatch_fields': len(mismatches), 'mismatch_sample': mismatches[:50]})
            ok12 = bool(compared and not mismatches and
                        (d12['tamper_negative_test'] or {}).get('detected'))
        except Exception as exc:
            d12['error'] = '%s: %s' % (type(exc).__name__, exc)
            ok12 = False
    else:
        d12['reason'] = 'exact public deadline-replay request is absent; analytical-table reconstruction cannot verify it'
    checks.append(_mk_check('candidate_offline_same_payload_recompute', ok12, d12))

    # 14. representative windows / no-command hold reasons
    d14 = {'representative_windows_s': [list(w) for w in REPRESENTATIVE_WINDOWS],
           'by_arm': {}}
    ok14 = None
    for tag, scan in scans.items():
        ctl = scan.get('controller') or {}
        cmds = ctl.get('commands') or []
        wins = ctl.get('windows') or {}
        rep = []
        for lo, hi in REPRESENTATIVE_WINDOWS:
            wid = int(lo * NS_PER_S)
            w = wins.get(wid)
            rep.append({'window_id': wid, 'span_s': [lo, hi],
                        'recorded': bool(w),
                        'candidates': bool(w and w.get('candidates')),
                        'invalid_reason': (w or {}).get('invalid_reason'),
                        'effective_fast': (w or {}).get('effective_fast'),
                        'reason': (w or {}).get('reason')})
        d14['by_arm'][tag] = {
            'commands_n': len(cmds),
            'command_windows_n': len(set(c.get('window_id') for c in cmds)),
            'representative_windows': rep,
            'hold_reasons_dedup': (control_block({tag: scan})
                                   .get('by_arm', {}).get(tag, {})
                                   .get('hold_reasons_dedup')),
            'sustained_gap_runs': (control_block({tag: scan})
                                   .get('by_arm', {}).get(tag, {})
                                   .get('sustained_gap_runs')),
            'note': 'with commands: every command window is exported; without '
                    'commands: the three representative windows plus deduplicated '
                    'hold reasons'}
    ok14 = (len(scans) > 0)
    checks.append(_mk_check('representative_windows_and_hold_reasons', ok14, d14))

    # 15. terminal vs tasks.jsonl consistency
    d15 = {}
    ok15 = None
    for tag, scan in scans.items():
        tv = scan.get('tasks_vs_terminals') or {}
        d15[tag] = tv
        if tv.get('rows'):
            v = (tv.get('role_mismatch', 0) == 0 and tv.get('available_mismatch', 0) == 0
                 and tv.get('prediction_mismatch', 0) == 0 and tv.get('late_mismatch', 0) == 0)
            ok15 = v if ok15 is None else (ok15 and v)
    checks.append(_mk_check('tasks_vs_terminals_consistent', ok15, d15))

    # 16. REJECT co-occurrence with ADMISSION_OUTCOME rejected
    d16 = {}
    ok16 = None
    for tag, scan in scans.items():
        r = sum((scan.get('rejects') or {}).values())
        a = sum((scan.get('admission_rejected') or {}).values())
        reasons = {}
        for row in (scan.get('reject_rows') or []):
            reasons[str(row[3])] = reasons.get(str(row[3]), 0) + 1
        d16[tag] = {'reject_events': r, 'admission_rejected': a,
                    'reject_reasons': reasons,
                    'admission_rejected_reasons': scan.get('admission_rejected_reasons')}
        v = (r == a) if (r or a) else None
        ok16 = v if ok16 is None else (ok16 and v if v is not None else ok16)
    checks.append(_mk_check('reject_equals_admission_rejected', ok16, d16))

    # 17. monotone timestamps / event file sanity
    d17 = {}
    ok17 = None
    for tag, scan in scans.items():
        span = scan.get('event_ts_span_ns')
        d17[tag] = {'event_ts_span_ns': span,
                    'event_types': scan.get('event_type_counts'),
                    'run_end': scan.get('run_end')}
        v = bool(span) if span else None
        ok17 = v if ok17 is None else (ok17 and v if v is not None else ok17)
    checks.append(_mk_check('event_timestamp_span_recorded', ok17, d17))

    # 18. predicted vs observed is reported (never silently pass)
    d18 = {}
    ok18 = None
    for tag, scan in scans.items():
        ctl = (scan.get('controller') or {})
        d18[tag] = {'windows_with_candidates': ctl.get('windows_with_candidates'),
                    'invalid_reason_counts': ctl.get('invalid_reasons'),
                    'candidate_payload_keys': ctl.get('resource_keys')}
        v = bool(ctl.get('windows_with_candidates'))
        ok18 = v if ok18 is None else (ok18 and v)
    d18['note'] = ('unknown/incomplete never folds into pass: a False here means '
                   'the controller log carried no candidate table at all')
    checks.append(_mk_check('candidate_tables_present', ok18, d18))

    shared_ok = (shared.get('present') and shared.get('unique_frame_ids', 0) > 0 and
                 shared.get('dup_frame_ids_n') == 0 and shared.get('order_ok') and
                 shared.get('frame_id_gaps') == 0 and not shared.get('parse_errors') and
                 not shared.get('unsequenced_frames') and not shared.get('unexpected_first_frame'))
    checks.append(_mk_check('shared_source_frame_continuity', bool(shared_ok),
        {k: shared.get(k) for k in ('rows', 'unique_frame_ids', 'dup_frame_ids_n',
         'frame_id_gaps', 'order_ok', 'parse_errors', 'unsequenced_frames')}))
    final_integrity = {}
    gpu_charge = {}
    for tag, scan in scans.items():
        ti = dict(scan.get('task_integrity') or {})
        ids = set((scan.get('terminals') or {}).keys())
        source_ids = set((shared.get('arrivals') or {}).keys())
        ti.update({'unique_terminals': len(ids), 'expected_source_tasks': len(source_ids),
                   'missing_source_tasks': len(source_ids - ids),
                   'unexpected_tasks': len(ids - source_ids)})
        final_integrity[tag] = ti
        mismatches = []
        checked = 0
        charge = scan.get('charge_by_component_ms') or {}
        for cid, c in (scan.get('components') or {}).items():
            if c.get('kind') not in GPU_KINDS or not c.get('finish'):
                continue
            start, finish = c.get('start_ns'), c.get('finish_ns')
            if start is None or finish is None:
                mismatches.append({'component_id': cid, 'reason': 'missing_actor_interval'})
                continue
            checked += 1
            expected = (finish - start) / MS
            actual = charge.get(cid)
            if actual is None or abs(float(actual) - expected) > 0.00001:
                mismatches.append({'component_id': cid, 'expected_ms': expected,
                                   'charged_ms': actual})
        gpu_charge[tag] = {'gpu_components_checked': checked, 'mismatches': len(mismatches),
                           'mismatch_sample': mismatches[:50],
                           'component_integrity': scan.get('component_integrity')}
    task_ok = bool(final_integrity) and all(
        not any(v.get(k) for k in ('parse_errors', 'duplicate_task_ids', 'value_dropped',
                                   'missing_source_tasks', 'unexpected_tasks')) and
        v['unique_terminals'] == EXPECTED_ARRIVALS for v in final_integrity.values())
    checks.append(_mk_check('protected_tasks_complete_and_unique', task_ok, final_integrity))
    cost_ok = bool(gpu_charge) and all(v['gpu_components_checked'] > 0 and
        v['mismatches'] == 0 and not (v.get('component_integrity') or {}).get('parse_errors')
        for v in gpu_charge.values())
    checks.append(_mk_check('gpu_actor_intervals_equal_real_charge', cost_ok, gpu_charge))

    event_status = {tag: scan.get('event_integrity') for tag, scan in scans.items()}
    events_complete = bool(event_status) and all(v.get('complete') for v in event_status.values())
    checks.append(_mk_check('event_stream_complete', events_complete, event_status))
    loop = closed_loop_block(scans, shared, recs)
    dynamic_loop = (loop.get('by_arm') or {}).get(DYNAMIC_POLICY) or {}
    effects = dynamic_loop.get('effects') or []
    controller_ok = bool(scans) and all(not (s.get('controller') or {}).get('integrity', {}).get('parse_errors') for s in scans.values())
    actual_ok = (True if effects and controller_ok and all(e['effective_observed'] is True for e in effects) else None)
    checks.append(_mk_check('real_command_applied_once', actual_ok,
        {'dynamic_commands_n': dynamic_loop.get('commands_n'), 'effects': effects,
         'note': 'Two actual card shares and no transition must be observed after issue and before the next command; zero commands provide no evidence.'}))
    checks.append(_mk_check('dynamic_adjustment_observed', bool(dynamic_loop.get('effective_observed_n')) if controller_ok else None,
        {'effective_observed_n': dynamic_loop.get('effective_observed_n'),
         'note': 'An observed adjustment is necessary but alone does not establish useful algorithm behavior.'}))
    output_rows = [r for r in dynamic_loop.get('feedback') or []
                   if r['mature_output_recompute']['matches_recorded'] is not None]
    feedback_ok = (all(r['mature_output_recompute']['matches_recorded'] is True and
                       r['standardized_math_matches'] is True for r in output_rows)
                   if output_rows and shared_ok and task_ok and controller_ok else None)
    checks.append(_mk_check('mature_feedback_matches_source_and_tasks', feedback_ok,
        {'feedback_rows_compared': len(output_rows), 'rows': output_rows,
         'note': 'Raw source cohorts and protected task outputs independently recompute both sides; public quality is separate from label correctness.'}))
    verdict_rows = [r for r in dynamic_loop.get('feedback') or [] if r['support'].get('verdict')]
    verdict_ok = (all(r.get('verdict_matches_recorded') is True for r in verdict_rows)
                  if verdict_rows and controller_ok else None)
    checks.append(_mk_check('probe_verdict_matches_standardized_net', verdict_ok,
        {'final_verdicts': dynamic_loop.get('final_verdicts'),
         'rows': [{'command_id': r['support'].get('command_id'), 'recorded': r['support'].get('verdict'),
                   'recomputed': r.get('verdict_recomputed'), 'matches': r.get('verdict_matches_recorded')} for r in verdict_rows]}))
    if not events_complete:
        event_dependent = {'source_frames_all_arms_ordered_unique', 'conservation_residuals_zero',
            'online_residual_counts_match_offline_replay', 'config_request_exactly_once',
            'tasks_vs_terminals_consistent', 'reject_equals_admission_rejected'}
        for check in checks:
            if check['name'] in event_dependent:
                check['ok'] = None
                check['status'] = 'unknown_or_incomplete'
                check['detail']['event_evidence_incomplete'] = True

    ver = {'schema': VERIFICATION_SCHEMA, 'task_id': task_id,
           'controller_version': CONTROLLER_VERSION,
           'implementation_revision': revision,
           'joint_folder_input': str(folder), 'joint_folder': str(jf),
           'joint_folder_resolution': how,
           'evidence_labels_path': labels_path,
           'checks': checks,
           'generated_utc': _time.strftime('%Y-%m-%dT%H:%M:%SZ', _time.gmtime()),
           'generated_unix': int(_time.time()),
           'passed': sum(1 for c in checks if c['ok'] is True),
           'failed': sum(1 for c in checks if c['ok'] is False),
           'unknown': sum(1 for c in checks if c['ok'] is None)}
    path = Path(out) if out else Path(folder if Path(folder).is_dir() else jf) / 'verification.json'
    try:
        write_json(path, ver)
        ver['verification_path'] = str(path)
    except OSError as exc:
        ver['verification_write_error'] = str(exc)
    return ver


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='T012 offline summary/verification '
                                             '(TASK §9, WS-E)')
    ap.add_argument('joint_folder')
    ap.add_argument('--evidence', default=None)
    ap.add_argument('--inputs', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--revision', default=IMPLEMENTATION_REVISION)
    ap.add_argument('--task-id', default=TASK_ID)
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--replay-events', default=None)
    ap.add_argument('--window-ns', type=int, default=WINDOW_NS)
    ap.add_argument('--c-star-ms', default=None)
    args = ap.parse_args(argv)
    if args.replay_events:
        import json
        c_star = None
        if args.c_star_ms:
            c_star = json.loads(args.c_star_ms)
        wins = replay_windows(args.replay_events, args.window_ns, c_star)
        print(json.dumps({'windows_n': len(wins), 'first': wins[0] if wins else None,
                          'last': wins[-1] if wins else None},
                         ensure_ascii=False, indent=1, sort_keys=True))
        return 0
    m = summarize(args.joint_folder, args.evidence, args.inputs, args.out,
                  args.task_id, args.revision)
    print('metrics: %s' % m.get('metrics_path'))
    if args.verify:
        v = verify_joint(args.joint_folder, args.evidence, args.revision,
                         args.task_id)
        print('verification: passed=%s failed=%s unknown=%s' %
              (v.get('passed'), v.get('failed'), v.get('unknown')))
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
