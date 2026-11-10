"""T011 QF-4 queue-feedback statistics (TASK.md sections 3.1, 3.2, 3.3, 4, 5).

Pure CPU, standard library only: no torch, no numpy, no Runtime, no model, no
label/quality bank, no demand_ledger/calendar worker and no private labels are
imported or read.  This is the qf4 rewrite of the N10 qf3 tracker.  The
review of T010 (TASK.md section 3) demanded the following concrete corrections,
all implemented here:

  B1  fully bounded state: a ring of 32 500 ms windows (each with 10 50 ms
      sub-buckets), a per-card-kind cost deque capped at 200 samples and a
      bounded recent root ledger.  Nothing is rebuilt from the whole history:
      the old ``block_series`` rebuild-all-then-truncate-32 must never reappear.
      Every event advances the accounting in O(1); work is repriced from the
      stored kind COUNTS, never by rescanning tasks/buckets/decisions.
  B2  exact component/root bookkeeping with unique ids: OFFER/ENQUEUE/DISPATCH/
      wait-REMOVE/TERMINAL are idempotent; Prefix DONE only ends the component,
      never the root; a root has exactly one terminal; in-flight components are
      never subtracted from the waiting W a second time.
  B3  both conservation residuals are recorded for every sealed full execution
      window:  dW = E - S - X_all  and  O - S = dW + X_all + R_all  (O counts
      unique component OFFERs, retries never recount).  growth = dW/T and
      resource_loss = (R_resource + X_resource)/T are kept SEPARATE from
      CPU/Head/quality/original-deadline/unknown loss.  The resource subset is
      never presented as the full balance.
  B4  the source input window and the parent event window are different clocks.
      A source-root cohort window seals only after the ordered frame_id
      watermark crosses it; missing/out-of-order frames mark the window
      incomplete and it is never used as a complete requirement window.  Raw
      lambda and the source-budget-supported lambda are computed over the most
      recent SEALED <=2 s cohort window using only information public at the
      time; the budget judgement never uses a consumer-delayed remaining.
  B5  the frozen reference cost c*_k is fixed once per run from the two cards'
      mean_wall at freeze time; it only unifies work units, execution keeps the
      per-card prices.  Every per-card-kind price carries
      mean_wall_ms/gate_ms/sample_n/sample_revision/cost_version/
      measurement_source and is validated BEFORE controllers/actors are built.
  B6  noise windows are capped at 32; the warmup gate floor is the measured P95
      of this run's own card; a formal gate is never silently set to the mean.

Units: event times are integer ns, reference work is reference-ms, windows T
are seconds.
"""

import bisect
import math
from collections import Counter, deque

MS = 1e6
NS = 1e9
NS_PER_S = 1e9
#: T014 s3.2.1: at least this many FORMAL samples before the live mean/gate
#: replace the warmup bootstrap prior.
MIN_LIVE_COST_SAMPLES_ = 30

GPU_ROLES = ('Fast', 'Accurate')
ALL_ROLES = ('Fast', 'Accurate', 'Local')

_FAST_KINDS = ('FAST_FULL', 'FAST_PREFIX', 'FAST_SUFFIX')
_LOCAL_KINDS = ('LOCAL_PROBE', 'LOCAL_MID2', 'LOCAL_SUFFIX')
GPU_KINDS = _FAST_KINDS + ('ACCURATE',)

#: root entry classes (TASK.md section 5): fixed at the public source stage,
#: never reclassified by the eventual successful route
ROOT_DIRECT_F = 'directF'
ROOT_DIRECT_A = 'directA'
ROOT_PREFIX = 'Prefix-origin'
ROOT_CLASSES = (ROOT_DIRECT_F, ROOT_DIRECT_A, ROOT_PREFIX)

#: loss reason classes (TASK.md section 5 final list), recorded per X and R
LOSS_REASONS = ('intrinsic_budget', 'receipt_delay', 'resource',
                'head_delay', 'done_delay', 'quality', 'unknown',
                'source_budget', 'receipt_stall', 'runtime_delay',
                'prediction_unknown')
RESOURCE_REASONS = ('resource',)

#: TASK.md section 3.3 asks for separate reason classes: a source-side budget
#: shortfall, a receipt/handling stall and a pure prediction gap must never be
#: folded into 'resource'.  The two legacy names are kept as accepted aliases so
#: an un-migrated producer cannot silently degrade to 'unknown', but they are
#: normalised to the canonical names before they reach a window.
ALIAS_REASONS = {'intrinsic_budget': 'source_budget',
                 'receipt_delay': 'receipt_stall'}

#: public first-visible quality intent roles
INTENT_ROLES = ('Local', 'Fast', 'Accurate')

#: T014 s3.2.2 deadline-opportunity shape: the arrival-time remaining budget is
#: binned in fixed-width classes; a pre/post pair whose histograms differ by
#: more than this TV, or where any class is thinner than the minimum on either
#: side, is NOT a usable window (unknown opportunity shape is never a pass).
DEADLINE_HIST_WIDTH_MS = 5.0
DEADLINE_SHAPE_TV_MAX = 0.1
DEADLINE_SHAPE_MIN_N = 30

WARMUP_SOURCE = 'warmup_same_startup'

#: parent-side timing names that must always exist in a perf snapshot (3.2)
PERF_NAMES = ('source_slice', 'loop_gap', 'controller_sync', 'worker')

#: bounded parent-side timing sample count per name (a TRUE full max is kept
#: separately, so the bounded list is never passed off as the full-run max)
PERF_SAMPLE_CAP = 4096


# ------------------------------------------------------------------ conversions


def ns_to_ms(value):
    return float(value) / MS


def role_of_kind(kind):
    if kind == 'ACCURATE':
        return 'Accurate'
    if kind in _LOCAL_KINDS:
        return 'Local'
    if kind in _FAST_KINDS:
        return 'Fast'
    return None


def parse_probability(value):
    """Strict public probability: a real float in [0, 1]; bool is rejected."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if math.isfinite(v) and 0.0 <= v <= 1.0:
            return v
    return None


def parse_probabilities(probabilities):
    if not isinstance(probabilities, dict):
        return None
    out = {}
    for name, value in probabilities.items():
        p = parse_probability(value)
        if p is None:
            return None
        out[str(name)] = p
    return out


# ------------------------------------------------------------------ small maths


def p95_linear(values):
    """Interpolated P95, identical to the old executor's np.percentile(.., 95)."""
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    if n == 0:
        return None
    if n == 1:
        return ordered[0]
    pos = 0.95 * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def median(values):
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def absolute_p95(values):
    """Empirical interpolated P95 of absolute values (the noise gate of TASK 7)."""
    return p95_linear([abs(float(v)) for v in values])


def tv_distance(counter_a, counter_b):
    """Total variation distance between two count distributions (section 8)."""
    keys = set(counter_a) | set(counter_b)
    total_a = sum(float(counter_a.get(k, 0)) for k in keys)
    total_b = sum(float(counter_b.get(k, 0)) for k in keys)
    if total_a <= 0 or total_b <= 0:
        return None
    return 0.5 * sum(abs(float(counter_a.get(k, 0)) / total_a
                         - float(counter_b.get(k, 0)) / total_b) for k in keys)


# ------------------------------------------------------------------ perf book


class PerfBook(object):
    """Bounded parent-side timing samples with a TRUE all-time maximum (3.2).

    Every name keeps at most ``cap`` samples (P50/P95/max come from that
    retained window) plus an independent ``full_max_ns`` that every single note
    updates.  Truncating the sample list therefore can never be presented as
    the full-run maximum.
    """

    def __init__(self, cap=PERF_SAMPLE_CAP):
        self.cap = max(int(cap), 1)
        self.samples = {}
        self.counts = {}
        self.full_max_ns = {}
        for name in PERF_NAMES:
            self._register(name)

    def _register(self, name):
        name = str(name)
        if name not in self.samples:
            self.samples[name] = deque(maxlen=self.cap)
            self.counts[name] = 0
            self.full_max_ns[name] = 0
        return name

    def note(self, name, ns):
        """Record one wall duration (ns); negative input is clamped to zero."""
        name = self._register(name)
        value = max(int(ns), 0)
        self.samples[name].append(float(value))
        self.counts[name] += 1
        if value > self.full_max_ns[name]:
            self.full_max_ns[name] = value
        return value

    def note_handler(self, name, ns):
        return self.note(name, ns)

    def note_source_slice(self, ns):
        return self.note('source_slice', ns)

    def note_loop_gap(self, ns):
        return self.note('loop_gap', ns)

    def note_controller_sync(self, ns):
        return self.note('controller_sync', ns)

    def note_worker(self, ns):
        return self.note('worker', ns)

    def stats(self, name):
        name = self._register(name)
        values = list(self.samples[name])
        n = int(self.counts[name])
        if not values:
            return {'n': n, 'p50_ms': 0.0, 'p95_ms': 0.0, 'max_ms': 0.0,
                    'full_max_ms': self.full_max_ns[name] / MS}
        p50 = median(values)
        p95 = p95_linear(values)
        return {'n': n,
                'p50_ms': (0.0 if p50 is None else float(p50)) / MS,
                'p95_ms': (0.0 if p95 is None else float(p95)) / MS,
                'max_ms': max(values) / MS,
                'full_max_ms': self.full_max_ns[name] / MS}

    def perf_snapshot(self):
        out = {}
        for name in sorted(self.samples):
            out[name] = self.stats(name)
        for name in PERF_NAMES:
            if name not in out:
                out[name] = self.stats(name)
        return out


# ------------------------------------------------------------------ cost book


class KindCostRecord(object):
    """One card x kind: bounded samples, mean/gate/floor/delta with versions."""

    __slots__ = ('card', 'kind', 'samples', 'gaps', 'floor_ns', 'gate_ns',
                 'mean_ns', 'delta_ns', 'sample_revision', 'cost_version',
                 'last_update_ns', 'measurement_source', 'n', 'history',
                 'live_n', 'warmup_n', 'warmup_mean_ns', 'warmup_gaps_ns')

    def __init__(self, card, kind, cap=200):
        self.card = int(card)
        self.kind = str(kind)
        self.samples = deque(maxlen=int(cap))   # LIVE service wall ns, <=200
        self.gaps = deque(maxlen=int(cap))      # LIVE dispatch -> actor start ns
        self.floor_ns = None
        self.gate_ns = None
        self.mean_ns = None
        self.delta_ns = None
        self.sample_revision = 0
        self.cost_version = 0
        self.last_update_ns = None
        self.measurement_source = None
        self.n = 0                              # live samples only
        self.history = deque(maxlen=32)         # (gate_ns, mean_ns) update trail
        # T014 s3.2.1: warmup and formal samples are counted independently.  A
        # warmup-seeded mean/gate is a bootstrap prior; live mean/gate update
        # only from the last <=200 FORMAL samples once live_n >= 30.
        self.live_n = 0
        self.warmup_n = 0
        self.warmup_mean_ns = None
        self.warmup_gaps_ns = None

    def observe(self, service_ns, gap_ns, now_ns):
        """O(1) FORMAL (live) sample intake; bootstrap before 30 live samples."""
        self.live_n += 1
        self.n = self.live_n
        self.samples.append(float(service_ns))
        if gap_ns is not None:
            self.gaps.append(float(gap_ns))
        if self.live_n >= MIN_LIVE_COST_SAMPLES_:
            self.mean_ns = sum(self.samples) / len(self.samples)
            if self.gaps:
                self.delta_ns = sum(self.gaps) / len(self.gaps)
        elif self.warmup_mean_ns is not None:
            self.mean_ns = self.warmup_mean_ns
        else:
            self.mean_ns = sum(self.samples) / len(self.samples)
            if self.gaps:
                self.delta_ns = sum(self.gaps) / len(self.gaps)
        self.sample_revision += 1
        return self

    def seed_warmup(self, samples_ns, gaps_ns, now_ns):
        """Warmup intake seeds the bootstrap mean + floor + gate; the warmup
        samples NEVER enter the live sample deque (T014 s3.2.1)."""
        if not samples_ns:
            return False
        self.warmup_n += len(samples_ns)
        self.warmup_mean_ns = sum(float(v) for v in samples_ns) / len(samples_ns)
        if gaps_ns:
            self.warmup_gaps_ns = sum(float(v) for v in gaps_ns) / len(gaps_ns)
        self.mean_ns = self.warmup_mean_ns
        floor = p95_linear(samples_ns)
        if floor is None:
            return False
        self.floor_ns = float(floor)
        self.gate_ns = float(floor)
        if self.warmup_gaps_ns is not None:
            self.delta_ns = self.warmup_gaps_ns
        self.measurement_source = WARMUP_SOURCE
        self.sample_revision += 1
        self.cost_version += 1
        self.last_update_ns = int(now_ns)
        self.history.append((float(self.gate_ns), float(self.mean_ns)))
        return True

    def try_update_gate(self, now_ns, min_samples=30, min_interval_ns=250_000_000):
        """Gate updates from the last <=200 REAL completion samples only, with
        >=30 samples and >=250 ms since the previous update; the mean is left
        independent and the floor keeps the warmup lower bound."""
        if self.floor_ns is None:
            return False
        if len(self.samples) < int(min_samples):
            return False
        if self.last_update_ns is not None and \
                (int(now_ns) - int(self.last_update_ns)) < int(min_interval_ns):
            return False
        gate = max(float(self.floor_ns), p95_linear(self.samples))
        delta = sum(self.gaps) / len(self.gaps) if self.gaps else self.delta_ns
        changed = (self.gate_ns is None or abs(gate - self.gate_ns) > 1e-9 or
                   (delta is not None and self.delta_ns is not None and
                    abs(delta - self.delta_ns) > 1e-9))
        self.gate_ns = gate
        if delta is not None:
            self.delta_ns = delta
        self.measurement_source = 'live_real_completions'
        self.last_update_ns = int(now_ns)
        self.history.append((float(self.gate_ns), float(self.mean_ns)))
        if changed:
            self.cost_version += 1
        return changed

    def record(self):
        """The canonical per-card-per-kind record (TASK.md section 3.3)."""
        return {
            'card': self.card, 'kind': self.kind,
            'mean_wall_ms': None if self.mean_ns is None else self.mean_ns / MS,
            'gate_ms': None if self.gate_ns is None else self.gate_ns / MS,
            'floor_ms': None if self.floor_ns is None else self.floor_ns / MS,
            'delta_ms': None if self.delta_ns is None else self.delta_ns / MS,
            'sample_n': int(self.live_n),
            'live_sample_n': int(self.live_n),
            'warmup_sample_n': int(self.warmup_n),
            'is_bootstrap': bool(self.live_n < MIN_LIVE_COST_SAMPLES_),
            'sample_revision': int(self.sample_revision),
            'cost_version': int(self.cost_version),
            'measurement_source': self.measurement_source,
            'window_samples': int(len(self.samples)),
        }

    def price_history_ms(self):
        """Bounded trail of (gate_ms, mean_ms) update points for the N_safe
        margin P95 (TASK.md section 6)."""
        return [(g / MS, m / MS) for g, m in self.history]


from resource_axis import CARD_COUNT, CARD_IDS


class CostBook(object):
    """The ONE bounded cost table for the two physical cards of one arm (qf4).

    Both arms use the same class; the fixed arm simply never receives commands.
    """

    KINDS = GPU_KINDS

    def __init__(self, cap=200):
        self.records = {(g, k): KindCostRecord(g, k, cap)
                        for g in CARD_IDS for k in self.KINDS}
        self._global_cost_version = 0
        self._sample_revision = 0
        self._last_fingerprint = self._table_fingerprint()

    def observe(self, card, kind, service_ns, gap_ns, now_ns):
        rec = self.records.get((int(card), str(kind)))
        if rec is None:
            return False
        rec.observe(service_ns, gap_ns, now_ns)
        self._sample_revision += 1
        return True

    def seed_warmup(self, card, kind, samples_ns, gaps_ns, now_ns):
        rec = self.records.get((int(card), str(kind)))
        if rec is None:
            return False
        ok = rec.seed_warmup(samples_ns, gaps_ns, now_ns)
        if ok:
            self._sample_revision += 1
        return ok

    def update_gates(self, now_ns, min_samples=30, min_interval_ns=250_000_000):
        changed = []
        for key in sorted(self.records):
            if self.records[key].try_update_gate(now_ns, min_samples, min_interval_ns):
                changed.append(key)
        return changed

    def record_obj(self, card, kind):
        return self.records.get((int(card), str(kind)))

    def mean_ns(self, card, kind):
        rec = self.record_obj(card, kind)
        return None if rec is None or rec.mean_ns is None else float(rec.mean_ns)

    def gate_ns(self, card, kind):
        rec = self.record_obj(card, kind)
        return None if rec is None or rec.gate_ns is None else float(rec.gate_ns)

    def floor_ns(self, card, kind):
        rec = self.record_obj(card, kind)
        return None if rec is None or rec.floor_ns is None else float(rec.floor_ns)

    def delta_ns(self, card, kind):
        rec = self.record_obj(card, kind)
        return None if rec is None or rec.delta_ns is None else float(rec.delta_ns)

    def _table_fingerprint(self):
        """Exact actual-value snapshot of every card x kind record."""
        out = []
        for key in sorted(self.records):
            rec = self.records[key]
            out.append((int(rec.card), str(rec.kind),
                        rec.mean_ns, rec.gate_ns, rec.floor_ns, rec.delta_ns,
                        int(rec.n), int(rec.sample_revision),
                        rec.measurement_source))
        return tuple(out)

    def global_cost_version(self):
        """Monotone version of the WHOLE price table (TASK.md section 3.4).

        Increments by exactly one whenever any card/kind mean/gate/floor/delta/
        sample actually differs from the previous read of this method.  It is
        NOT the max of the per-record versions, so a change to a kind whose own
        record version is below the table max is still visible.
        """
        fingerprint = self._table_fingerprint()
        if fingerprint != self._last_fingerprint:
            self._last_fingerprint = fingerprint
            self._global_cost_version += 1
        return int(self._global_cost_version)

    def sample_revision(self):
        """Total new samples taken in; independent of global_cost_version()."""
        direct = sum(int(rec.sample_revision) for rec in self.records.values())
        return int(max(self._sample_revision, direct))

    def cost_version(self):
        """Legacy alias kept for old callers: now the global table version."""
        return self.global_cost_version()

    def reference_c_star_ms(self):
        """c*_k = arithmetic mean of the two cards' CURRENT mean_wall (B5)."""
        out = {}
        for kind in self.KINDS:
            means = [self.mean_ns(g, kind) for g in CARD_IDS]
            if any(m is None for m in means):
                continue
            out[kind] = sum(means) / CARD_COUNT / MS
        return out

    def records_dict(self):
        return {g: {k: self.records[(g, k)].record() for k in self.KINDS}
                for g in CARD_IDS}

    def rebase_clock(self, t0_ns):
        """Re-anchor every record onto the formal relative experiment clock.

        T012 R1: warmup used to stamp an absolute ``perf_counter_ns()`` while the
        run path fed a relative ``now()``; the huge negative delta then froze the
        250 ms price/gate update for the whole experiment.  Stamps are shifted by
        ``t0_ns`` and clamped at zero so a relative stamp is never made negative.
        """
        t0 = int(t0_ns)
        for rec in self.records.values():
            if rec.last_update_ns is not None:
                rec.last_update_ns = max(0, int(rec.last_update_ns) - t0)
        return t0


def validate_cost_records(cost_records):
    """Uniform factory validation of dict per card x four kinds (section 3.3).

    Rejects missing/None/non-positive prices BEFORE the controller/actors are
    created, and refuses a list handed to a dict-only interface.
    """
    errors = []
    if not isinstance(cost_records, dict):
        return ['cost_records must be a dict {card: {kind: record}}, got %s'
                % type(cost_records).__name__]
    for kind in GPU_KINDS:
        for card in CARD_IDS:
            row = cost_records.get(card, cost_records.get(str(card)))
            if not isinstance(row, dict):
                errors.append('cost[%s][%s] missing' % (card, kind))
                continue
            rec = row.get(kind)
            if not isinstance(rec, dict):
                errors.append('cost[%s][%s] record missing' % (card, kind))
                continue
            for key in ('mean_wall_ms', 'gate_ms', 'sample_n', 'sample_revision',
                        'cost_version', 'measurement_source'):
                if key not in rec:
                    errors.append('cost[%s][%s].%s missing' % (card, kind, key))
            for key in ('mean_wall_ms', 'gate_ms'):
                v = rec.get(key)
                if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) \
                        or not math.isfinite(v) or v <= 0:
                    errors.append('cost[%s][%s].%s must be positive finite, got %r'
                                  % (card, kind, key, v))
            for key in ('sample_n', 'sample_revision', 'cost_version'):
                v = rec.get(key)
                if v is None or not isinstance(v, int) or isinstance(v, bool) or v < 0:
                    errors.append('cost[%s][%s].%s must be a non-negative int'
                                  % (card, kind, key))
            ms = rec.get('measurement_source')
            if not isinstance(ms, str) or not ms:
                errors.append('cost[%s][%s].measurement_source invalid' % (card, kind))
    return errors


# ------------------------------------------------------------------ observer


def _reason_map():
    return {r: {} for r in LOSS_REASONS}


def reason_component_counts(reason_map):
    """Collapse a ``role -> reason -> kind -> int`` loss map to reason totals.

    The shape is frozen by TASK.md section 3.3.  A non-dict reason bucket or a
    non-int leaf is an explicit assertion failure here, so no downstream code
    can ever take ``.values()`` of an integer counter by mistake.
    """
    out = {}
    assert isinstance(reason_map, dict), \
        'loss map must be a reason->kind->int dict, got %s' % type(reason_map).__name__
    for reason, kinds in reason_map.items():
        assert reason in LOSS_REASONS, 'unknown loss reason bucket %r' % (reason,)
        assert isinstance(kinds, dict), \
            'loss bucket %r must be a kind->int dict, got %s' % (
                reason, type(kinds).__name__)
        total = 0
        for kind, n in kinds.items():
            assert isinstance(n, int) and not isinstance(n, bool), \
                'loss count %r[%r] must be an int, got %r' % (reason, kind, n)
            total += int(n)
        out[reason] = total
    return out


def reason_subset_count(reason_totals, reasons):
    """Sum the reason totals whose reason is in ``reasons``."""
    return int(sum(int(reason_totals.get(r, 0)) for r in reasons))


def normalize_reason(reason):
    """Map a legacy loss reason onto its canonical TASK.md section 3.3 class."""
    return ALIAS_REASONS.get(reason, reason)


def reason_work_ms(reason_kinds, reasons, c_ref_of_kind):
    """Reference-ms carried by a ``reason -> kind -> int`` map.

    TASK.md section 3.3 defines the resource-loss *rate* as the lost components
    repriced by the frozen ``c*`` and divided by the window seconds.  A raw
    component count over T is a different unit, so every ``*_ms_s`` field must
    be produced through this helper.  ``reasons=None`` means "every reason".
    """
    total = 0.0
    for reason, kinds in reason_kinds.items():
        if reasons is not None and reason not in reasons:
            continue
        for kind, n in kinds.items():
            total += float(n) * float(c_ref_of_kind(kind))
    return total


def probe_flag_or(flags):
    """OR-fold a window's probe flags (T014 CPU_expected.probe_flag_or).

    A probe/transition flag is a property of the whole 500 ms WINDOW: once any
    tick inside the window was taken under a probe or a card transition, the
    window is flagged and a later clean tick must never clear it (that would
    let a window that really contained a probe masquerade as a clean one).
    """
    for f in (flags or ()):
        if bool(f):
            return True
    return False


class ResourceRootLoss(int):
    """Integer root-loss count that also carries honest resource-loss evidence."""

    def __new__(cls, val, info):
        obj = super(ResourceRootLoss, cls).__new__(cls, int(val))
        obj._info = dict(info)
        return obj

    def __getitem__(self, key):
        return self._info[key]

    def get(self, key, default=None):
        return self._info.get(key, default)

    def __contains__(self, key):
        return key in self._info

    def __iter__(self):
        return iter(self._info)

    def keys(self):
        return self._info.keys()

    def values(self):
        return self._info.values()

    def items(self):
        return self._info.items()


class QF4Observer(object):
    """Bounded incremental accounting over the raw public event stream (qf4)."""

    def __init__(self, config):
        cfg = dict(config)
        self.window_ns = int(round(float(cfg.get('window_ms', 500.0)) * MS))
        self.sub_ns = int(round(float(cfg.get('sub_window_ms', 50.0)) * MS))
        self.subs_per_window = max(int(self.window_ns // self.sub_ns), 1)
        self.probe_ring_s = float(cfg.get('probe_ring_s', 2.0))
        self.noise_ring_max = int(cfg.get('noise_ring_max', 16))
        derived_ring = max(int(self.probe_ring_s * NS / self.window_ns) + 2, 8)
        if 'ring_windows' in cfg:
            self.ring_windows = int(cfg['ring_windows'])
        else:
            self.ring_windows = derived_ring
        self.cohort_lookback = int(cfg.get('cohort_lookback', 8))
        self.deadline_ns = int(cfg.get('deadline_ns', 50 * MS))
        self.stall_lag_p95_ns = int(round(float(cfg.get('stall_lag_p95_ms', 12.5)) * MS))
        self.max_lag_samples = int(cfg.get('max_lag_samples_per_window', 4096))
        self.root_estimation_ns = int(round(float(cfg.get('root_estimation_ms', 2000.0)) * MS))
        self._static_c_ref_ms = cfg.get('c_ref_ms')
        self.c_star_ms = dict(self._static_c_ref_ms) if self._static_c_ref_ms else None
        self.cost_book = cfg.get('cost_book')
        self.perf_sample_cap = int(cfg.get('perf_sample_cap', PERF_SAMPLE_CAP))
        self.access_counts = {}
        self.access_max = {}
        self.access_calls = 0
        self.access_items = 0
        self.access_max_items = 0
        self.source_closed = False
        self.source_final_ns = None
        self.source_close_reason = None
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self):
        self.wait_counts = {r: {} for r in GPU_ROLES}    # role -> kind -> n
        self.resource_loss_counts = {r: Counter() for r in GPU_ROLES}
        self.open_offers = {r: {} for r in GPU_ROLES}    # offered, unresolved
        self.roots = {}
        self.components = {}
        self.cohorts = {}
        # T014 s3.1.3: insertion-ordered cohort keys + monotone cursors, so the
        # per-loop sealing/maturity checks never sort or rescan the ring.
        self.cohort_order = []
        self._cohort_keys = self.cohort_order
        self._seal_idx = 0
        self._mature_idx = 0
        self.sealed_windows = deque(maxlen=self.ring_windows)
        self.noise_records = deque(maxlen=self.noise_ring_max)
        self.cur = None
        self.window_index = -1
        self.last_t = None
        self.access_counts = {}
        self.access_max = {}
        self.access_calls = 0
        self.access_items = 0
        self.access_max_items = 0
        self.source_closed = False
        self.source_final_ns = None
        self.source_close_reason = None
        self.counters = {
            'offer': 0, 'enqueue': 0, 'dispatch': 0, 'remove': 0, 'reject': 0,
            'terminal': 0, 'dup_offer': 0, 'dup_enqueue': 0, 'dup_dispatch': 0,
            'dup_remove': 0, 'dup_terminal': 0, 'unknown_kind': 0,
            'orphan_event': 0, 'incomplete_cohorts': 0, 'stall_windows': 0,
            'unsupported_roots': 0, 'unknown_support_roots': 0,
            'root_publish': 0, 'bad_component_id': 0,
        }
        self.perf = PerfBook(self.perf_sample_cap)

    # -- access accounting (T014 s3.1.3) -----------------------------------
    def _note_access(self, name, n):
        self.access_calls += 1
        n = int(n)
        self.access_items += n
        if n > self.access_max_items:
            self.access_max_items = n
        self.access_counts[name] = self.access_counts.get(name, 0) + n
        if n > self.access_max.get(name, 0):
            self.access_max[name] = n

    def take_access_stats(self):
        out = {
            'counts': dict(self.access_counts),
            'max': dict(self.access_max),
            'calls': int(self.access_calls),
            'items': int(self.access_items),
            'max_items': int(self.access_max_items),
        }
        self.access_counts = {}
        self.access_calls = 0
        self.access_items = 0
        self.access_max_items = 0
        return out

    def access_snapshot(self):
        return {
            'counts': dict(self.access_counts),
            'max': dict(self.access_max),
            'calls': int(self.access_calls),
            'items': int(self.access_items),
            'max_items': int(self.access_max_items),
        }

    # -- units (B1: counts, repriced via frozen c*) -------------------------
    def freeze_c_star(self, c_star_ms=None):
        """Freeze the per-run common reference cost once (TASK.md section 5)."""
        if c_star_ms is not None:
            self.c_star_ms = {str(k): float(v) for k, v in c_star_ms.items()}
        elif self.cost_book is not None:
            self.c_star_ms = self.cost_book.reference_c_star_ms()
        if not self.c_star_ms or any(k not in self.c_star_ms for k in GPU_KINDS):
            raise ValueError('c* freeze incomplete: %r' % (self.c_star_ms,))
        return dict(self.c_star_ms)

    def c_ref_ms(self, kind):
        if not self.c_star_ms or kind not in self.c_star_ms:
            raise KeyError('no frozen c* for kind %r' % (kind,))
        return float(self.c_star_ms[kind])

    def work_of_counts(self, counts):
        return sum(float(n) * self.c_ref_ms(k) for k, n in counts.items())

    def waiting_work_ms(self, role):
        return self.work_of_counts(self.wait_counts[role])

    def commitment_ms(self, role=None):
        """Reference-ms still physically occupied by in-flight components.

        TASK.md section 3.3 keeps this OUT of W (it is not recoverable waiting
        work) and lets a candidate only subtract it from its own capacity.
        """
        total = 0.0
        for comp in self.components.values():
            if comp.get('state') != 'inflight':
                continue
            if role is not None and comp.get('role') != role:
                continue
            total += self.c_ref_ms(comp['kind'])
        return float(total)

    def rebase_clock(self, t0_ns):
        """Re-anchor every cost record onto the formal relative clock (TASK 3.1).

        The warmup path used to seed an absolute ``perf_counter_ns()`` stamp
        while the run path fed a relative ``now()``; the resulting huge negative
        delta froze the 250 ms price/gate update for the whole experiment.
        """
        if self.cost_book is None:
            return 0
        fn = getattr(self.cost_book, 'rebase_clock', None)
        if fn is not None:
            return int(fn(t0_ns))
        return int(t0_ns)

    def live_counts(self):
        """Per-role per-kind waiting component counts (bounded, O(kinds))."""
        return {r: dict(self.wait_counts[r]) for r in GPU_ROLES}

    def waiting_count(self, role):
        return sum(self.wait_counts[role].values())

    def cost_version(self):
        if self.cost_book is not None:
            return int(self.cost_book.cost_version())
        return 0

    def global_cost_version(self):
        """TASK.md 3.4 global table version (any real mean/gate change bumps it).

        ``cost_version`` is kept as the legacy alias so an older consumer cannot
        silently read a per-kind max instead of the global version.
        """
        if self.cost_book is not None:
            fn = getattr(self.cost_book, 'global_cost_version', None)
            if fn is not None:
                return int(fn())
            return int(self.cost_book.cost_version())
        return 0

    def sample_revision(self):
        if self.cost_book is not None:
            fn = getattr(self.cost_book, 'sample_revision', None)
            if fn is not None:
                return int(fn())
        return 0

    # -- ordered time advance ----------------------------------------------
    def _new_window(self, index, start_ns):
        idx = int(index)
        if not self.cohort_order or idx > self.cohort_order[-1]:
            self.cohort_order.append(idx)
        elif idx not in self.cohort_order:
            bisect.insort(self.cohort_order, idx)
        return {
            'index': int(index), 'start_ns': int(start_ns),
            'end_ns': int(start_ns) + self.window_ns, 'T_s': self.window_ns / NS_PER_S,
            'O': {r: {} for r in GPU_ROLES},
            'E': {r: {} for r in GPU_ROLES},
            'S': {r: {} for r in GPU_ROLES},
            'X': {r: _reason_map() for r in GPU_ROLES},
            'R': {r: _reason_map() for r in GPU_ROLES},
            'W_start': {r: dict(self.wait_counts[r]) for r in GPU_ROLES},
            'W_end': None,
            # TASK 3.3: the open-offer ledger needs the same start/end bounds as
            # W so that Open_end - Open_start reconciles against O - E - R.
            'Open_start': {r: dict(self.open_offers[r]) for r in GPU_ROLES},
            'Open_end': None,
            'effective_epoch': getattr(self, '_latest_epoch', 0),
            'lost_roots': {r: set() for r in GPU_ROLES},
            'root_arrivals': {c: 0 for c in ROOT_CLASSES},
            'intent': {r: 0 for r in INTENT_ROLES},
            'intent_unknown': 0,
            'reveal': {'EARLY': {'Fast': 0, 'Accurate': 0, 'Abstain': 0},
                       'prefix_origin_total': 0},
            'terminal': {'completed_F': 0, 'completed_A': 0, 'abstain': 0,
                         'ontime': 0, 'late': 0, 'known_ontime': 0,
                         'sum_p': 0.0, 'sum_p2': 0.0, 'late_discarded': 0},
            'lag': {'receipt_ns': [], 'done_ns': []},
            'sub': [{'start_ns': start_ns + i * self.sub_ns,
                     'end_ns': start_ns + (i + 1) * self.sub_ns,
                     'E': {r: {} for r in GPU_ROLES},
                     'S': {r: {} for r in GPU_ROLES},
                     'X': {r: 0 for r in GPU_ROLES},
                     'R': {r: 0 for r in GPU_ROLES}}
                    for i in range(self.subs_per_window)],
            'flags': {'transition': False, 'probe': False, 'stall': False,
                      'epoch': getattr(self, '_latest_epoch', 0),
                      'gF': getattr(self, '_latest_gF', 1.0), 'source_gap': False},
            'budget_hist': {},
            'cost_version_at_seal': None,
            'residuals': None,
            'sealed': False,
        }

    def advance_to(self, now_ns):
        now = int(now_ns)
        if self.cur is None:
            start = (now // self.window_ns) * self.window_ns
            self.window_index = start // self.window_ns
            self.cur = self._new_window(self.window_index, start)
            self.last_t = now
            self._note_access('advance_to', 1)
            return
        if now <= self.last_t:
            self._note_access('advance_to', 1)
            return
        inspected = 0
        while self.cur['end_ns'] <= now:
            inspected += 1
            self._seal_window()
        self.last_t = now
        self._note_access('advance_to', max(inspected, 1))

    def _current_sub(self):
        idx = int((self.last_t - self.cur['start_ns']) // self.sub_ns)
        idx = min(max(idx, 0), self.subs_per_window - 1)
        return self.cur['sub'][idx]

    def _seal_window(self):
        w = self.cur
        w['W_end'] = {r: dict(self.wait_counts[r]) for r in GPU_ROLES}
        w['Open_end'] = {r: dict(self.open_offers[r]) for r in GPU_ROLES}
        res = {}
        for role in GPU_ROLES:
            E = w['E'][role]
            S = w['S'][role]
            Xall = {}
            for reason, kinds in w['X'][role].items():
                for k, n in kinds.items():
                    Xall[k] = Xall.get(k, 0) + n
            Rall = {}
            for reason, kinds in w['R'][role].items():
                for k, n in kinds.items():
                    Rall[k] = Rall.get(k, 0) + n
            # shape-asserted per-reason totals (TASK 3.3: never .values() an int)
            X_by_reason = reason_component_counts(w['X'][role])
            R_by_reason = reason_component_counts(w['R'][role])
            O = w['O'][role]
            w_start = w['W_start'][role]
            w_end = w['W_end'][role]
            o_start = w['Open_start'][role]
            o_end = w['Open_end'][role]
            kinds = set(E) | set(S) | set(Xall) | set(Rall) | set(O) | set(w_start) | set(w_end)
            kinds |= set(o_start) | set(o_end)
            dW_counts = 0
            res_dW = 0.0
            res_OS = 0.0
            res_dOpen = 0.0
            E_ms = S_ms = X_ms = R_ms = O_ms = 0.0
            dW_ms = 0.0
            for k in kinds:
                c = self.c_ref_ms(k)
                e, s, x, rr, o = E.get(k, 0), S.get(k, 0), Xall.get(k, 0), Rall.get(k, 0), O.get(k, 0)
                dw = (w_end.get(k, 0) - w_start.get(k, 0)) - (e - s - x)
                dW_counts += abs(dw)
                res_dW += abs(dw) * c
                osr = o - (e + rr)
                res_OS += abs(osr) * c
                d_open = (o_end.get(k, 0) - o_start.get(k, 0)) - osr
                res_dOpen += abs(d_open) * c
                E_ms += e * c
                S_ms += s * c
                X_ms += x * c
                R_ms += rr * c
                O_ms += o * c
                dW_ms += (w_end.get(k, 0) - w_start.get(k, 0)) * c
            T = float(w['T_s'])
            res_X = reason_subset_count(X_by_reason, RESOURCE_REASONS)
            res_R = reason_subset_count(R_by_reason, RESOURCE_REASONS)
            all_X = sum(X_by_reason.values())
            all_R = sum(R_by_reason.values())
            # TASK 3.3: loss is reference-ms of the lost components divided by the
            # window seconds -- a raw component count over T is a different unit
            # and must never be labelled *_ms_s.
            X_kinds = w['X'][role]
            R_kinds = w['R'][role]
            res_work_ms = (reason_work_ms(X_kinds, RESOURCE_REASONS, self.c_ref_ms) +
                           reason_work_ms(R_kinds, RESOURCE_REASONS, self.c_ref_ms))
            all_work_ms = (reason_work_ms(X_kinds, None, self.c_ref_ms) +
                           reason_work_ms(R_kinds, None, self.c_ref_ms))
            x_res_work = reason_work_ms(X_kinds, RESOURCE_REASONS, self.c_ref_ms)
            r_res_work = reason_work_ms(R_kinds, RESOURCE_REASONS, self.c_ref_ms)
            lost_roots = w.get('lost_roots') or {}
            role_lost_roots = lost_roots.get(role) or set()
            res[role] = {
                'E_ms_s': E_ms / T, 'S_ms_s': S_ms / T, 'X_ms_s': X_ms / T,
                'R_ms_s': R_ms / T, 'O_ms_s': O_ms / T, 'dW_ms_s': dW_ms / T,
                'growth_ms_s': dW_ms / T,
                'resource_loss_ms_s': res_work_ms / T,
                'other_loss_ms_s': (all_work_ms - res_work_ms) / T,
                'X_resource_ms_s': x_res_work / T, 'R_resource_ms_s': r_res_work / T,
                'residual_dW_ms': res_dW, 'residual_O_minus_S_ms': res_OS,
                'residual_dOpen_ms': res_dOpen,
                'residual_dW_counts': int(dW_counts),
                'Open_start_n': int(sum(int(v) for v in o_start.values())),
                'Open_end_n': int(sum(int(v) for v in o_end.values())),
                # T012 WS-A: root dedup + component counts + window work (4/3.3)
                'resource_lost_roots': int(len(role_lost_roots)),
                'resource_lost_components': int(res_X + res_R),
                'other_lost_components': int(all_X + all_R - res_X - res_R),
                'E_n': int(sum(int(v) for v in E.values())),
                'S_n': int(sum(int(v) for v in S.values())),
                'X_n': int(all_X),
                'R_n': int(all_R),
                'O_n': int(sum(int(v) for v in O.values())),
                'W_start_ms': float(self.work_of_counts(w_start)),
                'W_end_ms': float(self.work_of_counts(w_end)),
            }
        w['residuals'] = res
        for key in ('receipt_ns', 'done_ns'):
            vals = w['lag'][key]
            w['lag'][key + '_p95'] = p95_linear(vals) if vals else None
        lag_max = max(w['lag'].get('receipt_ns_p95') or 0,
                      w['lag'].get('done_ns_p95') or 0)
        # A performance warning is not evidence that all GPU opportunities
        # are unavailable. Per-root remaining budgets stay in the scheduler
        # and replay; pause the whole window only on deadline-scale delay.
        w['flags']['delay_warning'] = lag_max > self.stall_lag_p95_ns
        if lag_max >= self.deadline_ns:
            w['flags']['stall'] = True
            self.counters['stall_windows'] += 1
        w['cost_version_at_seal'] = self.cost_version()
        w['sealed'] = True
        self.sealed_windows.append(w)
        self._cleanup(w['end_ns'])
        self.window_index += 1
        self.cur = self._new_window(self.window_index, self.window_index * self.window_ns)

    def _cleanup(self, now):
        """Bounded memory: drop cohorts older than the ring and their roots.

        T014 s3.1.3: prune from the FRONT of the insertion-ordered key deck and
        pull the two cursors back with it, so nothing ever scans the whole ring.
        """
        keep_from = now - int(self.ring_windows * self.window_ns * 1.5)
        while self.cohort_order:
            bucket = self.cohort_order[0]
            if (bucket + 1) * self.window_ns >= keep_from:
                break
            self.cohort_order.pop(0)
            coh = self.cohorts.pop(bucket, None)
            if coh is not None:
                for root in coh['roots']:
                    self.roots.pop(root['root_id'], None)
            if self._seal_idx > 0:
                self._seal_idx -= 1
            if self._mature_idx > 0:
                self._mature_idx -= 1
        if hasattr(self, 'noise_records'):
            while len(self.noise_records) > self.noise_ring_max:
                self.noise_records.popleft()
        if len(self.roots) > 100000:
            raise RuntimeError('root ledger exceeded the bounded safety cap')

    def mark_flags(self, transition=False, probe=False, gF=None, epoch=None,
                   log_backpressure=False):
        if self.cur is None:
            return
        self.cur['flags']['transition'] = probe_flag_or(
            (self.cur['flags']['transition'], transition))
        self.cur['flags']['probe'] = probe_flag_or(
            (self.cur['flags']['probe'], probe))
        self.cur['flags']['log_backpressure'] = probe_flag_or(
            (self.cur['flags'].get('log_backpressure'), log_backpressure))
        if gF is not None:
            self.cur['flags']['gF'] = float(gF)
            self._latest_gF = float(gF)
        if epoch is not None:
            old_epoch = self.cur['flags'].get('epoch')
            if old_epoch is not None and old_epoch != int(epoch):
                self.cur['flags']['transition'] = True
            self.cur['flags']['epoch'] = int(epoch)
            self._latest_epoch = int(epoch)
            # TASK 4: the sealed window itself must carry the effective epoch so
            # that the noise series can require one shared epoch.
            self.cur['effective_epoch'] = int(epoch)

    # -- source-root cohorts (B4) ------------------------------------------
    def root_publish(self, root_id, entry_class, source_publish_ns, deadline_ns,
                     frame_id=None, quality_role=None, probabilities=None,
                     gpu_candidates=None, first_kind=None, stage_rem_ns=None,
                     original_arrival_ns=None):
        """Register one public source root in its SOURCE input window (B4)."""
        self._note_access('root_publish', 1)
        if entry_class not in ROOT_CLASSES:
            raise ValueError('bad root entry class %r' % (entry_class,))
        if root_id in self.roots:
            return False
        root = {'root_id': root_id, 'entry_class': entry_class,
                'source_publish_ns': int(source_publish_ns),
                'deadline_ns': int(deadline_ns), 'frame_id': frame_id,
                'original_arrival_ns': int(original_arrival_ns if original_arrival_ns is not None
                                           else deadline_ns-self.deadline_ns),
                'quality_role': quality_role,
                'probabilities': parse_probabilities(probabilities),
                'gpu_candidates': list(gpu_candidates or []),
                'first_kind': first_kind, 'stage_rem_ns': stage_rem_ns,
                'budget_supported': None,
                'revealed_early': None, 'terminal': None, 'open_comps': 0,
                # T014 s3.1.4: a root can be mature-counted (deadline passed with
                # no timely output) before its physical terminal arrives; the
                # later DONE then only closes the physical account.
                'mature_failure': False}
        self.roots[root_id] = root
        bucket = int(source_publish_ns) // self.window_ns
        coh = self.cohorts.get(bucket)
        if coh is None:
            coh = {'start_ns': bucket * self.window_ns,
                   'end_ns': (bucket + 1) * self.window_ns,
                   'roots': [], 'wm_sealed': False, 'results_sealed': False,
                   'matured': False, 'matured_ns': None,
                   'incomplete': False, 'watermark_ns': None,
                   'effective_epoch': int(self.cur.get('effective_epoch') or 0),
                   'n': {c: 0 for c in ROOT_CLASSES},
                   'n_supported': {c: 0 for c in ROOT_CLASSES},
                   'n_unknown': {c: 0 for c in ROOT_CLASSES},
                   'reveal': {'Fast': 0, 'Accurate': 0, 'Abstain': 0,
                              'supported_Fast': 0, 'supported_Accurate': 0,
                              'sum_p_Fast': 0.0, 'sum_p_Accurate': 0.0,
                              'revealed_prefix_total': 0},
                   'prefix_total': 0, 'unrevealed_after_deadline': 0,
                   'terminal': {'completed': 0, 'ontime': 0, 'late': 0,
                                'abstain': 0, 'sum_p_ontime': 0.0},
                   'terminal_by_class': {c: {'n': 0, 'completed': 0, 'ontime': 0,
                                             'late': 0, 'abstain': 0,
                                             'sum_p_ontime': 0.0,
                                             'sum_quality_earliness': 0.0,
                                             'known_earliness': 0,
                                             'known_ontime': 0}
                                         for c in ROOT_CLASSES}}
            self.cohorts[bucket] = coh
            if not self.cohort_order or bucket > self.cohort_order[-1]:
                self.cohort_order.append(bucket)
            elif bucket not in self.cohort_order:
                bisect.insort(self.cohort_order, bucket)
        coh['roots'].append(root)
        coh['n'][entry_class] += 1
        # One source-root sample, in its SOURCE cohort, before GPU execution.
        # The old arrival-budget histogram is constantly 50 ms and cannot
        # detect changes in the Local time already consumed by this request.
        entry_role = 'Accurate' if entry_class == ROOT_DIRECT_A else 'Fast'
        remaining = max(0, int(deadline_ns) - int(source_publish_ns))
        hist = coh.setdefault('gpu_entry_budget_hist', {}).setdefault(
            entry_role, {'n': 0, 'bins': {}})
        bucket_key = str(remaining // int(DEADLINE_HIST_WIDTH_MS * MS))
        hist['n'] += 1
        hist['bins'][bucket_key] = hist['bins'].get(bucket_key, 0) + 1
        # Source probabilities are immutable after root_publish.  Accumulate
        # their direct-route means once instead of rescanning 2s of roots.
        if entry_class in (ROOT_DIRECT_F, ROOT_DIRECT_A):
            role = 'Fast' if entry_class == ROOT_DIRECT_F else 'Accurate'
            probability = (root['probabilities'] or {}).get(role)
            if probability is not None:
                sums = coh.setdefault('direct_quality_sums', {})
                counts = coh.setdefault('direct_quality_counts', {})
                sums[entry_class] = sums.get(entry_class, 0.0) + float(probability)
                counts[entry_class] = counts.get(entry_class, 0) + 1
        if entry_class == ROOT_PREFIX:
            coh['prefix_total'] += 1
        self.cur['root_arrivals'][entry_class] += 1
        if quality_role in INTENT_ROLES:
            self.cur['intent'][quality_role] += 1
        elif quality_role is not None:
            self.cur['intent_unknown'] += 1
        self.counters['root_publish'] += 1
        return True

    def seal_source_cohorts(self, watermark_ns, frame_gap_seen=False):
        """Seal source cohort windows the ordered watermark crossed (B4).

        ``watermark_ns`` is the source-publish time up to which THIS consumer
        has received every frame in order.  A window with a frame gap or out-of
        order receipt is marked incomplete and never used as complete.
        T014 s3.1.3: the key ring is scanned from the LAST cursor only, never
        sorted and never rebuilt from the whole history each loop.
        """
        wm = int(watermark_ns)
        keys = self.cohort_order
        i = self._seal_idx
        n = len(keys)
        inspected = 0
        while i < n:
            b = keys[i]
            inspected += 1
            coh = self.cohorts.get(b)
            end_ns = coh['end_ns'] if coh is not None else (b + 1) * self.window_ns
            if wm <= end_ns:
                break
            if coh is not None and not coh['wm_sealed']:
                coh['wm_sealed'] = True
                coh['watermark_ns'] = wm
                coh['incomplete'] = bool(frame_gap_seen)
                if coh['incomplete']:
                    self.counters['incomplete_cohorts'] += 1
            i += 1
        self._seal_idx = i
        self._note_access('seal_source_cohorts', inspected)

    def close_source(self, final_ns, reason='source_end'):
        """Seal cohorts up to final_ns on source termination (T014 s3.1.3)."""
        final_bucket = int(final_ns) // self.window_ns
        newly_sealed = 0
        inspected = 0
        i = self._seal_idx
        n = len(self.cohort_order)
        while i < n:
            b = self.cohort_order[i]
            if b > final_bucket:
                break
            inspected += 1
            coh = self.cohorts.get(b)
            if coh is not None and not coh['wm_sealed']:
                coh['wm_sealed'] = True
                coh['watermark_ns'] = int(final_ns)
                newly_sealed += 1
            i += 1
        self._seal_idx = i
        self.source_closed = True
        self.source_final_ns = int(final_ns)
        self.source_close_reason = str(reason)
        self._note_access('close_source', inspected)
        return newly_sealed

    def note_results_progress(self, now_ns):
        """Mature source cohorts whose absolute deadline has passed (T014 s3.1.4).

        Maturity is decided by the COMPLETE watermark plus the absolute deadline
        (``end_ns + deadline_ns``), never by ``open_comps == 0``: a root whose
        deadline passed with no timely output is recorded as a failure
        observation NOW, while a still-in-flight physical component keeps
        charging and is closed only when its real DONE arrives.  Unrevealed
        Prefix roots are counted as U at this same instant.
        """
        now = int(now_ns)
        horizon = int(self.deadline_ns)
        # TASK 3.2 forbids a full scan per loop.  Cohorts are inserted in
        # increasing bucket order and pruned from the front, so once a cohort is
        # still younger than one deadline the whole tail is younger too.
        # T014 s3.1.3: resume from the last cursor; a due-but-unsealed cohort
        # stops the cursor so the next loop revisits it.
        keys = self.cohort_order
        i = min(self._mature_idx, len(keys))
        n = len(keys)
        inspected = 0
        while i < n:
            b = keys[i]
            inspected += 1
            coh = self.cohorts.get(b)
            end_ns = coh['end_ns'] if coh is not None else (b + 1) * self.window_ns
            if end_ns + horizon > now:
                break
            if coh is not None:
                if coh['matured']:
                    i += 1
                    continue
                if not coh['wm_sealed']:
                    break
                coh['matured'] = True
                coh['matured_ns'] = now
                unrevealed = 0
                for root in coh['roots']:
                    if root['entry_class'] == ROOT_PREFIX and root['revealed_early'] is None:
                        unrevealed += 1
                    if root['terminal'] is not None:
                        continue
                    # deadline passed with no timely output: count the failure
                    # observation; never fabricate a TERMINAL here.
                    root['mature_failure'] = True
                    ctb = coh['terminal_by_class'][root['entry_class']]
                    ctb['n'] += 1
                    ctb['late'] += 1
                coh['unrevealed_after_deadline'] = unrevealed
                coh['results_sealed'] = True
            i += 1
        self._mature_idx = i
        self._note_access('note_results_progress', inspected)

    def recent_sealed_cohorts(self):
        take = max(int(self.root_estimation_ns // self.window_ns), 1)
        candidates = self.cohort_order[-self.cohort_lookback:] if self.cohort_lookback > 0 else self.cohort_order
        inspected = len(candidates)
        sealed = []
        for b in candidates:
            coh = self.cohorts.get(b)
            if coh is not None and coh['wm_sealed'] and not coh['incomplete']:
                sealed.append(coh)
        self._note_access('recent_sealed_cohorts', inspected)
        return sealed[-take:]

    def recent_sealed_cohorts_all(self):
        """Every sealed cohort still inside the bounded ring (probe windows)."""
        candidates = self.cohort_order[-self.cohort_lookback:] if self.cohort_lookback > 0 else self.cohort_order
        inspected = len(candidates)
        out = []
        for b in candidates:
            coh = self.cohorts.get(b)
            if coh is not None and coh['wm_sealed']:
                out.append(coh)
        self._note_access('recent_sealed_cohorts_all', inspected)
        return out

    def last_sealed_complete_cohorts(self, before_ns, span_ns=None, epoch=None):
        """The most recent matured, single-config, complete 500 ms source-root
        cohorts ending before ``before_ns`` (the probe's pre window).

        TASK.md section 6 requires the pre window to come from the SAME actual
        effective configuration epoch as the probe; ``epoch`` selects it.
        """
        take = max(int((span_ns or self.root_estimation_ns) // self.window_ns), 1)
        end_limit = int(before_ns)
        target_epoch = int(epoch) if epoch is not None else None
        out = []
        inspected = 0
        for b in reversed(self.cohort_order):
            inspected += 1
            coh = self.cohorts.get(b)
            if coh is None:
                continue
            if coh['end_ns'] > end_limit:
                continue
            if not (coh['wm_sealed'] and coh['results_sealed'] and not coh['incomplete']):
                continue
            if target_epoch is not None and coh.get('effective_epoch') != target_epoch:
                continue
            out.append(coh)
            if len(out) >= take:
                break
        out.reverse()
        self._note_access('last_sealed_complete_cohorts', inspected)
        return out

    def sealed_windows_since(self, index=None, count=None, since_ns=None):
        out = []
        inspected = 0
        idx = int(index) if index is not None else None
        s_ns = int(since_ns) if since_ns is not None else None
        cnt = int(count) if count is not None else None
        for w in reversed(self.sealed_windows):
            inspected += 1
            if idx is not None and w['index'] <= idx:
                break
            if s_ns is not None and w['end_ns'] <= s_ns:
                break
            out.append(w)
            if cnt is not None and len(out) >= cnt:
                break
        out.reverse()
        self._note_access('sealed_windows_since', inspected)
        return out

    def reveal_early(self, root_id, role, supported=None, p_quality=None):
        """Public EARLY reveal for a Prefix-origin root; quality intent and
        budget support are recorded separately (TASK.md section 5)."""
        root = self.roots.get(root_id)
        if root is None or root['revealed_early'] is not None:
            return False
        if role not in ('Fast', 'Accurate', 'Abstain'):
            return False
        p = parse_probability(p_quality)
        root['revealed_early'] = {'role': role, 'supported': supported,
                                  'p_quality': p}
        if role in ('Fast', 'Accurate') and p is not None:
            probs = dict(root.get('probabilities') or {})
            prior_probability = probs.get(role)
            probs[role] = float(p)
            root['probabilities'] = probs
        bucket = root['source_publish_ns'] // self.window_ns
        coh = self.cohorts.get(bucket)
        if coh is not None:
            if (role in ('Fast', 'Accurate') and p is not None and
                    root['entry_class'] == (ROOT_DIRECT_F if role == 'Fast' else ROOT_DIRECT_A)):
                cls = root['entry_class']
                sums = coh.setdefault('direct_quality_sums', {})
                counts = coh.setdefault('direct_quality_counts', {})
                sums[cls] = sums.get(cls, 0.0) + float(p) - float(prior_probability or 0.0)
                if prior_probability is None:
                    counts[cls] = counts.get(cls, 0) + 1
            coh['reveal'][role] += 1
            coh['reveal']['revealed_prefix_total'] += 1
            if supported:
                coh['reveal']['supported_' + role] += 1
            if role in ('Fast', 'Accurate') and p is not None:
                coh['reveal']['sum_p_' + role] += float(p)
        self.cur['reveal']['EARLY'][role] += 1
        return True

    # -- component events (B2: idempotent, unique ids) ----------------------
    def component_id_ok(self, comp_id):
        """True for the stable ``'<root_id>:<stage>:<attempt>'`` shape (3.3).

        Every segment must be non-empty and the LAST segment must parse as an
        integer.  A non-string (legacy integer job_id) is rejected, so the
        observer key can never silently degrade to the executor's job_id.
        """
        if not isinstance(comp_id, str):
            return False
        parts = comp_id.split(':')
        if len(parts) < 3:
            return False
        for part in parts:
            if not part:
                return False
        try:
            int(parts[-1])
        except (TypeError, ValueError):
            return False
        return True

    def component_offer(self, comp_id, kind, root_id=None, now_ns=None):
        if not self.component_id_ok(comp_id):
            self.counters['bad_component_id'] += 1
            return False
        role = role_of_kind(kind)
        if role not in GPU_ROLES:
            self.counters['unknown_kind'] += 1
            return False
        if comp_id in self.components:
            self.counters['dup_offer'] += 1
            return False
        if now_ns is not None:
            self.advance_to(now_ns)
        self.components[comp_id] = {'kind': kind, 'role': role, 'state': 'offered',
                                    'root_id': root_id, 'counted_open': True}
        if root_id is not None and root_id in self.roots:
            self.roots[root_id]['open_comps'] += 1
        self.open_offers[role][kind] = self.open_offers[role].get(kind, 0) + 1
        self.cur['O'][role][kind] = self.cur['O'][role].get(kind, 0) + 1
        self.counters['offer'] += 1
        return True

    def component_enqueue(self, comp_id, deadline_ns, now_ns, affinity=None):
        comp = self.components.get(comp_id)
        if comp is None or comp['state'] != 'offered':
            self.counters['dup_enqueue'] += 1
            return False
        self.advance_to(now_ns)
        role = comp['role']
        comp['state'] = 'waiting'
        comp['deadline_ns'] = int(deadline_ns)
        comp['enqueue_ns'] = int(now_ns)
        comp['affinity'] = affinity
        self.wait_counts[role][comp['kind']] = self.wait_counts[role].get(comp['kind'], 0) + 1
        self._release_open(comp)
        self.cur['E'][role][comp['kind']] = self.cur['E'][role].get(comp['kind'], 0) + 1
        self._sub_bump('E', role, comp['kind'])
        self.counters['enqueue'] += 1
        return True

    def component_dispatch(self, comp_id, now_ns, card=None):
        comp = self.components.get(comp_id)
        if comp is None or comp['state'] != 'waiting':
            self.counters['dup_dispatch'] += 1
            return False
        self.advance_to(now_ns)
        role = comp['role']
        comp['state'] = 'inflight'
        comp['dispatch_ns'] = int(now_ns)
        comp['card'] = card
        self.wait_counts[role][comp['kind']] -= 1
        if self.wait_counts[role][comp['kind']] <= 0:
            del self.wait_counts[role][comp['kind']]
        self.cur['S'][role][comp['kind']] = self.cur['S'][role].get(comp['kind'], 0) + 1
        self._sub_bump('S', role, comp['kind'])
        self.counters['dispatch'] += 1
        return True

    def component_remove(self, comp_id, reason, now_ns):
        """Wait removal of an UNDISPATCHED component (X).  In-flight components
        can never be removed here: only 'waiting' state qualifies (B2/B3)."""
        comp = self.components.get(comp_id)
        if comp is None:
            self.counters['orphan_event'] += 1
            return False
        if comp['state'] not in ('waiting', 'offered'):
            self.counters['dup_remove'] += 1
            return False
        reason = normalize_reason(reason)
        if reason not in LOSS_REASONS:
            reason = 'unknown'
        self.advance_to(now_ns)
        role = comp['role']
        if comp['state'] == 'waiting':
            self.wait_counts[role][comp['kind']] -= 1
            if self.wait_counts[role][comp['kind']] <= 0:
                del self.wait_counts[role][comp['kind']]
            self.cur['X'][role][reason][comp['kind']] = \
                self.cur['X'][role][reason].get(comp['kind'], 0) + 1
            self._sub_bump('X', role, comp['kind'])
        else:
            self._release_open(comp)
            self.cur['R'][role][reason][comp['kind']] = \
                self.cur['R'][role][reason].get(comp['kind'], 0) + 1
        if reason in RESOURCE_REASONS:
            self.resource_loss_counts[role][comp['kind']] += 1
            if comp.get('root_id') is not None:
                self.cur['lost_roots'][role].add(comp['root_id'])
        comp['state'] = 'removed'
        self._root_close_component(comp)
        self.components.pop(comp_id, None)
        self.counters['remove'] += 1
        return True

    def component_reject(self, comp_id, reason, now_ns, kind=None, role=None):
        """Final admission rejection (R) of a never-waiting component."""
        comp = self.components.get(comp_id)
        if comp is None:
            if kind is None:
                return False
            role2 = role_of_kind(kind)
            if role2 not in GPU_ROLES:
                return False
            comp = {'kind': kind, 'role': role2, 'state': 'offered',
                    'root_id': None, 'counted_open': False}
            self.components[comp_id] = comp
        if comp['state'] not in ('offered',):
            self.counters['dup_remove'] += 1
            return False
        reason = normalize_reason(reason)
        if reason not in LOSS_REASONS:
            reason = 'unknown'
        self.advance_to(now_ns)
        self._release_open(comp)
        self.cur['R'][comp['role']][reason][comp['kind']] = \
            self.cur['R'][comp['role']][reason].get(comp['kind'], 0) + 1
        if reason in RESOURCE_REASONS:
            self.resource_loss_counts[comp['role']][comp['kind']] += 1
            if comp.get('root_id') is not None:
                self.cur['lost_roots'][comp['role']].add(comp['root_id'])
        comp['state'] = 'rejected'
        self._root_close_component(comp)
        self.components.pop(comp_id, None)
        self.counters['reject'] += 1
        return True

    def component_inline_service(self, comp_id, kind, root_id, now_ns, finish_ns, card):
        """Observed GPU-local service: no waiting entry or second GPU command."""
        if not self.component_offer(comp_id, kind, root_id, now_ns=now_ns):
            return False
        comp = self.components[comp_id]
        role = comp['role']
        self._release_open(comp)
        comp.update(state='inflight', inline=True, card=card)
        for field in ('E', 'S'):
            self.cur[field][role][kind] = self.cur[field][role].get(kind, 0) + 1
            self._sub_bump(field, role, kind)
        self.counters['inline_service'] = self.counters.get('inline_service', 0) + 1
        return self.component_finish(comp_id, now_ns, finish_ns=finish_ns)

    def component_finish(self, comp_id, now_ns, finish_ns=None, dispatch_ns=None):
        """In-flight completion: never touches the waiting W (B2/B3)."""
        comp = self.components.get(comp_id)
        if comp is None:
            self.counters['orphan_event'] += 1
            return False
        if finish_ns is not None:
            lag = max(int(now_ns) - int(finish_ns), 0)
            if len(self.cur['lag']['done_ns']) < self.max_lag_samples:
                self.cur['lag']['done_ns'].append(lag)
        comp['state'] = 'finished'
        self._root_close_component(comp)
        self.components.pop(comp_id, None)
        return True

    def _release_open(self, comp):
        """Retire one unresolved-offer entry, exactly once per counted offer.

        ``component_reject`` used to leave the offer ledger untouched, so a
        rejected offer leaked forever and ``Open_end - Open_start`` could never
        be reconciled against ``O - E - R`` (TASK 3.3).
        """
        if not comp.get('counted_open'):
            return
        role = comp['role']
        kind = comp['kind']
        self.open_offers[role][kind] = self.open_offers[role].get(kind, 0) - 1
        if self.open_offers[role][kind] <= 0:
            del self.open_offers[role][kind]
        comp['counted_open'] = False

    def _root_close_component(self, comp):
        rid = comp.get('root_id')
        if rid is not None and rid in self.roots:
            self.roots[rid]['open_comps'] = max(0, self.roots[rid]['open_comps'] - 1)

    @staticmethod
    def _root_public_p(root, role):
        """The root's PUBLIC probability for the role it finally produced.

        A Prefix-origin root only reveals its route/quality at EARLY, so the
        registration-time table is empty and the EARLY public p is authoritative
        (T014 s3.2.4).  Unknown stays None: it is never imputed as 0.
        """
        rev = root.get('revealed_early') or {}
        if root.get('entry_class') == ROOT_PREFIX and rev.get('role') == role:
            return parse_probability(rev.get('p_quality'))
        probs = root.get('probabilities') or {}
        return parse_probability(probs.get(role))

    def root_terminal(self, root_id, role, ontime, now_ns, p_selected=None,
                      late_discarded=False, terminal_available_ns=None):
        """Exactly one terminal per root (B2); duplicates are rejected.

        T014 s3.1.4: a root whose cohort already matured as a failure keeps its
        failure observation; this late DONE only closes the physical account and
        never rewrites the mature Y/V/quality of that cohort.
        """
        self._note_access('root_terminal', 1)
        root = self.roots.get(root_id)
        if root is None:
            return False
        if root['terminal'] is not None:
            self.counters['dup_terminal'] += 1
            return False
        self.advance_to(now_ns)
        root['terminal'] = {'role': role, 'ontime': bool(ontime),
                            'p': parse_probability(p_selected),
                            'available_ns': int(terminal_available_ns if terminal_available_ns is not None else now_ns),
                            'late_discarded': bool(late_discarded)}
        if root.get('mature_failure'):
            self.counters['late_done_after_maturity'] = \
                self.counters.get('late_done_after_maturity', 0) + 1
            self.counters['terminal'] += 1
            return True
        bucket = root['source_publish_ns'] // self.window_ns
        coh = self.cohorts.get(bucket)
        t = self.cur['terminal']
        if role == 'Abstain':
            t['abstain'] += 1
        else:
            t['completed_F'] += int(role == 'Fast')
            t['completed_A'] += int(role == 'Accurate')
            t['ontime'] += int(ontime)
            t['late'] += int(not ontime)
            p = parse_probability(p_selected)
            if ontime and p is not None:
                t['known_ontime'] += 1
                t['sum_p'] += p
                t['sum_p2'] += p * p
        if coh is not None and not coh.get('matured'):
            ct = coh['terminal']
            if role == 'Abstain':
                ct['abstain'] += 1
            else:
                ct['completed'] += 1
                ct['ontime'] += int(ontime)
                ct['late'] += int(not ontime)
                if ontime:
                    p = self._root_public_p(root, role)
                    if p is not None:
                        ct['sum_p_ontime'] += float(p)
            ctb = coh['terminal_by_class'][root['entry_class']]
            ctb['n'] += 1
            if role == 'Abstain':
                ctb['abstain'] += 1
            else:
                ctb['completed'] += 1
                ctb['ontime'] += int(ontime)
                ctb['late'] += int(not ontime)
                if ontime:
                    p = self._root_public_p(root, role)
                    if p is not None:
                        ctb['sum_p_ontime'] += float(p)
                        ctb['known_ontime'] += 1
                        budget = root['deadline_ns']-root['original_arrival_ns']
                        if budget > 0:
                            fraction = max(0., min(1., (root['deadline_ns']-
                                root['terminal']['available_ns'])/float(budget)))
                            ctb['sum_quality_earliness'] += float(p)*fraction
                            ctb['known_earliness'] += 1
        self.counters['terminal'] += 1
        return True

    def record_receipt_lag(self, lag_ns, now_ns):
        self.advance_to(now_ns)
        if len(self.cur['lag']['receipt_ns']) < self.max_lag_samples:
            self.cur['lag']['receipt_ns'].append(max(int(lag_ns), 0))

    def _sub_bump(self, key, role, kind):
        sub = self._current_sub()
        if key in ('E', 'S'):
            sub[key][role][kind] = sub[key][role].get(kind, 0) + 1
        else:
            sub[key][role] += 1

    # -- demand estimation over the sealed cohort ring (section 5) ----------
    def demand_inputs(self, budget_ok=None, include_support=True):
        """Raw + source-budget-supported lambda over the most recent sealed
        <=2 s cohort window, EARLY coverage and the unified-work point estimate.

        ``budget_ok(root)`` judges, with the CURRENT gate and the stage-publish
        remaining stored at publication time, whether the zero-queue whole
        chain would have fitted.  It never uses a consumer-delayed remaining.
        """
        cohorts = self.recent_sealed_cohorts()
        if not cohorts:
            self._note_access('demand_inputs', 0)
            return None
        T = sum(c['end_ns'] - c['start_ns'] for c in cohorts) / NS_PER_S
        out = {'window_ns': [c['start_ns'] for c in cohorts],
               'T_s': T, 'n_cohorts': len(cohorts), 'incomplete': False}
        n = {c: 0 for c in ROOT_CLASSES}
        n_sup = {c: 0 for c in ROOT_CLASSES}
        n_unk = {c: 0 for c in ROOT_CLASSES}
        reveal = {'Fast': 0, 'Accurate': 0, 'Abstain': 0,
                  'supported_Fast': 0, 'supported_Accurate': 0,
                  'sum_p_Fast': 0.0, 'sum_p_Accurate': 0.0,
                  'revealed_prefix_total': 0}
        prefix_total = 0
        unrevealed = 0
        results_all_sealed = True
        for c in cohorts:
            for cls in ROOT_CLASSES:
                n[cls] += c['n'][cls]
                n_sup[cls] += c['n_supported'][cls]
                n_unk[cls] += c['n_unknown'][cls]
            for key, val in reveal.items():
                if isinstance(val, float):
                    reveal[key] += c['reveal'][key]
                else:
                    reveal[key] += c['reveal'][key]
            prefix_total += c['prefix_total']
            if c['results_sealed']:
                unrevealed += c['unrevealed_after_deadline']
            else:
                results_all_sealed = False
            if c['incomplete']:
                out['incomplete'] = True
        for root in ([r for c in cohorts for r in c['roots']] if include_support else []):
            if root['budget_supported'] is None and budget_ok is not None:
                root['budget_supported'] = bool(budget_ok(root))
            if root['budget_supported'] is True:
                n_sup[root['entry_class']] += 1
                coh = self.cohorts.get(root['source_publish_ns'] // self.window_ns)
                if root['revealed_early'] and root['revealed_early']['role'] in ('Fast', 'Accurate') \
                        and coh is not None:
                    pass  # support already counted at reveal time via supported_*
            elif root['budget_supported'] is None:
                n_unk[root['entry_class']] += 1
        sum_pDF = n_DF = 0
        sum_pDA = n_DA = 0
        if not include_support:
            n_unk = dict(n)
            sum_pDF = sum(c.get('direct_quality_sums', {}).get(ROOT_DIRECT_F, 0.0) for c in cohorts)
            sum_pDA = sum(c.get('direct_quality_sums', {}).get(ROOT_DIRECT_A, 0.0) for c in cohorts)
            n_DF = sum(c.get('direct_quality_counts', {}).get(ROOT_DIRECT_F, 0) for c in cohorts)
            n_DA = sum(c.get('direct_quality_counts', {}).get(ROOT_DIRECT_A, 0) for c in cohorts)
        for root in ([r for c in cohorts for r in c['roots']] if include_support else []):
            probs = root.get('probabilities') or {}
            if root['entry_class'] == ROOT_DIRECT_F and probs.get('Fast') is not None:
                sum_pDF += float(probs['Fast'])
                n_DF += 1
            elif root['entry_class'] == ROOT_DIRECT_A and probs.get('Accurate') is not None:
                sum_pDA += float(probs['Accurate'])
                n_DA += 1
        out.update({
            'support_evaluated': bool(include_support),
            'lambda_raw': {c: n[c] / T for c in ROOT_CLASSES},
            'lambda_supported': {c: n_sup[c] / T for c in ROOT_CLASSES},
            'n_raw': dict(n), 'n_supported': dict(n_sup), 'n_unknown': dict(n_unk),
            'EARLY': {'reveal': reveal, 'prefix_total_M': prefix_total,
                      'unrevealed_U': unrevealed,
                      'coverage': (reveal['revealed_prefix_total'] / prefix_total)
                      if prefix_total else None,
                      'knownBF_over_M': (reveal['supported_Fast'] / prefix_total)
                      if prefix_total else None,
                      'knownBA_over_M': (reveal['supported_Accurate'] / prefix_total)
                      if prefix_total else None,
                      'U_over_M': (unrevealed / prefix_total) if prefix_total else None,
                      'results_all_sealed': results_all_sealed,
                      'pBF': (reveal['Fast'] / reveal['revealed_prefix_total'])
                      if reveal['revealed_prefix_total'] else None,
                      'pBA': (reveal['Accurate'] / reveal['revealed_prefix_total'])
                      if reveal['revealed_prefix_total'] else None,
                      'pBF_quality_mean': (reveal['sum_p_Fast'] / reveal['Fast'])
                      if reveal['Fast'] else None,
                      'pBA_quality_mean': (reveal['sum_p_Accurate'] / reveal['Accurate'])
                      if reveal['Accurate'] else None,
                      'pDF_quality': (sum_pDF / n_DF) if n_DF else None,
                      'pDA_quality': (sum_pDA / n_DA) if n_DA else None,
                      'EARLY_samples': reveal['revealed_prefix_total']},
        })
        n_roots = sum(len(c.get('roots') or []) for c in cohorts)
        self._note_access('demand_inputs', len(cohorts) + (n_roots if include_support else 0))
        return out

    # -- stall / views ------------------------------------------------------
    def latest_lag_p95(self):
        wins = list(self.sealed_windows)[-2:]
        if len(wins) < 2:
            return None
        return {'receipt_ns': max((w['lag'].get('receipt_ns_p95') or 0) for w in wins),
                'done_ns': max((w['lag'].get('done_ns_p95') or 0) for w in wins),
                'stall': any(w['flags']['stall'] for w in wins)}

    # -- noise window (T012 WS-A, TASK.md section 4) ------------------------
    @staticmethod
    def window_epoch(window):
        """Effective epoch of a window: explicit flag first, legacy field next."""
        if not window:
            return 0
        flags = window.get('flags') or {}
        epoch = flags.get('epoch')
        if epoch is not None:
            return int(epoch)
        legacy = window.get('effective_epoch')
        if legacy is not None:
            return int(legacy)
        return 0

    def _window_has_zero_wait_opportunity(self, window):
        if not window:
            return False
        flags = window.get('flags') or {}
        if flags.get('transition') or flags.get('quota_block'):
            return False

        do = window.get('deadline_opportunity')
        if isinstance(do, dict):
            if do.get('zero_wait') or do.get('zero_wait_opportunity') or do.get('both_zero'):
                return True
        elif isinstance(do, bool) and do:
            return True

        w_start = window.get('W_start')
        w_end = window.get('W_end')
        subs = window.get('sub')

        if w_start is None and not subs and w_end is None:
            return False

        # Check at start of window
        if w_start is not None:
            fast_start = sum(int(v) for v in (w_start.get('Fast') or {}).values())
            acc_start = sum(int(v) for v in (w_start.get('Accurate') or {}).values())
            if fast_start == 0 and acc_start == 0:
                return True

        # Check across sub-windows
        if w_start is not None and subs:
            fast_cur = sum(int(v) for v in (w_start.get('Fast') or {}).values())
            acc_cur = sum(int(v) for v in (w_start.get('Accurate') or {}).values())
            for s in subs:
                fast_cur += (sum(int(v) for v in (s.get('E', {}).get('Fast') or {}).values())
                             - sum(int(v) for v in (s.get('S', {}).get('Fast') or {}).values())
                             - int(s.get('X', {}).get('Fast') or 0))
                acc_cur += (sum(int(v) for v in (s.get('E', {}).get('Accurate') or {}).values())
                            - sum(int(v) for v in (s.get('S', {}).get('Accurate') or {}).values())
                            - int(s.get('X', {}).get('Accurate') or 0))
                if fast_cur <= 0 and acc_cur <= 0:
                    return True

        # Check at end of window
        if w_end is not None:
            fast_end = sum(int(v) for v in (w_end.get('Fast') or {}).values())
            acc_end = sum(int(v) for v in (w_end.get('Accurate') or {}).values())
            if fast_end == 0 and acc_end == 0:
                return True

        return False

    def resource_root_loss(self, window, role):
        """Distinct roots hit by a resource-reason loss in this window (root
        de-duplicated; one lost component is enough to mark the root)."""
        if not window:
            self._note_access('resource_root_loss', 1)
            return ResourceRootLoss(0, {
                'healthy_evidence': {},
                'zero_wait_opportunity': False,
                'resource_attributed': False,
            })
        lost = (window.get('lost_roots') or {}).get(role)
        n = 0 if lost is None else int(len(lost))

        # Honest resource-loss evidence (TASK s3.2.2)
        w_epoch = self.window_epoch(window)
        healthy_wins = []
        inspected = 0
        for sw in self.sealed_windows:
            inspected += 1
            if sw is window:
                continue
            if sw.get('index') == window.get('index') and sw.get('start_ns') == window.get('start_ns'):
                continue
            if self.window_epoch(sw) != w_epoch:
                continue
            if self._window_is_healthy(sw, role):
                healthy_wins.append(sw)

        self._note_access('resource_root_loss', max(inspected, 1))

        if healthy_wins:
            healthy_loss_rates = []
            for hw in healthy_wins:
                r_hw = (hw.get('residuals') or {}).get(role) or {}
                lr = r_hw.get('resource_loss_ms_s')
                if lr is not None:
                    healthy_loss_rates.append(float(lr))
            h_rate = (sum(healthy_loss_rates) / len(healthy_loss_rates)) if healthy_loss_rates else 0.0

            cur_res = (window.get('residuals') or {}).get(role) or {}
            cur_lr = cur_res.get('resource_loss_ms_s')
            c_rate = float(cur_lr) if cur_lr is not None else None

            healthy_evidence = {
                'healthy_windows': int(len(healthy_wins)),
                'same_epoch': True,
                'healthy_loss_rate': float(h_rate),
                'current_loss_rate': c_rate,
            }
        else:
            healthy_evidence = {}

        zero_wait = self._window_has_zero_wait_opportunity(window)
        res_attr = bool(healthy_evidence) and bool(zero_wait)

        return ResourceRootLoss(n, {
            'healthy_evidence': healthy_evidence,
            'zero_wait_opportunity': bool(zero_wait),
            'resource_attributed': bool(res_attr),
        })

    def _window_usable(self, w, role=None):
        """T014 s3.2.2: is a window USABLE for a trigger/probe comparison?

        Complete and free of transition/probe/stall/source-gap flags, with the
        per-role residual block present.  A resource loss is the SIGNAL we are
        looking for and never makes the window unusable -- that stronger
        no-loss condition is the noise-bank ``_window_is_healthy``.
        """
        if not w:
            return False
        f = w.get('flags') or {}
        if f.get('transition') or f.get('probe') or f.get('stall'):
            return False
        if f.get('source_gap') or f.get('log_backpressure'):
            return False
        if role is not None:
            if ((w.get('residuals') or {}).get(role)) is None:
                return False
        return True

    def _window_is_healthy(self, w, role):
        f = w.get('flags') or {}
        if f.get('transition') or f.get('probe') or f.get('stall'):
            return False
        if f.get('source_gap') or f.get('log_backpressure'):
            return False
        res = (w.get('residuals') or {}).get(role)
        if res is None:
            return False
        if res.get('resource_loss_ms_s', 0) != 0:
            return False
        if res.get('other_loss_ms_s', 0) != 0:
            return False
        return True

    # -- T014 s3.2.2: deadline-opportunity shape -----------------------------
    def note_deadline_opportunity(self, role, remaining_ns):
        """Bin ONE arrival's remaining deadline room into 5 ms classes.

        The value is frozen at PUBLISH/arrival time (``deadline - arrival``), so
        it describes the input opportunity a direction was offered in this
        window and never the execution outcome of the root.
        """
        if self.cur is None:
            return
        try:
            rem_ms = max(0.0, float(remaining_ns) / float(MS))
        except (TypeError, ValueError):
            return
        if rem_ms != rem_ms or rem_ms == float('inf'):
            return
        role = str(role or 'Unknown')
        b = int(rem_ms // DEADLINE_HIST_WIDTH_MS)
        h = self.cur['budget_hist'].setdefault(
            role, {'n': 0, 'bins': {}, 'sum_ms': 0.0})
        h['n'] = int(h['n']) + 1
        h['bins'][str(b)] = int(h['bins'].get(str(b), 0)) + 1
        h['sum_ms'] = float(h['sum_ms']) + rem_ms

    def span_probe_flag(self, lo_ns, hi_ns):
        """OR of the probe flags of every sealed window inside a span."""
        lo, hi = int(lo_ns), int(hi_ns)
        flags = []
        if (self.cur is not None and int(self.cur['start_ns']) >= lo and
                int(self.cur['end_ns']) <= hi):
            flags.append(bool(self.cur['flags'].get('probe')))
        inspected = 0
        for w in reversed(self.sealed_windows):
            inspected += 1
            if int(w['end_ns']) < lo:
                break
            if int(w['start_ns']) >= lo and int(w['end_ns']) <= hi:
                flags.append(bool(w['flags'].get('probe')))
        self._note_access('span_probe_flag', inspected)
        return probe_flag_or(flags)

    def span_budget_hist(self, lo_ns, hi_ns, role):
        """Aggregated class histogram of the sealed windows fully inside a span."""
        lo, hi = int(lo_ns), int(hi_ns)
        agg = {'n': 0, 'bins': {}, 'spans': 0, 'role': str(role or 'Unknown')}
        inspected = 0
        for w in reversed(self.sealed_windows):
            inspected += 1
            if int(w['end_ns']) < lo:
                break
            if int(w['start_ns']) >= lo and int(w['end_ns']) <= hi:
                h = (w.get('budget_hist') or {}).get(str(role))
                if not h:
                    continue
                agg['spans'] += 1
                agg['n'] += int(h.get('n') or 0)
                for k, v in (h.get('bins') or {}).items():
                    agg['bins'][k] = int(agg['bins'].get(k, 0)) + int(v)
        self._note_access('span_budget_hist', inspected)
        return agg

    @staticmethod
    def cohort_entry_budget_hist(cohorts, role):
        """Immutable first-GPU source opportunity; never terminal/receipt time.

        Frozen pre cohorts carry these aggregates after leaving the live ring.
        Prefix counts once at first GPU entry, not again at Suffix admission.
        """
        result = {'n': 0, 'bins': {}, 'role': str(role)}
        for cohort in cohorts:
            hist = (cohort.get('gpu_entry_budget_hist') or {}).get(role) or {}
            result['n'] += int(hist.get('n') or 0)
            for key, count in (hist.get('bins') or {}).items():
                result['bins'][key] = result['bins'].get(key, 0) + int(count)
        return result

    @staticmethod
    def deadline_shape_ok(a, b, tv_max=DEADLINE_SHAPE_TV_MAX,
                          min_n=DEADLINE_SHAPE_MIN_N):
        """T014 s3.2.2: is the deadline-opportunity shape comparable?

        Both sides need at least one class sample, the total-variation distance
        of the two class histograms must stay <= ``tv_max`` and EVERY class that
        either side observed must carry at least ``min_n`` roots on BOTH sides.
        An empty side is never 'equal enough': it is explicitly NOT usable.
        """
        na = int((a or {}).get('n') or 0)
        nb = int((b or {}).get('n') or 0)
        if na <= 0 or nb <= 0:
            return False, {'reason': 'empty_side', 'n_a': na, 'n_b': nb,
                           'usable': False}
        ba = (a or {}).get('bins') or {}
        bb = (b or {}).get('bins') or {}
        tv = 0.0
        thin = []
        for k in sorted(set(ba) | set(bb)):
            ca = int(ba.get(k, 0))
            cb = int(bb.get(k, 0))
            tv += abs(ca / float(na) - cb / float(nb))
            if ca < int(min_n) or cb < int(min_n):
                thin.append({'bin': k, 'n_a': ca, 'n_b': cb})
        tv *= 0.5
        enough = na >= int(min_n) and nb >= int(min_n)
        ok = bool(not enough or tv <= float(tv_max))
        return ok, {'tv': round(tv, 6), 'n_a': na, 'n_b': nb,
                    'thin_classes': thin[:8], 'tv_max': float(tv_max),
                    'min_n': int(min_n), 'usable': ok, 'distribution_evidence_sufficient': enough,
                    'reason': None if ok else 'deadline_opportunity_shape'}

    def healthy_window_growth(self, role, limit=32, epoch=None):
        """Oldest -> newest growth of the most recent healthy same-epoch windows.

        A healthy window has no transition/probe/stall flag, no source gap, no
        resource loss and no other (non-resource) loss.  When ``epoch`` is None
        the epoch of the newest healthy window becomes the target, so windows
        from a different effective_epoch can never pollute the series.
        """
        limit = int(limit)
        out = []
        target = None if epoch is None else int(epoch)
        newest_ns = self.sealed_windows[-1]['end_ns'] if self.sealed_windows else 0
        min_start_ns = newest_ns - int(limit) * self.window_ns * 2
        inspected = 0
        for w in reversed(self.sealed_windows):
            inspected += 1
            if len(out) >= limit:
                break
            if w['start_ns'] < min_start_ns:
                break
            if target is not None and self.window_epoch(w) < target:
                break
            if not self._window_is_healthy(w, role):
                continue
            if target is None:
                target = self.window_epoch(w)
            if self.window_epoch(w) != target:
                continue
            out.append(w['residuals'][role]['growth_ms_s'])
        out.reverse()
        self._note_access('healthy_window_growth', inspected)
        return out

    def root_closure(self):
        """M_root = terminal_root + alive_root (TASK.md section 3.3).

        The live root ledger is bounded/recycled, so the alive count comes from
        successful publishes minus successful terminals.
        """
        terminal = int(self.counters.get('terminal', 0))
        published = int(self.counters.get('root_publish', 0))
        alive = max(published - terminal, 0)
        return {'terminal_root': terminal, 'alive_root': alive,
                'total': terminal + alive}

    # -- perf timing (T012 WS-A, TASK.md section 3.2) -----------------------
    def note_handler(self, name, ns):
        return self.perf.note_handler(name, ns)

    def note_source_slice(self, ns):
        return self.perf.note_source_slice(ns)

    def note_loop_gap(self, ns):
        return self.perf.note_loop_gap(ns)

    def note_controller_sync(self, ns):
        return self.perf.note_controller_sync(ns)

    def note_worker(self, ns):
        return self.perf.note_worker(ns)

    def perf_snapshot(self):
        return self.perf.perf_snapshot()

    def sealed_compact(self, limit=32):
        out = []
        for w in list(self.sealed_windows)[-int(limit):]:
            res = w['residuals'] or {}
            out.append({
                'index': w['index'], 'start_ns': w['start_ns'], 'end_ns': w['end_ns'],
                'flags': dict(w['flags']),
                'budget_hist': {k: {'n': int(v.get('n') or 0),
                                    'bins': dict(v.get('bins') or {}),
                                    'sum_ms': float(v.get('sum_ms') or 0.0)}
                                for k, v in (w.get('budget_hist') or {}).items()},
                'cost_version': w['cost_version_at_seal'],
                'root_arrivals': dict(w['root_arrivals']),
                'reveal_EARLY': dict(w['reveal']['EARLY']),
                'terminal': {k: v for k, v in w['terminal'].items()},
                'receipt_p95_ns': w['lag'].get('receipt_ns_p95'),
                'done_p95_ns': w['lag'].get('done_ns_p95'),
                'roles': {r: {kk: res.get(r, {}).get(kk) for kk in
                              ('E_ms_s', 'S_ms_s', 'X_ms_s', 'R_ms_s', 'O_ms_s',
                               'dW_ms_s', 'growth_ms_s', 'resource_loss_ms_s',
                               'other_loss_ms_s', 'residual_dW_ms',
                               'residual_O_minus_S_ms', 'residual_dOpen_ms',
                               'Open_start_n', 'Open_end_n',
                               'resource_lost_roots', 'resource_lost_components',
                               'other_lost_components', 'E_n', 'S_n', 'X_n',
                               'R_n', 'O_n', 'W_start_ms', 'W_end_ms')}
                           for r in GPU_ROLES},
                'W_end_counts': {r: dict(w['W_end'][r]) if w['W_end'] else {}
                                 for r in GPU_ROLES},
            })
        return out

    def snapshot(self):
        return {
            'waiting_counts': {r: dict(self.wait_counts[r]) for r in GPU_ROLES},
            'waiting_work_ms': {r: self.waiting_work_ms(r) for r in GPU_ROLES},
            'commitment_ms': {r: self.commitment_ms(r) for r in GPU_ROLES},
            'open_offers': {r: dict(self.open_offers[r]) for r in GPU_ROLES},
            'roots_live': len(self.roots),
            'components_live': len(self.components),
            'sealed_windows': len(self.sealed_windows),
            'counters': dict(self.counters),
            'cost_version': self.cost_version(),
            'global_cost_version': self.global_cost_version(),
            'c_star_ms': dict(self.c_star_ms) if self.c_star_ms else None,
            'root_closure': self.root_closure(),
            'access': self.access_snapshot(),
            'source_closed': bool(self.source_closed),
            'source_final_ns': self.source_final_ns,
            'source_close_reason': self.source_close_reason,
        }


def self_check():
    cfg = {'c_ref_ms': {'FAST_FULL': 4.0, 'FAST_PREFIX': 2.0, 'FAST_SUFFIX': 3.0,
                        'ACCURATE': 8.0}, 'deadline_ns': 50_000_000}
    ob = QF4Observer(cfg)
    ob.advance_to(0)
    for k in range(25):
        assert ob.component_offer('r%d:MID1:0' % k, 'FAST_FULL')
        assert ob.component_enqueue('r%d:MID1:0' % k, 500_000_000, 0)
    for k in range(15):
        assert ob.component_dispatch('r%d:MID1:0' % k, 0)
    for k in range(15, 25):
        assert ob.component_remove('r%d:MID1:0' % k, 'resource', 0)
    ob.advance_to(1_000_000_000)
    w = ob.sealed_windows[0]
    r = w['residuals']['Fast']
    T = w['T_s']
    ok = (abs(r['E_ms_s'] - 25 * 4.0 / T) < 1e-9
          and abs(r['S_ms_s'] - 15 * 4.0 / T) < 1e-9
          and abs(r['X_ms_s'] - 10 * 4.0 / T) < 1e-9
          and abs(r['dW_ms_s']) < 1e-9
          and r['residual_dW_counts'] == 0 and r['residual_O_minus_S_ms'] < 1e-9)
    # WS-A residual extension: root de-dup, component counts, window work
    assert r['resource_lost_components'] == 10
    assert r['resource_lost_roots'] == 0    # no component carried a root_id
    assert r['other_lost_components'] == 0
    assert (r['E_n'], r['S_n'], r['X_n'], r['R_n'], r['O_n']) == (25, 15, 10, 0, 25)
    assert abs(r['W_start_ms']) < 1e-9 and abs(r['W_end_ms']) < 1e-9
    assert ob.root_closure() == {'terminal_root': 0, 'alive_root': 0, 'total': 0}
    # idempotency probes (B2)
    assert not ob.component_dispatch('r0:MID1:0', 1_000_000_000)
    assert ob.counters['dup_dispatch'] == 1

    # ---- window_epoch: explicit flag first, legacy field as fallback ------
    assert QF4Observer.window_epoch({'flags': {}, 'effective_epoch': 5}) == 5
    assert QF4Observer.window_epoch({'flags': {'epoch': 3},
                                     'effective_epoch': 5}) == 3
    assert QF4Observer.window_epoch({'flags': {'epoch': 0}}) == 0
    assert QF4Observer.window_epoch(None) == 0

    # ---- noise window (TASK.md section 4) + root closure ------------------
    ob2 = QF4Observer(cfg)
    ob2.advance_to(0)
    for k in range(4):
        assert ob2.component_offer('r0:L%d:0' % k, 'FAST_FULL')
        assert ob2.component_enqueue('r0:L%d:0' % k, 400_000_000, 0)
        assert ob2.component_dispatch('r0:L%d:0' % k, 0)      # W_end 0 -> growth 0
    # stable '<root_id>:<stage>:<attempt>' id gate
    assert not ob2.component_offer('a:b', 'FAST_FULL')        # too few segments
    assert not ob2.component_offer(7, 'FAST_FULL')            # legacy int job_id
    assert not ob2.component_offer('r1::0', 'FAST_FULL')      # empty segment
    assert not ob2.component_offer('r1:MIDX:z', 'FAST_FULL')  # attempt not int
    assert ob2.counters['bad_component_id'] == 4
    assert ob2.component_offer('r1:MID1:0', 'FAST_FULL')      # stays offered

    ob2.advance_to(500_000_000)                               # seals w0 (clean)
    assert ob2.root_publish('rootR', ROOT_DIRECT_F, 600_000_000, 650_000_000,
                            quality_role='Fast')
    assert ob2.root_publish('rootK', ROOT_DIRECT_A, 610_000_000, 660_000_000,
                            quality_role='Accurate')
    for k in range(8):
        assert ob2.component_offer('w1:L%d:0' % k, 'FAST_FULL', 'rootR')
        assert ob2.component_enqueue('w1:L%d:0' % k, 900_000_000, 500_000_000)
    assert ob2.component_remove('w1:L0:0', 'resource', 500_000_000)
    assert ob2.component_remove('w1:L1:0', 'resource', 500_000_000)
    assert ob2.component_remove('w1:L2:0', 'head_delay', 500_000_000)
    assert ob2.component_offer('w1:L8:0', 'FAST_SUFFIX', 'rootK')
    assert ob2.component_reject('w1:L8:0', 'resource', 500_000_000)
    ob2.cur['W_start']['Fast']['FAST_FULL'] = -3              # white-box tilt
    ob2.advance_to(1_000_000_000)                             # seals w1 (lossy)

    ob2.mark_flags(epoch=1)
    ob2.cur['W_start']['Fast']['FAST_FULL'] = 0
    ob2.advance_to(1_500_000_000)                             # seals w2 clean
    ob2.mark_flags(epoch=1)
    ob2.cur['W_start']['Fast']['FAST_FULL'] = -2
    ob2.advance_to(2_000_000_000)                             # seals w3 clean
    ob2.mark_flags(epoch=1)
    ob2.cur['W_start']['Fast']['FAST_FULL'] = -1
    ob2.cur['flags']['source_gap'] = True
    ob2.advance_to(2_500_000_000)                             # seals w4 gapped

    r1 = ob2.sealed_windows[1]['residuals']['Fast']
    assert abs(r1['growth_ms_s'] - 64.0) < 1e-9
    assert r1['resource_lost_roots'] == 2       # rootR (x2 events) + rootK, dedup
    assert r1['resource_lost_components'] == 3  # 2 X resource + 1 R resource
    assert r1['other_lost_components'] == 1     # the head_delay X
    assert (r1['E_n'], r1['S_n'], r1['X_n'], r1['R_n'], r1['O_n']) == (8, 0, 3, 1, 9)
    assert abs(r1['W_start_ms'] + 12.0) < 1e-9 and abs(r1['W_end_ms'] - 20.0) < 1e-9
    assert ob2.resource_root_loss(ob2.sealed_windows[1], 'Fast') == 2
    assert ob2.resource_root_loss(ob2.sealed_windows[0], 'Fast') == 0
    rl_w1 = ob2.resource_root_loss(ob2.sealed_windows[1], 'Fast')
    assert isinstance(rl_w1, int)
    assert 'healthy_evidence' in rl_w1
    assert 'zero_wait_opportunity' in rl_w1
    assert 'resource_attributed' in rl_w1
    assert rl_w1['healthy_evidence'].get('healthy_windows') == 1
    assert rl_w1['healthy_evidence'].get('same_epoch') is True
    rl_w0 = ob2.resource_root_loss(ob2.sealed_windows[0], 'Fast')
    assert rl_w0['healthy_evidence'] == {}
    assert rl_w0['resource_attributed'] is False
    assert ob2.window_epoch(ob2.sealed_windows[0]) == 0
    assert ob2.window_epoch(ob2.sealed_windows[4]) == 1
    assert abs(ob2.sealed_windows[4]['residuals']['Fast']['growth_ms_s']
               - 48.0) < 1e-9
    h0 = ob2.healthy_window_growth('Fast', epoch=0)
    h1 = ob2.healthy_window_growth('Fast', epoch=1)
    hn = ob2.healthy_window_growth('Fast')      # inferred newest healthy epoch
    assert h0 == [0.0], h0
    assert h1 == [40.0, 56.0], h1
    assert hn == h1, hn
    assert 64.0 not in h0 and 64.0 not in h1    # resource/other-loss w1 excluded
    assert 48.0 not in h1                       # source_gap w4 excluded
    assert ob2.sealed_windows[4]['flags']['source_gap'] is True

    closure_before = ob2.root_closure()
    assert closure_before == {'terminal_root': 0, 'alive_root': 2, 'total': 2}
    assert ob2.root_terminal('rootR', 'Fast', True, 2_500_000_000)
    assert not ob2.root_terminal('rootR', 'Fast', True, 2_500_000_000)
    assert ob2.counters['dup_terminal'] == 1
    closure_after = ob2.root_closure()
    assert closure_after == {'terminal_root': 1, 'alive_root': 1, 'total': 2}
    assert closure_after['total'] == (closure_after['terminal_root']
                                      + closure_after['alive_root'])

    # ---- CostBook: global table version vs sample revision (3.4) ----------
    cb = CostBook()
    assert cb.global_cost_version() == 0
    assert cb.sample_revision() == 0
    assert cb.observe(0, 'FAST_FULL', 4_000_000, 0, 0)
    v1 = cb.global_cost_version()
    s1 = cb.sample_revision()
    assert v1 == 1 and s1 == 1
    assert cb.global_cost_version() == v1        # no table change -> no bump
    assert cb.cost_version() == v1               # legacy alias follows global

    cb2 = CostBook()
    for _ in range(2):
        assert cb2.seed_warmup(0, 'FAST_SUFFIX', [3_000_000], [], 0)
    assert cb2.seed_warmup(0, 'FAST_FULL', [4_000_000], [], 0)
    g_a = cb2.global_cost_version()
    max_a = max(rec.cost_version for rec in cb2.records.values())
    assert cb2.seed_warmup(0, 'FAST_FULL', [4_000_000], [], 0)
    g_b = cb2.global_cost_version()
    max_b = max(rec.cost_version for rec in cb2.records.values())
    assert max_b == max_a == 2                   # per-record max did NOT move
    assert g_b == g_a + 1                        # the table version still moved
    assert not cb2.update_gates(0)
    assert cb2.global_cost_version() == g_b

    # ---- perf: bounded samples + TRUE full max (3.2) ----------------------
    ob.note_source_slice(1_000_000)
    ob.note_loop_gap(2_000_000)
    ob.note_controller_sync(3_000_000)
    ob.note_worker(4_000_000)
    ps = ob.perf_snapshot()
    for name in PERF_NAMES:
        assert ps[name]['n'] == 1, name
        assert ps[name]['full_max_ms'] > 0.0, name
    pb = PerfBook(16)
    pb.note_handler('score', 9_000_000)
    for _ in range(64):
        pb.note_handler('score', 1_000_000)
    st = pb.stats('score')
    assert st['n'] == 65
    assert st['max_ms'] == 1.0                   # bounded retained samples only
    assert st['full_max_ms'] == 9.0              # independent, never truncated
    assert st['p50_ms'] == 1.0 and st['p95_ms'] == 1.0
    ob2.note_handler('recv', 7_000_000)
    assert ob2.perf_snapshot()['recv']['full_max_ms'] == 7.0

    # ---- R2-A: open-offer identity (O - E - R == Open_end - Open_start) ----
    # w1 offers 8 FAST_FULL + 1 FAST_SUFFIX (O_n 9), enqueues 8 (E_n), rejects 1
    # (R_n).  The rejected offer MUST be released from the open ledger, so the
    # per-kind open delta is exactly zero for every kind.
    assert (r1['O_n'], r1['E_n'], r1['R_n']) == (9, 8, 1)
    assert r1['Open_start_n'] == 1 and r1['Open_end_n'] == 1
    assert abs(r1['residual_dOpen_ms']) < 1e-9, r1['residual_dOpen_ms']
    assert ob2.open_offers['Fast'].get('FAST_SUFFIX') is None   # released
    assert ob2.open_offers['Fast']['FAST_FULL'] == 1            # still offered
    # a rejected fabricated component must not leak into the open ledger either
    assert ob2.component_reject('never-offered:MID1:0', 'resource', 2_500_000_000,
                                kind='FAST_FULL', role='Fast')
    assert ob2.open_offers['Fast']['FAST_FULL'] == 1
    assert not ob2.component_reject('no-kind:MID1:0', 'resource', 2_500_000_000)

    # ---- R2-A: loss rate is reference-ms per second, never a raw count ----
    # w1 lost 2x FAST_FULL (c* 4.0) + 1x FAST_SUFFIX (c* 3.0) in T = 0.5 s.
    assert abs(r1['resource_loss_ms_s'] - (2 * 4.0 + 3.0) / 0.5) < 1e-9
    assert abs(r1['X_resource_ms_s'] - (2 * 4.0) / 0.5) < 1e-9
    assert abs(r1['R_resource_ms_s'] - (1 * 3.0) / 0.5) < 1e-9
    assert abs(r1['other_loss_ms_s'] - (1 * 4.0) / 0.5) < 1e-9   # the head_delay X

    # ---- R2-A: commitment is in-flight work, reported but never in W -------
    assert abs(ob.commitment_ms('Fast') - 15 * 4.0) < 1e-9
    assert ob.commitment_ms('Accurate') == 0.0
    snap = ob.snapshot()
    assert abs(snap['commitment_ms']['Fast'] - 60.0) < 1e-9
    assert abs(snap['waiting_work_ms']['Fast']) < 1e-9   # dispatched work left W
    assert 'root_closure' in snap and 'open_offers' in snap

    # ---- R2-A: canonical reason classes + legacy aliases -------------------
    assert normalize_reason('intrinsic_budget') == 'source_budget'
    assert normalize_reason('receipt_delay') == 'receipt_stall'
    assert normalize_reason('resource') == 'resource'
    ob3 = QF4Observer(cfg)
    ob3.advance_to(0)
    assert ob3.component_offer('z:L0:0', 'FAST_FULL')
    assert ob3.component_enqueue('z:L0:0', 100_000_000, 0)
    assert ob3.component_remove('z:L0:0', 'intrinsic_budget', 0)
    assert ob3.cur['X']['Fast'].get('source_budget', {}).get('FAST_FULL') == 1
    assert ob3.cur['X']['Fast'].get('intrinsic_budget') == {}

    # ---- R2-A: rebase_clock moves warmup stamps onto the relative clock ----
    cb3 = CostBook()
    assert cb3.seed_warmup(0, 'FAST_FULL', [4_000_000], [], 5_000_000_000)
    rec3 = cb3.records[(0, 'FAST_FULL')]
    assert rec3.last_update_ns == 5_000_000_000
    ob4 = QF4Observer(dict(cfg, cost_book=cb3))
    assert ob4.rebase_clock(3_000_000_000) == 3_000_000_000
    assert rec3.last_update_ns == 2_000_000_000
    rec3.last_update_ns = 1_000_000_000
    assert ob4.rebase_clock(4_000_000_000) == 4_000_000_000
    assert rec3.last_update_ns == 0            # clamped, never negative

    # ---- R2-A: cohorts carry the effective epoch and mature in order -------
    assert ob2.cohorts[600_000_000 // ob2.window_ns]['effective_epoch'] == 0
    ob2.seal_source_cohorts(2_500_000_000)
    assert ob2.cohorts[600_000_000 // ob2.window_ns]['wm_sealed'] is True
    ends = [c['end_ns'] for c in ob2.cohorts.values()]
    assert ends == sorted(ends)                 # insertion order == time order
    # same-epoch pre-window selection (TASK.md section 6)
    same = ob2.last_sealed_complete_cohorts(2_500_000_000, epoch=0)
    assert all(c['effective_epoch'] == 0 for c in same)
    assert ob2.last_sealed_complete_cohorts(2_500_000_000, epoch=99) == []

    # ---- T014: SOURCE_END close_source and access stats -------------------
    ob5 = QF4Observer(cfg)
    ob5.advance_to(0)
    assert ob5.root_publish('cs_root', ROOT_DIRECT_F, 100_000_000, 150_000_000, quality_role='Fast')
    ob5.advance_to(1_000_000_000)
    assert ob5.close_source(1_200_000_000, reason='source_end') == 1
    assert ob5.source_closed is True
    assert ob5.source_final_ns == 1_200_000_000
    assert ob5.source_close_reason == 'source_end'
    snap5 = ob5.snapshot()
    assert 'access' in snap5 and snap5['source_closed'] is True
    astat = ob5.take_access_stats()
    assert astat['calls'] > 0 and astat['items'] > 0
    assert 'close_source' in astat['counts']
    astat2 = ob5.take_access_stats()
    assert astat2['calls'] == 0 and astat2['items'] == 0
    assert astat2['max'] == astat['max']

    return {'pass': bool(ok), 'E_ms_s': r['E_ms_s'], 'S_ms_s': r['S_ms_s'],
            'X_ms_s': r['X_ms_s'], 'dW_ms_s': r['dW_ms_s'],
            'residual_dW_counts': r['residual_dW_counts'],
            'residual_O_minus_S_ms': r['residual_O_minus_S_ms'],
            'resource_lost_components': r['resource_lost_components'],
            'resource_lost_roots': r['resource_lost_roots'],
            'noise_window_growth': r1['growth_ms_s'],
            'healthy_epoch0': h0, 'healthy_epoch1': h1,
            'root_closure': closure_after,
            'global_cost_version': g_b, 'perf_full_max_ms': st['full_max_ms'],
            'residual_dOpen_ms': r1['residual_dOpen_ms'],
            'Open_start_n': r1['Open_start_n'], 'Open_end_n': r1['Open_end_n'],
            'resource_loss_ms_s': r1['resource_loss_ms_s'],
            'commitment_ms_Fast': ob.commitment_ms('Fast'),
            'rebase_clock_ok': True}


if __name__ == '__main__':
    print(self_check())
