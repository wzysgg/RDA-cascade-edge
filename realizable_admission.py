"""PRAA M1: realizable admission from observed dispatch-to-result times.

Only public executor timestamps are used: batch dispatch, actual GPU result
time and the configured return delay. No labels, quality routes, future
arrivals or cached outcomes. A job whose remaining budget is below the
observed median end-to-end time of the smallest compatible batch cannot be
expected to return on time, so it is not given GPU service.
"""
from bisect import bisect_left, insort
from collections import deque

#: Batch kinds that can carry a queued job of this public kind.
COMPATIBLE = {'FAST_FULL': ('FAST_FULL', 'FAST_MIXED'),
              'FAST_PREFIX': ('FAST_PREFIX', 'FAST_MIXED'),
              'ACCURATE': ('ACCURATE',)}


class NeedTable:
    """Rolling quantile (median by default) of observed dispatch->result+return time per (kind, b)."""

    def __init__(self, window=256, minimum=30, quantile=.5, anchor=None, son_return_ns=None):
        if window < minimum or minimum < 1:
            raise ValueError('Need window must cover the minimum sample count')
        if not 0 < quantile < 1:
            raise ValueError('Need quantile must lie in (0, 1)')
        if anchor is not None and son_return_ns is not None:
            raise ValueError('Anchored need and safe-optimistic need are alternatives')
        self.window = window
        self.minimum = minimum
        self.quantile = quantile
        # Anchored need (opt-in): callable (kind, size) -> frozen profile ns or None.
        self.anchor = anchor
        # Safe-optimistic need (opt-in, CDC_DERIVATION.md 13b): the return delay counted
        # once per need, used by the subadditive upper value. median() stays observed-only.
        self.son_return_ns = son_return_ns
        # SON cache per (kind, card): (that key family's version, warm values, U vector).
        # Only an observation of the same kind and card can change them.
        self._son = {}
        self._son_versions = {}
        self.samples = {}
        self.ordered = {}
        self.version = 0
        self._floor = {}
        self._floor_version = -1
        self._anchored = {}
        self._anchored_version = -1
        self.warmings = []

    def add(self, kind, size, need_ns, gpu=None):
        """gpu keys a card separately: host paths of physical cards differ.

        With an anchor, returns a record when this key becomes warm: the anchored
        bound used while it was cold and the first observed value (a test of A).
        """
        if size < 1 or need_ns < 0:
            raise ValueError('Invalid need observation')
        key = (kind, int(size), gpu)
        recent = self.samples.setdefault(key, deque())
        ordered = self.ordered.setdefault(key, [])
        son = self.son_return_ns is not None
        warming = (self.anchor is not None or son) and len(ordered) == self.minimum - 1
        bound = (self.upper(kind, size, gpu) if son else self.anchored(kind, size, gpu)) if warming else None
        lower = self.optimistic(kind, size, gpu) if warming and son else None
        if len(recent) == self.window:
            old = recent.popleft()
            del ordered[bisect_left(ordered, old)]
        recent.append(need_ns)
        insort(ordered, need_ns)
        self.version += 1
        if son:
            family = (kind, gpu)
            self._son_versions[family] = self._son_versions.get(family, 0) + 1
        if warming:
            row = dict(kind=kind, size=int(size), gpu=gpu, bound_ns=bound,
                       observed_ns=self.observed(kind, size, gpu))
            if son:
                row['lower_ns'] = lower
            self.warmings.append(row)
            return row
        return None

    def observed(self, kind, size, gpu=None):
        ordered = self.ordered.get((kind, int(size), gpu))
        if not ordered or len(ordered) < self.minimum:
            return None
        # int(0.5*n) == n//2: the default is the original median exactly.
        return ordered[min(len(ordered) - 1, int(self.quantile * len(ordered)))]

    def median(self, kind, size, gpu=None):
        """Observed value; for a cold size the anchored bound if enabled, else None."""
        value = self.observed(kind, size, gpu)
        if value is not None or self.anchor is None:
            return value
        return self.anchored(kind, size, gpu)

    def _family(self, kind, gpu):
        """Cached (warm values {size: ns}, U vector) of one kind and card."""
        family = (kind, gpu)
        version = self._son_versions.get(family, 0)
        entry = self._son.get(family)
        if entry is None or entry[0] != version:
            warm = {s: v for (k, s, g) in self.ordered if k == kind and g == gpu
                    for v in (self.observed(k, s, g),) if v is not None}
            entry = (version, warm, [None], {})
            self._son[family] = entry
        return entry

    def _warm(self, kind, gpu):
        """Observed values of the warm sizes of one kind and card: {size: ns}."""
        return self._family(kind, gpu)[1]

    def upper(self, kind, size, gpu=None):
        """Safe value U(b) >= need(b) (Lemma 6), for realizability tests; None if undefined.

        Warm: the observed value. Cold: the smallest of U(b1) + U(b - b1) - return over
        splits (subadditivity S: one batch takes no longer than its two parts in sequence,
        the premise of the merge lemma) and, for b >= 2, the observed value of any warm
        larger size (monotonicity M, which the logs show for b >= 2 only). No profile.
        The vector is computed once per change of this kind and card, up to the largest
        size asked so far.
        """
        _, warm, values, _ = self._family(kind, gpu)
        size = int(size)
        ret = self.son_return_ns
        for b in range(len(values), size + 1):
            if b in warm:
                values.append(warm[b])
                continue
            cands = [values[c] + values[b - c] - ret for c in range(1, b // 2 + 1)
                     if values[c] is not None and values[b - c] is not None]
            if b >= 2:
                cands += [v for s, v in warm.items() if s > b]
            values.append(min(cands) if cands else None)
        return values[size]

    def optimistic(self, kind, size, gpu=None):
        """Observed value of the largest warm size s with 2 <= s < b (a lower value under M); else None."""
        warm = self._warm(kind, gpu)
        below = [s for s in warm if 2 <= s < int(size)]
        return warm[max(below)] if below else None

    def lower(self, kind, size, gpu=None):
        """Choice value D(b) for ranking (Proposition 7); None if undefined.

        Warm: the observed value. Cold: the optimistic value when one exists (D <= need
        under M); otherwise there is no optimistic value and the safe value U(b) is used.
        """
        _, warm, _, lowers = self._family(kind, gpu)
        size = int(size)
        if size not in lowers:
            if size in warm:
                value = warm[size]
            else:
                value = self.optimistic(kind, size, gpu)
                if value is None:
                    value = self.upper(kind, size, gpu)
            lowers[size] = value
        return lowers[size]

    def anchored(self, kind, size, gpu=None):
        """Conservative need of a cold size from the warm sizes of the same kind and card.

        Below: need(b0) + kappa (profile(b) - profile(b0)), b0 the largest warm size
        under b, kappa the largest observed need/profile ratio (Assumption A), never
        below need(b0). Above: the need of the smallest warm size over b (need is
        nondecreasing in b). The smaller of the available bounds; None if none.
        """
        if self._anchored_version != self.version:
            self._anchored = {}
            self._anchored_version = self.version
        key = (kind, int(size), gpu)
        if key in self._anchored:
            return self._anchored[key]
        warm = []
        for k, s, g in self.ordered:
            if k == kind and g == gpu:
                v = self.observed(k, s, g)
                if v is not None:
                    warm.append((s, v))
        warm.sort()
        bounds = []
        below = [(s, v) for s, v in warm if s < size]
        if below:
            s0, v0 = below[-1]
            profile, p0 = self.anchor(kind, size), self.anchor(kind, s0)
            ratios = []
            for s, v in warm:
                p = self.anchor(kind, s)
                if p:
                    ratios.append(v / p)
            if ratios and profile is not None and p0 is not None:
                bounds.append(max(v0, v0 + max(ratios) * (profile - p0)))
        above = [v for s, v in warm if s > size]
        if above:
            bounds.append(above[0])
        value = min(bounds) if bounds else None
        self._anchored[key] = value
        return value

    def floor(self, queue_kind):
        """Smallest observed median over all compatible batches; None if cold."""
        if self._floor_version != self.version:
            self._floor = {}
            self._floor_version = self.version
        if queue_kind not in self._floor:
            # Observed values only: the release floor never uses an anchored bound.
            values = [self.observed(kind, size, gpu) for kind, size, gpu in self.ordered
                      if kind in COMPATIBLE[queue_kind]]
            values = [v for v in values if v is not None]
            self._floor[queue_kind] = min(values) if values else None
        return self._floor[queue_kind]


class CudaEfficiency:
    """Recent CUDA service / actor occupancy, for capacity in CUDA ms per second."""

    def __init__(self, window=512):
        self.rows = deque(maxlen=window)

    def add(self, cuda_ns, occupancy_ns):
        if occupancy_ns > 0 and cuda_ns >= 0:
            self.rows.append((cuda_ns, occupancy_ns))

    def value(self):
        occupied = sum(o for _, o in self.rows)
        if len(self.rows) < 30 or occupied <= 0:
            return None
        return min(1., sum(c for c, _ in self.rows) / occupied)
