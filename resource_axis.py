"""Configured physical-card allocation axis and public-demand candidates."""
import math
import os

# Process-wide experiment geometry, set before importing any runtime/worker.
# The historical entry remains two cards; the four-card entry uses a fresh
# process and an explicit mask. Mixing geometries inside one process is refused.
CARD_COUNT = int(os.environ.get('EDGE_GPU_CARD_COUNT', '2'))
if CARD_COUNT not in (1, 2, 4):
    raise ValueError('EDGE_GPU_CARD_COUNT must be 1, 2 or 4')
CARD_IDS = tuple(range(CARD_COUNT))

AXIS_MIN_FAST = 0.0
AXIS_MAX_FAST = float(CARD_COUNT)
GRID = 0.1
CANDIDATE_GRID = tuple(round(i * GRID, 10) for i in range(CARD_COUNT * 10 + 1))


def card_shares(gf):
    """One fractional boundary card; every physical card conserves F+A=1."""
    g = float(gf)
    if not math.isfinite(g) or not 0 <= g <= CARD_COUNT:
        raise ValueError('allocation outside configured physical pool: %r' % gf)
    return tuple(round(min(1., max(0., g-card)), 12) for card in CARD_IDS)


def balanced_card_shares(gf):
    """Optional balanced split: every card carries gf/CARD_COUNT Fast, conserving the same total.

    With both roles positive on every card, the existing scorecard empty-turn
    rule makes each card work-conserving; no zero-share execution is allowed.
    """
    g = float(gf)
    if not math.isfinite(g) or not 0 <= g <= CARD_COUNT:
        raise ValueError('allocation outside configured physical pool: %r' % gf)
    return tuple(round(g/CARD_COUNT, 12) for _ in CARD_IDS)


def demand_reference(profile, costs):
    """Offered GPU work, including Prefix work and revealed continuations.

    This is a search anchor, not a deadline feasibility proof. Unrevealed
    continuations use the observed public mix; with no mix the anchor is unknown.
    """
    roots = list((profile or {}).get('roots') or [])
    known = [r.get('revealed_role') for r in roots
             if r.get('entry_class') == 'Prefix-origin'
             and r.get('revealed_role') in ('Fast', 'Accurate', 'Abstain')]
    pf = known.count('Fast') / len(known) if known else None
    pa = known.count('Accurate') / len(known) if known else None
    price = {k: min(float(row[k]) for row in costs.values())
             for k in ('FAST_FULL', 'FAST_PREFIX', 'FAST_SUFFIX', 'ACCURATE')}
    # Prefix and its Suffix cannot migrate between cards. Minima taken
    # separately can describe a physically impossible fast chain.
    chain_price = min(float(row['FAST_PREFIX']) + float(row['FAST_SUFFIX'])
                      for row in costs.values())
    expected_prefix_price = (min(float(row['FAST_PREFIX']) +
                                pf * float(row['FAST_SUFFIX'])
                                for row in costs.values())
                             if pf is not None else None)
    wf = wa = 0.0
    unknown = 0
    for root in roots:
        cls = root.get('entry_class')
        if cls == 'directA' or root.get('kind') == 'ACCURATE':
            wa += price['ACCURATE']
        elif cls == 'Prefix-origin' or root.get('kind') == 'FAST_PREFIX':
            wf += price['FAST_PREFIX']
            role = root.get('revealed_role')
            if role == 'Fast':
                wf += chain_price - price['FAST_PREFIX']
            elif role == 'Accurate':
                wa += price['ACCURATE']
            elif role != 'Abstain':
                unknown += 1
                if pf is not None:
                    wf += expected_prefix_price - price['FAST_PREFIX']
                    wa += pa * price['ACCURATE']
        else:
            wf += price['FAST_FULL']
    span = max(1, int((profile or {}).get('span_ns') or 2_000_000_000))
    reference = CARD_COUNT * wf / (wf + wa) if wf + wa > 0 else None
    if unknown and pf is None:
        reference = None
    return {'gF_ref': reference, 'D_F_ms_s': wf * 1000 / span,
            'D_A_ms_s': wa * 1000 / span, 'unknown_continuations': unknown,
            'basis': 'public_offered_component_work'}


def bounded_grid(current, reference, sequence, allowed=CANDIDATE_GRID):
    """Current, both neighbors, demand anchor and two rotating distant points."""
    allowed = sorted(set(float(v) for v in allowed))
    if not allowed or not math.isfinite(float(current)):
        return []
    out = {v for v in allowed if abs(v - current) <= GRID + 1e-9}
    if reference is not None and math.isfinite(float(reference)):
        out.add(min(allowed, key=lambda v: (abs(v-reference), v)))
    far = [v for v in allowed if v not in out]
    if far:
        offset = 2 * (int(sequence) % ((len(far)+1)//2))
        out.update(far[offset:offset+2])
    return sorted(out)


class PublicDemandLedger:
    """Incremental public-root counts in tick-sized time buckets.

    Each unique root contributes once; a public Prefix reveal replaces its
    category in the original bucket. Window queries aggregate bounded bucket
    counts, never sort or replay tasks. No labels/probabilities are retained.
    """
    CATEGORIES = ('direct_F', 'direct_A', 'prefix_F', 'prefix_A', 'prefix_X', 'prefix_U')

    def __init__(self, bucket_ns, span_ns, max_roots=16384):
        self.bucket_ns = int(bucket_ns)
        self.span_ns = int(span_ns)
        self.max_roots = int(max_roots)
        if self.bucket_ns <= 0 or self.span_ns < 2*self.bucket_ns or self.max_roots <= 0:
            raise ValueError('public ledger requires two positive tick intervals and positive capacity')
        self.buckets = {}
        self.indices = []
        self.roots = {}
        self.newest_ns = 0
        self.dropped_buckets = set()

    @staticmethod
    def category(root):
        if root.get('entry_class') == 'directA' or root.get('kind') == 'ACCURATE':
            return 'direct_A'
        if root.get('entry_class') == 'Prefix-origin' or root.get('kind') == 'FAST_PREFIX':
            return {'Fast': 'prefix_F', 'Accurate': 'prefix_A',
                    'Abstain': 'prefix_X'}.get(root.get('revealed_role'), 'prefix_U')
        return 'direct_F'

    def _drop_bucket(self, index):
        bucket = self.buckets.pop(index)
        for rid in bucket['roots']:
            self.roots.pop(rid, None)

    def _prune(self, now_ns):
        floor = (int(now_ns) - self.span_ns) // self.bucket_ns
        self.dropped_buckets = {index for index in self.dropped_buckets if index >= floor}
        while self.indices and self.indices[0] < floor:
            self._drop_bucket(self.indices.pop(0))

    def add(self, root):
        import bisect
        rid = root.get('root_id')
        if rid is None:
            return
        publish = int(root['source_publish_ns'])
        self.newest_ns = max(self.newest_ns, publish)
        self._prune(self.newest_ns)
        index = publish // self.bucket_ns
        if index < (self.newest_ns - self.span_ns) // self.bucket_ns:
            return
        category = self.category(root)
        old = self.roots.get(rid)
        if old is not None:
            old_index, old_category = old
            if old_index != index:
                raise ValueError('public root publish time changed')
            if old_category == category:
                return
            counts = self.buckets[index]['counts']
            counts[old_category] -= 1
            counts[category] += 1
            self.roots[rid] = (index, category)
            return
        if index not in self.buckets:
            bisect.insort(self.indices, index)
            self.buckets[index] = {'counts': dict.fromkeys(self.CATEGORIES, 0), 'roots': set()}
        bucket = self.buckets[index]
        bucket['roots'].add(rid)
        bucket['counts'][category] += 1
        self.roots[rid] = (index, category)
        while len(self.roots) > self.max_roots:
            dropped = self.indices.pop(0)
            self.dropped_buckets.add(dropped)
            self._drop_bucket(dropped)

    def work_summary(self, start_ns, end_ns, costs):
        start_ns, end_ns = int(start_ns), int(end_ns)
        if start_ns % self.bucket_ns or end_ns % self.bucket_ns or end_ns <= start_ns:
            raise ValueError('demand query must use complete positive tick intervals')
        self._prune(end_ns)
        first, last = start_ns // self.bucket_ns, end_ns // self.bucket_ns
        if end_ns - start_ns > self.span_ns:
            return {'valid': False, 'reason': 'direct_demand_outside_retention'}
        if any(first <= index < last for index in self.dropped_buckets):
            return {'valid': False, 'reason': 'direct_public_profile_overflow'}
        counts = dict.fromkeys(self.CATEGORIES, 0)
        for index in range(first, last):
            bucket = self.buckets.get(index)
            if bucket is not None:
                for key in self.CATEGORIES:
                    counts[key] += bucket['counts'][key]
        known = counts['prefix_F'] + counts['prefix_A'] + counts['prefix_X']
        pf = counts['prefix_F'] / known if known else None
        pa = counts['prefix_A'] / known if known else None
        price = {k: min(float(row[k]) for row in costs.values())
                 for k in ('FAST_FULL', 'FAST_PREFIX', 'FAST_SUFFIX', 'ACCURATE')}
        suffix = min(float(row['FAST_PREFIX']) + float(row['FAST_SUFFIX'])
                     for row in costs.values()) - price['FAST_PREFIX']
        prefix_n = sum(counts[k] for k in self.CATEGORIES if k.startswith('prefix_'))
        wf = counts['direct_F'] * price['FAST_FULL'] + prefix_n * price['FAST_PREFIX']
        wf += counts['prefix_F'] * suffix
        wa = (counts['direct_A'] + counts['prefix_A']) * price['ACCURATE']
        if known:
            expected_suffix = min(float(row['FAST_PREFIX']) + pf * float(row['FAST_SUFFIX'])
                                  for row in costs.values()) - price['FAST_PREFIX']
            wf += counts['prefix_U'] * expected_suffix
            wa += counts['prefix_U'] * pa * price['ACCURATE']
        span_ns = end_ns - start_ns
        return {'valid': True, 'D_F_ms_s': wf * 1000 / span_ns,
                'D_A_ms_s': wa * 1000 / span_ns, 'counts': counts,
                'root_count': sum(counts.values()), 'unknown_continuations': counts['prefix_U'],
                'continuation_basis': ('known_components_only' if counts['prefix_U'] and not known
                                       else 'revealed_public_mix'),
                'window_start_ns': start_ns, 'window_end_ns': end_ns,
                'span_ns': span_ns, 'basis': 'incremental_public_offered_component_work'}
