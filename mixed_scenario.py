"""T003 mixed random traffic with partial Local degradation: frozen input generator.

Schema T003_mixed_v1.  Every random stream is an independent random.Random instance
seeded by SHA256('seed|T003_mixed_v1|stream|node'); Python's builtin hash() is never
used and no two streams share an RNG.  This module is CPU-only: it imports neither
torch nor CUDA and reads neither the evaluation labels nor the quality-demand
reference.

Traffic design (fixed by the task contract):
  base per-node rate      lambda_i = traffic_rho / CPU_noMID2_pipeline_mean_ns[i] * 1e9
  per-node 10s rate pair  u drawn uniformly from RATE_CHOICES, order bit drawn
                          independently; the two 5s multipliers are u and 2-u, so every
                          complete pair has time-average 1.0 and each phase integral
                          equals lambda_i * phase_duration exactly (phases are whole
                          numbers of pairs).
  arrivals in a 5s bin   exponential process at lambda_i * multiplier; an interval that
                          crosses the bin end is discarded and the next bin restarts
                          from its own start.
  per arrival             source uniform over the 270 EVALUATION sources; condition
                          Bernoulli(p) with p = degraded_blur_prob for affected nodes in
                          [change_s, recover_s) and background_blur_prob otherwise.
"""
import hashlib
import json
import random

SCHEMA = 'T003_mixed_v1'
SCENARIO = 'T003_mixed_partial_blur'
T004_SCHEMA = 'T004_mixed_v1'
RUN_LABEL = 'T004'  # executing-task label for run names; run_t005 sets 'T005'
T004_SCENARIO = 'T004_mixed_partial_blur'
STREAMS = ('rate', 'arrival', 'blur', 'source')
RATE_CHOICES = (0.6, 0.8, 1.0, 1.2, 1.4)
AFFECTED_NODES = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
BACKGROUND = 'background'
DEGRADED = 'degraded'
RECOVER = 'recover'
PHASE_ORDER = (BACKGROUND, DEGRADED, RECOVER)
PHASE_LABELS = {BACKGROUND: '背景混合', DEGRADED: '部分退化', RECOVER: '恢复背景'}
CLEAN = 'clean'
BLUR = 'blur_0.55'


def identity(seed, stream, node, schema=None):
    return '%d|%s|%s|%d' % (int(seed), schema or SCHEMA, stream, int(node))


def stream_seed(seed, stream, node, schema=None):
    if stream not in STREAMS:
        raise ValueError('unknown stream %r' % (stream,))
    return int.from_bytes(hashlib.sha256(identity(seed, stream, node, schema).encode('utf-8')).digest(), 'big')


def base_node_rates(calibration, rho):
    """rho is the node-side traffic intensity; 0.6 / 1.0 / 1.4 are supported sizes."""
    means = calibration['CPU_noMID2_pipeline_mean_ns']
    return [float(rho) / (float(x) / 1e9) for x in means]


def evaluation_index(meta):
    rows = meta['rows'] if isinstance(meta, dict) else meta
    bank = {}
    for r in rows:
        if r.get('split') == 'EVALUATION':
            bank[(r['condition'], r['source'])] = int(r['bank_index'])
    sources = sorted({s for (c, s) in bank if c == CLEAN})
    blur_sources = sorted({s for (c, s) in bank if c == BLUR})
    if not sources:
        raise ValueError('no EVALUATION sources in bank metadata')
    if sources != blur_sources:
        raise ValueError('clean and blur_0.55 must cover the same evaluation sources')
    return sources, bank


def phase_boundaries(spec):
    return [(0.0, float(spec['change_s']), BACKGROUND),
            (float(spec['change_s']), float(spec['recover_s']), DEGRADED),
            (float(spec['recover_s']), float(spec['duration_s']), RECOVER)]


def blur_probability(node, affected, offset_s, background_prob, degraded_prob, change_s, recover_s):
    if int(node) in affected and change_s <= offset_s < recover_s:
        return float(degraded_prob)
    return float(background_prob)


def build_rate_schedule(spec):
    """20 nodes x (duration/rate_bin_s) multipliers; each 10s pair sums to exactly 2."""
    nodes = int(spec['total_nodes'])
    bin_s = float(spec['rate_bin_s'])
    nbins = int(round(float(spec['duration_s']) / bin_s))
    if nbins <= 0 or nbins % 2:
        raise ValueError('duration_s must be a positive multiple of 2*rate_bin_s')
    schedule = {}
    for node in range(nodes):
        rng = random.Random(stream_seed(spec['seed'], 'rate', node, spec.get('schema_version', SCHEMA)))
        mult = []
        for _ in range(nbins // 2):
            u = float(rng.choice(RATE_CHOICES))
            first = u if rng.randrange(2) == 0 else 2.0 - u
            mult.extend((first, 2.0 - first))
        schedule[str(node)] = [float(x) for x in mult]
    return schedule, nbins


def build_arrivals(spec, meta, base_rates, rate_schedule=None, override_degraded=None):
    """Deterministic frozen arrival plan for spec.

    override_degraded exists only for the CPU check that must show the arrival,
    node and source sequences are independent of the blur probability.
    """
    if rate_schedule is None:
        rate_schedule, _ = build_rate_schedule(spec)
    sources, bank = evaluation_index(meta)
    nodes = int(spec['total_nodes'])
    bin_s = float(spec['rate_bin_s'])
    change = float(spec['change_s'])
    recover = float(spec['recover_s'])
    affected = set(int(n) for n in spec['affected_nodes'])
    background_prob = float(spec['background_blur_prob'])
    degraded_prob = float(spec['degraded_blur_prob'] if override_degraded is None else override_degraded)
    per_node = {}
    for node in range(nodes):
        rng = random.Random(stream_seed(spec['seed'], 'arrival', node, spec.get('schema_version', SCHEMA)))
        base = float(base_rates[node])
        mult = rate_schedule[str(node)]
        rows = []
        seq = 0
        for k, m in enumerate(mult):
            start = k * bin_s
            end = start + bin_s
            rate = base * float(m)
            if not rate > 0 or rate != rate:
                raise ValueError('non-positive arrival rate at node %d bin %d' % (node, k))
            at = start
            while True:
                at += rng.expovariate(rate)
                if at >= end:
                    break
                rows.append((int(round(at * 1e9)), node, seq))
                seq += 1
        per_node[node] = rows
    plan = []
    for node in range(nodes):
        source_rng = random.Random(stream_seed(spec['seed'], 'source', node, spec.get('schema_version', SCHEMA)))
        blur_rng = random.Random(stream_seed(spec['seed'], 'blur', node, spec.get('schema_version', SCHEMA)))
        for off, _node, seq in per_node[node]:
            source = sources[source_rng.randrange(len(sources))]
            probability = blur_probability(node, affected, off / 1e9, background_prob, degraded_prob,
                                           change, recover)
            condition = BLUR if blur_rng.random() < probability else CLEAN
            plan.append({'task_id': -1, 'scheduled_offset_ns': off, 'origin_node': node,
                         'source': source, 'image_condition': condition,
                         'blur_probability': probability,
                         'bank_index': bank[(condition, source)], '_seq': seq})
    plan.sort(key=lambda r: (r['scheduled_offset_ns'], r['origin_node'], r['_seq']))
    for i, r in enumerate(plan):
        r['task_id'] = i
        del r['_seq']
    return plan


def build_periodic_arrivals(spec, meta):
    """Periodic sampling in integrated-rate time; busy devices never stop arrivals.

    Each node keeps one seeded fractional phase across every rate change.
    Source/blur streams are separate from timing, including stress substitution.
    """
    sources, bank = evaluation_index(meta)
    affected = set(spec['affected_nodes'])
    duration = float(spec['duration_s'])
    load_start, load_end = spec.get('load_start_s'), spec.get('load_end_s')
    boundaries = sorted({0.0, duration} | ({float(load_start), float(load_end)}
                         if load_start is not None else set()))
    plan = []
    for node in range(int(spec['total_nodes'])):
        phase = random.Random(stream_seed(spec['seed'], 'arrival', node)).random()
        source_rng = random.Random(stream_seed(spec['seed'], 'source', node))
        blur_rng = random.Random(stream_seed(spec['seed'], 'blur', node))
        sequence, integral = 0, 0.0
        threshold = phase
        for start, end in zip(boundaries, boundaries[1:]):
            high = load_start is not None and node in affected and load_start <= start < load_end
            rate = float(spec['high_node_tps'] if high else spec['normal_node_tps'])
            next_integral = integral + rate * (end-start)
            while threshold < next_integral:
                at = start + (threshold-integral)/rate
                source = source_rng.choice(sources)
                condition = BLUR if blur_rng.random() < spec['background_blur_prob'] else CLEAN
                plan.append({'task_id': -1, 'scheduled_offset_ns': round(at*1e9),
                    'origin_node': node, 'source': source, 'image_condition': condition,
                    'blur_probability': spec['background_blur_prob'],
                    'bank_index': bank[(condition, source)], '_seq': sequence})
                sequence += 1
                threshold = sequence + phase
            integral = next_integral
    plan.sort(key=lambda r: (r['scheduled_offset_ns'], r['origin_node'], r['_seq']))
    for i, row in enumerate(plan):
        row['task_id'] = i
        del row['_seq']
    return plan


def plan_bytes(plan):
    return json.dumps(plan, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def logical_plan_sha(plan):
    """Must stay byte-identical to live_runtime/run_v24 plan_sha256 computation."""
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_plan(spec, plan, meta):
    """Return a list of human-readable errors; empty means the plan is admissible."""
    errors = []
    sources, bank = evaluation_index(meta)
    source_set = set(sources)
    nodes = int(spec['total_nodes'])
    bin_s = float(spec['rate_bin_s'])
    change = float(spec['change_s'])
    recover = float(spec['recover_s'])
    affected = set(int(n) for n in spec['affected_nodes'])
    background_prob = float(spec['background_blur_prob'])
    degraded_prob = float(spec['degraded_blur_prob'])
    limit = int(round(float(spec['duration_s']) * 1e9))
    if [r['task_id'] for r in plan] != list(range(len(plan))):
        errors.append('task_id must be contiguous from 0')
    previous = (-1, -1)
    for r in plan:
        off = r['scheduled_offset_ns']
        node = r['origin_node']
        if not (0 <= off < limit):
            errors.append('offset out of [0,duration): %s' % (off,))
            break
        if not (0 <= node < nodes):
            errors.append('origin_node out of range: %s' % (node,))
            break
        if (off, node) < previous:
            errors.append('plan is not sorted by (offset,node) at task %s' % (r['task_id'],))
            break
        previous = (off, node)
    counts = {'source': 0, 'bank': 0, 'prob': 0, 'condition': 0}
    for r in plan:
        node = r['origin_node']
        if r['source'] not in source_set:
            counts['source'] += 1
        if bank.get((r['image_condition'], r['source'])) != r['bank_index']:
            counts['bank'] += 1
        expect = blur_probability(node, affected, r['scheduled_offset_ns'] / 1e9,
                                  background_prob, degraded_prob, change, recover)
        if abs(expect - r['blur_probability']) > 1e-15:
            counts['prob'] += 1
        if r['image_condition'] not in (CLEAN, BLUR):
            counts['condition'] += 1
    for key, value in counts.items():
        if value:
            errors.append('%d rows violate %s consistency' % (value, key))
    return errors


def plan_stats(spec, plan, base_rates, rate_schedule):
    """Descriptive statistics used by quick checks and by the input manifest."""
    duration = float(spec['duration_s'])
    bin_s = float(spec['rate_bin_s'])
    nodes = int(spec['total_nodes'])
    nbins = int(round(duration / bin_s))
    boundaries = phase_boundaries(spec)
    phase_stats = []
    for start, end, name in boundaries:
        arrivals = 0
        blurred = 0
        affected_arrivals = 0
        expected_blurred = 0.0
        for r in plan:
            t = r['scheduled_offset_ns'] / 1e9
            if not (start <= t < end):
                continue
            arrivals += 1
            if r['origin_node'] in tuple(spec['affected_nodes']):
                affected_arrivals += 1
            if r['image_condition'] == BLUR:
                blurred += 1
            expected_blurred += r['blur_probability']
        theoretical = sum(float(x) for x in base_rates) * (end - start)
        phase_stats.append({'phase': name, 'phase_cn': PHASE_LABELS[name], 'start_s': start, 'end_s': end,
                            'arrivals': arrivals, 'affected_arrivals': affected_arrivals,
                            'other_arrivals': arrivals - affected_arrivals, 'blurred': blurred,
                            'blur_fraction': (blurred / arrivals if arrivals else 0.0),
                            'expected_blurred': expected_blurred,
                            'expected_blur_fraction': (expected_blurred / arrivals if arrivals else 0.0),
                            'theoretical_arrivals': theoretical})
    node_phase = [[0, 0, 0, 0, 0, 0] for _ in range(nodes)]
    bins = [[0] * nbins for _ in range(nodes)]
    total_blur = 0
    for r in plan:
        t = r['scheduled_offset_ns'] / 1e9
        index = 0 if t < boundaries[0][1] else 1 if t < boundaries[1][1] else 2
        node = r['origin_node']
        node_phase[node][index * 2] += 1
        if r['image_condition'] == BLUR:
            node_phase[node][index * 2 + 1] += 1
            total_blur += 1
        bins[node][min(nbins - 1, int(t / bin_s))] += 1
    intensity = []
    for node in range(nodes):
        actual = [c / bin_s for c in bins[node]]
        ordered = sorted(actual)
        index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        intensity.append({'node': node, 'base_rate_tps': float(base_rates[node]),
                          'planned_multipliers': rate_schedule[str(node)],
                          'actual_counts': bins[node], 'actual_tps': actual,
                          'actual_mean': sum(actual) / len(actual), 'actual_p95': ordered[index],
                          'actual_max': max(actual)})
    base_total = sum(float(x) for x in base_rates)
    pairs = []
    for node in range(nodes):
        mult = rate_schedule[str(node)]
        for k in range(0, len(mult), 2):
            pairs.append(mult[k] + mult[k + 1])
    return {'schema_version': spec.get('schema_version', SCHEMA), 'seed': int(spec['seed']), 'realized_total_tasks': len(plan),
            'theoretical_total_tasks': base_total * duration,
            'poisson_sigma': (base_total * duration) ** 0.5,
            'total_blurred': total_blur,
            'total_blur_fraction': (total_blur / len(plan) if plan else 0.0),
            'base_total_tps': base_total,
            'base_node_rates_tps': [float(x) for x in base_rates],
            'affected_base_total_tps': sum(float(base_rates[n]) for n in spec['affected_nodes']),
            'rate_pair_sum_min': min(pairs), 'rate_pair_sum_max': max(pairs),
            'rate_pair_count': len(pairs),
            'phase_stats': phase_stats,
            'node_phase': [{'node': n, 'background_arrivals': node_phase[n][0], 'background_blurred': node_phase[n][1],
                            'degraded_arrivals': node_phase[n][2], 'degraded_blurred': node_phase[n][3],
                            'recover_arrivals': node_phase[n][4], 'recover_blurred': node_phase[n][5]}
                           for n in range(nodes)],
            'intensity_5s': intensity}


def build_spec(args, calibration=None):
    """Spec shared by run_t003.py and quick_checks_t003.py; policy is filled per arm."""
    def opt(name, default):
        value = getattr(args, name, None)
        return default if value is None else value
    schema = str(opt('schema_version', SCHEMA))
    t004 = schema == T004_SCHEMA
    change = float(opt('change_s', 70.0 if t004 else 140.0))
    recover = float(opt('recover_s', 190.0 if t004 else 380.0))
    duration = float(opt('duration_s', 250.0 if t004 else 500.0))
    deadline = float(opt('deadline_ms', 50.0))
    spec = {'name': None,
            'policy': str(opt('policy', 'fixed_1a1f')),
            'scenario': str(opt('scenario', T004_SCENARIO if t004 else SCENARIO)),
            'schema_version': schema,
            'controller_version': str(opt('controller_version', 'v25' if t004 else 'v24')),
            'quality_need_enabled': True,
            'protect_accurate_deadline': True,
            'seed': int(opt('seed', 7904001)),
            'traffic_rho': float(opt('traffic_rho', 0.60)),
            'rho': float(opt('traffic_rho', 0.60)),
            'total_nodes': 20,
            'affected_nodes': list(AFFECTED_NODES),
            'rate_bin_s': int(opt('rate_bin_s', 5)),
            'rate_pair_s': int(opt('rate_pair_s', 10)),
            'blur_radius': float(opt('radius', 0.55)),
            'changed_condition': BLUR,
            'background_blur_prob': float(opt('background_blur_prob', 0.10)),
            'degraded_blur_prob': float(opt('degraded_blur_prob', 0.80)),
            'duration_s': duration,
            'change_s': change,
            'recover_s': recover,
            'controller_start_s': float(opt('controller_start_s', 10.0 if t004 else 60.0)),
            'initial_fast': 1.0,
            'requests': [],
            'deadline_ms': deadline,
            'forecast_horizon_ms': float(opt('forecast_horizon_ms', None) or deadline),
            'deadline_ns': int(round(deadline * 1e6)),
            'controller_axis': [round(0.5 + 0.1 * i, 1) for i in range(11)],
            'quality_optimizer_connected': False,
            'phases': [{'start_s': 0.0, 'end_s': change, 'condition': 'mixed_background', 'load_increase': False},
                       {'start_s': change, 'end_s': recover, 'condition': 'mixed_degraded', 'load_increase': False},
                       {'start_s': recover, 'end_s': duration, 'condition': 'mixed_recover', 'load_increase': False}],
            'algorithm_description': ('T004 mixed random per-task blur traffic; paired 5s rates average to 1x base'
                                      if t004 else 'T003 mixed random per-task blur traffic; paired 5s rates average to 1x base')}
    if calibration is not None:
        spec['controller_prices_ms'] = {k: float(v) for k, v in calibration['controller_prices_ms'].items()}
        rates = base_node_rates(calibration, spec['traffic_rho'])
        spec['base_node_rates_tps'] = [float(x) for x in rates]
        spec['rate_tps'] = float(sum(rates))
    return spec


def run_name(policy, tag, schema=None):
    # T005 reuses the T004_mixed_v1 schema and inputs; only the run-name label moves
    # to the executing task (T005 TASK section 7).  run_t005 sets RUN_LABEL='T005'.
    label = RUN_LABEL if (schema or SCHEMA) == T004_SCHEMA else 'T003'
    # T008 TASK section 8 names the two successful arms
    # TRIAL_T008_mixed_d50_n20_blur0.55_<policy>; the T004/T005 branches are
    # unchanged, only the label list is generalised.
    # T009 TASK section 8 names the two arms
    # TRIAL_T009_mixed_d50_n20_blur0.55_<policy>; only the label list grows.
    # T010 TASK section 9 keeps the same naming rule for the joint run's two arms
    # TRIAL_T010_mixed_d50_n20_blur0.55_<policy>; only the label list grows again.
    # T012 TASK section 1 keeps the same naming rule for the joint run's two arms
    # TRIAL_T012_mixed_d50_n20_blur0.55_<policy>; only the label list grows again.
    # T014 TASK s5 keeps the same naming rule; only the label list grows.
    if label in ('T004', 'T005', 'T008', 'T009', 'T010', 'T011', 'T012',
                 'T014', 'T015', 'T016') and not tag:
        name = 'TRIAL_%s_mixed_d50_n20_blur0.55_%s' % (label, policy)
    else:
        name = 'TRIAL_%s_mixed_d50_n20_blur0.55_%s_protect1_need1_%s' % (label, policy, tag)
    for token in (label, 'mixed', 'd50', policy):
        if token not in name:
            raise ValueError('run name must contain %r: %s' % (token, name))
    return name
