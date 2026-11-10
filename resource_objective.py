"""Unweighted resource objective; model quality belongs to routing only.

Per input loss = min(return latency / original deadline budget, 1) + miss.
Missing/late/rejected inputs have loss 2. Utility = 2 - loss, or
1 + normalized deadline slack for an on-time output. All inputs stay in the
denominator. Equal coefficients are an explicit engineering choice, not a
measurement: preventing one deadline miss is worth one budget of aggregate
latency reduction. Numerical epsilon is not a statistical confidence bound.
"""
import math

OBJECTIVE = 'latency_timeout'
ROLES = ('Fast', 'Accurate')
ROUNDOFF = 1e-9


def timely_utility(arrival_ns, deadline_ns, available_ns):
    budget = int(deadline_ns) - int(arrival_ns)
    if budget <= 0 or available_ns is None or available_ns > deadline_ns:
        return 0.0
    latency = max(0, int(available_ns) - int(arrival_ns))
    return 2.0 - min(1.0, latency / float(budget))


def summarize_role(n, ontime, utility):
    missed = int(n) - int(ontime)
    if missed < 0 or not math.isfinite(float(utility)):
        raise ValueError('invalid resource accounting')
    # Capped delay includes a whole budget for every missing/late input.
    capped_delay = 2.0 * n - float(utility) - missed
    return {'input': int(n), 'ontime': int(ontime), 'missed': missed,
            'miss_rate': missed / float(n) if n else 0.0,
            'utility': float(utility), 'loss': 2.0 * n - float(utility),
            'capped_delay_sum': capped_delay}


def paired_effect(candidate, reference, span_s):
    """One common input profile/ledger, two allocations. Protect both miss counts.

    Speed may trade between roles; aggregate combined loss must improve.
    New role deadline failures cannot be hidden by the receiver's improvement.
    """
    if not isinstance(candidate, dict) or not isinstance(reference, dict) or span_s <= 0:
        return {'available': False, 'reason': 'resource_comparison_missing'}
    delta = {}
    for role in ROLES:
        a, b = candidate.get(role), reference.get(role)
        if not isinstance(a, dict) or not isinstance(b, dict):
            return {'available': False, 'reason': 'resource_role_missing'}
        for row in (a, b):
            for key in ('input', 'ontime', 'missed', 'miss_rate', 'utility', 'capped_delay_sum'):
                value = row.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    return {'available': False, 'reason': 'resource_value_invalid'}
            if (row['input'] < 0 or row['ontime'] < 0 or row['ontime'] > row['input']
                    or row['missed'] != row['input'] - row['ontime']
                    or row['utility'] < -ROUNDOFF or row['utility'] > 2*row['ontime']+ROUNDOFF):
                return {'available': False, 'reason': 'resource_accounting_invalid'}
        if a['input'] != b['input']:
            return {'available': False, 'reason': 'resource_input_mismatch'}
        delta[role] = {'missed_delta': a['missed'] - b['missed'],
                       'miss_rate_delta': a['miss_rate'] - b['miss_rate'],
                       'utility_gain': (a['utility'] - b['utility']) / span_s,
                       'capped_delay_reduction': (b['capped_delay_sum'] - a['capped_delay_sum']) / span_s}
    gain = sum(v['utility_gain'] for v in delta.values())
    protected = all(v['missed_delta'] <= 0 for v in delta.values())
    return {'available': True, 'gain': gain, 'role_deltas': delta,
            'roles_protected': protected, 'epsilon': ROUNDOFF,
            'permitted': protected and gain > ROUNDOFF}


def resource_verdict(effect, ready=True):
    if not ready or not effect.get('available'):
        return 'UNCERTAIN'
    if not effect['roles_protected'] or effect['gain'] < -ROUNDOFF:
        return 'ROLLBACK'
    return 'CONFIRM' if effect['gain'] > ROUNDOFF else 'UNCERTAIN'
