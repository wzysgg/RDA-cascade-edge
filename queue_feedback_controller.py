"""QF-4 feedback controller: unified demand, real capacity candidates, one
selection rule and a real short-window probe transaction (TASK.md sections
5-8).  Pure standard library; single ``control_step`` entry shared by the
Runtime and the CPU checks.  It never imports torch, numpy, the Runtime, any
demand_ledger/calendar worker or private labels.

Substantive qf4 changes over the N10 qf3 controller (TASK.md sections 5-8):

1. demand is the UNIFIED root work (direct F/A + Prefix origin with public
   EARLY probabilities), with raw and source-budget-supported lambda, coverage
   and unknown bounds; unknown is never filled with zero and probabilities are
   never imputed to 1.
2. capacity comes from the REAL per-card mean/gate/delta tables: a continuous
   reference-GPU model for the mixed Fast role (marked approx) and the
   conditional single-kind cycle formula for Accurate with the N_safe
   opportunity margin; candidates are verified against the REAL select_job
   gate on a next-cycle profile copy (basis = ordinary head or the recent
   rejected profile), never by floor(P95) arithmetic alone.
3. one selection rule over the frozen 11-point grid: exclusion, feasibility
   band K >= L + c_role/(1000*T), nearest valid step (never several valid
   bands skipped to the extreme), max-Yhat fallback with the documented tie
   order; max(L/K) stays a diagnostic.
4. the trigger gate needs two fresh windows of persistent gap (growth above
   the bounded noise P95 or a full resource-lost request in each window); the
   low-coverage/model-miss path is supply-margin gated; LOAD/QUALITY tags
   never switch anything.
5. the probe is a state machine IDLE->REQUESTED->EFFECTIVE->PROBE with
   KEEP/CONTINUE/ROLLBACK/UNCERTAIN outcomes on standardized mature GPU root
   cohorts, fixed comparability gates and a bounded epsilon; the fixed arm
   runs the identical math as a shadow with zero commands.
"""

import collections
import math
import os
import time
from collections import Counter, OrderedDict, deque
from resource_axis import AXIS_MIN_FAST, AXIS_MAX_FAST, GRID, CANDIDATE_GRID, CARD_COUNT, CARD_IDS, card_shares

from queue_feedback_stats import (
    GPU_ROLES, GPU_KINDS, MS, NS_PER_S, ROOT_CLASSES, ROOT_DIRECT_A,
    ROOT_DIRECT_F, ROOT_PREFIX, WARMUP_SOURCE, CostBook, QF4Observer,
    absolute_p95, parse_probability, p95_linear, role_of_kind, tv_distance,
    validate_cost_records,
)

CONTROLLER_VERSION = 'qf4r'
ALGORITHM_REVISION = 'QF-4R'
TASK_ID = 'T016'
IMPLEMENTATION_REVISION = 'T016_qf4r_physical_start_guard_49'
DELIVERY_GAIN_ROUNDOFF = 1e-9  # numerical precision, not the statistical noise epsilon
DECISION_SCHEMA = 'QF4R_decision_v3'
CONTROL_LOG_SCHEMA = 'QF4R_control_v3'
FIXED_POLICY = 'fixed_1a1f'
DYNAMIC_POLICY = 'qf4r_feedback'
POLICIES = (FIXED_POLICY, DYNAMIC_POLICY)


def capacity_confirmed_roles(context):
    """R019: two already observed halves exceed current role capacity.

    This selects evidence only at an existing demand-change joint. It never
    triggers a clock, changes a threshold or overrides the reversal filter.
    """
    if not context.get('joint') or context.get('pending'):
        return []
    checks = context.get('role_checks', {})
    if not all(checks.get(r) for r in GPU_ROLES):
        return []
    result = []
    for role in GPU_ROLES:
        if 'demand_change_'+role not in context.get('trigger_reasons', []):
            continue
        checked = checks[role]
        cap = context['effective_fast'] if role == 'Fast' else CARD_COUNT-context['effective_fast']
        halves = checked['half_offered_ms_s']
        if len(halves) != 2:
            raise ValueError('Capacity confirmation requires two observed halves')
        if all(d/1000. > cap+checked['change_threshold_gpu'] for d in halves):
            result.append(role)
    return result


def capacity_confirmed_window(context, waiting_ms, legal):
    """Select a coherent span only if its target serves the confirmed role."""
    roles = capacity_confirmed_roles(context)
    if not roles:
        return None
    checks = context['role_checks']
    starts = {checks[r]['window_start_ns'] for r in GPU_ROLES}
    ends = {checks[r]['window_end_ns'] for r in GPU_ROLES}
    if len(starts) != 1 or len(ends) != 1:
        return None
    span = next(iter(ends))-next(iter(starts))
    weights = {}
    for role in GPU_ROLES:
        h = context['intervals_ns'][role]/NS_PER_S
        weights[role] = checks[role]['offered_ms_s']+max(0.,waiting_ms[role]+checks[role]['growth_ms_s']*h)/h
    if sum(weights.values()) <= 0:
        return None
    raw = CARD_COUNT*weights['Fast']/sum(weights.values())
    target = min(legal, key=lambda g:(abs(g-raw),g))
    delta = target-context['effective_fast']
    receiver = 'Fast' if delta > 0 else 'Accurate'
    if abs(delta) < 1e-9 or receiver not in roles:
        return None
    return dict(span_ns=span, window_end_ns=next(iter(ends)),
                confirmed_roles=roles, receiver=receiver,
                preliminary_raw=raw, preliminary_target=target)


def overload_unit_cost_target(proportional_fast, weights, units, eta, margin):
    """PRAA M3 raw Fast target; unchanged proportional target unless overloaded.

    weights/units are ms/s of offered GPU work and ms per job. eta is measured
    CUDA service per actor-occupied second, so one card supplies 1000*eta ms/s.
    Under overload, maximizing completions per GPU time is a fractional
    knapsack: the role with the lower unit price receives its weighted demand
    plus the existing change floor, the other role the remainder.
    """
    info = {'eta': eta, 'proportional_raw_fast': proportional_fast, 'active': False}
    if eta is None or not eta > 0:
        return proportional_fast, info
    card_ms_s = 1000.*eta
    demand = sum(weights[r] for r in GPU_ROLES)
    info.update(capacity_ms_s=CARD_COUNT*card_ms_s, demand_ms_s=demand)
    if demand <= CARD_COUNT*card_ms_s:
        return proportional_fast, info
    cheap = min(GPU_ROLES, key=lambda r: (units[r], r))
    cards = min(float(CARD_COUNT), weights[cheap]/card_ms_s+margin)
    raw = cards if cheap == 'Fast' else CARD_COUNT-cards
    info.update(active=True, priority_role=cheap, priority_cards=cards,
                unit_ms={r: units[r] for r in GPU_ROLES}, raw_fast=raw)
    return raw, info


def _env_flag(name):
    """Opt-in runtime switches kept OUT of the frozen reference identity.

    The T014 reference document is issued from the task mailbox and pins the
    configuration, so a behaviour change that must not invalidate it is enabled
    per process instead of by a config key."""
    return str(os.environ.get(name) or '').strip().lower() in ('1', 'true', 'yes', 'on')

PERIOD_MS = 50.0
PERIOD_NS = 50_000_000
TICK_MS = 100.0
TICK_NS = 100_000_000
WINDOW_MS = 500.0
SERVICE_MS_S_PER_STEP = GRID * WINDOW_MS / 0.5   # one 0.1 grid step = 100 ms/s of card service
WINDOW_NS = 500_000_000
PROBE_MS = 500.0
PROBE_EXTENDED_MS = 2000.0
ROOT_ESTIMATION_MS = 2000.0
STATES = ('IDLE', 'REQUESTED', 'EFFECTIVE', 'PROBE')
#: TASK section 6 names the confirmable verdict CONFIRM; KEEP stays an accepted
#: alias for one revision so an older control log stays readable.
OUTCOMES = ('CONFIRM', 'KEEP', 'CONTINUE', 'ROLLBACK', 'UNCERTAIN')
CONFIRM_OUTCOMES = ('CONFIRM', 'KEEP')
NOISE_WINDOW_LIMIT = 32
HEALTHY_MIN_FOR_NOISE = 8
#: fixed engineering parameters of section 8 (not confidence bounds)
COMPARABLE_TV_MAX = 0.1
COMPARABLE_MEAN_REL_MAX = 0.1
COMPARABLE_LAG_MAX_NS = 12_500_000
COMPARABLE_MIN_GPU_ROOTS = 20
EARLY_MIN_SAMPLES = 20
EARLY_MIN_COVERAGE = 0.5
MAX_UNCONFIRMED_TRIALS_PER_DIRECTION = 2
LAMBDA_UNLOCK_REL = 0.20

#: TASK section 5.3 / REFERENCE.algorithm: the candidate epsilon_select and the
#: exploratory (low-coverage) branch.
TPROFILE_S = 2.0
SELECT_NOISE_MAX_PROFILES = 16
SELECT_NOISE_MIN_PROFILES = 8
SELECT_RESOLUTION_JOBS_S = 0.5
EXPLORATORY_MIN_REVEAL = 30
CACHE_MAX_INPUT_AGE_MS = 750.0
CACHE_MAX_INPUT_AGE_NS = int(CACHE_MAX_INPUT_AGE_MS * MS)
#: TASK section 6: probe cohort alignment and the standardized output noise.
SETTLE_MS = 50.0
PROBE_PRE_MS = 1000.0
PROBE_FIRST_POST_MS = 500.0
PROBE_EXTENDED_POST_MS = 1000.0
PROBE_POST_ALIGN_MS = 500.0
PROBE_PRE_COHORTS = 4
OUTPUT_NOISE_MAX_PAIRS = 16
OUTPUT_NOISE_MIN_PAIRS = 8
#: T014 s3.4.5 output-noise span banks: pre 1 s with a 0.5 s / 1 s post span.
#: A 0.5 s and a 1 s difference are NEVER mixed into one epsilon series.
NOISE_SPAN_BANKS = ('2.0|0.5', '2.0|1.0', '2.0|2.0')
#: TASK section 6: unlock evidence of a failed edge, compared with the REAL
#: snapshot saved at rollback time.
LAMBDA_CHANGE_INVALIDATE_REL = 0.20
SOURCE_COMPOSITION_TV_LIMIT = 0.1
MEAN_GATE_CHANGE_REL = 0.1
#: TASK section 3.4: ONE cost book per arm, live mean only at >=30 samples.
MIN_LIVE_COST_SAMPLES = 30
COST_UPDATE_MS = 250.0
#: REFERENCE.algorithm.gF_grid, in the order the replay worker enumerates it.
#: the receiving kinds of the Fast role (used for c_role_F).
_FAST_RECEIVE_KINDS = ('FAST_FULL', 'FAST_PREFIX', 'FAST_SUFFIX')


# ---------------------------------------------------------------- pure demand


def root_work_demand(lambda_DF, lambda_DA, lambda_B, pBF, pBA, c_star_ms):
    """The ONE unified demand formula (TASK.md section 5 / D002 section 4).

        aF = lambda_DF*cFull + lambda_B*(cPrefix + pBF*cSuffix)
        aA = lambda_DA*cA + lambda_B*pBA*cA
    """
    c = {str(k): float(v) for k, v in c_star_ms.items()}
    aF = float(lambda_DF) * c['FAST_FULL'] + \
        float(lambda_B) * (c['FAST_PREFIX'] + float(pBF) * c['FAST_SUFFIX'])
    aA = float(lambda_DA) * c['ACCURATE'] + \
        float(lambda_B) * float(pBA) * c['ACCURATE']
    return aF, aA


def phi_cards(gf):
    """Card0/card1 Fast shares of a gF target; Accurate is 1 - Fast.

    Shares are rounded to 12 decimals so the 0.1 grid never produces a
    9.999999...ms budget that flips the b == m boundary of the conditional
    capacity formula.
    """
    g = float(gf)
    return card_shares(g)


def on_grid(value, axis_min=AXIS_MIN_FAST, grid=GRID):
    steps = (float(value) - float(axis_min)) / float(grid)
    return abs(steps - round(steps)) < 1e-9


def grid_value(value, axis_min=AXIS_MIN_FAST, axis_max=AXIS_MAX_FAST, grid=GRID):
    steps = int(round((float(value) - float(axis_min)) / float(grid)))
    return round(min(max(float(axis_min) + steps * float(grid), float(axis_min)),
                     float(axis_max)), 10)


def demand_weights(lambda_DF, lambda_B, pBF, c_star_ms, role='Fast'):
    """w_k = nu_k*c*_k/a_r with the nu of the full-request formula (section 6)."""
    if role == 'Fast':
        nu = {'FAST_FULL': float(lambda_DF),
              'FAST_PREFIX': float(lambda_B),
              'FAST_SUFFIX': float(lambda_B) * float(pBF)}
    else:
        nu = {'ACCURATE': 1.0}
    a = sum(nu[k] * float(c_star_ms[k]) for k in nu)
    if a <= 0:
        return None, 0.0
    w = {k: (nu[k] * float(c_star_ms[k])) / a for k in nu}
    return w, a


def opportunity_margin_ms(price_history, n_point, min_margin_ms=0.1, max_points=8):
    """h_gk = max(0.1 ms, empirical P95 of |delta_b| + (N_point-1)*|delta_mean|
    over the last <=8 same-kind price updates (TASK.md section 6)."""
    hist = list(price_history or [])[-int(max_points):]
    if len(hist) < 2:
        return float(min_margin_ms), True
    db = [abs(hist[i][0] - hist[i - 1][0]) for i in range(1, len(hist))]
    dm = [abs(hist[i][1] - hist[i - 1][1]) for i in range(1, len(hist))]
    reps = max(int(n_point or 1) - 1, 0)
    vals = [b + reps * m for b, m in zip(db, dm)]
    p = absolute_p95(vals)
    if p is None:
        return float(min_margin_ms), True
    return max(float(min_margin_ms), p), False


def conditional_single_kind(phi, m_ms, b_ms, delta_ms, price_history=None,
                            min_margin_ms=0.1):
    """The REAL conditional capacity of one partial single-kind card (D002 s6).

    With constant charge m, gate b >= m, no reservation/cross-period spill:
        N(B) = 0                      if B < b
             = 1 + floor((B - b)/m)   otherwise,  B = share*50 ms
        mu  = min(20*N, share*1000/(m+delta))
    The scored / valid-increase capacity uses N_safe with the opportunity
    margin h subtracted from B; the unmargined N is kept as a diagnostic.
    Dedicated whole cards use mu = 1000/(m+delta) and need no N.  b < m falls
    back to the continuous estimate with approx=True; the real gate still has
    to verify every start.
    """
    out = {'capacity_model': 'conditional_single_kind', 'N_point': None,
           'N_safe': None, 'mu_point': None, 'mu_safe': None,
           'approx': False, 'low_confidence': False, 'dedicated': False,
           'b_lt_m': False, 'margin_ms': None}
    phi = float(phi)
    if phi <= 0.0:
        out['capacity_model'] = 'zero_share'
        out['mu_point'] = out['mu_safe'] = 0.0
        return out
    m = float(m_ms)
    b = float(b_ms)
    dl = float(delta_ms or 0.0)
    if m <= 0:
        out['capacity_model'] = 'invalid_price'
        return out
    if phi >= 1.0 - 1e-12:
        out['dedicated'] = True
        mu = 1000.0 / (m + dl)
        out['mu_point'] = out['mu_safe'] = mu
        return out
    if b < m:
        out['b_lt_m'] = True
        out['approx'] = True
        cont = phi * 1000.0 / (m + dl)
        out['mu_point'] = out['mu_safe'] = cont
        return out
    B = phi * PERIOD_MS

    def n_of(budget):
        if budget < b:
            return 0
        return 1 + int(math.floor((budget - b) / m))

    n_point = n_of(B)
    h, low_conf = opportunity_margin_ms(price_history, n_point,
                                        min_margin_ms=min_margin_ms)
    n_safe = n_of(B - h)
    mu_cap = phi * 1000.0 / (m + dl)
    out['N_point'] = int(n_point)
    out['N_safe'] = int(n_safe)
    out['mu_point'] = min(20.0 * n_point, mu_cap)
    out['mu_safe'] = min(20.0 * n_safe, mu_cap)
    out['margin_ms'] = h
    out['low_confidence'] = bool(low_conf)
    return out


def fast_continuous_capacity(phi, weights, c_star_ms, mean_ms, delta_ms):
    """Continuous reference-GPU capacity of one partial mixed-Fast card:
    d_g = sum_k w_k*(m_gk + delta_gk)/c*_k ;  K_g = phi/d_g  (marked approx)."""
    if phi <= 0.0:
        return 0.0, 0.0, None
    d = 0.0
    for kind, w in weights.items():
        m = mean_ms.get(kind)
        if m is None:
            return None, None, 'missing_mean_%s' % kind
        dl = float(delta_ms.get(kind, 0.0) or 0.0)
        d += float(w) * (float(m) + dl) / float(c_star_ms[kind])
    if d <= 0:
        return None, None, 'nonpositive_d'
    K = float(phi) / d
    mu = K * 1000.0 / float(c_star_ms['FAST_FULL'])
    return K, mu, None


def evaluate_candidates(current_gF, demand, cost, c_star_ms, T_s=1.0,
                        min_margin_ms=0.1):
    """The fixed 11-point CPU candidate table of TASK.md section 6.

    ``demand`` carries lambda_DF/lambda_DA/lambda_B/pBF/pBA (+ optional public
    quality probabilities pDF/pDA/pBF_quality/pBA_quality).  ``cost`` maps
    card -> kind -> {'mean_wall_ms','gate_ms','delta_ms','price_history'}.
    Nothing here runs a GPU experiment: pure CPU arithmetic over the frozen
    grid, marked approx where the model is mixed.
    """
    lam_DF = float(demand.get('lambda_DF') or 0.0)
    lam_DA = float(demand.get('lambda_DA') or 0.0)
    lam_B = float(demand.get('lambda_B') or 0.0)
    pBF = float(demand.get('pBF') or 0.0)
    pBA = float(demand.get('pBA') or 0.0)
    aF, aA = root_work_demand(lam_DF, lam_DA, lam_B, pBF, pBA, c_star_ms)
    L_F = aF / 1000.0
    L_A = aA / 1000.0
    wF, _ = demand_weights(lam_DF, lam_B, pBF, c_star_ms, 'Fast')
    if wF is None:
        # a_r == 0: diagnostic capacity uses the Full idle proxy; never a gap
        wF = {'FAST_FULL': 1.0, 'FAST_PREFIX': 0.0, 'FAST_SUFFIX': 0.0}
        aF_basis = 'idle_proxy_Full'
    else:
        aF_basis = 'profile_mix'
    aA_basis = 'idle_proxy_Accurate' if aA <= 0 else 'profile_mix'

    means = {g: {k: float(cost[g][k]['mean_wall_ms']) for k in GPU_KINDS}
             for g in CARD_IDS}
    gates = {g: {k: float(cost[g][k]['gate_ms']) for k in GPU_KINDS}
             for g in CARD_IDS}
    deltas = {g: {k: float(cost[g][k].get('delta_ms') or 0.0) for k in GPU_KINDS}
              for g in CARD_IDS}
    hist = {g: {k: cost[g][k].get('price_history') for k in GPU_KINDS}
            for g in CARD_IDS}
    c_roleF = max(float(c_star_ms[k]) for k in _FAST_RECEIVE_KINDS)
    c_roleA = float(c_star_ms['ACCURATE'])
    feas_margin_F = c_roleF / (1000.0 * float(T_s))
    feas_margin_A = c_roleA / (1000.0 * float(T_s))

    pDF = demand.get('pDF')
    pDA = demand.get('pDA')
    pBF_q = demand.get('pBF_quality')
    pBA_q = demand.get('pBA_quality')

    cands = []
    v_ok_any = False
    for gF in CANDIDATE_GRID:
        gF = round(gF, 10)
        fast_shares = phi_cards(gF)
        accurate_shares = tuple(round(1.0 - f, 12) for f in fast_shares)
        K_F = 0.0
        muF_parts = []
        fast_approx = False
        fast_basis = []
        for g, phi in enumerate(fast_shares):
            if phi <= 0:
                continue
            K, mu, err = fast_continuous_capacity(phi, wF, c_star_ms,
                                                  means[g], deltas[g])
            if err is not None:
                K, mu = 0.0, 0.0
                fast_basis.append('card%d:%s' % (g, err))
            if abs(phi - 1.0) > 1e-12:
                fast_approx = True
                fast_basis.append('card%d:mixed_partial_approx' % g)
            K_F += K or 0.0
            muF_parts.append(mu or 0.0)
        cardA = {}
        muA_safe_total = 0.0
        muA_point_total = 0.0
        for g, phi in enumerate(accurate_shares):
            rec = conditional_single_kind(phi, means[g]['ACCURATE'],
                                          gates[g]['ACCURATE'],
                                          deltas[g]['ACCURATE'],
                                          price_history=hist[g]['ACCURATE'],
                                          min_margin_ms=min_margin_ms)
            rec['card'] = g
            cardA[g] = rec
            muA_safe_total += rec['mu_safe'] or 0.0
            muA_point_total += rec['mu_point'] or 0.0
        K_A = muA_safe_total * c_roleA / 1000.0
        K_A_point = muA_point_total * c_roleA / 1000.0

        thetaF = 1.0 if L_F <= 0 else min(1.0, K_F / L_F)
        thetaA = 1.0 if L_A <= 0 else min(1.0, K_A / L_A)
        thetaA_point = 1.0 if L_A <= 0 else min(1.0, K_A_point / L_A)
        theta_min = min(thetaF, thetaA)
        theta_min_point = min(thetaF, thetaA_point)
        Yhat = lam_DF * thetaF + lam_DA * thetaA + \
            lam_B * (pBF * thetaF + pBA * theta_min)
        Yhat_point = lam_DF * thetaF + lam_DA * thetaA_point + \
            lam_B * (pBF * thetaF + pBA * theta_min_point)
        v_terms = 0.0
        v_terms_point = 0.0
        v_ok = True
        if lam_DF > 0:
            if pDF is None:
                v_ok = False
            else:
                v_terms += lam_DF * float(pDF) * thetaF
                v_terms_point += lam_DF * float(pDF) * thetaF
        if lam_DA > 0:
            if pDA is None:
                v_ok = False
            else:
                v_terms += lam_DA * float(pDA) * thetaA
                v_terms_point += lam_DA * float(pDA) * thetaA_point
        if lam_B > 0:
            if pBF_q is None and pBF > 0:
                v_ok = False
            if pBA_q is None and pBA > 0:
                v_ok = False
            if pBF_q is not None:
                v_terms += lam_B * pBF * float(pBF_q) * thetaF
                v_terms_point += lam_B * pBF * float(pBF_q) * thetaF
            if pBA_q is not None:
                v_terms += lam_B * pBA * float(pBA_q) * theta_min
                v_terms_point += lam_B * pBA * float(pBA_q) * theta_min_point
        Vhat = v_terms if v_ok else None
        Vhat_point = v_terms_point if v_ok else None
        v_ok_any = bool(v_ok_any or v_ok)
        feas_F = K_F >= L_F + feas_margin_F
        feas_A = K_A >= L_A + feas_margin_A
        feas_A_point = K_A_point >= L_A + feas_margin_A
        cands.append({
            'gF': gF, 'gA': round(CARD_COUNT - gF, 10),
            'phi_F': list(fast_shares), 'phi_A': list(accurate_shares),
            'K_F': K_F, 'K_A': K_A, 'K_A_point': K_A_point,
            'L_F': L_F, 'L_A': L_A,
            'muF_tasks_s': sum(muF_parts),
            'muA_point_tasks_s': muA_point_total,
            'muA_safe_tasks_s': muA_safe_total,
            'thetaF': thetaF, 'thetaA': thetaA, 'thetaA_point': thetaA_point,
            'Yhat': Yhat, 'Yhat_point': Yhat_point,
            'Vhat': Vhat, 'Vhat_point': Vhat_point, 'V_available': bool(v_ok),
            'feasible_F': bool(feas_F), 'feasible_A': bool(feas_A),
            'feasible_A_point': bool(feas_A_point),
            'feasible_both': bool(feas_F and feas_A),
            'feasible_both_point': bool(feas_F and feas_A_point),
            'margin_F': K_F - L_F, 'margin_A': K_A - L_A,
            'margin_min': min(K_F - L_F, K_A - L_A),
            'distance': round(abs(gF - float(current_gF)), 10),
            'capacity_model_F': 'continuous_mixed_approx' if fast_approx
                                else 'continuous_mixed',
            'fast_approx': bool(fast_approx), 'fast_basis': fast_basis,
            'aF_basis': aF_basis, 'aA_basis': aA_basis,
            'card_A': cardA,
            'feas_margin_F': feas_margin_F, 'feas_margin_A': feas_margin_A,
            'effective_increase': True, 'commitment_protected': True,
        })
    return {
        'demand': {'lambda_DF': lam_DF, 'lambda_DA': lam_DA, 'lambda_B': lam_B,
                   'pBF': pBF, 'pBA': pBA, 'pDF': pDF, 'pDA': pDA,
                   'pBF_quality': pBF_q, 'pBA_quality': pBA_q},
        'aF': aF, 'aA': aA, 'L_F': L_F, 'L_A': L_A,
        'aF_basis': aF_basis, 'aA_basis': aA_basis,
        'c_role_F_ms': c_roleF, 'c_role_A_ms': c_roleA,
        'T_s': float(T_s), 'current_gF': float(current_gF),
        'grid': GRID, 'point_invalid': not bool(v_ok_any),
        'candidates': cands,
    }


def current_candidate(table, current_gF):
    if table is None:
        return None
    for c in table['candidates']:
        if abs(c['gF'] - round(float(current_gF), 10)) < 1e-9:
            return c
    return None


def select_epsilon(profile_series, T_profile_s=TPROFILE_S,
                   min_profiles=SELECT_NOISE_MIN_PROFILES):
    """TASK 5.3: eps_select = max(1/T_profile, absolute P95 of the last <=16
    comparable same-epoch candidate net-gain drifts), kept per unit (Y and V).

    Fewer than ``min_profiles`` comparable profiles -> the one-candidate
    resolution ``1/T_profile`` (0.5 jobs/s for T_profile = 2 s) and the
    low_confidence mark; a zero drift never lowers the floor.
    """
    vals = [abs(float(v)) for v in (profile_series or [])][-SELECT_NOISE_MAX_PROFILES:]
    base = 1.0 / float(T_profile_s)
    if len(vals) < int(min_profiles):
        return {'epsilon': base, 'low_confidence': True, 'n_profiles': len(vals),
                'basis': 'one_candidate_resolution'}
    p = absolute_p95(vals)
    return {'epsilon': max(base, p or 0.0), 'low_confidence': False,
            'n_profiles': len(vals), 'basis': 'same_epoch_profile_drift_p95'}


def _receiver_gain(cur, cand, receiver):
    """Predicted receiving-role opportunity gained by ``cand`` (per unit of
    that role's own work), plus the same-unit resolution of the table."""
    if receiver == 'Accurate':
        return (float(cand['muA_safe_tasks_s']) - float(cur['muA_safe_tasks_s']),
                float(cur.get('muA_safe_tasks_s') or 0.0))
    return (float(cand['K_F']) - float(cur['K_F']), float(cur.get('K_F') or 0.0))


def _exploratory_target(table, cur, epsY, epsV, receiver, excluded):
    """TASK 5.3 exploratory branch: one minimal legal step only.

    The step must be known to increase the receiving role's own on-time
    opportunity, and (when the point model is valid) must not lower the total
    point Y/V beyond the same-unit epsilon.  Whether the donor really has idle
    margin, or whether all three sensitivity scenarios stay non-worse, is
    checked by the controller against the live view; when neither holds the
    controller holds explicitly and lists the numbers.
    """
    cur_v = cur['Vhat'] if cur['V_available'] else None
    point_valid = bool(cur.get('V_available')) and \
        not bool(table.get('point_invalid')) and \
        not bool(cur.get('point_invalid'))
    best = None
    for c in table['candidates']:
        if abs(c['gF'] - cur['gF']) < 1e-9 or c['gF'] in tuple(excluded or ()):
            continue
        if abs(abs(c['gF'] - cur['gF']) - table.get('grid', GRID)) > 1e-6:
            continue            # exploratory: one minimal grid step only
        if not c.get('effective_increase', True):
            continue
        if not c.get('commitment_protected', True):
            continue
        gain, _base = _receiver_gain(cur, c, receiver)
        if gain <= 1e-12:
            continue
        dY = c['Yhat'] - cur['Yhat']
        dV = None if (cur_v is None or not c['V_available'] or c['Vhat'] is None) \
            else c['Vhat'] - cur_v
        if point_valid and (dY < -epsY or (dV is not None and dV < -epsV)):
            continue
        rec = dict(c)
        rec['receiver_gain'] = gain
        rec['gain_Y'] = dY
        rec['gain_V'] = dV
        key = (gain, -(abs(c['gF'] - cur['gF'])), -c['gF'])
        if best is None or key > best[0]:
            best = (key, rec)
    if best is None:
        return {'action': 'hold', 'reason': 'exploratory_no_receiver_opportunity',
                'current': cur, 'epsilon_Y': epsY, 'epsilon_V': epsV,
                'exploratory': True, 'point_valid': point_valid,
                'receiver': receiver, 'excluded': list(excluded or ())}
    chosen = best[1]
    return {'action': 'move', 'target_gF': chosen['gF'], 'chosen': chosen,
            'current': cur, 'branch': 'exploratory_min_step',
            'gain_Y': chosen['gain_Y'], 'gain_V': chosen['gain_V'],
            'receiver_gain': chosen['receiver_gain'],
            'epsilon_Y': epsY, 'epsilon_V': epsV, 'exploratory': True,
            'minimal': True, 'point_valid': point_valid, 'receiver': receiver,
            'supply_check_required': True,
            'branches': (['sensitivity_scenarios'] if point_valid
                         else ['supply_margin_only'])}


def select_target(table, current_gF, eps_Y, eps_V, receiver=None, exclude_gF=(),
                  exploratory=False):
    """The ONE selection rule of TASK 5.3 (normal branch).

    1. exclude the current band, the rejected bands, candidates with no real
       receiving increase and candidates that drop V by more than eps_V (the
       unit of V, never the probe's 1/s resolution);
    2. an admissible gain is ``dYhat >= eps_Y`` OR ``dVhat >= eps_V``, each in
       its own unit;
    3. among admissible candidates take the highest Yhat, ties inside eps_Y by
       the highest Vhat, then the smallest displacement, then ascending gF;
    4. a move is issued only when the chosen gain is STRICTLY positive: a zero
       gain never passes a zero epsilon.
    ``exploratory`` switches to the one-minimal-step branch.
    """
    cur = current_candidate(table, current_gF)
    if cur is None:
        return {'action': 'hold', 'reason': 'current_off_grid',
                'epsilon_Y': eps_Y, 'epsilon_V': eps_V}
    if not valid_epsilon(eps_Y) or not valid_epsilon(eps_V):
        # T014 3.2.4: an invalid comparison bound keeps the action UNKNOWN
        # (hold) and reports the untouched values instead of widening the
        # comparison.  ``epsilon_observed_must_remain`` is the raw input.
        return {'action': 'hold', 'reason': 'epsilon_invalid', 'current': cur,
                'epsilon_Y': eps_Y, 'epsilon_V': eps_V,
                'invalid_epsilon_Y': not valid_epsilon(eps_Y),
                'invalid_epsilon_V': not valid_epsilon(eps_V),
                'exploratory': bool(exploratory)}
    epsY = float(eps_Y)
    epsV = float(eps_V)
    excluded = tuple(exclude_gF or ())
    if exploratory:
        return _exploratory_target(table, cur, epsY, epsV, receiver, excluded)
    cur_v = cur['Vhat'] if cur['V_available'] else None
    pool = []
    for c in table['candidates']:
        if abs(c['gF'] - cur['gF']) < 1e-9:
            continue
        if c['gF'] in excluded:
            continue
        if not c.get('effective_increase', True):
            continue
        if not c.get('commitment_protected', True):
            continue
        if cur_v is not None and c['V_available'] and c['Vhat'] is not None and \
                c['Vhat'] < cur_v - epsV:
            continue
        gain_y = c['Yhat'] - cur['Yhat']
        gain_v = (None if (cur_v is None or not c['V_available'] or
                           c['Vhat'] is None) else c['Vhat'] - cur_v)
        if gain_y >= epsY or (gain_v is not None and gain_v >= epsV):
            rec = dict(c)
            rec['gain_Y'] = gain_y
            rec['gain_V'] = gain_v
            pool.append(rec)
    if not pool:
        return {'action': 'hold', 'reason': 'all_candidates_excluded',
                'current': cur, 'epsilon_Y': epsY, 'epsilon_V': epsV,
                'excluded': list(excluded), 'exploratory': False}
    top_y = max(c['Yhat'] for c in pool)
    near = [c for c in pool if c['Yhat'] >= top_y - epsY]
    chosen = sorted(near, key=lambda c: (
        -(c['Vhat'] if (c['V_available'] and c['Vhat'] is not None) else -1e18),
        c['distance'], c['gF']))[0]
    gain_y = chosen['Yhat'] - cur['Yhat']
    gain_v = (None if (cur_v is None or not chosen['V_available'] or
                       chosen['Vhat'] is None) else chosen['Vhat'] - cur_v)
    out = {'current': cur, 'chosen': chosen, 'branch': 'max_Yhat',
           'gain_Y': gain_y, 'gain_V': gain_v, 'epsilon_Y': epsY,
           'epsilon_V': epsV, 'exploratory': False, 'receiver': receiver}
    if not ((gain_y > 0.0) or (gain_v is not None and gain_v > 0.0)):
        out['action'] = 'hold'
        out['reason'] = 'benefit_below_epsilon_select'
        return out
    out['action'] = 'move'
    out['target_gF'] = chosen['gF']
    return out


def first_effective_step(current_gF, target_gF, table, basis='A_safe',
                         minimal=False):
    """Nearest valid step toward the target: never across several valid bands.

    For an Accurate receiver the guarded mu (N_safe) must strictly increase;
    Fast Full receivers use the same margin logic on the K_F model.  Equal
    mu neighbours (1.3 -> 1.2 in the REFERENCE fixture) are skipped, not sent.
    ``minimal=True`` (TASK 5.3 exploratory branch) allows exactly ONE adjacent
    grid step: when that step adds no receiving opportunity the exploratory
    branch holds instead of skipping ahead.
    """
    cur = current_candidate(table, current_gF)
    if cur is None:
        return None
    start = cur['gF']
    tgt = float(target_gF)
    if abs(tgt - start) < 1e-9:
        return None
    step_dir = 1.0 if tgt > start else -1.0
    g = round(start, 10)
    skipped = []
    while True:
        g = round(g + step_dir * GRID, 10)
        if g < AXIS_MIN_FAST - 1e-9 or g > AXIS_MAX_FAST + 1e-9:
            return None
        cand = None
        for c in table['candidates']:
            if abs(c['gF'] - g) < 1e-9:
                cand = c
                break
        if cand is None:
            return None
        if basis == 'A_safe':
            gain = cand['muA_safe_tasks_s'] - cur['muA_safe_tasks_s']
        else:
            gain = cand['K_F'] - cur['K_F']
        if gain > 1e-12:
            return {'target_gF': g, 'delta_fast': round(g - start, 10),
                    'skipped': skipped, 'gain': gain, 'candidate': cand,
                    'minimal': bool(minimal), 'basis': basis}
        if minimal:
            # exploratory: one minimal grid step only, never a skipped band
            return None
        skipped.append(g)
        if abs(g - tgt) < 1e-9:
            return None


# ---------------------------------------------------------------- probe maths


def standardized_probe(pre, post, T_pre_s, T_post_s, layers=ROOT_CLASSES):
    """Standardized mature-cohort rates (TASK.md section 6).

    ``pre``/``post`` map layer -> {'n': source roots, 'y': on-time GPU finals,
    'v': sum of on-time p_selected, 'l': resource-lost roots}.  lambda_bar_c =
    (n_pre+n_post)/(Tpre+Tpost); Ystd = sum lambda_bar * y/n.  Both sides empty is
    an IGNORED layer (it carries no evidence at all); exactly one empty side is
    not comparable and is never imputed with zero (REFERENCE standardized
    fixture).  ``T_pre_s``/``T_post_s`` are the REAL span lengths of each side,
    never a shared (T, T) (T014 s3.4.2).

    ``v_available`` is False when any compared layer has an on-time output with
    a missing public probability: the quality unit is then UNKNOWN (Y-only),
    never a fabricated 0 (T014 s3.4.4).  ``dLoss`` is the same standardisation
    applied to the per-class resource-loss rate (diagnostic + rollback input).
    """
    out = {'layers': {}, 'comparable': True, 'missing_layers': [],
           'ignored_layers': [], 'one_side_zero': False,
           'v_available': True}
    Ttot = float(T_pre_s) + float(T_post_s)
    Ystd_pre = Ystd_post = Vstd_pre = Vstd_post = 0.0
    Hstd_pre = Hstd_post = 0.0
    out['h_available'] = True
    loss_pre = loss_post = 0.0
    gpu_roots = 0
    for layer in layers:
        p = pre.get(layer) or {}
        q = post.get(layer) or {}
        npre = int(p.get('n') or 0)
        npost = int(q.get('n') or 0)
        gpu_roots += npre + npost
        if npre <= 0 and npost <= 0:
            out['layers'][layer] = {'n_pre': 0, 'n_post': 0, 'ignored': True}
            out['ignored_layers'].append(layer)
            continue
        if npre <= 0 or npost <= 0:
            out['comparable'] = False
            out['missing_layers'].append(layer)
            out['one_side_zero'] = True
            out['layers'][layer] = {'n_pre': npre, 'n_post': npost,
                                    'comparable': False, 'ignored': False}
            continue
        lam_bar = (npre + npost) / Ttot
        s_pre = float(p.get('y') or 0) / npre
        s_post = float(q.get('y') or 0) / npost
        v_av = bool(p.get('v_available', True)) and bool(q.get('v_available', True))
        v_pre = (float(p.get('v') or 0) / npre) if v_av else None
        v_post = (float(q.get('v') or 0) / npost) if v_av else None
        h_av = bool(p.get('h_available')) and bool(q.get('h_available'))
        h_pre = (float(p.get('h') or 0)/npre) if h_av else None
        h_post = (float(q.get('h') or 0)/npost) if h_av else None
        l_pre = float(p.get('l') or 0) / npre
        l_post = float(q.get('l') or 0) / npost
        out['layers'][layer] = {
            'n_pre': npre, 'n_post': npost, 'lambda_bar': lam_bar,
            's_pre': s_pre, 's_post': s_post, 'v_pre': v_pre, 'v_post': v_post,
            'y_pre': float(p.get('y') or 0), 'y_post': float(q.get('y') or 0),
            'v_sum_pre': float(p.get('v') or 0), 'v_sum_post': float(q.get('v') or 0),
            'v_available': v_av,
            'h_pre': h_pre, 'h_post': h_post, 'h_available': h_av,
            'loss_pre_n': int(p.get('l') or 0), 'loss_post_n': int(q.get('l') or 0),
            'loss_rate_pre': l_pre, 'loss_rate_post': l_post,
            'loss_rate_delta': lam_bar * (l_post - l_pre),
        }
        Ystd_pre += lam_bar * s_pre
        Ystd_post += lam_bar * s_post
        if v_av:
            Vstd_pre += lam_bar * v_pre
            Vstd_post += lam_bar * v_post
        else:
            out['v_available'] = False
        if h_av:
            Hstd_pre += lam_bar*h_pre
            Hstd_post += lam_bar*h_post
        else:
            out['h_available'] = False
        loss_pre += lam_bar * l_pre
        loss_post += lam_bar * l_post
    out.update({'Ystd_pre': Ystd_pre, 'Ystd_post': Ystd_post,
                'Vstd_pre': Vstd_pre, 'Vstd_post': Vstd_post,
                'loss_pre': loss_pre, 'loss_post': loss_post,
                'dLoss': loss_post - loss_pre,
                'T_pre_s': float(T_pre_s), 'T_post_s': float(T_post_s),
                'dY': Ystd_post - Ystd_pre, 'dV': Vstd_post - Vstd_pre,
                'Hstd_pre': Hstd_pre, 'Hstd_post': Hstd_post,
                'dH': Hstd_post-Hstd_pre if out['h_available'] else None,
                'gpu_roots': gpu_roots,
                'n_layers_compared': len(out['layers']) - len(out['ignored_layers'])})
    return out


def valid_epsilon(value):
    """T014 3.2.4: a comparison epsilon must be a finite, strictly positive
    number.

    ``True``/``False`` (a bool is a flag, not a magnitude), NaN, +-inf, a
    negative or zero value and a missing value are all INVALID.  An invalid
    epsilon must never be silently used: a negative epsilon would invert the
    comparison and admit every candidate, and a missing one would raise.  The
    caller has to fall back to the safe floor / keep instead.
    """
    if value is None or isinstance(value, bool):
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if v != v or v in (float('inf'), float('-inf')):
        return False
    return v > 0.0


def operational_verdict(dY, eps_Y, dV, eps_V, v_available, comparable,
                        dH=None, eps_H=None, h_available=False, require_speed=False,
                        require_correct_ontime=False):
    """The ONE post-probe verdict rule (TASK s6, T014 s3.4.5).

    A cost_version change and a donor loss-rate rise are diagnostic only.  The
    gain comparison is INCLUSIVE of its epsilon (``dY >= eps_Y``) while the
    bound itself is always strictly positive (see ``valid_epsilon``), so a
    zero-gain pair can never pass a zero epsilon.
    """
    if not comparable:
        return 'UNCERTAIN'
    if (dY <= -eps_Y) or (v_available and dV <= -eps_V):
        return 'ROLLBACK'
    if require_correct_ontime:
        # Public quality-weighted timely deliveries, not their earliness.
        # Unknown quality can never confirm a correctness improvement from Y alone.
        return 'CONFIRM' if v_available and dV >= eps_V else 'UNCERTAIN'
    if require_speed:
        if not h_available or dH is None or not valid_epsilon(eps_H):
            return 'UNCERTAIN'
        if dH <= -eps_H:
            return 'ROLLBACK'
        return 'CONFIRM' if dH >= eps_H else 'UNCERTAIN'
    if (dY >= eps_Y) or (v_available and dV >= eps_V):
        return 'CONFIRM'
    return 'UNCERTAIN'


def probe_epsilon(diff_series, T_s=1.0, healthy_min=HEALTHY_MIN_FOR_NOISE, resolution=1.):
    """eps = max(1/T, empirical P95 of <=32 same-config healthy comparable
    adjacent-window standardized output diffs); Y/V are kept separate."""
    vals = [abs(float(v)) for v in (diff_series or [])][-NOISE_WINDOW_LIMIT:]
    t_s = float(T_s) if T_s else 0.0
    floor = (float(resolution) / t_s) if t_s > 0 else float('inf')
    if len(vals) < int(healthy_min):
        return {'epsilon': floor, 'low_confidence': True, 'n_windows': len(vals)}
    p = absolute_p95(vals)
    eps = max(floor, p or 0.0)
    if not valid_epsilon(eps):
        # T014 3.2.4: an unusable/noise-less sample falls back to the floor,
        # never to a NaN/zero/infinite comparison bound.
        eps = floor if valid_epsilon(floor) else 1.0
    return {'epsilon': eps, 'low_confidence': False,
            'n_windows': len(vals)}


def probability_histogram_tv(probs_a, probs_b, width=0.1):
    """TV over a fixed 0.1-width histogram of the public quality probability."""
    from collections import Counter as _C

    def hist(probs):
        h = _C()
        for p in probs or []:
            v = parse_probability(p)
            if v is None:
                continue
            h[min(int(v / width), 9)] += 1
        return h
    return tv_distance(hist(probs_a), hist(probs_b))


# ------------------------------------------------------------------- factory


def make_controller_inputs_qf4(reference, cost_records, deadline_ns, uuid_order,
                               policy, start_ms=10_000.0, initial_fast=1.0,
                               folder=None, replay=None, worker=None,
                               cost_book=None):
    """Factory used by the Runtime BEFORE any actor work is accepted (s3.3).

    The complete per-card x kind price records (mean_wall_ms, gate_ms,
    sample_n, sample_revision, cost_version, measurement_source) are validated
    here; a list is never accepted where the dict contract exists.

    ``cost_book`` is the ONE CostBook of the arm, owned by the Runtime: when it
    is passed the controller never builds a private second book (TASK s3.4).
    ``replay`` / ``worker`` are the optional candidate-replay worker plumbing of
    TASK s5: the controller only ever CONSUMES ``view['replay']``.
    """
    ref = reference or {}
    configuration = ref.get('configuration') or {}
    errors = []
    if not isinstance(deadline_ns, int) or isinstance(deadline_ns, bool) or deadline_ns <= 0:
        errors.append('deadline_ns must be a positive integer')
    if not isinstance(uuid_order, (list, tuple)) or len(uuid_order) != CARD_COUNT or len(set(uuid_order)) != CARD_COUNT:
        errors.append('uuid_order must list every configured GPU UUID exactly once')
    if policy not in POLICIES:
        errors.append('policy %r not in %r' % (policy, POLICIES))
    if configuration.get('controller_version') not in (None, CONTROLLER_VERSION):
        errors.append('reference controller_version is not %s' % CONTROLLER_VERSION)
    errors.extend(validate_cost_records(cost_records))
    if errors:
        raise ValueError('qf4 factory validation failed: ' + '; '.join(errors))
    cfg = {
        'controller_version': CONTROLLER_VERSION,
        'implementation_revision': IMPLEMENTATION_REVISION,
        'algorithm_revision': configuration.get('algorithm_revision', 'QF-4'),
        'experiment_boundary': configuration.get(
            'experiment_boundary', 'shared_local_frontend_gpu_control'),
        'policy': str(policy),
        'delivery_objective': configuration.get('delivery_objective', 'latency_timeout'),
        'policies': list(POLICIES),
        'c_ref_ms': {str(k): float(v) for k, v in (ref.get('c_ref_ms') or {}).items()},
        'cost_records': {int(g): {k: dict(v) for k, v in row.items()}
                         for g, row in cost_records.items()},
        'deadline_ns': int(deadline_ns),
        'uuid_order': [str(u) for u in uuid_order],
        'start_ms': float(start_ms),
        'start_ns': int(round(float(start_ms) * MS)),
        'initial_fast': float(initial_fast),
        'axis_min_fast': float(configuration.get('axis_min_fast', AXIS_MIN_FAST)),
        'axis_max_fast': float(configuration.get('axis_max_fast', AXIS_MAX_FAST)),
        'grid': float(configuration.get('grid', GRID)),
        'minimum_fast_share': float(configuration.get('minimum_fast_share', configuration.get('minimum_role_share', GRID))),
        'minimum_accurate_share': float(configuration.get('minimum_accurate_share', configuration.get('minimum_role_share', GRID))),
        'allocation_hysteresis_floor': float(configuration.get('allocation_hysteresis_floor', GRID)),
        'period_ms': float(configuration.get('period_ms', PERIOD_MS)),
        'window_ms': float(configuration.get('window_ms', WINDOW_MS)),
        'tick_ms': float(configuration.get('tick_ms', TICK_MS)),
        'probe_ms': float(configuration.get('probe_ms', PROBE_MS)),
        'probe_extended_ms': float(configuration.get('probe_extended_ms', PROBE_EXTENDED_MS)),
        # T014 algorithm: the post span ladder [500, 1000, 2000] ms and its
        # extension budget (probe_post_levels_ms / probe_max_extensions).
        'probe_post_levels_ms': list(configuration.get('probe_post_levels_ms') or []),
        'probe_max_extensions': int(configuration.get('probe_max_extensions', 2)),
        'root_estimation_ms': float(configuration.get('root_estimation_ms', ROOT_ESTIMATION_MS)),
        'stall_lag_p95_ms': float(configuration.get('stall_lag_p95_ms', 12.5)),
        'min_price_opportunity_margin_ms': float(
            configuration.get('min_price_opportunity_margin_ms', 0.1)),
        'transition_timeout_ms': float(configuration.get('transition_timeout_ms', 2000.0)),
        'decision_budget_ns': int(configuration.get('decision_budget_ns', 800_000)),
        # TASK section 6 aligned probe clock (REFERENCE.algorithm).
        'settle_ms': float(configuration.get('settle_ms', SETTLE_MS)),
        'probe_pre_ms': float(configuration.get('probe_pre_ms', 2000.0)),
        'probe_first_post_ms': float(configuration.get(
            'probe_first_post_ms', PROBE_FIRST_POST_MS)),
        'probe_extended_post_ms': float(configuration.get(
            'probe_extended_post_ms', PROBE_EXTENDED_POST_MS)),
        'probe_post_alignment_ms': float(configuration.get(
            'probe_post_alignment_ms', PROBE_POST_ALIGN_MS)),
        'candidate_interval_ms': float(configuration.get('candidate_interval_ms', 500.0)),
        'cache_max_input_age_ms': float(configuration.get(
            'cache_max_input_age_ms', CACHE_MAX_INPUT_AGE_MS)),
        'tprofile_s': float(configuration.get('source_profile_ms', 2000.0)) / 1000.0,
        'min_live_cost_samples': int(configuration.get('min_live_cost_samples', 30)),
        'cost_update_ms': float(configuration.get('cost_update_ms', COST_UPDATE_MS)),
        'lambda_change_invalidate_rel': float(configuration.get(
            'lambda_change_invalidate_rel', LAMBDA_CHANGE_INVALIDATE_REL)),
        'source_composition_TV_limit': float(configuration.get(
            'source_composition_TV_limit', SOURCE_COMPOSITION_TV_LIMIT)),
        'price_change_invalidate_rel': float(configuration.get(
            'price_change_invalidate_rel', 0.1)),
        # ONE book for the whole arm; the controller never builds a second one.
        'cost_book': cost_book,
        'replay': replay,
        'worker': worker,
        'controller_private_inputs_allowed': False,
        'folder': str(folder) if folder else None,
    }
    return cfg


def make_executor_view_qf4(now_ns, cards, heads, cost_records, deadline_ns,
                           pinned=None, waiting=None, waiting_work_ms=None,
                           rejected_profiles=None, cost_version=0,
                           receipt_lag_p95_ns=None, done_lag_p95_ns=None,
                           stall=False, idle_ns=None, idle_horizon_ns=None,
                           watermark_ns=None, frame_gap_seen=False,
                           decision_budget_ns=800_000, gate_probe=None,
                           snapshot_ns=None, replay=None, worker=None):
    """The Runtime's single public view for one light tick (TASK.md s3.3).

    Carries the real snapshot_ns, per-card spent/busy/pending/pinned,
    effective/target/transition, the per-card mean/gate records and the recent
    rejected profiles.  Every light and full decision receives this dict.

    ``replay`` is the candidate-replay worker RESULT (interface s6.5); the
    controller consumes it and never asks the worker itself.  A missing or
    invalid result is reported as ``prediction_unknown``.
    """
    return {
        'now_ns': int(now_ns),
        'snapshot_ns': int(snapshot_ns if snapshot_ns is not None else now_ns),
        'deadline_ns': int(deadline_ns),
        'period_ns': PERIOD_NS,
        'cards': [dict(c) for c in cards],
        'cost_records': {int(g): {k: dict(v) for k, v in row.items()}
                         for g, row in (cost_records or {}).items()},
        'cost_version': int(cost_version),
        'heads': {'Fast': dict(heads['Fast']) if heads.get('Fast') else None,
                  'Accurate': dict(heads['Accurate']) if heads.get('Accurate') else None},
        'pinned': [dict(p) for p in (pinned or [])],
        'waiting': {r: dict((waiting or {}).get(r) or {}) for r in GPU_ROLES},
        'waiting_work_ms': dict(waiting_work_ms or {}),
        'rejected_profiles': dict(rejected_profiles or {}),
        'receipt_lag_p95_ns': receipt_lag_p95_ns,
        'done_lag_p95_ns': done_lag_p95_ns,
        'stall': bool(stall),
        'idle_ns': dict(idle_ns or {}),
        'idle_horizon_ns': int(idle_horizon_ns or (2 * WINDOW_NS)),
        'watermark_ns': watermark_ns,
        'frame_gap_seen': bool(frame_gap_seen),
        'decision_budget_ns': int(decision_budget_ns),
        'gate_probe': gate_probe,
        'replay': replay,
        'worker': worker,
        'replay_required': worker is not None,
    }


# ----------------------------------------------------------------- controller


class QueueFeedbackController(object):
    """QF-4 decision controller.  One instance per two-GPU arm."""

    def __init__(self, config):
        self.config = dict(config)
        self.policy = str(self.config.get('policy', DYNAMIC_POLICY))
        self.dynamic = (self.policy != FIXED_POLICY)
        # Current resource control computes one demand target from observed
        # work/queue trends on an adaptive timer. Historical post-probe/return
        # policies remain available only for historical objectives.
        self.continuous_resource = self.config.get('delivery_objective', 'latency_timeout') == 'latency_timeout'
        # T016 C2 (opt-in, env `T016_HOLD_ON_UNCERTAIN`): hold a moved share
        # through an UNCERTAIN probe instead of snapping back to last_confirmed.
        # An unchanged reference keeps the default (return), so the identity
        # checks and every historical run are untouched.
        self.hold_on_uncertain = _env_flag('T016_HOLD_ON_UNCERTAIN')
        # T016 C3 (opt-in, env `T016_SPEED_CHANNEL_REQUIRES_HEALTH`): the
        # no-pressure speed channel (H) may only fire when NEITHER role shows any
        # measured resource loss.  Without it "no_persistent_gap" (the double
        # window persistence test merely lapsed) opened an H-driven move toward
        # Fast even while both sides were losing work.
        self.speed_channel_requires_health = _env_flag('T016_SPEED_CHANNEL_REQUIRES_HEALTH')
        self.last_held = None
        self.held_review = None
        self.axis_min = float(self.config.get('axis_min_fast', AXIS_MIN_FAST))
        self.axis_max = float(self.config.get('axis_max_fast', AXIS_MAX_FAST))
        self.grid = float(self.config.get('grid', GRID))
        # Each role has its own reservation, independent of grid precision.
        # Historical configurations may still supply the symmetric key.
        legacy_minimum = self.config.get('minimum_role_share', GRID)
        self.minimum_fast_share = float(self.config.get('minimum_fast_share', legacy_minimum))
        self.minimum_accurate_share = float(self.config.get('minimum_accurate_share', legacy_minimum))
        # Note: expressing allocations more precisely must not
        # lower the pre-existing noise/change threshold. Never round noise.
        self.allocation_hysteresis_floor = float(self.config.get('allocation_hysteresis_floor', GRID))
        self.grid_points = [round(self.axis_min + i * self.grid, 10)
                            for i in range(int(round(
                                (self.axis_max - self.axis_min) / self.grid)) + 1)]
        self.period_ns = int(round(float(self.config.get('period_ms', PERIOD_MS)) * MS))
        self.tick_ns = int(round(float(self.config.get('tick_ms', TICK_MS)) * MS))
        self.window_ns = int(round(float(self.config.get('window_ms', WINDOW_MS)) * MS))
        self.probe_ns = int(round(float(self.config.get('probe_ms', PROBE_MS)) * MS))
        self.probe_extended_ns = int(round(
            float(self.config.get('probe_extended_ms', PROBE_EXTENDED_MS)) * MS))
        # T014: the post span is a LADDER (probe_post_levels_ms) with a bounded
        # number of extensions; the last level is the maximum post span that
        # the alignment fixture pins (post_start + 2000 ms).
        levels = self.config.get('probe_post_levels_ms') or []
        self.probe_post_levels_ns = [int(round(float(x) * MS)) for x in levels
                                     if float(x) > 0.0] or \
            [int(PROBE_MS * MS), self.probe_extended_ns]
        self.probe_max_extensions = int(self.config.get('probe_max_extensions', 2))
        self.deadline_ns = int(self.config.get('deadline_ns', 50 * MS))
        self.transition_timeout_ns = int(
            max(float(self.config.get('transition_timeout_ms', 2000.0)) * MS,
                4 * self.deadline_ns))
        self.min_margin_ms = float(self.config.get('min_price_opportunity_margin_ms', 0.1))
        self.start_ns = (0 if self.continuous_resource else int(self.config.get('start_ns')))
        self.direct_stable_ns = 0
        self.direct_samples = deque()
        self.direct_last_calc_ns = None
        self.direct_interval_ns = self.tick_ns
        self.direct_offered_anchor = None
        self.direct_demand_evidence = None
        self.direct_timing = None
        self.decision_budget_ns = int(self.config.get('decision_budget_ns', 800_000))
        self.uuid_order = list(self.config.get('uuid_order') or [])
        self.folder = self.config.get('folder')
        self.dual_clock = None
        self.dual_clock_fallback_reason = None
        dual_mode = __import__('os').environ.get('EDGE_DUAL_CLOCK_MODE', 'OFF').upper()
        if dual_mode != 'OFF':
            from dual_clock_state import DualClockGovernor
            try:
                self.dual_clock = DualClockGovernor(dual_mode, self.folder)
            except (ValueError, TypeError, KeyError, ArithmeticError, FileNotFoundError) as exc:
                self.dual_clock_fallback_reason = 'initialization_fallback:'+str(exc)
        # TASK section 6: the aligned 500 ms probe clock (REFERENCE.algorithm).
        self.settle_ns = int(round(float(self.config.get('settle_ms', SETTLE_MS)) * MS))
        self.post_align_ns = int(round(float(self.config.get(
            'probe_post_alignment_ms', PROBE_POST_ALIGN_MS)) * MS))
        self.probe_first_post_ns = int(round(float(self.config.get(
            'probe_first_post_ms', PROBE_FIRST_POST_MS)) * MS))
        self.probe_pre_ns = int(round(float(self.config.get(
            'probe_pre_ms', 2000.0)) * MS))
        self.tprofile_s = float(self.config.get('tprofile_s', TPROFILE_S))
        # Two adjacent intervals must fit in the existing public retention.
        self.direct_max_interval_ns = max(self.tick_ns,
            int(self.tprofile_s * NS_PER_S / (2*self.tick_ns)) * self.tick_ns)
        self.direct_history_ns = 2*self.direct_max_interval_ns
        # Demand memory is independent of the adaptive decision interval.
        # Queue slopes, in contrast, must describe only the current allocation.
        self.direct_trend_ns = 2*self.tick_ns
        self.cost_sample_cap = int(self.config.get('cost_sample_limit', 200))
        self.min_live_cost_samples = int(self.config.get(
            'min_live_cost_samples', MIN_LIVE_COST_SAMPLES))
        self.cost_update_ns = int(round(float(self.config.get(
            'cost_update_ms', COST_UPDATE_MS)) * MS))
        self.cache_max_age_ns = int(round(float(self.config.get(
            'cache_max_input_age_ms', CACHE_MAX_INPUT_AGE_MS)) * MS))
        self.lambda_invalidate_rel = float(self.config.get(
            'lambda_change_invalidate_rel', LAMBDA_CHANGE_INVALIDATE_REL))
        self.composition_tv_limit = float(self.config.get(
            'source_composition_TV_limit', SOURCE_COMPOSITION_TV_LIMIT))
        # TASK section 3.4: ONE CostBook for the whole arm.  When the Runtime
        # passes its book the controller reuses it and never builds a second.
        self.cost_book = self.config.get('cost_book') or CostBook(
            cap=self.cost_sample_cap)
        stats_cfg = {
            'window_ms': 500.0, 'sub_window_ms': 50.0, 'ring_windows': 32,
            'deadline_ns': self.deadline_ns,
            'stall_lag_p95_ms': self.config.get('stall_lag_p95_ms', 12.5),
            'cost_book': self.cost_book,
        }
        # The longest noise pair needs BOTH spans plus the unsealed current
        # bucket and one deadline-maturity bucket. Eight inspected buckets
        # cannot supply a 2 s + 2 s pair when the current bucket is present.
        noise_span_ns = self.probe_pre_ns + max(self.probe_post_levels_ns)
        stats_window_ns = int(stats_cfg['window_ms'] * MS)
        stats_cfg['cohort_lookback'] = max(8,
            (noise_span_ns + stats_window_ns - 1) // stats_window_ns + 2)
        if self.config.get('c_ref_ms'):
            stats_cfg['c_ref_ms'] = self.config['c_ref_ms']
        self.observer = QF4Observer(stats_cfg)
        if self.config.get('c_ref_ms'):
            self.observer.freeze_c_star(self.config['c_ref_ms'])
        self.observer.cost_book = self.cost_book
        # TASK section 5: the controller only CONSUMES the worker result.
        self.replay_worker = self.config.get('worker') or self.config.get('replay')
        self.table_id = 0
        self.last_table_id = 0
        self.gain_history = OrderedDict()
        self.gain_history_max = 16
        self._last_decision_row = None
        self.reset()

    # -- lifecycle ----------------------------------------------------------
    def reset(self):
        self.state = 'IDLE'
        self.target_fast = float(self.config.get('initial_fast', 1.0))
        self.effective_fast = float(self.config.get('initial_fast', 1.0))
        self.confirmed_fast = float(self.config.get('initial_fast', 1.0))
        self.epoch = 0
        self.command_seq = 0
        self.commands = []
        # TASK section 6: this tick's NEW commands only (history stays in
        # ``commands``); ``_issue`` appends to both.
        self.new_commands_this_tick = []
        self.new_command_count = 0
        self.pending = None
        self.pending_command = None
        self.apply_count = 0
        self.config_request_count = 0
        self.tick_index = 0
        self.table_id = 0
        self.last_table_id = 0
        self.gain_history = OrderedDict()
        self.gain_history_max = 16
        self._last_decision_row = None
        # TASK section 4: per-role window consumption, advanced AFTER both
        # directions were evaluated in the same tick.
        self.last_consumed_index = {r: None for r in GPU_ROLES}
        self.last_scored_window = None
        self.last_resource_table = None
        self.last_trigger = {r: None for r in GPU_ROLES}
        self.triggers = {r: None for r in GPU_ROLES}
        self.counters = Counter()
        self.decisions = deque(maxlen=4096)
        self.last_reason = 'init'
        # one lock per DIRECTION ('A'/'F'): a failed A edge never locks F
        self.locked_edges = {}
        self.trials_by_direction = Counter()
        self.trial_context = {}
        self.last_trial_renewal = None
        self.noise_seed_version = None
        # TASK section 6: separate Y and V standardized output-noise pairs,
        # held per SPAN BANK ((T_pre_s, T_post_s)); a 0.5 s and a 1 s post span
        # are never mixed into one epsilon series (T014 s3.4.5).
        self.noise_windows = {'Y': deque(maxlen=OUTPUT_NOISE_MAX_PAIRS),
                              'V': deque(maxlen=OUTPUT_NOISE_MAX_PAIRS)}
        self.noise_banks = {key: {'Y': deque(maxlen=OUTPUT_NOISE_MAX_PAIRS),
                                  'V': deque(maxlen=OUTPUT_NOISE_MAX_PAIRS),
                                  'H': deque(maxlen=OUTPUT_NOISE_MAX_PAIRS)}
                            for key in ('%.1f|%.1f' % (self.probe_pre_ns / NS_PER_S,
                                                      span / NS_PER_S)
                                        for span in self.probe_post_levels_ns)}
        # TASK section 5.3: same-epoch candidate gain drifts, per unit.
        self.profile_series = {'Y': deque(maxlen=SELECT_NOISE_MAX_PROFILES),
                               'V': deque(maxlen=SELECT_NOISE_MAX_PROFILES)}
        self.eps_select = {'Y': select_epsilon((), self.tprofile_s),
                           'V': select_epsilon((), self.tprofile_s)}
        self.last_profile = None
        self.demand_cache = None
        self.last_table = None
        self.last_candidates = None
        self.last_selection = None
        self.last_score = None
        self.last_shadow = None
        self.last_probe = None
        self.last_replay = None
        self.last_budget_basis = None
        self.last_confirmed = {'gF': float(self.effective_fast), 'ns': None,
                               'epoch': 0, 'command_id': None}
        self.last_held = None
        self.failed_edges = []
        # TASK section 6: provenance of the last output-noise pair + the start
        # of the current effective epoch (same-epoch noise/window filter).
        self.last_noise_pair = None
        self.last_noise_source = None
        self.epoch_start_ns = None
        self.last_windows = []
        self._lock_reference = None

    # -- event intake -------------------------------------------------------
    def observe(self, event_type, kw, observed_ns):
        try:
            self.observer.advance_to(observed_ns)
        except Exception:
            self.counters['advance_errors'] += 1

    # -- cost book seeding ----------------------------------------------------
    def seed_costs(self, cost_records, now_ns):
        """Validate + seed the measured warmup records into the runtime book."""
        errors = validate_cost_records(cost_records)
        if errors:
            raise ValueError('seed_costs validation failed: ' + '; '.join(errors))
        for g in CARD_IDS:
            for kind in GPU_KINDS:
                rec = cost_records[g][kind]
                book = self.cost_book.record_obj(g, kind)
                book.gate_ns = float(rec['gate_ms']) * MS
                book.mean_ns = float(rec['mean_wall_ms']) * MS
                book.floor_ns = float(rec.get('floor_ms') or rec['gate_ms']) * MS
                book.delta_ns = float(rec.get('delta_ms') or 0.0) * MS
                book.n = int(rec.get('sample_n') or 0)
                book.sample_revision = int(rec.get('sample_revision') or 0)
                book.cost_version = int(rec.get('cost_version') or 0)
                book.measurement_source = str(rec.get('measurement_source'))
                book.last_update_ns = int(now_ns)
                book.history.append((book.gate_ns, book.mean_ns))
        self.observer.cost_book = self.cost_book
        if not self.observer.c_star_ms:
            self.observer.freeze_c_star(self.cost_book.reference_c_star_ms())
        return dict(self.observer.c_star_ms)

    # -- helpers ---------------------------------------------------------------
    def _cost_view(self, view):
        out = {}
        for g in CARD_IDS:
            row = {}
            for kind in GPU_KINDS:
                rec = ((view or {}).get('cost_records') or {}).get(g, {}).get(kind)
                if rec is None:
                    rec = self.cost_book.record_obj(g, kind).record()
                rec = dict(rec)
                rec['price_history'] = self.cost_book.record_obj(g, kind).price_history_ms()
                row[kind] = rec
            out[g] = row
        return out

    def _bootstrap_costs(self, view):
        """TASK section 3.4: card/kinds with ZERO real samples are bootstrap.

        A record only counts as bootstrap when it says so explicitly, or when
        it carries a sample count of zero -- an unknown record is never
        assumed to be cheap.
        """
        out = {}
        cost = (view or {}).get('cost_records') or {}
        for g in CARD_IDS:
            for kind in GPU_KINDS:
                rec = ((cost.get(g) or {}).get(kind)) or {}
                boot = bool(rec.get('bootstrap'))
                if 'sample_n' in rec and int(rec.get('sample_n') or 0) <= 0:
                    boot = True
                if boot:
                    out.setdefault(g, []).append(kind)
        return out

    def _budget_ok_factory(self, view):
        """Zero-wait whole-chain budget judge with the CURRENT gates (s5).

        Uses the stage-publish remaining stored at root publication; never a
        consumer-delayed remaining.  Missing prices count as unknown, an
        unrevealed Prefix continuation is never dropped by assumption.
        """
        cost = (view or {}).get('cost_records')
        head_ns = int((view or {}).get('decision_budget_ns') or
                      self.decision_budget_ns)
        basis = {}

        def gate(card, kind):
            rec = ((cost or {}).get(card) or {}).get(kind)
            if not rec or rec.get('gate_ms') is None:
                return None
            return float(rec['gate_ms']) * MS

        def warmup_floor(card, kind):
            """Best-case start price of a card that has not served this kind
            yet: its warmup floor.  REFERENCE counter-example: source
            remaining 10 ms, current A gate 16.19471175 ms, standby card
            warmup gate 4.5899131 ms -> the standby card supports the root."""
            rec = ((cost or {}).get(card) or {}).get(kind)
            if rec is not None and rec.get('floor_ms') is not None:
                return float(rec['floor_ms']) * MS
            try:
                fl = self.cost_book.floor_ns(card, kind)
            except Exception:
                fl = None
            return None if fl is None else float(fl)

        def axis_min_gate(kind, f_side):
            """Cheapest zero-wait start of ``kind`` over BOTH cards: cards on
            that axis use their measured gate, the other card is judged with
            its warmup floor because it can be re-targeted."""
            best = None
            cards = (view or {}).get('cards') or []
            for g in CARD_IDS:
                share = cards[g].get('effective', 1.0) if g < len(cards) else 1.0
                on_axis = (share > 1e-12) if f_side else (share < 1.0 - 1e-12)
                v = gate(g, kind) if on_axis else None
                src = 'card%d:measured_gate' % g
                if v is None:
                    v = warmup_floor(g, kind)
                    src = 'card%d:%s' % (
                        g, 'warmup_floor' if on_axis else 'warmup_floor_off_axis')
                if v is not None and (best is None or v < best[0]):
                    best = (v, src)
            if best is not None:
                basis[kind] = best[1]
            return None if best is None else best[0]

        def ok(root):
            rem = root.get('stage_rem_ns')
            if rem is None:
                return None
            rem = float(rem)
            cls = root['entry_class']
            if cls == ROOT_DIRECT_F:
                v = axis_min_gate('FAST_FULL', True)
            elif cls == ROOT_DIRECT_A:
                v = axis_min_gate('ACCURATE', False)
            else:
                rev = root.get('revealed_early') or {}
                role = rev.get('role')
                pcard = root.get('prefix_card', 0)
                if role == 'Fast':
                    v_p = gate(pcard, 'FAST_PREFIX')
                    v_s = gate(rev.get('suffix_card', pcard), 'FAST_SUFFIX')
                    v = None if (v_p is None or v_s is None) else v_p + head_ns + v_s
                elif role == 'Accurate':
                    v_p = gate(pcard, 'FAST_PREFIX')
                    v_a = axis_min_gate('ACCURATE', False)
                    v = None if (v_p is None or v_a is None) else v_p + head_ns + v_a
                else:
                    return None  # unrevealed: unknown, never dropped by guess
            if v is None:
                return None
            return bool(rem >= v)
        self.last_budget_basis = basis
        return ok

    def _current_lambda(self, view, now_ns):
        # Selection uses raw source demand and the deadline replay.  The old
        # per-root gate-based support classifier is diagnostic, and must not
        # spend the allocation loop's budget or filter its source demand.
        demand = self.observer.demand_inputs(include_support=False)
        return demand

    def _high_coverage(self, demand):
        early = (demand or {}).get('EARLY') or {}
        lam_B = float((demand or {}).get('lambda_raw', {}).get(ROOT_PREFIX, 0.0))
        samples = int(early.get('EARLY_samples') or 0)
        coverage = early.get('coverage')
        if lam_B <= 0:
            # direct-only: no EARLY coverage needed when the direct windows are
            # complete and the direct probabilities are available (TASK s5)
            ok = bool(demand and not demand.get('incomplete') and
                      (early.get('pDF_quality') is not None or
                       float((demand.get('lambda_raw') or {}).get(ROOT_DIRECT_F, 0.0)) <= 0) and
                      (early.get('pDA_quality') is not None or
                       float((demand.get('lambda_raw') or {}).get(ROOT_DIRECT_A, 0.0)) <= 0))
            return {'high': ok, 'reason': 'no_prefix_demand' if ok
                    else 'direct_probability_missing'}
        if samples >= EARLY_MIN_SAMPLES and coverage is not None and \
                coverage >= EARLY_MIN_COVERAGE:
            return {'high': True, 'reason': 'early_coverage_ok'}
        return {'high': False, 'reason': 'coverage_insufficient',
                'samples': samples, 'coverage': coverage}

    # -- TASK section 4: unified gap, both directions -------------------------
    def _window_epoch(self, w):
        """Observer helper if present, else the window's own epoch record."""
        fn = getattr(self.observer, 'window_epoch', None)
        if fn is not None:
            try:
                ep = fn(w)
            except Exception:
                self.counters['window_epoch_errors'] += 1
                ep = None
            if ep is not None:
                return int(ep)
        flags = w.get('flags') or {}
        if flags.get('epoch') is not None:
            return int(flags['epoch'])
        if w.get('effective_epoch') is not None:
            return int(w['effective_epoch'])
        return None

    def _resource_root_loss(self, w, role):
        """Deduplicated resource-lost ROOT COUNT of one window and role.

        Uses the frozen observer API when present and otherwise the window's
        own ``lost_roots`` set; both are root-deduplicated, never a raw
        request counter.
        """
        fn = getattr(self.observer, 'resource_root_loss', None)
        if fn is not None:
            try:
                n = fn(w, role)
            except Exception:
                self.counters['resource_root_loss_errors'] += 1
                n = None
            if n is not None:
                if isinstance(n, (int, float)):
                    return int(n)
                return len(set(n))
        return len(self._loss_root_ids(w, role))

    @staticmethod
    def _loss_root_ids(w, role):
        """The raw root-identity set of one window and role."""
        return set(((w.get('lost_roots') or {}).get(role)) or ())

    def _theta_growth(self, role, c_role_ms):
        """theta_r = max(P95(|healthy growth|), c_role/0.5 s); <8 healthy
        same-epoch windows -> one job resolution + low_confidence."""
        try:
            series = self.observer.healthy_window_growth(
                role, NOISE_WINDOW_LIMIT, epoch=self.epoch)
        except TypeError:
            series = self.observer.healthy_window_growth(role, NOISE_WINDOW_LIMIT)
        one_work = float(c_role_ms) / 0.5  # one reference job per 0.5 s
        if len(series) < HEALTHY_MIN_FOR_NOISE:
            return {'theta': one_work, 'low_confidence': True, 'n': len(series),
                    'basis': 'one_unit_resolution', 'epoch': self.epoch}
        p = absolute_p95(series)
        return {'theta': max(p or 0.0, one_work), 'low_confidence': False,
                'n': len(series), 'basis': 'healthy_p95_and_one_unit',
                'epoch': self.epoch}

    def _role_pressure(self, role, sealed, trig_role, c_role_ms):
        """Resource pressure of one role over its two fresh trigger windows.

        deficit_ms_s = mean over the role's fresh windows of
            max(0.0, growth_ms_s) + X_resource_ms_s + R_resource_ms_s
        i.e. positive standing-queue growth PLUS the work lost or refused because
        of resource shortage.  Those last two fields exist in every sealed window
        (queue_feedback_stats.py:987-992) but the trigger never reads them, so a
        role whose queue is NOT growing yet which loses roots to resource
        shortage (the t=73.5 Accurate case: growth=-13.49, loss=53.95 ms/s) is
        pressureless by growth alone and pressured once the loss terms count.
        steps_needed = ceil(deficit_ms_s / SERVICE_MS_S_PER_STEP), at least 1.
        """
        marks = list((trig_role or {}).get('windows') or ())
        by_index = {}
        for w in sealed or ():
            try:
                by_index[int(w['index'])] = w
            except (TypeError, KeyError, ValueError):
                continue
        per_window = []
        for m in marks:
            try:
                w = by_index.get(int(m['window_index']))
            except (TypeError, KeyError, ValueError):
                w = None
            res = ((w or {}).get('residuals') or {}).get(role) or {}
            growth = float(res.get('growth_ms_s') or 0.0)
            x = float(res.get('X_resource_ms_s') or 0.0)
            r = float(res.get('R_resource_ms_s') or 0.0)
            per_window.append(max(0.0, growth) + x + r)
        deficit_ms_s = (sum(per_window) / float(len(per_window))) if per_window else 0.0
        steps_needed = max(1, int(math.ceil(deficit_ms_s / SERVICE_MS_S_PER_STEP)))
        pressured = bool((trig_role or {}).get('trigger')
                         and (trig_role or {}).get('reason') == 'persistent_gap')
        return {'role': role, 'pressured': pressured, 'deficit_ms_s': deficit_ms_s,
                'steps_needed': steps_needed, 'windows': [m.get('window_index') for m in marks],
                'resolved_windows': len(per_window), 'basis': 'growth_pos_plus_X_plus_R'}

    def _self_test_pressure(self):
        """Offline (no observer, no GPU) invariants of the v19 pressure law.

        Direction comes from resource pressure, magnitude from the measured
        deficit.  These assertions fail ``self_check`` if either the pressure
        arithmetic or the deficit-driven admission regresses.
        """
        checks = {}

        def _sealed(r_loss):
            res = {'growth_ms_s': -13.4862766875,
                   'X_resource_ms_s': 13.4862766875,
                   'R_resource_ms_s': r_loss}
            return [{'index': 145, 'residuals': {'Accurate': dict(res)}},
                    {'index': 146, 'residuals': {'Accurate': dict(res)}}]

        marks = [{'window_index': 145}, {'window_index': 146}]
        trig = {'trigger': True, 'reason': 'persistent_gap', 'windows': marks}
        p1 = self._role_pressure('Accurate', _sealed(40.4588300625), trig, 8.0)
        checks['deficit_53_steps_1'] = bool(
            p1['pressured'] and abs(p1['deficit_ms_s'] - 53.94510675) < 1e-9
            and p1['steps_needed'] == 1 and p1['resolved_windows'] == 2)
        p2 = self._role_pressure('Accurate', _sealed(240.0), trig, 8.0)
        checks['deficit_253_steps_3'] = bool(
            p2['pressured'] and abs(p2['deficit_ms_s'] - 253.4862766875) < 1e-9
            and p2['steps_needed'] == 3)
        healthy = [{'index': 145, 'residuals': {'Accurate': {
            'growth_ms_s': 0.0, 'X_resource_ms_s': 0.0, 'R_resource_ms_s': 0.0}}},
            {'index': 146, 'residuals': {'Accurate': {
                'growth_ms_s': 0.0, 'X_resource_ms_s': 0.0,
                'R_resource_ms_s': 0.0}}}]
        p3 = self._role_pressure('Accurate', healthy, {
            'trigger': False, 'reason': 'no_persistent_gap', 'windows': marks}, 8.0)
        checks['healthy_zero'] = bool(
            not p3['pressured'] and p3['deficit_ms_s'] == 0.0
            and p3['steps_needed'] == 1)

        # A deficit does not authorize a zero-benefit transfer.
        cur = {'gF': 1.0, 'quality_earliness_hat': 0.0}

        def _row(g):
            return {'gF': g, 'epsilon_Y': 1.0, 'epsilon_V': 1.0,
                    'net_gain_Yhat': 0.0, 'net_gain_Vhat': 0.0,
                    'V_available': True, 'net_gain_quality_earliness': 0.0,
                    'point_valid': True}

        rows = [_row(1.1), _row(1.2)]
        none_step, _ = self._replay_minimal_effective_step(
            rows, cur, 'F', False, deficit_steps=None)
        two_step, _ = self._replay_minimal_effective_step(
            rows, cur, 'F', False, deficit_steps=2)
        checks['pressure_requires_net_benefit'] = bool(
            none_step is None and two_step is None)

        # A step that cannot reach the deficit must still move -- to the axis
        # bound -- and say so.
        bound_step, _ = self._replay_minimal_effective_step(
            rows, cur, 'F', False, deficit_steps=3)
        checks['axis_bound_does_not_invent_benefit'] = bound_step is None

        # A historical (deadline_output) table with Vhat absent must neither
        # crash nor lose the deficit-driven magnitude.
        def _hist_row(g):
            return {'gF': g, 'epsilon_Y': 1.0, 'epsilon_V': 1.0,
                    'net_gain_Yhat': 0.0, 'V_available': False,
                    'point_valid': True, 'is_effective_step': False}

        hist_rows = [_hist_row(1.1), _hist_row(1.2)]
        hist_cur = {'gF': 1.0}
        hist_open, _ = self._replay_minimal_effective_step(
            hist_rows, hist_cur, 'F', False, deficit_steps=None)
        hist_step, _ = self._replay_minimal_effective_step(
            hist_rows, hist_cur, 'F', False, deficit_steps=2)
        checks['deficit_magnitude_on_historical_table'] = bool(
            hist_open is None and hist_step is not None
            and abs(hist_step['gF'] - 1.2) < 1e-9)
        return {'ok': bool(all(checks.values())), 'checks': checks}

    def _trigger_role(self, role, c_role_ms):
        """Two consecutive NEW 500 ms windows, each with growth > theta_r OR
        at least one complete root's attributable resource loss.  A loss is
        NOT cancelled by expiry cleanup, and an isolated single-window burst
        never triggers.  Window consumption happens in the caller, AFTER both
        directions were evaluated (TASK section 4)."""
        sealed = list(self.observer.sealed_windows)
        if len(sealed) < 2:
            return {'trigger': False, 'reason': 'need_two_fresh_windows',
                    'role': role, 'windows': [], 'theta_growth': None,
                    'resource_attributed_windows': 0,
                    'loss_for_evaluation_windows': 0}
        last = self.last_consumed_index.get(role)
        # ``>=``: the newest ALREADY evaluated window is re-used as the older
        # half of the next overlapping pair (TASK section 4); the newest index
        # of the pair is always new, an isolated burst can never trigger.
        fresh = [w for w in sealed
                 if last is None or int(w['index']) >= int(last)]
        windows = fresh[-2:]
        if len(windows) < 2:
            return {'trigger': False, 'reason': 'windows_already_consumed',
                    'role': role, 'windows': [], 'theta_growth': None,
                    'resource_attributed_windows': 0,
                    'loss_for_evaluation_windows': 0}
        if self._window_epoch(windows[-1]) != self._window_epoch(windows[0]):
            return {'trigger': False, 'reason': 'epoch_boundary_pair',
                    'role': role, 'windows': [], 'theta_growth': None,
                    'resource_attributed_windows': 0,
                    'loss_for_evaluation_windows': 0}
        # T014 s3.2.5: BOTH windows must be complete + USABLE (no transition /
        # probe / stall flag, no source gap) and their deadline-opportunity
        # shape must be comparable; an unusable pair is never evidence.
        unusable = [int(w['index']) for w in windows
                    if not self.observer._window_usable(w, role)]
        shape_ok, shape = QF4Observer.deadline_shape_ok(
            self.observer.span_budget_hist(int(windows[0]['start_ns']),
                                           int(windows[0]['end_ns']), role),
            self.observer.span_budget_hist(int(windows[-1]['start_ns']),
                                           int(windows[-1]['end_ns']), role))
        if unusable or not shape_ok:
            return {'trigger': False,
                    'reason': 'pair_not_usable' if unusable
                              else 'deadline_opportunity_shape',
                    'role': role, 'windows': [], 'theta_growth': None,
                    'unusable_windows': unusable, 'deadline_shape': shape,
                    'resource_attributed_windows': 0,
                    'loss_for_evaluation_windows': 0}
        tg = self._theta_growth(role, c_role_ms)
        marks = []
        for w in windows:
            res = (w.get('residuals') or {}).get(role) or {}
            growth = float(res.get('growth_ms_s') or 0.0)
            fn = getattr(self.observer, 'resource_root_loss', None)
            if fn is not None:
                try:
                    ev = fn(w, role)
                except Exception:
                    self.counters['resource_root_loss_errors'] += 1
                    ev = None
                if ev is not None:
                    n_roots = int(ev)
                    try:
                        resource_attributed = bool(ev['resource_attributed'])
                    except (TypeError, KeyError, IndexError):
                        resource_attributed = False
                    try:
                        healthy_evidence = dict(ev['healthy_evidence'])
                    except (TypeError, KeyError, IndexError):
                        healthy_evidence = {}
                    try:
                        zero_wait_opportunity = bool(ev['zero_wait_opportunity'])
                    except (TypeError, KeyError, IndexError):
                        zero_wait_opportunity = False
                else:
                    n_roots = self._resource_root_loss(w, role)
                    resource_attributed = False
                    healthy_evidence = {}
                    zero_wait_opportunity = False
            else:
                n_roots = self._resource_root_loss(w, role)
                resource_attributed = False
                healthy_evidence = {}
                zero_wait_opportunity = False

            ids = self._loss_root_ids(w, role)
            growth_over = bool(growth > tg['theta'])
            # TASK s3.1.7 / s3.2.5: a real loss is worth re-evaluating a candidate
            # on its own (loss_for_evaluation only WAKES a candidate; it is not a
            # resource attribution), so the sustained-condition test is
            # 'growth over theta OR at least one complete root's real loss'.
            loss_for_eval = bool(growth_over or int(n_roots) >= 1)
            marks.append({
                'window_index': int(w['index']),
                'epoch': self._window_epoch(w),
                'growth': growth,
                'resource_lost_requests': int(res.get('resource_lost_components') or 0),
                'resource_lost_roots': int(n_roots),
                'loss_roots': sorted(str(r) for r in ids),
                'growth_over': growth_over,
                'resource_loss': bool(int(n_roots) >= 1),
                'resource_attributed': resource_attributed,
                'healthy_evidence': healthy_evidence,
                'zero_wait_opportunity': zero_wait_opportunity,
                'loss_for_evaluation': loss_for_eval,
                'low_confidence': bool(tg['low_confidence']),
            })
        trigger = bool(marks) and all(
            m['growth_over'] or m['loss_for_evaluation'] for m in marks)
        resource_attributed_windows = sum(
            1 for m in marks if m.get('resource_attributed'))
        loss_for_evaluation_windows = sum(
            1 for m in marks if m.get('loss_for_evaluation'))
        return {'trigger': bool(trigger), 'theta_growth': tg, 'windows': marks,
                'role': role, 'epoch': self.epoch,
                'reason': 'persistent_gap' if trigger else 'no_persistent_gap',
                'consumed_index': self.last_consumed_index.get(role),
                'resource_attributed_windows': int(resource_attributed_windows),
                'loss_for_evaluation_windows': int(loss_for_evaluation_windows)}

    def _trigger_both(self, c_role_ms):
        """TASK section 4: BOTH directions are computed in the same tick.

        ``c_role_ms`` may be a per-role dict (Fast/Accurate reference cost) or
        a single number.  The reasons of both directions are kept; nothing is
        discarded by taking max(L/K) first.
        """
        if isinstance(c_role_ms, dict):
            cmap = {'Fast': float(c_role_ms.get('Fast') or 0.0),
                    'Accurate': float(c_role_ms.get('Accurate') or 0.0)}
        else:
            cmap = {'Fast': float(c_role_ms), 'Accurate': float(c_role_ms)}
        out = {r: self._trigger_role(r, cmap[r]) for r in GPU_ROLES}
        self.last_trigger = out
        self.triggers = out
        # TASK section 4: BOTH directions were evaluated first; only now are
        # these two windows marked consumed.  A second call without a new
        # window therefore answers 'windows_already_consumed'.
        self._mark_windows_consumed(self.observer.sealed_windows)
        return out

    def _mark_windows_consumed(self, sealed):
        """Called only AFTER both directions were evaluated in this tick."""
        if not sealed:
            return
        newest = int(sealed[-1]['index'])
        for role in GPU_ROLES:
            prev = self.last_consumed_index.get(role)
            if prev is None or newest > int(prev):
                self.last_consumed_index[role] = newest
        self.last_scored_window = newest

    def record_gain(self, epoch, receiver, gF, dY, dV, eps_Y, eps_V,
                    comparable, source):
        key = '%s|%s|%.1f' % (int(epoch), str(receiver), float(gF))
        rec = {
            'epoch': int(epoch),
            'receiver': str(receiver),
            'gF': float(gF),
            'dY': float(dY),
            'dV': None if dV is None else float(dV),
            'eps_Y': float(eps_Y),
            'eps_V': None if eps_V is None else float(eps_V),
            'comparable': bool(comparable),
            'source': str(source),
            'seq': int(self.command_seq),
        }
        if key in self.gain_history:
            del self.gain_history[key]
        self.gain_history[key] = rec
        while len(self.gain_history) > self.gain_history_max:
            self.gain_history.popitem(last=False)
        return rec

    def net_gain_view(self, receiver, gF, epoch):
        ep_val = self.epoch if epoch is None else int(epoch)
        matching = [
            v for v in self.gain_history.values()
            if str(v.get('receiver')) == str(receiver) and int(v.get('epoch')) == ep_val
        ]
        n = int(len(matching))
        same_epoch = bool(ep_val == int(self.epoch))
        low_confidence = bool(n < 8)
        dY_vals = [float(v['dY']) for v in matching if v.get('dY') is not None]
        dV_vals = [float(v['dV']) for v in matching if v.get('dV') is not None]
        mean_dY = (sum(dY_vals) / len(dY_vals)) if dY_vals else None
        mean_dV = (sum(dV_vals) / len(dV_vals)) if dV_vals else None
        return {
            'n': n,
            'same_epoch': same_epoch,
            'mean_dY': mean_dY,
            'mean_dV': mean_dV,
            'low_confidence': low_confidence,
            'records': n,
        }

    def _annotate_candidates(self, table, receiver):
        cur = current_candidate(table, self.effective_fast)
        for c in (table.get('candidates') or []):
            if cur is not None:
                if receiver == 'Accurate':
                    c['effective_increase'] = bool(
                        c['muA_safe_tasks_s'] > cur['muA_safe_tasks_s'] + 1e-12)
                else:
                    c['effective_increase'] = bool(c['K_F'] > cur['K_F'] + 1e-12)
            c['net_gain_history'] = self.net_gain_view(receiver, c['gF'], self.epoch)

    # -- command issue -----------------------------------------------------------
    def _issue(self, now_ns, view, apply, target_gF, info, channel):
        # T014 s3.2.4: "return with no pre-observation" means return to a share
        # that was ALREADY effective, never to a fresh command that would then
        # need a pre-observation it does not have.  A fresh issue therefore
        # snaps to the real effective share (an on-grid, axis-legal value) and
        # reports the snap instead of silently commanding the axis floor.
        if info.get('return_to_confirmed') or info.get('no_probe'):
            back = float(info.get('return_gF', target_gF))
            snapped = grid_value(back, self.axis_min, self.axis_max, self.grid)
            if abs(snapped - float(target_gF)) > 1e-9:
                info['return_snapped_from'] = float(target_gF)
                info['return_snapped_to'] = float(snapped)
            target_gF = snapped
        # T012: the quantum gate must test the REQUESTED target.  Testing the
        # snapped value made ``limit_quantum`` unreachable dead code (grid_value
        # always returns an on-grid value).
        if not on_grid(target_gF, self.axis_min, self.grid):
            info['reason'] = 'limit_quantum'
            self.last_reason = 'limit_quantum'
            self.counters['issue_blocked_quantum'] += 1
            return None
        # ... and so must the range gate: grid_value CLAMPS to the axis, so a
        # clamped 1.9 silently became a legal 1.5 command.
        if (float(target_gF) < self.axis_min - 1e-9
                or float(target_gF) > self.axis_max + 1e-9):
            info['reason'] = 'limit_range'
            self.last_reason = 'limit_range'
            self.counters['issue_blocked_range'] += 1
            return None
        target = grid_value(target_gF, self.axis_min, self.axis_max, self.grid)
        if self.pending is not None:
            # TASK section 6: at most ONE pending command at any moment.
            info['reason'] = 'one_pending_only'
            self.last_reason = 'one_pending_only'
            self.counters['issue_blocked_pending'] += 1
            return None
        direction = info.get('direction') or (
            'F' if float(target) > float(self.effective_fast) else 'A')
        # A rollback/return command goes back to an already known band: it is
        # not a new trial, so the 2-trial cap must not block it.
        returning = bool(info.get('return_to_confirmed'))
        if not self.continuous_resource and not returning and \
                self.trials_by_direction['unconfirmed_' + direction] >= \
                MAX_UNCONFIRMED_TRIALS_PER_DIRECTION and \
                not self._renew_trial_budget(direction, now_ns):
            # TASK section 6: the 2-trial cap is checked at the ISSUE entry,
            # not only when a probe is evaluated (REFERENCE
            # third_unconfirmed_issue_allowed false).
            info['reason'] = 'unconfirmed_trials_exhausted_' + direction
            self.last_reason = info['reason']
            self.counters['issue_blocked_trials'] += 1
            return None
        cards = (view or {}).get('cards') or []
        if any(bool(c.get('transition')) for c in cards):
            info['reason'] = 'limit_transition'
            return None
        command = {
            'command_id': 'qf4r-%06d' % (self.command_seq + 1),
            'from_fast': round(float(self.effective_fast), 10),
            'target_fast': float(target),
            'delta_fast': round(float(target) - float(self.effective_fast), 10),
            'proposed_ns': int(now_ns),
            'channel': channel,
            'direction': direction,
            # The full pre-cohort ledger stays in pending, where the probe
            # actually reads it.  Carry aggregate evidence and root IDs in the
            # command, rather than serializing the entire ledger three times
            # in every control row while this command remains pending.
            'info': {k: v for k, v in info.items()
                     if k not in ('trend', 'pre_cohorts')},
        }
        if info.get('pre_cohorts'):
            command['info']['pre_cohorts'] = [
                dict({k: v for k, v in cohort.items() if k != 'roots'},
                     root_ids=[r.get('root_id') for r in cohort.get('roots', [])])
                for cohort in info['pre_cohorts']]
        if not self.dynamic:
            # TASK section 6 / C10: the fixed arm never issues a command, never
            # applies and never enters a fake EFFECTIVE/PROBE state.
            self.counters['shadow_commands_suppressed'] += 1
            info['reason'] = 'fixed_shadow_no_command'
            return None
        try:
            result = apply(command)
        except Exception:
            self.counters['apply_exceptions'] += 1
            info['reason'] = 'apply_exception'
            return None
        if not bool((result or {}).get('accepted')):
            self.counters['apply_rejected'] += 1
            info['reason'] = 'apply_rejected'
            return None
        self.command_seq += 1
        # A later accepted command cancels evidence for the old configuration.
        # Never attribute its future results to the share it replaced.
        if self.held_review is not None:
            self.counters['held_review_cancelled'] += 1
            self.held_review = None
        command['command_id'] = 'qf4r-%06d' % self.command_seq
        self.pending = {
            'command': command,
            'issued_ns': int(now_ns),
            'pre_fast': round(float(self.effective_fast), 10),
            'direction': direction,
            'extended': False,
            'extensions': 0,
            'pre_prices': self._cost_view(view),
            'pre_cohorts': info.get('pre_cohorts'),
            'noise_banks': {k: {'Y': list(v['Y']), 'V': list(v['V']), 'H': list(v['H']), 'key': k, 'span_known': True} for k, v in self.noise_banks.items()},
            # a return to a confirmed band is verified for real effectiveness
            # but is not probed again (that edge already has a verdict)
            'no_probe': bool(info.get('return_to_confirmed')),
            'continuous_resource': self.continuous_resource,
            'info': info,
        }
        self.target_fast = float(target)
        self.state = 'REQUESTED'
        self.commands.append(command)
        # TASK section 6: only THIS tick's new command is reported downstream.
        self.new_commands_this_tick.append(command)
        self.new_command_count = len(self.new_commands_this_tick)
        self.pending_command = command
        self.counters['commands_issued'] += 1
        self.apply_count = int(self.counters.get('commands_issued', 0))
        # exactly one CONFIG_REQUEST per command_id
        self.config_request_count += 1
        info['reason'] = 'command_issued'
        return command

    def _both_effective(self, view, target):
        cards = (view or {}).get('cards') or []
        if len(cards) != CARD_COUNT:
            return False
        for card, share in zip(cards, phi_cards(target)):
            if abs(float(card.get('effective', 0.0)) - float(share)) > 1e-9:
                return False
            if bool(card.get('transition')):
                return False
        return True

    # -- probe evaluation -----------------------------------------------------------
    def _cohort_layer_stats(self, cohorts, include_probabilities=True):
        """Per-class source-root statistics of one/more complete source cohorts.

        T014 s3.1.3: the denominator ``n`` is the number of REGISTERED source
        roots of the class, never the terminal count (a terminal-only
        denominator is a survivor bias).  ``v`` accumulates only KNOWN public
        probabilities and ``known == y`` is the availability flag, so a class
        whose on-time outputs carry a missing probability is reported Y-only
        instead of a fabricated 0 (T014 s3.4.4).  ``l`` is the per-class
        resource-loss root count and is filled by the probe caller.
        """
        layers = {c: {'n': 0, 'y': 0, 'v': 0.0, 'known': 0, 'h': 0., 'h_known': 0, 'l': 0}
                  for c in ROOT_CLASSES}
        probs = []
        for coh in cohorts:
            for cls in ROOT_CLASSES:
                layers[cls]['n'] += int(coh['n'][cls])
                t = coh['terminal_by_class'][cls]
                layers[cls]['y'] += int(t['ontime'])
                layers[cls]['v'] += float(t['sum_p_ontime'])
                layers[cls]['known'] += int(t['known_ontime'])
                layers[cls]['h'] += float(t.get('sum_quality_earliness') or 0.)
                layers[cls]['h_known'] += int(t.get('known_earliness') or 0)
            for root in (coh['roots'] if include_probabilities else ()):
                if root['entry_class'] == ROOT_PREFIX:
                    continue
                role = ('Fast' if root['entry_class'] == ROOT_DIRECT_F
                        else 'Accurate')
                p = (root.get('probabilities') or {}).get(role)
                if p is not None:
                    probs.append(p)
        for cls in ROOT_CLASSES:
            layers[cls]['v_available'] = layers[cls]['known'] >= layers[cls]['y']
            layers[cls]['h_available'] = layers[cls]['h_known'] >= layers[cls]['y']
        return layers, probs

    def _mean_changes(self, item_prices, view):
        """Per-card-kind mean relative change pre->post (comparability gate)."""
        post = self._cost_view(view)
        rels = []
        for g in CARD_IDS:
            for kind in GPU_KINDS:
                m_pre = ((item_prices or {}).get(g) or {}).get(kind, {}).get('mean_wall_ms')
                m_post = post.get(g, {}).get(kind, {}).get('mean_wall_ms')
                if m_pre and m_post:
                    rels.append(abs(m_post - m_pre) / m_pre)
        return {'max_rel_change': max(rels) if rels else None, 'n': len(rels)}

    def _loss_roots_over(self, role, t_lo_ns, t_hi_ns):
        """Resource-lost ROOT IDENTITIES of one role over a sealed-window span
        (kept for diagnostics; the probe compares loss RATES per source class
        and never the raw cross-window ID sets, T014 s3.4.3)."""
        roots = set()
        for w in self.observer.sealed_windows:
            if int(w['start_ns']) >= int(t_lo_ns) and int(w['end_ns']) <= int(t_hi_ns):
                lr = (w.get('lost_roots') or {}).get(role) or set()
                roots |= set(lr)
        return roots

    def _loss_by_class(self, t_lo_ns, t_hi_ns):
        """Resource-lost source roots per ROOT CLASS over a span, deduplicated
        per root within that span (T014 s3.4.3: loss/n and the unified lambda)."""
        seen = set()
        by = {c: 0 for c in ROOT_CLASSES}
        for w in self.observer.sealed_windows:
            if int(w['start_ns']) < int(t_lo_ns) or int(w['end_ns']) > int(t_hi_ns):
                continue
            for role in GPU_ROLES:
                for rid in ((w.get('lost_roots') or {}).get(role) or ()):
                    if rid in seen:
                        continue
                    seen.add(rid)
                    root = self.observer.roots.get(rid)
                    if root is not None:
                        by[root['entry_class']] = by.get(root['entry_class'], 0) + 1
        return by

    def _noise_bank(self, T_pre_s, T_post_s):
        """The noise bank for one exact span pair (never mixes spans)."""
        key = '%.1f|%.1f' % (round(float(T_pre_s), 3), round(float(T_post_s), 3))
        bank = self.noise_banks.get(key)
        if bank is None:
            return {'Y': [], 'V': [], 'H': [], 'key': key, 'span_known': False}
        return {'Y': list(bank['Y']), 'V': list(bank['V']), 'H': list(bank['H']), 'key': key,
                'span_known': True}

    def _append_output_noise(self):
        coh = [c for c in self.observer.recent_sealed_cohorts_all()
               if c.get('results_sealed') and not c.get('incomplete')]
        npre = max(1, self.probe_pre_ns // self.window_ns)
        for span in self.probe_post_levels_ns:
            npost = max(1, span // self.window_ns)
            cs = coh[-(npre+npost):]
            if len(cs) != npre+npost or any(self._cohort_flagged(c) for c in cs):
                continue
            if any(int(c['effective_epoch']) != self.epoch for c in cs):
                continue
            if any(cs[i]['end_ns'] != cs[i+1]['start_ns'] for i in range(len(cs)-1)):
                continue
            key = '%.1f|%.1f' % (self.probe_pre_ns/NS_PER_S, span/NS_PER_S)
            ident = (key, cs[0]['start_ns'], cs[-1]['end_ns'])
            if ident == getattr(self, '_noise_seen_'+key, None):
                continue
            # Noise uses the per-class terminal aggregates only. Probability
            # histograms remain part of actual pre/post comparability, where
            # _cohort_layer_stats keeps its default full behaviour.
            pre, _ = self._cohort_layer_stats(cs[:npre], include_probabilities=False)
            post, _ = self._cohort_layer_stats(cs[npre:], include_probabilities=False)
            std = standardized_probe(pre, post, self.probe_pre_ns/NS_PER_S, span/NS_PER_S)
            if std['comparable']:
                bank = self.noise_banks[key]
                bank['Y'].append(abs(std['dY']))
                if std['v_available']:
                    bank['V'].append(abs(std['dV']))
                if std['h_available']:
                    bank['H'].append(abs(std['dH']))
                setattr(self, '_noise_seen_'+key, ident)

    def _freeze_pre_cohorts(self, now_ns):
        """T014 s3.4.1: the pre window is frozen BEFORE the command is issued.

        It is the last two adjacent, complete, healthy, same-OLD-epoch 500 ms
        source cohorts fully inside the current 500 ms boundary.  Health means
        complete execution evidence with no transition/probe/stall/source gap;
        a real resource loss is allowed (overload is exactly the situation the
        adjustment is for).  Fewer than two -> None and the caller HOLDS; the
        probe must never go looking for a pre window after the config changed.
        """
        cut = (int(now_ns) // max(int(self.window_ns), 1)) * int(self.window_ns)
        if cut <= 0:
            return None
        cohorts = list(self.observer.last_sealed_complete_cohorts(
            before_ns=cut, span_ns=self.probe_pre_ns) or [])
        cohorts = [c for c in cohorts
                   if isinstance(c, dict) and c.get('end_ns') is not None
                   and int(c['end_ns']) <= cut
                   and not c.get('incomplete')
                   and not self._cohort_flagged(c)]
        required = max(1, int(self.probe_pre_ns // self.window_ns))
        if len(cohorts) < required:
            return None
        cohorts = cohorts[-required:]
        if any(cohorts[i]['end_ns'] != cohorts[i+1]['start_ns'] for i in range(len(cohorts)-1)):
            return None
        own = self._cohort_epoch(cohorts[0])
        if not all(self._cohort_epoch(c) == own for c in cohorts):
            return None
        # the pre side is the configuration in force AT ISSUE TIME (the old one
        # from the command's point of view); it must never be a mixed or a
        # post-command epoch.
        if int(own) != int(self.epoch):
            return None
        if not all(int(c['end_ns']) - int(c['start_ns']) == int(self.window_ns)
                   for c in cohorts):
            return None
        return cohorts

    def _cohort_flagged(self, coh):
        for w in self.observer.sealed_windows:
            if int(w['end_ns']) <= int(coh['start_ns']) or \
                    int(w['start_ns']) >= int(coh['end_ns']):
                continue
            f = w.get('flags') or {}
            if f.get('transition') or f.get('probe') or f.get('stall') or \
                    f.get('source_gap') or f.get('log_backpressure'):
                return True
        return False

    def _cohort_epoch(self, coh):
        e = self._window_epoch(coh)
        if e:
            return int(e)
        if self.epoch_start_ns is None:
            return int(self.epoch)
        if int(coh['start_ns']) >= int(self.epoch_start_ns):
            return int(self.epoch)
        return int(self.epoch) - 1

    def _probe_support(self, now_ns, view, post_cohorts, pre_cohorts, probe_item=None):
        item = self.pending if probe_item is None else probe_item
        if (item.get('info') or {}).get('selection_objective') == 'latency_timeout':
            return self._resource_probe_support(now_ns, view, post_cohorts, pre_cohorts, item)
        pre_layers, pre_probs = self._cohort_layer_stats(pre_cohorts)
        post_layers, post_probs = self._cohort_layer_stats(post_cohorts)
        # T014 s3.4.2: the REAL span of each side is passed, never a shared (T,T).
        pre_lo = min([int(c['start_ns']) for c in pre_cohorts] or [now_ns])
        pre_hi = max([int(c['end_ns']) for c in pre_cohorts] or [now_ns])
        post_lo = min([int(c['start_ns']) for c in post_cohorts] or [0])
        post_hi = max([int(c['end_ns']) for c in post_cohorts] or [0])
        T_pre = max(pre_hi - pre_lo, 0) / float(NS_PER_S)
        T_post = max(post_hi - post_lo, 0) / float(NS_PER_S)
        # T014 s3.4.3: loss is compared as a rate per source class over the
        # unified lambda, deduplicated per root inside each span.
        loss_pre = self._loss_by_class(pre_lo, pre_hi)
        loss_post = self._loss_by_class(post_lo, post_hi)
        for cls in ROOT_CLASSES:
            pre_layers[cls]['l'] = int(loss_pre.get(cls, 0))
            post_layers[cls]['l'] = int(loss_post.get(cls, 0))
        std = standardized_probe(pre_layers, post_layers, T_pre, T_post)
        bank = (item.get('noise_banks') or {}).get('%.1f|%.1f' % (T_pre, T_post)) or self._noise_bank(T_pre, T_post)
        span_min = min(T_pre, T_post)
        epsY = probe_epsilon(list(bank['Y']), span_min,
                             healthy_min=OUTPUT_NOISE_MIN_PAIRS)
        epsV = probe_epsilon(list(bank['V']), span_min,
                             healthy_min=OUTPUT_NOISE_MIN_PAIRS)
        speed_mode = ((item.get('info') or {}).get('selection_objective') ==
                      'quality_weighted_earliness')
        resolution = float((item.get('info') or {}).get('quality_speed_quantum') or 1.)
        epsH = probe_epsilon(list(bank.get('H') or []), span_min,
                             healthy_min=OUTPUT_NOISE_MIN_PAIRS, resolution=resolution)
        lag = self.observer.latest_lag_p95() or {}
        comparable = bool(std['comparable']
                          and not std['one_side_zero']
                          and std['gpu_roots'] >= COMPARABLE_MIN_GPU_ROOTS
                          and not lag.get('stall')
                          and T_pre > 0 and T_post > 0
                          and bank['span_known'])
        tv_layers = tv_distance({c: pre_layers[c]['n'] for c in ROOT_CLASSES},
                                {c: post_layers[c]['n'] for c in ROOT_CLASSES})
        tv_probs = probability_histogram_tv(pre_probs, post_probs)
        if tv_layers is None or tv_layers > COMPARABLE_TV_MAX:
            comparable = False
        if tv_probs is not None and tv_probs > COMPARABLE_TV_MAX:
            comparable = False
        # T014 s3.4.4: a post price drift is DIAGNOSTIC only and never a
        # comparability veto -- a real net loss must still be able to roll back.
        mean_change = self._mean_changes(item.get('pre_prices'), view)
        cmd = item.get('command') or {}
        direction = item.get('direction') or \
            ('F' if float(cmd.get('target_fast', 0.0)) >
             float(cmd.get('from_fast', 0.0)) else 'A')
        receiver = 'Fast' if direction == 'F' else 'Accurate'
        donor = 'Accurate' if direction == 'F' else 'Fast'
        recv_cls = ROOT_DIRECT_F if receiver == 'Fast' else ROOT_DIRECT_A
        sup_cls = ROOT_DIRECT_A if receiver == 'Fast' else ROOT_DIRECT_F
        recv_row = std['layers'].get(recv_cls) or {}
        lam_bar = float(recv_row.get('lambda_bar') or 0.0)
        recv_std_gain = lam_bar * (float(recv_row.get('s_post') or 0.0) -
                                   float(recv_row.get('s_pre') or 0.0))
        pre_recv = self._loss_roots_over(receiver, pre_lo, pre_hi)
        post_recv = self._loss_roots_over(receiver, post_lo, post_hi)
        pre_donor = self._loss_roots_over(donor, pre_lo, pre_hi)
        post_donor = self._loss_roots_over(donor, post_lo, post_hi)
        donor_row = std['layers'].get(sup_cls) or {}
        donor_loss_delta = float(donor_row.get('loss_rate_delta') or 0.0)
        cost_ver = None
        # T014 s3.2.2: the DEADLINE-OPPORTUNITY shape of the two spans must be
        # comparable (5 ms classes, TV <= 0.1, >= 30 roots per class each side);
        # an unusable shape is NOT comparable and can never confirm a gain.
        shape_ok, shape = QF4Observer.deadline_shape_ok(
            self.observer.span_budget_hist(pre_lo, pre_hi, receiver),
            self.observer.span_budget_hist(post_lo, post_hi, receiver))
        shape['role'] = receiver
        if not shape_ok and shape.get('reason') not in ('insufficient_samples', 'insufficient_sample', 'too_few_samples'):
            comparable = False
        entry_shapes = {}
        if self.hold_on_uncertain:
            for role in GPU_ROLES:
                pre_entry = QF4Observer.cohort_entry_budget_hist(pre_cohorts, role)
                post_entry = QF4Observer.cohort_entry_budget_hist(post_cohorts, role)
                # Sparse samples remain weak evidence; they cannot bypass a
                # visible opportunity-distribution change through min_n.
                ok, detail = QF4Observer.deadline_shape_ok(pre_entry, post_entry, min_n=0)
                detail['n_pre'] = pre_entry['n']
                detail['n_post'] = post_entry['n']
                entry_shapes[role] = detail
                comparable = comparable and ok
        try:
            cost_ver = int(self.cost_book.global_cost_version())
        except Exception:
            cost_ver = None
        support = {
            'command_id': cmd.get('command_id'),
            'selection_objective': ((item.get('info') or {}).get('selection_objective') or 'deadline_output'),
            'epsilon_H': epsH, 'h_available': bool(std['h_available']),
            'direction': direction, 'receiver': receiver, 'donor': donor,
            'pre_layers': pre_layers, 'post_layers': post_layers,
            'std': std, 'epsilon_Y': epsY, 'epsilon_V': epsV,
            'v_available': bool(std['v_available']),
            'dLoss': float(std['dLoss']),
            'donor_loss_rate_delta': donor_loss_delta,
            'loss_by_class_pre': loss_pre, 'loss_by_class_post': loss_post,
            # T014 s3.3.8/3.4.4: ONE comparable flag; there is no separate
            # comparable_total that could bypass the real gates.
            'comparable': comparable, 'comparable_total': bool(comparable),
            'deadline_shape_ok': bool(shape_ok), 'deadline_shape': shape,
            'source_entry_deadline_shapes': entry_shapes,
            'one_side_zero': bool(std['one_side_zero']),
            'missing_layers': list(std['missing_layers']),
            'ignored_layers': list(std['ignored_layers']),
            'tv_layers': tv_layers, 'tv_probs': tv_probs,
            'mean_change': mean_change, 'lag': lag,
            'T_s': span_min, 'T_pre_s': T_pre, 'T_post_s': T_post,
            'noise_bank_key': bank['key'],
            'noise_bank_n_Y': len(bank['Y']), 'noise_bank_n_V': len(bank['V']),
            'receiver_std_gain': float(recv_std_gain),
            'receiver_epsilon': float(epsY['epsilon']),
            'receiver_loss_down_roots': int(max(0, len(pre_recv) - len(post_recv))),
            'donor_new_loss': bool(post_donor - pre_donor),
            'supplier_resource_loss_up': bool(donor_loss_delta > 0),
            'receiver_loss_roots_pre': len(pre_recv),
            'receiver_loss_roots_post': len(post_recv),
            'donor_loss_roots_pre': len(pre_donor),
            'donor_loss_roots_post': len(post_donor),
            'pre_prices': item.get('pre_prices'),
            'post_prices': self._cost_view(view),
            'price_sensitivity': {'pre_prices': item.get('pre_prices'), 'post_prices': self._cost_view(view)},
            'cost_version_now': cost_ver,
            'noise_source_Y': self.last_noise_source,
            'noise_source_V': self.last_noise_source,
            'pre_span_ns': [pre_lo, pre_hi], 'post_span_ns': [post_lo, post_hi],
            # T014 CPU_expected.probe_flag_or: a probe flag anywhere in either
            # span is carried onto the whole span pair (OR, never last-tick).
            'probe_flag': bool(self.observer.span_probe_flag(pre_lo, pre_hi)
                               or self.observer.span_probe_flag(post_lo, post_hi)),
            'verdict': None,
        }
        self.last_probe = support
        return support

    def _resource_probe_support(self, now_ns, view, post_cohorts, pre_cohorts, item):
        """Compare keeping this allocation versus its previous band on one
        fresh public profile, with identical input/route/price/initial ledger.

        This is a MODEL comparison for the current continuation decision, not
        an observed counterfactual nor a statistical causal-effect estimate.
        Mature real outcomes are carried separately for audit. No confidence
        head, true label or other arm's future output affects the verdict.
        """
        from resource_objective import paired_effect, summarize_role, timely_utility, ROUNDOFF
        cmd = item['command']
        replay = view.get('replay') or {}
        valid = self._replay_table_valid(replay, now_ns)
        by_band = {round(float(r['gF']), 10): r for r in replay.get('candidates', ())}
        old = by_band.get(round(float(cmd['from_fast']), 10))
        new = by_band.get(round(float(cmd['target_fast']), 10))
        post_start = int(item.get('post_start_ns') or 0)
        post_end = max([int(c['end_ns']) for c in post_cohorts] or [now_ns])
        mature = bool(post_cohorts) and all(c.get('results_sealed') and c.get('wm_sealed')
            and not c.get('incomplete') for c in post_cohorts)
        ready = bool(valid['ok'] and replay.get('selection_objective') == 'latency_timeout'
            and replay.get('snapshot_ns', -1) >= post_end
            and replay.get('profile_origin_ns', -1) >= post_start
            and abs(float(replay.get('reference_gF', -99))-float(cmd['target_fast'])) < 1e-9
            and old and new and mature and not view.get('stall')
            and not view.get('frame_gap_seen') and not view.get('log_backpressure'))
        effect = paired_effect((new or {}).get('resource_outcomes'),
            (old or {}).get('resource_outcomes'), (new or {}).get('resource_span_s', 0))
        sensitivities = {}
        if ready:
            for scen in ('unknown_F', 'unknown_A', 'unknown_Abstain'):
                a = (new.get('sensitivity') or {}).get(scen) or {}
                b = (old.get('sensitivity') or {}).get(scen) or {}
                test = paired_effect(a.get('resource_outcomes'), b.get('resource_outcomes'), new['resource_span_s'])
                sensitivities[scen] = test
                if not test.get('available'):
                    ready = False
                elif not test['roles_protected'] or test['gain'] < -ROUNDOFF:
                    effect = dict(effect, roles_protected=False, sensitivity_harm=scen)

        def observed(cohorts):
            totals = {r: [0, 0, 0.] for r in GPU_ROLES}
            unresolved = 0
            for c in cohorts:
                for root in c['roots']:
                    terminal = root.get('terminal') or {}
                    role = ('Fast' if root['entry_class'] == ROOT_DIRECT_F else
                        'Accurate' if root['entry_class'] == ROOT_DIRECT_A else
                        (root.get('revealed_early') or {}).get('role'))
                    if role not in totals:
                        # Unknown Prefix outcomes remain explicitly unknown.
                        unresolved += 1
                        continue
                    u = timely_utility(root['original_arrival_ns'], root['deadline_ns'],
                        terminal.get('available_ns')) if terminal.get('ontime') else 0.
                    totals[role][0] += 1
                    totals[role][1] += int(u > 0)
                    totals[role][2] += u
            return {'roles': {r: summarize_role(*v) for r, v in totals.items()},
                    'unresolved_prefix_inputs': unresolved}

        old_y = sum(v['ontime'] for v in (old or {}).get('resource_outcomes', {}).values())
        new_y = sum(v['ontime'] for v in (new or {}).get('resource_outcomes', {}).values())
        span = (new or {}).get('resource_span_s') or 1.
        support = {'command_id': cmd['command_id'], 'selection_objective': 'latency_timeout',
            'basis': 'same_public_profile_model_comparison_with_mature_observed_diagnostics',
            'comparable': ready, 'comparable_total': ready, 'mature_real_outputs': mature,
            'resource_effect': effect, 'sensitivity': sensitivities,
            'paired_profile_sha': replay.get('profile_sha'), 'snapshot_ns': replay.get('snapshot_ns'),
            'price_version': replay.get('price_version'), 'from_gF': cmd['from_fast'],
            'target_gF': cmd['target_fast'], 'old_resource_outcomes': (old or {}).get('resource_outcomes'),
            'new_resource_outcomes': (new or {}).get('resource_outcomes'),
            'observed_pre': observed(pre_cohorts), 'observed_post': observed(post_cohorts),
            'pre_span_ns': [min([c['start_ns'] for c in pre_cohorts] or [now_ns]),
                            max([c['end_ns'] for c in pre_cohorts] or [now_ns])],
            'post_span_ns': [post_start, post_end], 'validity': valid,
            'std': {'dY': (new_y-old_y)/span, 'dV': 0., 'v_available': False,
                    'dH': effect.get('gain'), 'h_available': effect.get('available', False)},
            'epsilon_Y': {'epsilon': ROUNDOFF}, 'epsilon_V': {'epsilon': ROUNDOFF},
            'epsilon_H': {'epsilon': ROUNDOFF}, 'h_available': effect.get('available', False),
            'v_available': False, 'verdict': None}
        self.last_probe = support
        return support

    @staticmethod
    def _support_verdict(support):
        if support['selection_objective'] == 'latency_timeout':
            from resource_objective import resource_verdict
            return resource_verdict(support['resource_effect'], support['comparable'])
        std = support['std']
        return operational_verdict(std['dY'], support['epsilon_Y']['epsilon'],
            std['dV'], support['epsilon_V']['epsilon'], std['v_available'], support['comparable'],
            dH=std.get('dH'), eps_H=support['epsilon_H']['epsilon'],
            h_available=support['h_available'],
            require_speed=support['selection_objective'] == 'quality_weighted_earliness',
            require_correct_ontime=support['selection_objective'] == 'correct_ontime')

    def _price_sensitivity(self, view):
        """Old/new band Yhat/Vhat recomputed under BOTH pre and post prices
        (section 8: a cost_version change never cancels a probe)."""
        out = {}
        if self.last_table is None or self.pending is None:
            return out
        c_star_now = self.observer.c_star_ms
        demand = self.last_table['demand']
        for tag, prices in (('pre', self.pending.get('pre_prices')),
                            ('post', self._cost_view(view))):
            if not prices:
                continue
            t = evaluate_candidates(self.effective_fast, demand, prices,
                                    c_star_now, T_s=1.0,
                                    min_margin_ms=self.min_margin_ms)
            cur = current_candidate(t, self.effective_fast)
            out[tag] = {'current_Yhat': cur['Yhat'] if cur else None,
                        'current_Vhat': cur['Vhat'] if cur else None}
        return out

    def _finish_probe(self, now_ns, view, apply, verdict, reason, support):
        item = self.pending
        self.pending = None
        self.pending_command = None
        self.state = 'IDLE'
        self.counters['verdict_' + str(verdict)] += 1
        self.last_reason = 'verdict_%s_%s' % (verdict, reason)
        if isinstance(support, dict):
            support['verdict'] = verdict
            support['verdict_reason'] = reason
            self.last_probe = support

        if item is not None and self.continuous_resource:
            # Reassess from the effective band; neither uncertainty nor a
            # negative evaluation issues a historical return command.
            self.last_held = {'gF': float(self.effective_fast), 'ns': int(now_ns),
                              'reason': str(reason), 'verdict': verdict}
            self.held_review = None
            self.counters['resource_reassessment_due'] += 1
            return None

        if item is not None and not item.get('no_probe'):
            cmd = item.get('command') or {}
            direction = item.get('direction') or (
                'F' if float(cmd.get('target_fast', 0.0)) > float(cmd.get('from_fast', 0.0)) else 'A')
            receiver = None
            if isinstance(support, dict) and support.get('receiver'):
                receiver = support['receiver']
            elif isinstance(item, dict):
                receiver = (item.get('info') or {}).get('receiver')
            if not receiver:
                receiver = 'Fast' if direction == 'F' else 'Accurate'
            gF = float(cmd.get('target_fast', self.effective_fast))

            std = (support.get('std') or {}) if isinstance(support, dict) else {}
            dY = None
            if 'dY' in std and std['dY'] is not None:
                dY = float(std['dY'])
            elif isinstance(support, dict) and support.get('dY') is not None:
                dY = float(support['dY'])
            elif isinstance(support, dict) and support.get('receiver_gain') is not None:
                dY = float(support['receiver_gain'])
            else:
                dY = 0.0

            v_av = bool(std.get('v_available', False)) if std else bool((support or {}).get('v_available', False))
            dV = None
            if v_av:
                if 'dV' in std and std['dV'] is not None:
                    dV = float(std['dV'])
                elif isinstance(support, dict) and support.get('dV') is not None:
                    dV = float(support['dV'])

            eps_Y = None
            if isinstance(support, dict):
                if 'epsilon_Y' in support:
                    ey = support['epsilon_Y']
                    eps_Y = float(ey.get('epsilon') if isinstance(ey, dict) else ey)
                elif 'eps' in support:
                    eps_Y = float(support['eps'])
                elif 'eps_Y' in support:
                    eps_Y = float(support['eps_Y'])
            if eps_Y is None:
                eps_Y = 0.5

            eps_V = None
            if v_av and isinstance(support, dict):
                if 'epsilon_V' in support:
                    ev = support['epsilon_V']
                    eps_V = float(ev.get('epsilon') if isinstance(ev, dict) else ev)
                elif 'eps_V' in support:
                    eps_V = float(support['eps_V'])

            comparable = bool((support or {}).get('comparable', False)) if isinstance(support, dict) else False

            self.record_gain(
                epoch=self.epoch,
                receiver=receiver,
                gF=gF,
                dY=dY,
                dV=dV,
                eps_Y=eps_Y,
                eps_V=eps_V,
                comparable=comparable,
                source=verdict,
            )

        if item is None:
            return None
        cmd = item['command']
        direction = item.get('direction') or (
            'F' if float(cmd['target_fast']) > float(cmd['from_fast']) else 'A')
        if verdict in CONFIRM_OUTCOMES:
            # NOTE: a no-probe return command carries no pre-observation, so the
            # APPROVED share it returns to is evidence.  A no-probe command must
            # never overwrite confirmed_fast from an unobserved post span.
            if item.get('no_probe'):
                self.counters['no_probe_confirm_ignored'] += 1
                self.last_reason = 'verdict_%s_%s_no_probe' % (verdict, reason)
                return None
            # TASK section 6: CONFIRM updates last_confirmed and clears the
            # trial counter / the matching lock of THIS direction only.
            self.confirmed_fast = float(cmd['target_fast'])
            self.last_confirmed = {'gF': float(cmd['target_fast']),
                                   'ns': int(now_ns), 'epoch': int(self.epoch),
                                   'command_id': cmd['command_id']}
            self.trials_by_direction['unconfirmed_' + direction] = 0
            self.trial_context.pop(direction, None)
            self.locked_edges.pop(direction, None)
            if not self.locked_edges:
                self._lock_reference = None
            self.counters['confirm_confirmed' if verdict == 'CONFIRM'
                           else 'keep_confirmed'] += 1
            return None
        if verdict == 'ROLLBACK':
            # TASK section 6: save the REAL snapshot and lock ONLY this edge.
            snap = self._real_snapshot(view, now_ns)
            self._lock_reference = {'snapshot': snap, 'created_ns': int(now_ns),
                                    'direction': direction, 'reason': str(reason)}
            edge = {'from_fast': round(float(cmd['from_fast']), 10),
                    'target_fast': round(float(cmd['target_fast']), 10),
                    'reason': str(reason), 'created_ns': int(now_ns),
                    'direction': direction, 'snapshot': snap, 'kind': 'rollback'}
            self.locked_edges[direction] = edge
            self.failed_edges.append(edge)
            self.trials_by_direction['unconfirmed_' + direction] = 0
            self.trial_context.pop(direction, None)
            back = float(cmd['from_fast'])
            if abs(back - float(self.effective_fast)) > 1e-9:
                return self._issue(now_ns, view, apply, back,
                                   {'reason': 'rollback', 'support': support,
                                    'return_to_confirmed': True},
                                   'rollback')
            return None
        if verdict == 'CONTINUE':
            self.trials_by_direction['unconfirmed_' + direction] += 1
            self.trial_context[direction] = {
                'snapshot': self._real_snapshot(view, now_ns),
                'taken_ns': int(now_ns), 'reason': str(reason)}
            return None
        # UNCERTAIN: back to last_confirmed, record the failed edge + snapshot
        snap = self._real_snapshot(view, now_ns)
        edge = {'from_fast': round(float(cmd['from_fast']), 10),
                'target_fast': round(float(cmd['target_fast']), 10),
                'reason': str(reason), 'created_ns': int(now_ns),
                'direction': direction, 'snapshot': snap, 'kind': 'uncertain'}
        self.failed_edges.append(edge)
        # T016: an UNCERTAIN probe is NOT a confirmed failure, so it must not
        # lock the direction here the way a ROLLBACK does.  The bounded trial
        # counter (two per direction, REFERENCE
        # max_unconfirmed_trials_per_direction=2 / third_unconfirmed_issue_
        # allowed=false) is the only guard it spends; _renew_trial_budget()
        # reopens that budget once the public workload really changed.
        self.trial_context[direction] = {
            'snapshot': snap, 'taken_ns': int(now_ns), 'reason': str(reason)}
        self.counters['uncertain_hold'] += 1
        self.trials_by_direction['unconfirmed_' + direction] += 1
        if self.hold_on_uncertain:
            # T016 C2: an UNCERTAIN probe means the move's evidence has not
            # matured yet (the 0.1-step effect sits below the window noise), NOT
            # that the move hurt.  Keep the moved share -- do not snap back to
            # last_confirmed -- so the deficit-driven step persists.  Only a
            # measured net loss (ROLLBACK) or a direction lock releases it.
            std = (support or {}).get('std') or {}
            self.counters['uncertain_kept'] += 1
            self.last_held = {'gF': float(cmd['target_fast']),
                              'from_gF': float(cmd['from_fast']),
                              'direction': direction, 'ns': int(now_ns),
                              'reason': str(reason),
                              'comparable': bool((support or {}).get('comparable')),
                              'dY': std.get('dY'), 'dH': std.get('dH')}
            self.last_reason = 'uncertain_held_' + str(direction)
            # Two later, bounded 2 s observations. This is monitoring of the
            # same command, not another trial or permission to reduce epsilon.
            if not item.get('held_review'):
                self.held_review = {'item': item, 'epoch': self.epoch,
                                    'due_ns': int(now_ns) + 2600 * MS,
                                    'remaining': 2}
            return None
        back = self.last_confirmed.get('gF')
        if back is None:
            back = float(cmd['from_fast'])
        if abs(float(back) - float(self.effective_fast)) > 1e-9:
            return self._issue(now_ns, view, apply, float(back),
                               {'reason': 'return_to_last_confirmed',
                                'support': support,
                                'return_to_confirmed': True},
                               'uncertain_return')
        return None

    def _review_held(self, now_ns, view, apply):
        review = self.held_review
        if review is None:
            return
        item = review['item']
        target = float(item['command']['target_fast'])
        if (review['epoch'] != self.epoch or
                abs(self.effective_fast - target) > 1e-9):
            self.held_review = None
            self.counters['held_review_cancelled'] += 1
            return
        if now_ns < review['due_ns'] or self.pending is not None:
            return
        self.counters['held_reviews'] += 1
        review['remaining'] -= 1
        review['due_ns'] = int(now_ns) + 2600 * MS
        if review['remaining'] <= 0:
            self.held_review = None
        span = self._probe_post_span(item)
        end = ((now_ns - self.deadline_ns - self.settle_ns) //
               self.post_align_ns) * self.post_align_ns
        start = end - span
        cohorts = [c for c in self.observer.recent_sealed_cohorts_all()
                   if start <= c['start_ns'] and c['end_ns'] <= end]
        usable = (bool(cohorts) and cohorts[0]['start_ns'] == start and
                  cohorts[-1]['end_ns'] == end and
                  all(c['results_sealed'] and c['wm_sealed'] and
                      not c['incomplete'] and c.get('effective_epoch') == self.epoch
                      for c in cohorts) and
                  all(a['end_ns'] == b['start_ns']
                      for a, b in zip(cohorts, cohorts[1:])))
        if not usable or view.get('stall') or view.get('frame_gap_seen'):
            self.counters['held_review_unknown'] += 1
            self.last_probe = {'command_id': item['command']['command_id'],
                               'held_review': True, 'verdict': 'UNCERTAIN',
                               'reason': 'held_review_cohorts_unusable'}
            return
        support = self._probe_support(now_ns, view, cohorts,
                                      item.get('pre_cohorts') or [], probe_item=item)
        if support['selection_objective'] == 'latency_timeout':
            verdict = self._support_verdict(support)
            support.update(held_review=True, verdict=verdict,
                verdict_reason='paired_resource_continuation')
            self.last_probe = support
            if verdict == 'UNCERTAIN':
                self.counters['held_review_uncertain'] += 1
                return
            self.held_review = None
            self.pending = dict(item, held_review=True)
            self.counters['held_review_' + verdict] += 1
            self._finish_probe(now_ns, view, apply, verdict,
                'paired_resource_continuation', support)
            return
        # Late windows are more exposed to workload changes. Require the
        # existing deadline-opportunity gate to pass for BOTH roles; missing
        # evidence must not be promoted to confirmation of a tiny effect.
        shapes = {}
        for role in GPU_ROLES:
            shapes[role] = QF4Observer.deadline_shape_ok(
                self.observer.span_budget_hist(*support['pre_span_ns'], role),
                self.observer.span_budget_hist(start, end, role))
        support['comparable'] = bool(support['comparable'] and
                                     all(ok and detail.get('distribution_evidence_sufficient')
                                         and not detail.get('thin_classes')
                                         for ok, detail in shapes.values()))
        entry_shapes = support.get('source_entry_deadline_shapes') or {}
        support['comparable'] = bool(support['comparable'] and
            all((entry_shapes.get(role) or {}).get('n_pre', 0) >= 30 and
                (entry_shapes.get(role) or {}).get('n_post', 0) >= 30
                for role in GPU_ROLES))
        support['held_review'] = True
        support['review_deadline_shapes'] = {r: detail for r, (_, detail) in shapes.items()}
        std = support['std']
        verdict = self._support_verdict(support)
        support.update(verdict=verdict, verdict_reason='held_review_mature_outputs')
        self.last_probe = support
        if verdict == 'UNCERTAIN':
            self.counters['held_review_uncertain'] += 1
            return
        self.held_review = None
        self.pending = dict(item, held_review=True)
        self.counters['held_review_' + verdict] += 1
        self._finish_probe(now_ns, view, apply, verdict,
                           'held_review_mature_outputs', support)

    def _probe_post_span(self, item):
        """The post span of the CURRENT ladder level (T014 probe ladder)."""
        lvl = int((item or {}).get('extensions') or 0)
        ladder = self.probe_post_levels_ns
        return ladder[min(max(lvl, 0), len(ladder) - 1)]

    def _extend_probe(self, item, reason='probe_extended'):
        """Advance the post ladder by ONE level, bounded by the frozen
        ``probe_max_extensions`` and by the ladder length.

        Returns False when no further level exists; the caller then has to
        decide with the evidence it already has (never by extending forever).
        """
        if not isinstance(item, dict):
            return False
        lvl = int(item.get('extensions') or 0)
        if lvl >= int(self.probe_max_extensions) or \
                lvl + 1 >= len(self.probe_post_levels_ns):
            return False
        item['extensions'] = lvl + 1
        item['extended'] = True
        item['extend_reason'] = str(reason)
        return True

    def _lambda_snapshot(self):
        demand = self.demand_cache or {}
        return {c: float((demand.get('lambda_raw') or {}).get(c, 0.0))
                for c in ROOT_CLASSES}

    def _evaluate_probe(self, now_ns, view, apply):
        item = self.pending
        if item is None:
            self.state = 'IDLE'
            return
        post_start = int(item.get('post_start_ns') or 0)
        span = self._probe_post_span(item)
        post_end = post_start + span
        cohorts = [c for c in self.observer.recent_sealed_cohorts_all()
                   if post_start <= c['start_ns'] and c['end_ns'] <= post_end]
        complete = bool(cohorts) and all(c['results_sealed'] and c['wm_sealed']
                                         and not c['incomplete'] for c in cohorts)
        covered = bool(cohorts) and \
            min(c['start_ns'] for c in cohorts) <= post_start and \
            max(c['end_ns'] for c in cohorts) >= post_end
        if not (complete and covered):
            if now_ns < post_end + self.deadline_ns + 500 * MS:
                self.last_reason = 'probe_waiting_mature_results'
                return
            if self._extend_probe(item, 'probe_extended_1s'):
                self.counters['probe_extended'] += 1
                self.last_reason = 'probe_extended_%d' % int(item['extensions'])
                self.last_probe = {'command_id': item['command']['command_id'],
                                   'extended': True, 'comparable': False,
                                   'reason': 'cohorts_not_yet_complete',
                                   'extensions': int(item['extensions']),
                                   'post_span_ns': self._probe_post_span(item)}
                return
            self._finish_probe(now_ns, view, apply, 'UNCERTAIN',
                               'cohorts_never_sealed',
                               {'command_id': item['command']['command_id'],
                                'reason': 'cohorts_never_sealed',
                                'comparable': False})
            return
        support = self._probe_support(now_ns, view, cohorts,
                                      item.get('pre_cohorts') or [])
        std = support['std']
        epsY = float(support['epsilon_Y']['epsilon'])
        epsV = float(support['epsilon_V']['epsilon'])
        dY = float(std['dY'])
        dV = float(std['dV'])
        v_av = bool(std['v_available'])
        comparable = bool(support['comparable'])
        if (support['selection_objective'] == 'latency_timeout' and not comparable
                and now_ns < post_end + self.window_ns + self.tick_ns):
            # A complete 2s observed profile is anchored to its newest source
            # event, not the control clock. The replay at post_end therefore
            # still starts slightly before post_start. Wait one bounded replay
            # publication interval before extending/declaring UNCERTAIN.
            self.last_reason = 'probe_waiting_fresh_paired_profile'
            self.counters['probe_waiting_fresh_paired_profile'] += 1
            return
        # T014 s3.4.5: the verdict is driven ONLY by the comparable real
        # output; a donor loss-rate rise is diagnostic and can never veto a
        # clear net gain on its own (and a post price drift is diagnostic).
        verdict = self._support_verdict(support)
        if verdict == 'ROLLBACK':
            self._finish_probe(now_ns, view, apply, 'ROLLBACK',
                               'comparable_net_loss', support)
            return
        if verdict == 'CONFIRM':
            self._finish_probe(now_ns, view, apply, 'CONFIRM',
                               'comparable_net_gain', support)
            return
        # order 4: extend the post window once to 1 s, then UNCERTAIN back to
        # the last confirmed band recording the failed edge + snapshot.
        if self._extend_probe(item, 'no_positive_evidence_yet'):
            self.counters['probe_extended'] += 1
            self.last_reason = 'probe_extended_%d' % int(item['extensions'])
            self.last_probe = dict(support, extended=True,
                                   extensions=int(item['extensions']),
                                   post_span_ns=self._probe_post_span(item),
                                   reason='no_positive_evidence_yet')
            return
        self._finish_probe(now_ns, view, apply, 'UNCERTAIN',
                           'no_evidence_after_extended_probe', support)

    # -- scoring tick -----------------------------------------------------------------
    def _score_tick(self, now_ns, view, apply):
        sealed = list(self.observer.sealed_windows)
        if not sealed or (not self.continuous_resource and self.last_scored_window is not None
                          and sealed[-1]['index'] <= self.last_scored_window):
            return
        self.last_windows = sealed[-2:]
        self.last_score = {'tick_ns': int(now_ns),
                           'window_index': int(sealed[-1]['index']),
                           'epoch': int(self.epoch), 'reason': None}
        self._append_output_noise()
        self.demand_cache = self._current_lambda(view, now_ns)
        if self.demand_cache is None or self.demand_cache.get('incomplete'):
            self.last_reason = 'demand_window_incomplete_or_missing'
            self.last_score['reason'] = self.last_reason
            return
        if view.get('stall') or view.get('log_backpressure') or view.get('frame_gap_seen'):
            self.last_reason = 'executor_stall_hold'
            self.last_score.update(reason=self.last_reason,
                                   stall=bool(view.get('stall')),
                                   log_backpressure=bool(view.get('log_backpressure')),
                                   frame_gap_seen=bool(view.get('frame_gap_seen')))
            self.counters['stall_holds'] += 1
            return
        replay = view.get('replay')
        validity = self._replay_table_valid(replay, now_ns)
        self.last_replay = dict(validity)
        self.last_score['replay'] = dict(validity)
        if not validity['ok']:
            self.last_reason = validity['reason']
            self.last_score['reason'] = self.last_reason
            return
        if self.continuous_resource:
            identity = (self.epoch, replay.get('snapshot_ns'), replay.get('profile_sha'),
                        replay.get('price_version'))
            if self.last_resource_table == identity:
                self.last_reason = 'resource_waiting_fresh_table'
                self.last_score['reason'] = self.last_reason
                return
            self.last_resource_table = identity
            self.counters['resource_tables_reassessed'] += 1
        c = self.observer.c_star_ms
        roles = {'Fast': min(float(c[k]) for k in ('FAST_FULL', 'FAST_PREFIX', 'FAST_SUFFIX')),
                 'Accurate': float(c['ACCURATE'])}
        trig = self._trigger_both(roles)
        pressure = {r: self._role_pressure(r, sealed, trig[r], roles[r]) for r in GPU_ROLES}
        # C3: the speed channel is a "nothing is going wrong anywhere" channel.
        # A non-zero deficit on EITHER role means work is being lost, so the
        # pressure path (not the H path) must own the tick.
        no_loss_anywhere = not any(pressure[r]['deficit_ms_s'] > 1e-9 for r in GPU_ROLES)
        self.last_score['no_loss_anywhere'] = bool(no_loss_anywhere)
        pressured_roles = [r for r in GPU_ROLES if pressure[r]['pressured']]
        # Pressure triggers an audit of BOTH directions. It is not evidence
        # that transferring service to the largest deficit maximizes net gain.
        direction_role = None
        pressure_audit = bool(pressured_roles)
        self.last_score['pressure'] = pressure
        self.last_score['direction_role'] = direction_role
        high = self._high_coverage(self.demand_cache)
        boot = self._bootstrap_costs(view)
        exploratory = not high['high'] or bool(boot.get(0) or boot.get(1))
        proposals = []
        receivers = []
        for role in GPU_ROLES:
            is_pressured = pressure[role]['pressured']
            other = 'Accurate' if role == 'Fast' else 'Fast'
            speed_audit = (replay.get('selection_objective') == 'quality_weighted_earliness'
                           and trig[role].get('reason') == 'no_persistent_gap'
                           and (no_loss_anywhere or not self.speed_channel_requires_health))
            delivery_audit = (replay.get('selection_objective') == 'correct_ontime'
                              and trig[role].get('reason') == 'no_persistent_gap')
            resource_audit = replay.get('selection_objective') == 'latency_timeout'
            compute = bool(pressure_audit or trig[role].get('trigger') or speed_audit or delivery_audit or resource_audit)
            prop = self._propose_step(
                replay, validity, role, exploratory,
                deficit_steps=(pressure[role]['steps_needed'] if is_pressured else None)) if compute else None
            if (prop and prop.get('action') == 'move'
                    and 'quality_earliness_hat' in prop['row']
                    and prop['row'].get('selection_objective') not in ('correct_ontime', 'latency_timeout')
                    and not prop['row'].get('delivery_positive')
                    and self.speed_channel_requires_health and not no_loss_anywhere):
                self.last_score.setdefault('quality_speed_shadow', {})[role] = prop
                prop = None
            if speed_audit and prop and prop.get('action') == 'move':
                if direction_role is not None or trig[role].get('trigger') or trig[other].get('trigger'):
                    # Resource pressure wins: a no_persistent_gap role never upgrades
                    # to a real trigger while any side is pressured.  On a pressured
                    # tick the speed arm yields entirely (shadow only).
                    self.last_score['quality_speed_shadow'] = prop
                    if direction_role is not None:
                        prop = None
                else:
                    trig[role] = dict(trig[role], trigger=True,
                                      reason='quality_speed_candidate_gain',
                                      audit_basis='fresh_healthy_windows_and_permitted_public_H_gain')
            if delivery_audit and prop and prop.get('action') == 'move':
                trig[role] = dict(trig[role], trigger=True,
                    reason='correct_ontime_candidate_gain',
                    audit_basis='joint_public_correct_ontime_gain_without_speed_gate')
            if resource_audit and prop and prop.get('action') == 'move':
                trig[role] = dict(trig[role], trigger=True,
                    reason='latency_timeout_candidate_gain',
                    audit_basis='same_public_inputs_combined_loss_both_role_timeout_protection')
            if direction_role is not None and not is_pressured and prop is not None:
                # One pressure direction per tick.  The yielding side keeps its
                # proposal as an audit trail only, never as a candidate: with two
                # pressured roles the larger deficit owns the tick, and the other
                # role must not compete for it through its own trigger.
                self.last_score.setdefault('yielded_proposals', {})[role] = prop
                prop = None
            locked = bool(prop and prop.get('action') == 'move' and
                          self._direction_locked('F' if role == 'Fast' else 'A', now_ns))
            receivers.append({'receiver': role, 'trigger': trig[role],
                              'pressure': pressure[role],
                              'proposal': prop, 'direction_locked': locked})
            if (prop and prop.get('action') == 'move' and not locked
                    and (direction_role is None or is_pressured)):
                proposals.append((role, prop))
        self.last_selection = {'trigger': trig, 'triggers': trig, 'receivers': receivers,
                               'high_coverage': high, 'exploratory': exploratory,
                               'pressure': pressure, 'direction_role': direction_role,
                               'score_tick_ns': int(now_ns),
                               'score_window_index': int(sealed[-1]['index'])}
        self.last_table = replay if validity['ok'] else None
        self.last_table_id += 1
        if not proposals:
            self.last_reason = validity['reason'] if not validity['ok'] else 'no_permitted_effective_step'
            self.last_score['reason'] = self.last_reason
            return
        chosen = self._choose_proposal(proposals)
        self.last_score['chosen_receiver'] = chosen[0]
        self.last_shadow = {'receiver': chosen[0], 'proposal': chosen[1], 'not_applied': not self.dynamic}
        if not self.dynamic:
            self.last_reason = 'fixed_shadow_scored'
            self.last_score['reason'] = self.last_reason
            return
        self._select_and_issue(now_ns, view, apply, replay, high, trig, chosen[0],
                               selected_proposal=chosen[1])
        self.last_score['reason'] = self.last_reason
        self.last_score['command_seq_after'] = int(self.command_seq)
        self.last_score['state_after'] = self.state

    def _choose_proposal(self, proposals):
        """Choose an already permitted probe without a Fast-first tie bias.

        Pareto dominance concerns the point predictions, not significance:
        a dominated proposal has no predicted advantage to justify trying it
        first.  All proposals still pass the existing gain/sensitivity gates,
        and only the real post-change probe can confirm an improvement.
        Conflicting predictions retain the Y/V tolerances and prefer a smaller
        move.  Exact ties alternate by epoch, independently of input order.
        """
        preferred = 'Fast' if self.epoch % 2 == 0 else 'Accurate'
        ordered = sorted(proposals, key=lambda p: (p[0] != preferred, p[0]))
        if all(p[1]['row'].get('selection_objective') == 'latency_timeout' for p in ordered):
            from resource_objective import ROUNDOFF
            peak = max(p[1]['row']['resource_effect']['gain'] for p in ordered)
            tied = [p for p in ordered if p[1]['row']['resource_effect']['gain'] >= peak-ROUNDOFF]
            return min(tied, key=lambda p: (abs(p[1]['step_gF']-self.effective_fast), p[0] != preferred))
        if all(p[1]['row'].get('selection_objective') == 'correct_ontime' for p in ordered):
            known = all(p[1]['row'].get('net_gain_Vhat') is not None for p in ordered)
            # Equal sums can differ with completion order. Resolve their tie by
            # delivery count and movement, rather than an insignificant last bit.
            if known:
                peak = max(float(p[1]['row']['net_gain_Vhat']) for p in ordered)
                ordered = [p for p in ordered if float(p[1]['row']['net_gain_Vhat'])
                           >= peak-DELIVERY_GAIN_ROUNDOFF]
            peak = max(float(p[1]['row']['net_gain_Yhat']) for p in ordered)
            ordered = [p for p in ordered if float(p[1]['row']['net_gain_Yhat'])
                       >= peak-DELIVERY_GAIN_ROUNDOFF]
            def delivery_rank(p):
                return (-abs(p[1]['step_gF']-self.effective_fast), p[0] == preferred)
            return max(ordered, key=delivery_rank)
        if all(p[1]['row'].get('net_gain_quality_earliness') is not None for p in ordered):
            # Delivery gains are the primary key; H only breaks ties so a Fast
            # proposal can never beat an Accurate one with better deliveries.
            known = all(p[1]['row'].get('net_gain_Vhat') is not None for p in ordered)
            def rank(p):
                row = p[1]['row']
                gains = ((float(row['net_gain_Vhat']), float(row['net_gain_Yhat']))
                         if known else (float(row['net_gain_Yhat']),))
                return gains + (float(row['net_gain_quality_earliness'])
                    - float(row.get('epsilon_quality_earliness') or 0.0),
                    -abs(p[1]['step_gF']-self.effective_fast), p[0] == preferred)
            return max(ordered, key=rank)
        chosen = ordered[0]
        for other in ordered[1:]:
            a, b = chosen[1]['row'], other[1]['row']
            ay, by = float(a['net_gain_Yhat']), float(b['net_gain_Yhat'])
            av, bv = a.get('net_gain_Vhat'), b.get('net_gain_Vhat')
            quality_known = av is not None and bv is not None
            if quality_known:
                av, bv = float(av), float(bv)
            b_dominates = by >= ay and (not quality_known or bv >= av) and (
                by > ay or (quality_known and bv > av))
            a_dominates = ay >= by and (not quality_known or av >= bv) and (
                ay > by or (quality_known and av > bv))
            if b_dominates:
                chosen = other
                continue
            if a_dominates:
                continue
            ey = max(float(a['epsilon_Y']), float(b['epsilon_Y']))
            if by - ay > ey:
                chosen = other
            elif abs(by - ay) <= ey:
                ev = max(float(a['epsilon_V']), float(b['epsilon_V']))
                if quality_known and bv - av > ev:
                    chosen = other
                elif (not quality_known or abs(bv - av) <= ev) and (
                    abs(other[1]['step_gF'] - self.effective_fast)
                    < abs(chosen[1]['step_gF'] - self.effective_fast) - 1e-12):
                    chosen = other
        return chosen

    def _replay_worker_stats(self):
        fn = getattr(self.replay_worker, 'stats', None)
        if fn is None:
            return None
        try:
            return fn()
        except Exception:
            return None

    def _replay_candidate(self, replay, now_ns):
        """The WS-D worker result for THIS band (interface s6.5).

        Missing / stale / invalid worker output is an explicit
        ``prediction_unknown``; it is never silently replaced by the local
        point model and never reported as resource evidence.
        """
        if not isinstance(replay, dict) or not bool(replay.get('valid')):
            reason = 'prediction_unknown'
            if isinstance(replay, dict) and replay.get('invalid_reason'):
                reason = str(replay['invalid_reason'])
            return None, False, reason, None
        snap_ns = replay.get('snapshot_ns')
        age = None if snap_ns is None else int(now_ns) - int(snap_ns)
        if replay.get('overflow'):
            return None, False, 'prediction_unknown', age
        if age is not None and age > int(self.cache_max_age_ns):
            return None, False, 'prediction_unknown', age
        for c in (replay.get('candidates') or []):
            try:
                if abs(float(c.get('gF', -1.0)) -
                       round(float(self.effective_fast), 10)) < 1e-9:
                    return c, True, 'replay_current_band', age
            except (TypeError, ValueError):
                continue
        return None, True, 'replay_no_current_band', age

    def _sensitivity_not_worse(self, scen, cur):
        """TASK section 5.3 exploratory branch (b): all THREE sensitivity
        scenarios of the WS-D worker predict the point total Y/V not worse."""
        if not scen or cur is None:
            return False
        rows = []
        if isinstance(scen, dict):
            for name in ('point', 'unknown_F', 'unknown_A', 'unknown_Abstain'):
                row = scen.get(name)
                if isinstance(row, dict):
                    rows.append(row)
        elif isinstance(scen, list):
            rows = [r for r in scen if isinstance(r, dict)]
        if len(rows) < 3:
            return False
        for r in rows:
            y = r.get('Yhat')
            v = r.get('Vhat')
            if y is None and v is None:
                return False
            if y is not None and cur.get('Yhat') is not None and \
                    float(y) < float(cur['Yhat']) - 1e-9:
                return False
            if v is not None and cur.get('Vhat') is not None and \
                    float(v) < float(cur['Vhat']) - 1e-9:
                return False
        return True

    # -- T014 s3.3: ONE authoritative deadline-replay decision ----------------
    def _replay_table_valid(self, replay, now_ns):
        """Validity + fingerprint guard for the ONE authoritative replay table.

        T014 s3.3.9: missing / stale / overflow / other-epoch worker output is an
        explicit ``prediction_unknown``; the local point model never silently
        replaces it and the fixed arm uses the same guard.
        """
        out = {'ok': False, 'reason': 'prediction_unknown', 'age_ns': None,
               'epoch': None, 'price_version': None, 'profile_sha': None,
               'snapshot_ns': None, 'input_n': None}
        if not isinstance(replay, dict) or not bool(replay.get('valid')):
            if isinstance(replay, dict) and replay.get('invalid_reason'):
                out['reason'] = str(replay['invalid_reason'])
            return out
        if self.config.get('delivery_objective') in ('correct_ontime', 'latency_timeout') and replay.get('selection_objective') != self.config['delivery_objective']:
            out['reason'] = 'cache_invalid_delivery_objective'
            return out
        out['snapshot_ns'] = replay.get('snapshot_ns')
        out['epoch'] = replay.get('epoch')
        out['price_version'] = replay.get('price_version')
        out['profile_sha'] = replay.get('profile_sha')
        out['input_n'] = replay.get('input_n')
        out['age_ns'] = (None if out['snapshot_ns'] is None
                         else int(now_ns) - int(out['snapshot_ns']))
        if replay.get('overflow'):
            out['reason'] = 'profile_overflow_hold'
            return out
        if out['age_ns'] is None or out['age_ns'] > int(self.cache_max_age_ns):
            out['reason'] = 'prediction_unknown_stale'
            return out
        if out['epoch'] is not None and int(out['epoch']) != int(self.epoch):
            out['reason'] = 'cache_invalid_epoch'
            return out
        out['ok'] = True
        out['reason'] = 'replay_table_valid'
        return out

    @staticmethod
    def _finite_target(value):
        """A numeric target field: never a bool, NaN, inf or off-axis value."""
        if isinstance(value, bool) or value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(v) or math.isinf(v):
            return None
        if v < AXIS_MIN_FAST - 1e-9 or v > AXIS_MAX_FAST + 1e-9:
            return None
        return v

    @staticmethod
    def _finite_epsilon(value):
        if value is None or isinstance(value, bool):
            return None
        try:
            value = float(value)
        except (ValueError, TypeError):
            return None
        return value if math.isfinite(value) and value >= 0 else None

    def _replay_rows_guard(self, rows, replay=None):
        """T014 s3.3.3: every candidate row carries a finite float target and an
        explicit bool ``is_effective_step``; a bool/NaN/inf/off-grid/out-of-range
        numeric target is refused (``float(False)`` can never become a share)."""
        if not rows:
            return {'ok': False, 'reason': 'no_candidates'}
        grid_vals = []
        for g in ((replay or {}).get('grid') or []):
            v = self._finite_target(g)
            if v is not None:
                grid_vals.append(round(v, 10))
        for row in rows:
            if (replay or {}).get('selection_objective') in ('correct_ontime', 'latency_timeout') and row.get('selection_objective') != replay['selection_objective']:
                return {'ok': False, 'reason': 'row_delivery_objective_mismatch'}
            v = self._finite_target(row.get('gF'))
            if v is None:
                return {'ok': False, 'reason': 'gF_not_finite', 'row_gF': row.get('gF')}
            if 'is_effective_step' not in row:
                return {'ok': False, 'reason': 'missing_is_effective_step'}
            if not isinstance(row.get('is_effective_step'), bool):
                return {'ok': False, 'reason': 'is_effective_step_not_bool'}
            if row.get('selection_objective') == 'latency_timeout':
                from resource_objective import paired_effect
                outcomes = row.get('resource_outcomes')
                effect = paired_effect(outcomes, outcomes, row.get('resource_span_s', 0))
                if not effect.get('available'):
                    return {'ok': False, 'reason': 'resource_outcomes_unavailable'}
            if ((replay or {}).get('selection_objective') == 'quality_weighted_earliness'
                    and row.get('V_available', row.get('Vhat') is not None)):
                for key in ('quality_earliness_hat', 'net_gain_quality_earliness',
                            'epsilon_quality_earliness'):
                    value = row.get(key)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                            not math.isfinite(float(value)):
                        return {'ok': False, 'reason': key + '_unavailable'}
            for key in ('Yhat', 'Vhat', 'net_gain_Yhat', 'net_gain_Vhat',
                        'epsilon_Y', 'epsilon_V'):
                val = row.get(key)
                if val is None:
                    continue
                if isinstance(val, bool):
                    return {'ok': False, 'reason': key + '_bool'}
                try:
                    f = float(val)
                except (TypeError, ValueError):
                    return {'ok': False, 'reason': key + '_not_numeric'}
                if math.isnan(f) or math.isinf(f):
                    return {'ok': False, 'reason': key + '_nan_inf'}
            if grid_vals and round(v, 10) not in grid_vals:
                return {'ok': False, 'reason': 'gF_off_grid', 'row_gF': v}
        return {'ok': True, 'reason': None, 'n_rows': len(rows)}

    def _replay_current_row(self, rows):
        cur = float(self.effective_fast)
        for row in rows:
            v = self._finite_target(row.get('gF'))
            if v is not None and abs(v - cur) < 1e-9:
                return row
        return None

    def _replay_sensitivity_ok(self, row):
        for name in ('unknown_F', 'unknown_A', 'unknown_Abstain'):
            scen = (row.get('sensitivity') or {}).get(name) or {}
            dy, dv = scen.get('dYhat'), scen.get('dVhat')
            ey = self._finite_epsilon(row.get('epsilon_Y'))
            ev = self._finite_epsilon(row.get('epsilon_V'))
            if ey is None or dy is None or not math.isfinite(float(dy)) or float(dy) < -ey:
                return False
            if dv is not None and (ev is None or not math.isfinite(float(dv)) or float(dv) < -ev):
                return False
            if ('quality_earliness_hat' in row
                    and row.get('selection_objective') != 'correct_ontime'
                    and row.get('V_available', row.get('net_gain_Vhat') is not None)):
                dh = scen.get('dQualityEarliness')
                eh = self._finite_epsilon(row.get('epsilon_quality_earliness'))
                if dh is None or eh is None or not math.isfinite(float(dh)) or float(dh) < -eh:
                    return False
        return True

    @staticmethod
    def _role_outcomes_nonharm(delta, delivery_only=False):
        if not isinstance(delta, dict):
            return False
        for role in GPU_ROLES:
            result = delta.get(role)
            if not isinstance(result, dict):
                return False
            keys = ('ontime', 'quality') if delivery_only else ('completed', 'ontime', 'quality', 'quality_earliness', 'all_quality')
            for key in keys:
                if key not in result:
                    return False
                value = result.get(key)
                if value is None and key in ('quality', 'quality_earliness', 'all_quality'):
                    continue  # Preserve public-quality-unknown Y fallback.
                if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                        not math.isfinite(value) or value < -1e-9):
                    return False
        return True

    def _replay_minimal_effective_step(self, rows, cur, direction, exploratory,
                                       deficit_ms_s=None, deficit_steps=None):
        """Select a permitted quality/speed improvement on the triggered axis.

        Service-debt rows maximize quality-weighted delivery earliness;
        historical rows retain the established minimal-step protocol.
        """
        cur_g = self._finite_target(cur.get('gF'))
        if cur_g is None:
            return None, [{'gF': None, 'reason': 'current_row_gF_invalid'}]
        if cur.get('selection_objective') == 'latency_timeout':
            from resource_objective import paired_effect, ROUNDOFF
            permitted, rejected = [], []
            for row in rows:
                g = self._finite_target(row.get('gF'))
                if g is None or abs(g-cur_g) < 1e-9 or (
                    direction == 'F' and g < cur_g) or (direction == 'A' and g > cur_g):
                    continue
                effect = paired_effect(row.get('resource_outcomes'),
                    cur.get('resource_outcomes'), row.get('resource_span_s', 0))
                sensitivity_ok = True
                for scen in ('unknown_F', 'unknown_A', 'unknown_Abstain'):
                    a = (row.get('sensitivity') or {}).get(scen) or {}
                    b = (cur.get('sensitivity') or {}).get(scen) or {}
                    test = paired_effect(a.get('resource_outcomes'), b.get('resource_outcomes'),
                        row.get('resource_span_s', 0))
                    sensitivity_ok &= bool(test.get('available') and test.get('roles_protected')
                        and test['gain'] >= -ROUNDOFF)
                if row.get('point_valid') is False or not effect.get('permitted') or not sensitivity_ok:
                    rejected.append({'gF': g, 'reason': 'resource_gain_or_role_timeout_protection',
                        'resource_effect': effect, 'sensitivity_ok': sensitivity_ok})
                    continue
                permitted.append({'gF': g, 'move': abs(g-cur_g), 'row': row,
                    'resource_effect': effect, 'net_gain_history_used': False})
            if not permitted:
                return None, rejected
            peak = max(p['resource_effect']['gain'] for p in permitted)
            tied = [p for p in permitted if p['resource_effect']['gain'] >= peak-ROUNDOFF]
            return min(tied, key=lambda p: (p['move'], p['gF'])), rejected
        best = None
        rejected = []
        admissible = []
        delivery_only = cur.get('selection_objective') == 'correct_ontime'
        quality_speed = 'quality_earliness_hat' in cur and not delivery_only
        joint_delivery = quality_speed or delivery_only
        for row in rows:
            g = self._finite_target(row.get('gF'))
            if g is None or abs(g - cur_g) < 1e-9:
                continue
            if direction == 'F' and not (g > cur_g + 1e-9):
                continue
            if direction == 'A' and not (g < cur_g - 1e-9):
                continue
            eps_y = self._finite_epsilon(row.get('epsilon_Y'))
            eps_v = self._finite_epsilon(row.get('epsilon_V'))
            if eps_y is None or eps_v is None:
                rejected.append({'gF': g, 'reason': 'invalid_epsilon'})
                continue
            dy = row.get('net_gain_Yhat')
            dv = row.get('net_gain_Vhat')
            if isinstance(dy, bool) or dy is None:
                rejected.append({'gF': g, 'reason': 'dYhat_missing'})
                continue
            dy = float(dy)
            dv = None if dv is None else float(dv)
            v_av = bool(row.get('V_available', dv is not None)) and dv is not None
            y_branch = dy >= eps_y and (not v_av or dv >= -eps_v)
            v_branch = bool(v_av and dv >= eps_v and dy >= -eps_y)
            strictly_positive = dy > 0 or (v_av and dv > 0)
            # ``dv`` is None whenever Vhat is unavailable (the row guard tolerates
            # a missing net_gain_Vhat), so every test below must short-circuit on
            # v_av / dv is None instead of comparing None with a number.
            strict_positive = dy > 0 or (v_av and dv > 0)
            if delivery_only:
                # 139.5s of the no-loan run had equal Y and V differing by
                # 2.84e-14 solely from summation order, yet moved F20% -> F70%.
                # One nanofraction of a predicted task/s is numerical equality;
                # this does not reduce any empirical probe/noise threshold.
                permit = dy >= -DELIVERY_GAIN_ROUNDOFF and (
                    dv > DELIVERY_GAIN_ROUNDOFF if v_av else dy > DELIVERY_GAIN_ROUNDOFF)
            elif quality_speed:
                # A measured loss opens an audit; it cannot turn a zero-gain
                # prediction into evidence for a resource transfer. Compare
                # total deliveries, including the donor, on every grid point.
                dh = row.get('net_gain_quality_earliness')
                eh = self._finite_epsilon(row.get('epsilon_quality_earliness'))
                speed_gain = (v_av and dv == 0 and dy == 0 and eh is not None
                              and isinstance(dh, (int, float)) and not isinstance(dh, bool)
                              and math.isfinite(float(dh)) and float(dh) > max(eh, 0.0))
                speed_nonharm = (not v_av or
                    (isinstance(dh, (int, float)) and not isinstance(dh, bool)
                     and math.isfinite(float(dh)) and float(dh) >= 0))
                permit = (dy >= 0 and (not v_av or dv >= 0) and speed_nonharm
                          and (strict_positive or speed_gain))
            elif deficit_steps is not None:
                # Pressure-driven tick: the MEASURED deficit justifies the move.
                # A non-harmful row is therefore admissible without a predicted
                # gain (that is the whole point of v19) and Vhat may be absent.
                permit = dy >= 0 and (dv is None or dv >= 0)
            else:
                permit = (y_branch or v_branch) and strictly_positive
            if not permit:
                rejected.append({'gF': g, 'reason': 'permit_not_met',
                                 'dY': dy, 'dV': dv, 'epsY': eps_y, 'epsV': eps_v})
                continue
            if joint_delivery and row.get('role_protection_required'):
                if (not self._role_outcomes_nonharm(row.get('role_outcome_delta'), delivery_only) or
                        not all(self._role_outcomes_nonharm(
                            (row.get('sensitivity') or {}).get(scen, {}).get('role_outcome_delta'), delivery_only)
                            for scen in ('unknown_F', 'unknown_A', 'unknown_Abstain'))):
                    rejected.append({'gF': g, 'reason': 'role_outcome_harm_or_unknown',
                                     'role_outcome_delta': row.get('role_outcome_delta')})
                    continue
            # ``point_valid`` is preserved explicitly even where the H-derived
            # marker is not a gate: a candidate built on an invalid grid point
            # must never become a share.  (The replay table as a whole is also
            # refused upstream when point_valid is false --
            # candidate_replay_t012.py:1737 ``base['valid'] = bool(point_valid)``
            # checked by _replay_table_valid -- so this is defence in depth.)
            if row.get('point_valid') is False:
                rejected.append({'gF': g, 'reason': 'point_invalid'})
                continue
            if not joint_delivery and deficit_steps is None and not bool(row.get('is_effective_step')):
                # For service_debt rows ``is_effective_step`` IS the H-derived
                # marker (candidate_replay_t012.py: effective = point_valid and
                # direction_ok and gain_y >= 0 and gain_v >= 0 and gain_h > 0
                # and gain_h >= h_eps).  H must not gate a candidate that
                # strictly adds deliveries, so it applies to historical rows only.
                rejected.append({'gF': g, 'reason': 'not_effective_step'})
                continue
            # Known-quality service_debt candidates must preserve H as well as
            # deliveries. H's margin ranks otherwise equal delivery gains.
            if quality_speed:
                dh = row.get('net_gain_quality_earliness')
                if v_av and (isinstance(dh, bool) or dh is None or not math.isfinite(float(dh))):
                    rejected.append({'gF': g, 'reason': 'quality_gain_invalid', 'gain': dh})
                    continue
                row = dict(row)
                row['h_shadow'] = float(dh) if dh is not None else None
                row['delivery_positive'] = bool(strict_positive)
                row['direction_source'] = ('resource_pressure' if deficit_steps is not None
                                           else 'quality_speed_channel')
            if (exploratory or bool(row.get('low_confidence'))) \
                    and not self._replay_sensitivity_ok(row):
                rejected.append({'gF': g, 'reason': 'sensitivity_unavailable'})
                continue
            move = abs(g - cur_g)
            admissible.append({'gF': g, 'move': move, 'row': row})
        if not admissible:
            return None, rejected

        deficit_uncovered = False
        if deficit_steps is not None and not joint_delivery:
            # v19: a pressure-driven tick must move FAR ENOUGH to cover the
            # measured deficit, not merely as little as possible.  Narrow the
            # admissible set to the smallest step that covers ``deficit_steps``;
            # when no admitted step reaches it keep the axis-bound largest step
            # and record the shortfall.  Applied to BOTH table kinds so the
            # magnitude law is not lost on a historical (deadline_output) table.
            need = float(deficit_steps) * GRID
            sufficient = [c for c in admissible if c['move'] >= need - 1e-12]
            if sufficient:
                sel_move = min(c['move'] for c in sufficient)
            else:
                sel_move = max(c['move'] for c in admissible)
                deficit_uncovered = True
            admissible = [c for c in admissible if abs(c['move'] - sel_move) < 1e-12]

        if joint_delivery:
            # Joint outcome gain owns amplitude. Deficit-derived steps remain
            # diagnostics; clipping to them can discard the best permitted row.
            known = all(c['row'].get('net_gain_Vhat') is not None for c in admissible)
            if delivery_only:
                if known:
                    peak = max(float(c['row']['net_gain_Vhat']) for c in admissible)
                    admissible = [c for c in admissible if float(c['row']['net_gain_Vhat'])
                                  >= peak-DELIVERY_GAIN_ROUNDOFF]
                peak = max(float(c['row']['net_gain_Yhat']) for c in admissible)
                admissible = [c for c in admissible if float(c['row']['net_gain_Yhat'])
                              >= peak-DELIVERY_GAIN_ROUNDOFF]
            def rank(c):
                r = c['row']
                gains = ((float(r['net_gain_Vhat']), float(r['net_gain_Yhat']))
                         if known else (float(r['net_gain_Yhat']),))
                if delivery_only:
                    return (-c['move'], -c['gF'])
                speed_margin = (float(r.get('h_shadow') or 0.0)
                                - float(r.get('epsilon_quality_earliness') or 0.0))
                return gains + (speed_margin, -c['move'], -c['gF'])
            best = max(admissible, key=rank)
            best['net_gain_history_used'] = False
            if deficit_uncovered:
                best['deficit_uncovered'] = True
            return best, rejected

        min_move = min(c['move'] for c in admissible)
        tied = [c for c in admissible if abs(c['move'] - min_move) < 1e-12]
        receiver = 'Fast' if direction == 'F' else 'Accurate'
        if len(tied) == 1:
            best = tied[0]
            best['net_gain_history_used'] = False
        else:
            default_best = sorted(tied, key=lambda c: c['gF'])[0]
            def _tie_key(c):
                v = self.net_gain_view(receiver, c['gF'], self.epoch)
                mdy = v['mean_dY']
                return (mdy is not None, mdy if mdy is not None else -1e18, -c['gF'])
            hist_views = [self.net_gain_view(receiver, c['gF'], self.epoch) for c in tied]
            has_hist = any(v['mean_dY'] is not None for v in hist_views)
            sorted_tied = sorted(tied, key=_tie_key, reverse=True)
            best = sorted_tied[0]
            best['net_gain_history_used'] = bool(has_hist and best['gF'] != default_best['gF'])
        if deficit_uncovered:
            best['deficit_uncovered'] = True
        return best, rejected

    def _propose_step(self, replay, validity, receiver, exploratory, deficit_steps=None):
        """The ONE proposal function shared by both arms (T014 s3.3.2/s3.3.9)."""
        direction = 'F' if receiver == 'Fast' else 'A'
        out = {'action': 'hold', 'reason': None, 'direction': direction,
               'step_gF': None, 'step': None, 'basis': 'replay_deadline_table',
               'rejected': [], 'current_row': None, 'row_guard': None,
               'deficit_steps': deficit_steps}
        if not validity.get('ok'):
            out['reason'] = validity.get('reason') or 'prediction_unknown'
            return out
        rows = [r for r in (replay.get('candidates') or []) if isinstance(r, dict)]
        guard = self._replay_rows_guard(rows, replay)
        out['row_guard'] = guard
        if not guard['ok']:
            out['reason'] = 'replay_row_invalid_' + str(guard['reason'])
            return out
        cur = self._replay_current_row(rows)
        out['current_row'] = cur
        if cur is None:
            out['reason'] = 'replay_current_band_missing'
            return out
        if replay.get('selection_objective') == 'latency_timeout':
            # Empty score turns now advance explicitly. Each role needs a
            # positive allocation, not a dedicated card. The minimum share
            # is independent of the precision used to express that share.
            lower, upper = self.minimum_fast_share, float(CARD_COUNT)-self.minimum_accurate_share
            out['structural_role_reservation'] = {'min_gF': lower, 'max_gF': upper,
                'min_gA': self.minimum_accurate_share,
                'basis': 'explicit_asymmetric_minimum_shares_with_empty_score_turns'}
            rows = [r for r in rows if r is cur or lower <= float(r['gF']) <= upper]
        demand = (replay or {}).get('demand_reference') or {}
        if demand.get('gF_ref') is not None:
            df, da = demand.get('D_F_ms_s'), demand.get('D_A_ms_s')
            if df is not None and da is not None and 0 < df+da <= CARD_COUNT * 1000:
                # Reserve at least the measured offered mean work for each
                # present flow. Endpoint idle loans are extra service, not a
                # substitute for a flow's future resource contract. These are
                # work-rate bounds, never a 50ms deadline feasibility proof.
                lower, upper = float(df)/1000., CARD_COUNT-float(da)/1000.
                out['work_reservation'] = {'min_gF': lower, 'max_gF': upper,
                                            'demand': demand}
                rows = [r for r in rows if r is cur or
                        lower-1e-9 <= float(r['gF']) <= upper+1e-9]
        step, rejected = self._replay_minimal_effective_step(
            rows, cur, direction, exploratory, deficit_steps=deficit_steps)
        out['rejected'] = rejected
        if step is None:
            out['reason'] = 'no_effective_step_' + direction
            return out
        # T014 s3.2.5: the DONOR is the card that gives the time away, i.e. the
        # role OPPOSITE to the receiver.  The old row echoed the receiver here,
        # which made the recorded supplier equal to the demander.
        donor = 'Accurate' if receiver == 'Fast' else 'Fast'
        out.update({'action': 'move', 'step_gF': float(step['gF']),
                    'step': {'target_gF': float(step['gF']), 'donor': donor,
                             'receiver': receiver,
                             'basis': 'replay_effective_step'},
                    'row': step['row'], 'target_gF': float(step['gF']),
                    'net_gain_history_used': bool(step.get('net_gain_history_used', False)),
                    'reason': 'replay_effective_step'})
        return out

    def _select_and_issue(self, now_ns, view, apply, table, high, trig, receiver,
                          selected_proposal=None):
        direction = 'F' if receiver == 'Fast' else 'A'
        if self._direction_locked(direction, now_ns):
            self.last_reason = 'direction_locked_' + direction
            self.counters['direction_locked_holds'] += 1
            return None
        # T014 s3.3.2: the deadline replay table is the ONLY authority for a real
        # move; evaluate_candidates/select_target/L,K/idle stay diagnostic and
        # can no longer veto the move or substitute its step.
        replay = (view or {}).get('replay')
        validity = self._replay_table_valid(replay, now_ns)
        self.last_replay = dict(validity, worker=self._replay_worker_stats())
        if not validity['ok']:
            self.last_reason = validity['reason']
            self.counters['prediction_unknown_holds'] += 1
            return None
        if selected_proposal is not None:
            # Revalidation may reject a scored decision, but must never silently
            # substitute a different table or discard its pressure magnitude.
            identity = ('snapshot_ns', 'profile_sha', 'price_version', 'epoch')
            if (not isinstance(table, dict) or
                    any(replay.get(k) != table.get(k) for k in identity) or
                    selected_proposal.get('action') != 'move' or
                    selected_proposal.get('direction') != direction):
                self.last_reason = 'scored_proposal_invalidated'
                self.counters['select_holds'] += 1
                return None
        bootstrap = self._bootstrap_costs(view)
        # T014 s3.3.6: an exploratory situation still needs a buildable point and
        # the three sensitivity scenarios of the SAME candidate.
        exploratory = bool(not high['high'] or bootstrap.get(0) or bootstrap.get(1))
        prop = self._propose_step(
            replay, validity, receiver, exploratory,
            deficit_steps=(selected_proposal.get('deficit_steps')
                           if selected_proposal is not None else None))
        if selected_proposal is not None and (
                prop.get('action') != 'move' or
                prop.get('step_gF') != selected_proposal.get('step_gF') or
                prop.get('row') != selected_proposal.get('row') or
                prop.get('current_row') != selected_proposal.get('current_row')):
            self.last_reason = 'scored_proposal_invalidated'
            self.counters['select_holds'] += 1
            return None
        self.last_selection = dict(self.last_selection or {}, selection=prop,
                                   receiver=receiver,
                                   exploratory=bool(exploratory),
                                   basis='replay_deadline_table',
                                   net_gain_history_used=bool(prop.get('net_gain_history_used', False)))
        if prop.get('action') != 'move':
            self.last_reason = str(prop.get('reason'))
            self.counters['select_holds'] += 1
            return None
        guard_fn = (view or {}).get('live_candidate_guard')
        live_guard = None
        if callable(guard_fn):
            try:
                live_guard = guard_fn(prop['step_gF'])
                if not isinstance(live_guard, dict):
                    live_guard = {'ok': False, 'reason': 'live_calendar_unknown'}
            except Exception as exc:
                live_guard = {'ok': False, 'reason': 'live_calendar_unknown',
                              'error': repr(exc)}
        if live_guard is not None and not live_guard.get('ok'):
            self.last_selection['live_guard'] = live_guard
            self.last_reason = str(live_guard.get('reason') or 'live_calendar_unknown')
            self.counters['live_calendar_holds'] += 1
            return None
        # Only historical objectives need a frozen pre/post probe. Current
        # resource control resumes evaluation as soon as the change settles.
        pre = None if self.continuous_resource else self._freeze_pre_cohorts(now_ns)
        if pre is None and not self.continuous_resource:
            self.last_reason = 'probe_waiting_pre_cohorts'
            self.counters['probe_waiting_pre'] += 1
            return None
        info = {'trigger': trig, 'selection': prop, 'step': prop['step'],
                'selection_objective': (prop['row'].get('selection_objective') or
                    ('quality_weighted_earliness' if 'quality_earliness_hat' in prop['row'] else 'deadline_output')),
                'quality_speed_quantum': (float(prop['row'].get('quality_speed_quantum'))
                    if prop['row'].get('quality_speed_quantum') is not None else
                    float(prop['row'].get('epsilon_quality_earliness') or .5)/.5),
                'high_coverage': high, 'exploratory': bool(exploratory),
                'receiver': receiver, 'direction': direction,
                'pre_cohorts': pre,
                'replay': {'gF': (prop.get('current_row') or {}).get('gF'),
                           'Yhat': (prop.get('current_row') or {}).get('Yhat'),
                           'Vhat': (prop.get('current_row') or {}).get('Vhat'),
                           'step_gF': prop.get('step_gF'),
                           'profile_sha': validity.get('profile_sha'),
                           'snapshot_ns': validity.get('snapshot_ns'),
                           'price_version': validity.get('price_version')},
                'gate_probe': None, 'live_guard': live_guard}
        return self._issue(now_ns, view, apply, prop['step_gF'], info, 'normal')

    def _real_snapshot(self, view, now_ns=None):
        """The REAL values a failed direction is unlocked against (TASK s6).

        Saved at the failure: source lambdas, the public source composition,
        every participating card/kind mean + gate, and the sealed index.
        """
        demand = self.demand_cache or {}
        lam = {c: float((demand.get('lambda_raw') or {}).get(c, 0.0))
               for c in ROOT_CLASSES}
        reveal = ((demand.get('EARLY') or {}).get('reveal') or {})
        composition = {role: float(reveal.get(role) or 0.0)
                       for role in ('Fast', 'Accurate', 'Abstain')}
        means = {}
        gates = {}
        for g in CARD_IDS:
            for kind in GPU_KINDS:
                key = '%d:%s' % (g, kind)
                try:
                    means[key] = self.cost_book.mean_ns(g, kind)
                    gates[key] = self.cost_book.gate_ns(g, kind)
                except Exception:
                    continue
        return {'lambda': lam, 'composition': composition,
                'mean_ns': means, 'gate_ns': gates,
                'taken_ns': int(now_ns) if now_ns is not None else None,
                'epoch': int(self.epoch),
                'sealed_index': (int(self.observer.sealed_windows[-1]['index'])
                                 if self.observer.sealed_windows else None)}

    def _unlock_evidence(self, ref):
        """TASK section 6 unlock evidence: at least two NEW healthy same-epoch
        windows since the failure AND one real change versus the saved REAL
        snapshot (lambda > 20%, composition TV > 0.1, or a participating
        mean/gate move > 10%)."""
        ref = ref or {}
        snap = ref.get('snapshot') or {}
        taken_ns = int(ref.get('taken_ns') or snap.get('taken_ns') or 0)
        fresh = 0
        for w in self.observer.sealed_windows:
            if int(w['start_ns']) < taken_ns or self._window_epoch(w) != self.epoch:
                continue
            f = w.get('flags') or {}
            if f.get('transition') or f.get('probe') or f.get('stall') or \
                    f.get('source_gap') or f.get('log_backpressure'):
                continue
            fresh += 1
        out = {'fresh_healthy_windows': fresh, 'reasons': []}
        if fresh < 2:
            return False, out
        lam = self._lambda_snapshot()
        base = snap.get('lambda') or {}
        for cls in ROOT_CLASSES:
            a = float(base.get(cls) or 0.0)
            b = float(lam.get(cls) or 0.0)
            if a > 0.0 and abs(b - a) / a > self.lambda_invalidate_rel:
                out['reasons'].append('lambda_change_' + cls)
            elif a <= 0.0 and b >= 2.0:
                # 0 -> at least one root per 500 ms counts as a change
                out['reasons'].append('lambda_new_' + cls)
        reveal = (((self.demand_cache or {}).get('EARLY') or {})
                  .get('reveal') or {})
        now_comp = {role: float(reveal.get(role) or 0.0)
                    for role in ('Fast', 'Accurate', 'Abstain')}
        tv = tv_distance(snap.get('composition') or {}, now_comp)
        out['composition_tv'] = tv
        if tv is not None and tv > self.composition_tv_limit:
            out['reasons'].append('source_composition_tv')
        base_mean = snap.get('mean_ns') or {}
        base_gate = snap.get('gate_ns') or {}
        for key, old, fn in (
                [(k, v, self.cost_book.mean_ns) for k, v in base_mean.items()] +
                [(k, v, self.cost_book.gate_ns) for k, v in base_gate.items()]):
            if old in (None, 0):
                continue
            try:
                g_txt, kind = key.split(':', 1)
                new = fn(int(g_txt), kind)
            except Exception:
                continue
            if new is None:
                continue
            if abs(float(new) - float(old)) / abs(float(old)) > \
                    MEAN_GATE_CHANGE_REL:
                out['reasons'].append('cost_change_' + key)
        return bool(out['reasons']), out

    def _renew_trial_budget(self, direction, now_ns):
        """A changed public workload earns a new bounded trial context.

        The two-trial limit remains at issue entry. It is renewed only after
        the existing failure-snapshot guard proves two fresh usable windows
        and a real source/price change; time passing or queue flicker cannot
        renew it. The opposite direction and prior failure records stay intact.
        """
        contexts = getattr(self, 'trial_context', {})
        ref = contexts.get(direction)
        if ref is None:
            return False
        unlocked, evidence = self._unlock_evidence(ref)
        ref['unlock_evidence'] = evidence
        if not unlocked:
            return False
        previous = int(self.trials_by_direction['unconfirmed_' + direction])
        self.trials_by_direction['unconfirmed_' + direction] = 0
        contexts.pop(direction, None)
        self.counters['trial_budget_renewed_' + direction] += 1
        self.last_trial_renewal = {'direction': direction, 'tick_ns': int(now_ns),
                                   'previous_unconfirmed': previous,
                                   'snapshot': ref.get('snapshot'),
                                   'evidence': evidence}
        return True

    def _direction_locked(self, direction, now_ns):
        """TASK section 6: only the FAILED edge direction stays locked and it
        is released by real evidence compared with the snapshot saved at THAT
        failure; a failed A edge must never lock F (or the reverse)."""
        if self.continuous_resource:
            return False
        edge = self.locked_edges.get(direction)
        if edge is None:
            if self.trials_by_direction['unconfirmed_' + direction] >= \
                    MAX_UNCONFIRMED_TRIALS_PER_DIRECTION:
                return not self._renew_trial_budget(direction, now_ns)
            return False
        ref = {'snapshot': edge.get('snapshot') or {},
               'taken_ns': edge.get('created_ns'),
               'direction': direction, 'reason': edge.get('reason')}
        unlocked, ev = self._unlock_evidence(ref)
        edge['unlock_evidence'] = ev
        if not unlocked:
            return True
        self.locked_edges.pop(direction, None)
        # A locked ROLLBACK edge and the per-direction trial budget can share
        # one saved context. Releasing the edge must release both guards.
        if direction in getattr(self, 'trial_context', {}):
            self._renew_trial_budget(direction, now_ns)
        if not self.locked_edges:
            self._lock_reference = None
        self.counters['direction_unlocked'] += 1
        return False

    def _donor_margin_ok(self, view, direction, sel):
        """Low-coverage supply-margin gate (TASK.md section 7): one minimal
        effective probe only when the donor side shows two fresh windows
        without resource loss / growth and real idle capacity covering the
        candidate's predicted supply loss for 1 s."""
        # TASK section 5.3: donor for increasing A is Fast, for increasing F
        # it is Accurate (always the reverse of the receiver).
        donor = 'Fast' if direction == 'A' else 'Accurate'
        for w in (self.last_windows or [])[-2:]:
            res = (w.get('residuals') or {}).get(donor) or {}
            if res.get('resource_loss_ms_s', 0) > 0 or \
                    res.get('growth_ms_s', 0) > 0:
                return {'ok': False, 'reason': 'donor_not_healthy'}
        if self.last_table is None:
            return {'ok': False, 'reason': 'no_candidate_table'}
        cur = current_candidate(self.last_table, self.effective_fast)
        tgt = None
        for c in self.last_table['candidates']:
            if abs(c['gF'] - float(sel['target_gF'])) < 1e-9:
                tgt = c
                break
        if cur is None or tgt is None:
            return {'ok': False, 'reason': 'candidates_missing'}
        lost = max(0.0, cur['K_F'] - tgt['K_F']) if direction == 'A' \
            else max(0.0, cur['K_A'] - tgt['K_A'])
        if lost <= 0:
            return {'ok': True, 'reason': 'no_supply_loss_predicted'}
        need = lost * 1000.0  # reference-ms of supply work per second
        cards = (view or {}).get('cards') or []
        idle_ns = (view or {}).get('idle_ns') or {}
        horizon = max(float((view or {}).get('idle_horizon_ns') or
                            (2 * WINDOW_NS)), 1.0)
        idle = 0.0
        for g in CARD_IDS:
            eff = float(cards[g].get('effective', 1.0)) if g < len(cards) else 1.0
            # card 0 is Fast (available share = gF), card 1 is Accurate (1-gF)
            share = eff if g == 0 else (1.0 - eff)
            idle_ratio = float(idle_ns.get(g, 0.0)) / horizon
            idle += idle_ratio * share * 1000.0
        if idle + 1e-9 >= need:
            return {'ok': True, 'reason': 'donor_margin_ok',
                    'idle_ms': idle, 'needed_ms': need}
        return {'ok': False, 'reason': 'no_safe_donor',
                'idle_ms': idle, 'needed_ms': need}

    def _gate_probe(self, view, sel):
        """REAL select_job/ETA verification on the candidate band (s6/s5.3).

        The runtime supplies ``gate_probe`` in the view; it builds the
        next-cycle profile copy (spent=0, commitments kept) and calls the
        frozen quota_forecast select_job.  T012 C9: a missing callable FAILS
        CLOSED, and the answer must be about the selected component.
        """
        probe = (view or {}).get('gate_probe')
        if probe is None:
            return {'ok': False, 'reason': 'no_gate_probe_in_view',
                    'basis': 'missing'}
        sel = sel or {}
        sel_comp = sel.get('checked_comp_id') or sel.get('comp_id')
        kwargs = {'target_gF': sel.get('target_gF'),
                  'current_gF': self.effective_fast}
        if sel.get('receiver') is not None:
            kwargs['receiver'] = sel.get('receiver')
        if sel_comp is not None:
            kwargs['checked'] = sel_comp
        try:
            result = probe(**kwargs)
        except TypeError:
            try:
                result = probe(target_gF=sel.get('target_gF'),
                               current_gF=self.effective_fast)
            except Exception as exc:
                return {'ok': False,
                        'reason': 'gate_probe_exception:%r' % (exc,)}
        except Exception as exc:
            return {'ok': False, 'reason': 'gate_probe_exception:%r' % (exc,)}
        if not isinstance(result, dict):
            return {'ok': False, 'reason': 'gate_probe_bad_result'}
        checked = result.get('checked_comp_id')
        started = result.get('started_comp_id') or checked
        if sel_comp is not None and checked is not None and \
                str(checked) != str(sel_comp):
            return {'ok': False, 'reason': 'gate_probe_other_component',
                    'checked_comp_id': checked, 'selected_comp_id': sel_comp,
                    'basis': result.get('basis')}
        if not bool(result.get('ok')):
            return {'ok': False, 'reason': result.get('reason') or
                    'gate_probe_not_ok', 'checked_comp_id': checked,
                    'started_comp_id': started,
                    'basis': result.get('basis'), 'result': result}
        return {'ok': True, 'checked_comp_id': checked,
                'started_comp_id': started, 'result': result,
                'basis': result.get('basis')}

    def _direct_sample_work(self, sample, role, prices, field='waiting'):
        return sum((float(prices[k]) +
                    (float(prices['FAST_SUFFIX']) if k == 'FAST_PREFIX' and
                     self.config.get('inline_fast_continuation', False) else 0.)) * int(n)
                   for k, n in sample[field][role].items())

    def _direct_demand_tick(self, now_ns, view, apply):
        """Cheap tick monitoring; one target calculation only when due.

        Time to one lost/backlogged job and public queue slack shorten the next
        interval. Stable intervals grow by merging two intervals, bounded by
        half the existing source retention. No candidate outcome replay.
        """
        aligned = (int(now_ns)//self.tick_ns)*self.tick_ns
        independent = bool(self.dual_clock is not None and not self.dual_clock.disabled
            and self.dual_clock.mode=='FULL' and self.dual_clock.c.get('independent_checks_enabled'))
        if independent and not self.dual_clock.context.get('joint'):
            # The independent role checks own the trigger. No legacy polling
            # query, shared due time or periodic ratio calculation on this path.
            self.last_reason = 'independent_role_checks_no_material_change'
            return
        if self.direct_samples and self.direct_samples[-1]['aligned_ns'] == aligned:
            return
        sample = {'aligned_ns': aligned, 'now_ns': int(now_ns),
                  'waiting': {r: dict(self.observer.wait_counts[r]) for r in GPU_ROLES},
                  'loss': {r: dict(self.observer.resource_loss_counts[r]) for r in GPU_ROLES}}
        self.direct_samples.append(sample)
        while self.direct_samples and self.direct_samples[0]['aligned_ns'] < aligned-self.direct_history_ns:
            self.direct_samples.popleft()
        samples = list(self.direct_samples)
        prices = {k: float(self.observer.c_ref_ms(k)) for k in GPU_KINDS}
        units = {'Fast': max(prices['FAST_FULL'], prices['FAST_PREFIX']+prices['FAST_SUFFIX']),
                 'Accurate': prices['ACCURATE']}
        recent_start = max(samples[0]['aligned_ns'], aligned-2*self.tick_ns)
        recent = next((p for p in samples if p['aligned_ns'] >= recent_start), samples[0])
        dt = max(1, now_ns-recent['now_ns']) / NS_PER_S
        gap_times = {}
        recent_loss = False
        for role in GPU_ROLES:
            growth = (self._direct_sample_work(sample, role, prices)
                      - self._direct_sample_work(recent, role, prices)) / dt
            lost = (self._direct_sample_work(sample, role, prices, 'loss')
                    - self._direct_sample_work(recent, role, prices, 'loss')) / dt
            # Queue clipping/expiry can hide an overload; include real resource
            # refusals in the timer, while offered work already includes them.
            pressure = max(0., growth) + max(0., lost)
            gap_times[role] = units[role] / pressure if pressure > 0 else None
            if len(samples)>1 and sample['loss'][role] != samples[-2]['loss'][role]:
                recent_loss = True
        read_demand = view.get('direct_demand')
        # The governor's cheap view has the same causal cutoff. Reuse it when
        # its physical span matches; otherwise this legacy span is distinct.
        if independent:
            quick = None
        elif (self.dual_clock is not None and not self.dual_clock.disabled and self.dual_clock.context.get('quick_span_ns') == aligned-recent_start):
            quick = self.dual_clock.context.get('quick')
        else:
            quick = (read_demand(aligned, aligned-recent_start)
                     if callable(read_demand) and aligned>recent_start else None)
        mix_alarm = False
        if quick and quick.get('valid') and self.direct_offered_anchor is not None:
            df, da = float(quick['D_F_ms_s']), float(quick['D_A_ms_s'])
            total = df+da
            if total>0:
                # Cross multiplication detects a measured mix shift without
                # evaluating any candidate. One-job resolution filters a lone
                # rare arrival; the allocation quantum is already configured.
                resolution = CARD_COUNT*max(units.values()) / ((aligned-recent_start)/NS_PER_S)
                mix_alarm = abs(CARD_COUNT*df-self.direct_offered_anchor*total) > self.allocation_hysteresis_floor*total+resolution
        times_ns = [float(self.direct_max_interval_ns)]
        times_ns.extend(t*NS_PER_S for t in gap_times.values() if t is not None)
        slack = view.get('direct_min_queue_slack_ns')
        if slack is not None:
            times_ns.append(max(0, int(slack)))
        if recent_loss or mix_alarm or self.direct_last_calc_ns is None:
            times_ns.append(float(self.tick_ns))
        requested_ns = min(times_ns)
        desired_ns = max(self.tick_ns, min(self.direct_max_interval_ns,
            int(requested_ns//self.tick_ns)*self.tick_ns))
        # Shorten immediately, but lengthen only after a completed observation.
        self.direct_interval_ns = min(self.direct_interval_ns, desired_ns)
        first = samples[0]['aligned_ns']
        due = (first+2*self.direct_interval_ns if self.direct_last_calc_ns is None
               else self.direct_last_calc_ns+self.direct_interval_ns)
        self.direct_timing = {'interval_ms': self.direct_interval_ns/MS,
            'requested_interval_ms': requested_ns/MS, 'desired_interval_ms': desired_ns/MS,
            'min_interval_ms': self.tick_ns/MS, 'max_interval_ms': self.direct_max_interval_ns/MS,
            'next_due_ns': due, 'gap_time_ms': {r: None if t is None else t*1000 for r,t in gap_times.items()},
            'queue_slack_ms': None if slack is None else slack/MS,
            'poll_resolution_limited': requested_ns < self.tick_ns,
            'resource_loss_alarm': bool(recent_loss), 'source_mix_alarm': bool(mix_alarm)}
        dc = self.dual_clock
        original_due = now_ns >= due
        enhanced = dc is not None and not dc.disabled and dc.mode not in ('SHADOW', 'HYSTERESIS')
        if enhanced:
            # Advance the legacy horizon law independently of detailed checks.
            # Neither role's timer is substituted for this physical horizon.
            if original_due:
                self.direct_last_calc_ns = aligned
                self.direct_interval_ns = min(desired_ns, 2*self.direct_interval_ns)
            permitted = bool(dc.context.get('joint'))
            if dc.context.get('fallback_reason') and not independent:
                permitted = original_due
            if not permitted:
                return
        elif now_ns < due:
            return
        if view.get('frame_gap_seen') or view.get('log_backpressure'):
            self.last_reason = 'adaptive_public_evidence_gap'
            return
        if independent:
            self.direct_interval_ns = min(dc.context['intervals_ns'].values())
            self.direct_timing.update(mode='independent_role_checks',next_due_ns=None,
                role_intervals_ns=dict(dc.context['intervals_ns']))
        interval = self.direct_interval_ns
        # Use all retained, complete public demand buckets, even across a
        # resource change. Two equal halves keep the empirical noise estimate
        # on the same time scale as the offered-work estimate. Startup uses
        # the available complete pairs, not invented zero-filled history.
        available_ticks = max(0, (aligned-self.start_ns)//self.tick_ns)
        demand_half = min(self.direct_history_ns//(2*self.tick_ns),
                          available_ticks//2)*self.tick_ns
        capacity_window = None
        if independent and getattr(self, 'capacity_confirmed_window_enabled', False):
            legal_span = [g for g in self.grid_points
                if self.minimum_fast_share-1e-9<=g<=CARD_COUNT-self.minimum_accurate_share+1e-9]
            waiting_ms = {r:self._direct_sample_work(sample,r,prices) for r in GPU_ROLES}
            proposed = capacity_confirmed_window(dc.context, waiting_ms, legal_span)
            if proposed and proposed['window_end_ns'] == aligned:
                short_half = proposed['span_ns']//2
                if proposed['span_ns'] == 2*short_half and short_half >= self.tick_ns and short_half%self.tick_ns == 0:
                    demand_half = short_half
                    capacity_window = proposed
        if demand_half < self.tick_ns:
            self.last_reason = 'adaptive_wait_public_history'
            return
        begin = aligned-2*demand_half
        middle = aligned-demand_half
        halves = [read_demand(middle, demand_half),
                  read_demand(aligned, demand_half)] if callable(read_demand) else []
        if not enhanced:
            self.direct_last_calc_ns = aligned
        self.last_scored_window = aligned//self.tick_ns
        if len(halves)!=2 or not all(d.get('valid') for d in halves):
            self.direct_timing['next_due_ns'] = aligned+interval
            self.last_reason = 'adaptive_demand_unavailable'
            self.direct_demand_evidence = {'valid': False, 'halves': halves}
            return
        next_interval = (self.direct_interval_ns if enhanced else min(desired_ns, 2*interval))
        horizon_s = next_interval/NS_PER_S
        span_s = 2*demand_half/NS_PER_S
        # These samples are cleared on allocation changes, unlike demand
        # memory. No slope is allowed to straddle two allocations.
        trend_samples = [p for p in samples
                         if p['aligned_ns'] >= max(self.direct_stable_ns,
                                                  aligned-self.direct_trend_ns)]
        if not trend_samples:
            trend_samples=[sample]
        early = trend_samples[0]
        trend_known = len(trend_samples) >= 2
        trend_span_s = max(0, sample['now_ns']-early['now_ns'])/NS_PER_S
        split_known = len(trend_samples) >= 3
        mid = trend_samples[len(trend_samples)//2]
        trends = {}
        for role, key in (('Fast', 'D_F_ms_s'), ('Accurate', 'D_A_ms_s')):
            b0 = self._direct_sample_work(early, role, prices)
            b1 = self._direct_sample_work(mid, role, prices)
            b2 = self._direct_sample_work(sample, role, prices)
            growth = (b2-b0)/trend_span_s if trend_known else 0.
            # Zero is only the arithmetic fallback when the slope is unknown;
            # the explicit flags distinguish it from measured stable queues.
            v0 = v1 = None
            growth_noise = 0.
            if split_known:
                dt0 = (mid['now_ns']-early['now_ns'])/NS_PER_S
                dt1 = (sample['now_ns']-mid['now_ns'])/NS_PER_S
                v0, v1 = (b1-b0)/dt0, (b2-b1)/dt1
                growth_noise = abs(v1-v0)/2
            d0, d1 = float(halves[0][key]), float(halves[1][key])
            offered = (d0+d1)/2
            loss = max(0., (self._direct_sample_work(sample, role, prices, 'loss')
                           - self._direct_sample_work(early, role, prices, 'loss'))/trend_span_s) if trend_known else 0.
            role_horizon_s = horizon_s
            role_growth_known, role_split_known = trend_known, split_known
            if independent:
                checked = dc.context['role_checks'][role]
                growth = checked['growth_ms_s']
                role_growth_known = checked['growth_known']
                role_split_known = checked.get('growth_scatter_known',False)
                growth_noise = checked.get('growth_scatter_ms_s',0.)
                loss = checked['loss_rate_ms_s']
                role_horizon_s = dc.context['intervals_ns'][role]/NS_PER_S
            forecast = max(0., b2+growth*role_horizon_s)
            # Observed half-window scatter + one-job measurement resolution.
            # This tolerance is an empirical filter, not a statistical bound.
            noise = abs(d1-d0)/2 + growth_noise + units[role]/span_s
            trends[role] = {'growth_ms_s': growth, 'resource_loss_ms_s': loss,
                'growth_known': role_growth_known, 'growth_scatter_known': role_split_known,
                'unknown_growth_fallback': None if role_growth_known else 'current_backlog_only',
                'backlog_ms': b2, 'forecast_backlog_ms': forecast, 'offered_ms_s': offered,
                'allocation_weight_ms_s': offered+forecast/role_horizon_s,
                'forecast_horizon_ms':role_horizon_s*1000,
                'input_noise_ms_s': noise, 'half_growth_ms_s': [v0,v1],
                'half_offered_ms_s': [d0,d1]}
        wf, wa = (trends[r]['allocation_weight_ms_s'] for r in GPU_ROLES)
        evidence = {'valid': True, 'tick_ns': int(now_ns), 'score_tick_ns': int(now_ns),
            'score_sample_index': self.last_scored_window, 'observed_gaps': trends,
            'trend_span_ms': trend_span_s*1000, 'demand_span_ms': span_s*1000,
            'trend_start_ns': early['now_ns'],
            'forecast_horizon_ms': next_interval/MS,
            'window_start_ns': begin, 'window_end_ns': aligned, 'public_halves': halves,
            'selection_objective': 'latency_timeout',
            'basis': 'retained_offered_work_plus_current_epoch_projected_backlog'}
        if capacity_window is not None:
            evidence['capacity_confirmed_window'] = capacity_window
            evidence['basis'] = 'two_observed_capacity_deficits_with_aligned_demand_backlog_trend'
        self.direct_demand_evidence = self.last_selection = self.last_score = evidence
        self.counters['direct_demand_calculations'] += 1
        # Flat measurements can merge adjacent intervals, up to retained data.
        self.direct_interval_ns = next_interval
        self.direct_timing.update(next_interval_ms=next_interval/MS,
            next_due_ns=None if independent else aligned+next_interval)
        if wf+wa<=0:
            if independent:dc.joint_evaluated(now_ns)
            self.direct_offered_anchor = None
            self.last_reason = 'adaptive_no_work_hold'
            return
        df, da = (trends[r]['offered_ms_s'] for r in GPU_ROLES)
        self.direct_offered_anchor = CARD_COUNT*df/(df+da) if df+da>0 else None
        raw = CARD_COUNT*wf/(wf+wa)
        if getattr(self, 'overload_unit_cost_allocation_enabled', False):
            # PRAA M3. Only when the weighted demand exceeds the measured CUDA
            # capacity: a fractional knapsack serves the role with the lower
            # observed unit price first (its demand plus the existing 0.1 GPU
            # change floor), the other role receives the remainder. Underload
            # keeps the original proportional target. No deadline term.
            read_eta = view.get('cuda_efficiency')
            margin = float(dc.c.get('demand_change_floor_gpu', .1)) if dc is not None else .1
            raw, overload = overload_unit_cost_target(
                raw, {'Fast': wf, 'Accurate': wa}, units,
                read_eta() if callable(read_eta) else None, margin)
            evidence['overload_unit_cost_allocation'] = overload
        noise_g = CARD_COUNT*(wa*trends['Fast']['input_noise_ms_s']
                             + wf*trends['Accurate']['input_noise_ms_s'])/(wf+wa)**2
        tolerance = max(self.allocation_hysteresis_floor, noise_g)
        if dc is not None and not dc.disabled and dc.mode == 'HYSTERESIS':
            tolerance = max(tolerance, dc.c['ordinary_hysteresis_floor'])
        legal = [g for g in self.grid_points
                 if self.minimum_fast_share-1e-9<=g<=CARD_COUNT-self.minimum_accurate_share+1e-9]
        target = min(legal, key=lambda g:(abs(g-raw),g))
        deadline_enabled = bool(dc is not None and not dc.disabled and dc.mode == 'FULL'
                                and dc.c.get('deadline_target_enabled'))
        if deadline_enabled:
            from deadline_control import target as deadline_target
            read_deadlines = view.get('deadline_envelope')
            if not callable(read_deadlines):
                self.last_reason = 'deadline_target_public_view_unavailable'
                return
            deadline_started = time.process_time_ns()
            deadline_evidence = read_deadlines()
            target, deadline_selection = deadline_target(raw, deadline_evidence, legal, CARD_COUNT,
                current=float(self.effective_fast) if independent else None)
            evidence['deadline_target_cpu_ns'] = time.process_time_ns()-deadline_started
            evidence['deadline_envelope'] = deadline_evidence
            evidence['deadline_selection'] = deadline_selection
            evidence['basis'] = 'offered_work_backlog_trend_with_ready_deadline_bounds'
        delta = target-float(self.effective_fast)
        directional_floor = self.allocation_hysteresis_floor
        if dc is not None and dc.mode == 'FULL':
            tolerance, directional_floor = dc.allocation_tolerance(
                delta, self.allocation_hysteresis_floor, noise_g)
        evidence.update(raw_target_fast=raw, target_fast=target,
            target_accurate=round(CARD_COUNT-target,10), from_fast=float(self.effective_fast),
            minimum_fast_share=self.minimum_fast_share,
            minimum_accurate_share=self.minimum_accurate_share,
            hysteresis_fast=tolerance, empirical_noise_fast=noise_g,
            hysteresis_floor_fast=self.allocation_hysteresis_floor,
            directional_hysteresis_floor=directional_floor,
            hysteresis_receiving_role='Fast' if delta>0 else 'Accurate',
            commanded_delta_fast=delta, hysteresis_basis='quantized_command_delta')
        if independent:
            dc.joint_evaluated(now_ns)
            self.direct_timing['role_next_checks_ns']=dict(
                Fast=dc.next_fast_check_ns,Accurate=dc.next_accurate_check_ns)
            evidence['trigger_basis']='material_single_role_change_only'
        if dc is not None and dc.mode != 'HYSTERESIS':
            row = dict(dc.context, record_kind='qf_candidate', qf_target_fast=target,
                       final_target_fast=None, target_status='proposal_before_hysteresis_and_pending_gate', qf_evidence=evidence)
            row.pop('quick', None)
            dc.log('JOINT_DECISIONS', row)
        net_gain_enabled = bool(dc is not None and dc.mode == 'FULL' and not dc.disabled
                                and dc.c.get('net_gain_enabled') and not independent)
        if abs(target-self.effective_fast)<1e-9 or (not net_gain_enabled and abs(delta)<tolerance-1e-9):
            if independent:
                dc.reverse_candidate = dc.reverse_evidence = dc.reverse_window_end_ns = None
            self.last_reason = 'adaptive_noise_hysteresis_hold'
            return
        evidence['direction'] = 'F' if delta>0 else 'A'
        if net_gain_enabled:
            if self.state != 'IDLE' or self.pending is not None or any(c.get('transition') for c in view.get('cards',[])):
                self.last_reason = 'net_gain_wait_pending_or_transition'
                return
            evaluate = view.get('net_gain_evaluate')
            gain = evaluate(target) if callable(evaluate) else dict(valid=False,accept=False,reason='net_gain_view_unavailable')
            evidence['net_gain'] = gain
            evidence['hysteresis_basis'] = 'two_public_window_net_timeliness_and_capped_latency'
            self.counters['net_gain_evaluations'] += 1
            if not gain.get('valid') or not gain.get('accept'):
                self.last_reason = 'net_gain_hold_'+gain.get('reason','unknown')
                self.counters['net_gain_holds'] += 1
                return
            self.counters['net_gain_accepts'] += 1
        if not net_gain_enabled and dc is not None and not dc.disabled and dc.mode != 'HYSTERESIS':
            held = dc.filter_target(now_ns, evidence, self.effective_fast)
            if held and dc.mode == 'FULL':
                if independent:
                    self.direct_timing['role_next_checks_ns']=dict(
                        Fast=dc.next_fast_check_ns,Accurate=dc.next_accurate_check_ns)
                self.last_reason = held
                return
        if enhanced and self.state != 'IDLE':
            self.last_reason = 'dual_clock_pending_no_conflicting_command'
            dc.log('JOINT_DECISIONS',dict(tick_ns=now_ns,record_kind='pending_hold',
                qf_target_fast=target,final_target_fast=float(self.effective_fast),
                command_issued=False,fallback_reason=self.last_reason))
            return
        command = self._issue(now_ns, view, apply, target, evidence, 'adaptive_demand')
        self.last_reason = ('adaptive_demand_issued' if command is not None
                            else evidence.get('reason','adaptive_demand_apply_blocked'))
        if dc is not None and dc.mode != 'HYSTERESIS':
            dc.log('JOINT_DECISIONS',dict(tick_ns=now_ns,record_kind='issuance_outcome',
                qf_target_fast=target,final_target_fast=target if command is not None else float(self.effective_fast),
                command_issued=command is not None,command_id=None if command is None else command.get('command_id'),
                effective_fast=float(self.effective_fast),fallback_reason=self.last_reason))

    # -- main tick ---------------------------------------------------------------------
    def tick(self, now_ns, view, apply, full=True):
        now_ns = int(now_ns)
        settle = int(self.settle_ns)
        self.tick_index += 1
        self.observer.advance_to(now_ns)
        wm = (view or {}).get('watermark_ns')
        if wm is not None:
            self.observer.seal_source_cohorts(wm,
                                              bool((view or {}).get('frame_gap_seen')))
            self.observer.note_results_progress(now_ns)
        cards = (view or {}).get('cards') or []
        self.observer.mark_flags(
            transition=any(bool(c.get('transition')) for c in cards),
            probe=self.state == 'PROBE',
            gF=self.effective_fast, epoch=self.epoch)
        window_id = (now_ns // self.tick_ns) * self.tick_ns
        if self.state == 'REQUESTED' and self.pending is not None:
            target = float(self.pending['command']['target_fast'])
            if self._both_effective(view, target):
                if self.dual_clock is not None:
                    self.dual_clock.command_applied(self.effective_fast, target)
                self.epoch += 1
                self.epoch_start_ns = now_ns
                self.effective_fast = target
                # a new effective epoch invalidates every epoch-scoped series
                for unit in ('Y', 'V'):
                    self.profile_series[unit].clear()
                self.last_profile = None
                self.last_consumed_index = {r: None for r in GPU_ROLES}
                self.pending['effective_ns'] = now_ns
                self.state = 'EFFECTIVE'
                self.last_reason = 'both_cards_effective'
            elif now_ns - int(self.pending['issued_ns']) > self.transition_timeout_ns:
                self.counters['transition_timeout'] += 1
                self.pending = None
                self.state = 'IDLE'
                self.last_reason = 'transition_timeout'
        if self.state == 'EFFECTIVE' and self.pending is not None:
            eff = int(self.pending['effective_ns'])
            if self.pending.get('continuous_resource'):
                # Retain demand memory and the adaptive due time. Only queue
                # slopes are invalidated by the new allocation; do not restart
                # a short-window demand/decision cycle after every command.
                if now_ns >= eff + settle:
                    self.last_held = {'gF': float(self.effective_fast),
                                      'ns': now_ns, 'reason': 'applied_pending_reassessment'}
                    self.pending = None
                    self.pending_command = None
                    self.state = 'IDLE'
                    self.counters['resource_allocations_applied'] += 1
                    self.last_reason = 'resource_effective_reassess'
                    self.direct_stable_ns = eff + settle
                    self.direct_samples.clear()
        if self.state == 'EFFECTIVE' and self.pending is not None and not self.pending.get('continuous_resource'):
            eff = int(self.pending['effective_ns'])
            # TASK section 6 / C8: post_start = ceil((effective + settle 50ms) /
            # 500ms) * 500ms -- the first COMPLETE post cohort starts at the
            # next 500 ms cohort boundary (150 ms + 50 ms -> 500 ms).
            align = max(int(self.post_align_ns), 1)
            settle = int(self.settle_ns)
            post_start = ((eff + settle + align - 1) // align) * align
            self.pending['post_start_ns'] = post_start
            self.pending['post_first_ns'] = post_start + self.probe_first_post_ns
            # T014 s3.4.1: the pre window was frozen BEFORE the command was
            # issued (``_select_and_issue``).  It is never searched for after
            # the configuration changed, so a missing frozen pre is an explicit
            # state instead of a silently rebuilt one.
            if not self.pending.get('pre_cohorts') and not self.pending.get('no_probe'):
                self.counters['probe_missing_frozen_pre'] += 1
                self.last_reason = 'probe_missing_frozen_pre'
            # T014 s3.2.4 / CPU_expected.return_without_pre: a return to the
            # already-effective share carries NO pre-observation by definition,
            # so it must still leave EFFECTIVE for IDLE at its post boundary.
            # A normal probe with a missing frozen pre keeps the explicit
            # probe_missing_frozen_pre state instead of silently rebuilding one.
            if now_ns >= int(self.pending['post_start_ns']) and \
                    (self.pending.get('pre_cohorts')
                     or self.pending.get('no_probe')):
                if self.pending.get('no_probe'):
                    # C8 return-to-last-confirmed: the command really became
                    # effective but must NOT enter a new probe.
                    self.pending = None
                    self.state = 'IDLE'
                    self.last_reason = 'return_to_confirmed_effective'
                else:
                    self.state = 'PROBE'
                    self.last_reason = 'probe_started'
        if self.state == 'PROBE' and self.pending is not None:
            post_start = int(self.pending.get('post_start_ns') or 0)
            span = self._probe_post_span(self.pending)
            # TASK section 6: evaluate only after the post deadline + settle,
            # with a complete source watermark and every root closed once.
            if now_ns >= post_start + span + settle:
                self._evaluate_probe(now_ns, view, apply)
        if self.state == 'IDLE' and not self.continuous_resource:
            self._review_held(now_ns, view, apply)
        remaining_control_ns = self.settle_ns if self.continuous_resource else 3 * NS_PER_S
        if (self.state != 'IDLE' and self.continuous_resource and self.dual_clock is not None and not self.dual_clock.disabled
                and self.dual_clock.mode in ('DUAL_CLOCK', 'FULL', 'UNIFORM_100MS')
                and not view.get('source_end') and now_ns + remaining_control_ns < int(view.get('duration_ns') or (now_ns + 4*NS_PER_S))):
            self._direct_demand_tick(now_ns, view, apply)
        if self.state == 'IDLE' and now_ns >= self.start_ns and not view.get('source_end') and now_ns + remaining_control_ns < int(view.get('duration_ns') or (now_ns + 4 * NS_PER_S)):
            # The fixed arm observes the same demand but _issue suppresses
            # commands. Direct mode never calls candidate scoring.
            if self.continuous_resource:
                self._direct_demand_tick(now_ns, view, apply)
            else:
                sealed = self.observer.sealed_windows
                if sealed and sealed[-1]['end_ns'] <= now_ns:
                    self._score_tick(now_ns, view, apply)
        min_dec = self._decision_min(now_ns, view, window_id)
        self._last_decision_row = min_dec
        if full:
            decision = self._decision(now_ns, view, window_id)
        else:
            decision = min_dec
        self.decisions.append(decision)
        return decision

    def _decision_min(self, now_ns, view, window_id):
        return {
            'schema': DECISION_SCHEMA,
            'valid': True,
            'tick_ns': int(now_ns),
            'window_id': int(window_id),
            'policy': self.policy,
            'controller_version': CONTROLLER_VERSION,
            'implementation_revision': IMPLEMENTATION_REVISION,
            'resource_control_mode': ('adaptive_demand_feedback' if self.continuous_resource else 'post_probe'),
            'adaptive_timing': self.direct_timing,
            'state': self.state,
            'effective_fast': float(self.effective_fast),
            'target_fast': float(self.target_fast),
            'reason': self.last_reason,
            'epoch': self.epoch,
            'command_seq': self.command_seq,
            'tick_index': self.tick_index,
            'counters': dict(self.counters),
            'table_id': self.last_table_id,
            'last_score': getattr(self, 'last_score', None),
            'last_trial_renewal': getattr(self, 'last_trial_renewal', None),
            # A table can become usable on a later 100 ms tick. Preserve that
            # actual comparison instead of waiting for the next full snapshot.
            'score_selection': (self._selection_diagnostics()
                                if (getattr(self, 'last_score', None) or {}).get('tick_ns') == int(now_ns)
                                else None),
        }

    @staticmethod
    def _proposal_diagnostics(proposal):
        if not isinstance(proposal, dict):
            return None
        # Candidate rows already live in resource.candidates; carry references
        # and the exact rejection reasons without copying the full table again.
        keys = ('action', 'reason', 'direction', 'step_gF', 'step', 'basis',
                'rejected', 'row_guard', 'net_gain_history_used', 'deficit_steps')
        out = {k: proposal.get(k) for k in keys if k in proposal}
        for name in ('row', 'current_row'):
            row = proposal.get(name)
            if isinstance(row, dict):
                out[name] = {k: row.get(k) for k in (
                    'gF', 'Yhat', 'Vhat', 'net_gain_Yhat', 'net_gain_Vhat',
                    'epsilon_Y', 'epsilon_V', 'V_available', 'is_effective_step',
                    'quality_earliness_hat', 'net_gain_quality_earliness',
                    'epsilon_quality_earliness',
                    'low_confidence', 'donor_final', 'receiver_final')}
        return out

    def _selection_diagnostics(self):
        selection = self.last_selection
        if self.continuous_resource:
            return self.direct_demand_evidence
        if not isinstance(selection, dict):
            return None
        out = {k: selection.get(k) for k in (
            'score_tick_ns', 'score_window_index', 'trigger', 'high_coverage',
            'receiver', 'exploratory', 'eps_select', 'exploratory_hold',
            'net_gain_history_used')}
        out['receivers'] = [
            {'receiver': item.get('receiver'), 'trigger': item.get('trigger'),
             'direction_locked': bool(item.get('direction_locked')),
             'proposal': self._proposal_diagnostics(item.get('proposal'))}
            for item in (selection.get('receivers') or [])]
        out['net_gain_history_used'] = bool(selection.get('net_gain_history_used', False))
        out['selection'] = self._proposal_diagnostics(selection.get('selection'))
        return out

    def decision_row(self):
        """Minimal dict for the last tick (used for 100 ms log row)."""
        if self._last_decision_row is not None:
            return dict(self._last_decision_row)
        return self._decision_min(0, None, 0)

    def _candidate_payload(self):
        return None if self.last_table is None else self.last_table.get('input_payload')

    def _window_residuals(self):
        """TASK s9.8: the newest sealed window's per-role E/S/X/R/O/Open counts."""
        wins = getattr(self.observer, 'sealed_windows', None) or []
        if not wins:
            return None
        w = wins[-1]
        return {'index': w.get('index'), 'start_ns': w.get('start_ns'),
                'end_ns': w.get('end_ns'), 'T_s': w.get('T_s'),
                'residuals': w.get('residuals'),
                'Open_start': w.get('Open_start'), 'Open_end': w.get('Open_end'),
                'effective_epoch': w.get('effective_epoch')}

    def _decision(self, now_ns, view, window_id):
        cards = (view or {}).get('cards') or []
        command = None
        if self.pending is not None:
            command = self.pending['command']
        resource = {
            'schema': 'QF4_resource_v1',
            'snapshot_ns': int((view or {}).get('snapshot_ns') or now_ns),
            'window_id': int(window_id),
            'gF_effective': float(self.effective_fast),
            'gF_target': float(self.target_fast),
            'cards': [dict(c) for c in cards],
            'waiting': (view or {}).get('waiting'),
            'waiting_work_ms': (view or {}).get('waiting_work_ms'),
            'cost_version': (view or {}).get('cost_version'),
            'direct_demand': self.direct_demand_evidence,
            'stall': bool((view or {}).get('stall')),
            'log_backpressure': bool((view or {}).get('log_backpressure')),
            'frame_gap_seen': bool((view or {}).get('frame_gap_seen')),
            'table_id': self.last_table_id,
            'observer': self.observer.snapshot(),
            'cost_records': (view or {}).get('cost_records'),
            'demand': self._demand_summary(),
            'table_summary': None if self.last_table is None else {
                'basis': 'deadline_replay', 'profile_sha': self.last_table.get('profile_sha'),
                'snapshot_ns': self.last_table.get('snapshot_ns'),
            },
            # TASK s9: every control window must carry the candidate table and
            # the payload it was computed from; analysis_t012 reads these keys.
            'candidates': (None if self.last_table is None
                           else [dict(c) for c in self.last_table['candidates']]),
            'candidates_valid': self.last_table is not None,
            'candidate_payload': self._candidate_payload(),
            'residuals': self._window_residuals(),
        }
        decision = {
            'schema': DECISION_SCHEMA,
            'valid': True,
            'tick_ns': int(now_ns),
            'window_id': int(window_id),
            'policy': self.policy,
            'controller_version': CONTROLLER_VERSION,
            'implementation_revision': IMPLEMENTATION_REVISION,
            'resource_control_mode': ('adaptive_demand_feedback' if self.continuous_resource else 'post_probe'),
            'adaptive_timing': self.direct_timing,
            'state': self.state,
            'table_id': self.last_table_id,
            'resource': resource,
            'command': command,
            # TASK section 6 / C8: pending state and this tick's NEW commands are
            # separate counters -- a pending command is never a new command.
            'new_command': (self.new_commands_this_tick[-1]
                            if self.new_commands_this_tick else None),
            'new_command_count': int(self.new_command_count),
            'pending_command': (self.pending or {}).get('command'),
            'apply_count': int(self.apply_count),
            'config_request_count': int(self.config_request_count),
            'reason': self.last_reason,
            'epoch': self.epoch,
            'command_seq': self.command_seq,
            'tick_index': self.tick_index,
            'counters': dict(self.counters),
            'triggers': {r: dict(self.triggers.get(r) or {}) for r in GPU_ROLES},
            'last_score': getattr(self, 'last_score', None),
            'trials_by_direction': dict(self.trials_by_direction),
            'trial_context': {k: dict(v) for k, v in self.trial_context.items()},
            'last_trial_renewal': getattr(self, 'last_trial_renewal', None),
        }
        if self.last_selection is not None:
            decision['last_selection'] = self._selection_diagnostics()
        if self.last_probe is not None:
            decision['last_probe'] = self.last_probe
        if self.last_replay is not None:
            decision['replay'] = self.last_replay
        if self.last_shadow is not None:
            decision['shadow'] = self.last_shadow
        decision['failed_edges'] = [dict(e) for e in self.failed_edges]
        # C10: the fixed arm must keep the REAL 1F1A configuration and never
        # report a fake EFFECTIVE/PROBE; expose the assertion explicitly.
        decision['effective_fast_assert'] = float(self.effective_fast)
        decision['fixed_arm_effective_ok'] = bool(
            self.dynamic or abs(float(self.effective_fast) - float(self.config.get('initial_fast', 1.0))) < 1e-9)
        return decision

    def _demand_summary(self):
        d = self.demand_cache
        if not d:
            return None
        return {
            'T_s': d.get('T_s'), 'incomplete': d.get('incomplete'),
            'lambda_raw': d.get('lambda_raw'),
            'lambda_supported': d.get('lambda_supported'),
            'n_raw': d.get('n_raw'), 'n_supported': d.get('n_supported'),
            'n_unknown': d.get('n_unknown'),
            'EARLY': {k: v for k, v in (d.get('EARLY') or {}).items()},
        }

    def snapshot(self):
        return {
            'controller_version': CONTROLLER_VERSION,
            'implementation_revision': IMPLEMENTATION_REVISION,
            'policy': self.policy,
            'resource_control_mode': ('adaptive_demand_feedback' if self.continuous_resource else 'post_probe'),
            'adaptive_timing': self.direct_timing,
            'direct_demand': self.direct_demand_evidence,
            'state': self.state,
            'target_fast': self.target_fast,
            'effective_fast': self.effective_fast,
            'confirmed_fast': self.confirmed_fast,
            'command_seq': self.command_seq,
            'epoch': self.epoch,
            'counters': dict(self.counters),
            'observer': self.observer.snapshot(),
            'cost_records': self.cost_book.records_dict(),
            'global_cost_version': int(self.cost_book.global_cost_version()),
            'sample_revision': int(self.cost_book.sample_revision()),
            # TASK section 6 / C8: separate pending / new / applied counters.
            'new_command_count': int(self.new_command_count),
            'pending_command': (self.pending or {}).get('command'),
            'pending_command_id': (self.pending or {}).get('command_id'),
            'apply_count': int(self.apply_count),
            'config_request_count': int(self.config_request_count),
            'triggers': {r: dict(self.triggers.get(r) or {}) for r in GPU_ROLES},
            'eps_select': dict(self.eps_select or {}),
            'last_probe': self.last_probe,
            'last_shadow': self.last_shadow,
            'last_replay': self.last_replay,
            'last_confirmed': self.last_confirmed,
            'hold_on_uncertain': bool(self.hold_on_uncertain),
            'speed_channel_requires_health': bool(self.speed_channel_requires_health),
            'last_held': self.last_held,
            'held_review': ({'command_id': self.held_review['item']['command']['command_id'],
                             'epoch': self.held_review['epoch'],
                             'due_ns': self.held_review['due_ns'],
                             'remaining': self.held_review['remaining']}
                            if self.held_review else None),
            'failed_edges': [dict(e) for e in self.failed_edges],
            'trials_by_direction': dict(self.trials_by_direction),
            'trial_context': {k: dict(v) for k, v in self.trial_context.items()},
            'last_trial_renewal': getattr(self, 'last_trial_renewal', None),
            'locked_edges': {k: dict(v) for k, v in self.locked_edges.items()},
            'noise': {u: {'n': len(self.noise_windows[u]),
                          'epsilon': self.last_noise_pair}
                      for u in ('Y', 'V')},
            'budget_basis': self.last_budget_basis,
            'gain_history_n': len(self.gain_history),
            'gain_history': [dict(v) for v in self.gain_history.values()],
        }


def control_step(controller, now_ns, view, apply, full=True):
    """The one adapter shared by the Runtime and the CPU checks.

    TASK section 6 / C8: ``commands`` carries ONLY the commands issued in THIS
    tick.  A pending command is reported through ``pending_command`` on every
    later tick but is never counted as a new command again, and one issued
    command_id applies exactly once and emits exactly one CONFIG_REQUEST.
    """
    controller.new_commands_this_tick = []
    controller.new_command_count = 0
    decision = controller.tick(now_ns, view, apply, full=full)
    new_commands = list(getattr(controller, 'new_commands_this_tick', []) or [])
    pending_command = (getattr(controller, 'pending', None) or {}).get('command')
    log = {
        'schema': CONTROL_LOG_SCHEMA,
        'controller_version': CONTROLLER_VERSION,
        'implementation_revision': IMPLEMENTATION_REVISION,
        'policy': decision.get('policy'),
        'tick_ns': decision.get('tick_ns'),
        'window_id': decision.get('window_id'),
        'state': decision.get('state'),
        'reason': decision.get('reason'),
        'command': decision.get('command'),
        'new_command': (new_commands[-1] if new_commands else None),
        'new_command_count': len(new_commands),
        'pending_command': pending_command,
        'apply_count': int(getattr(controller, 'apply_count', 0)),
        'config_request_count': int(getattr(controller, 'config_request_count', 0)),
        'resource': decision.get('resource'),
    }
    return {'decision': decision,
            'log': log,
            'commands': new_commands,
            'new_commands': new_commands,
            'new_command_count': len(new_commands),
            'pending_command': pending_command}


def self_check():
    """Cheap, file-free invariants for the repaired qf4r controller.

    Covers the repaired surfaces: ONE shared CostBook per arm, the monotone
    global cost version, per-tick command accounting (commands issued only on
    the issuing tick), the probe post-alignment, the epsilon_select floor of
    1/T_profile and the fixed-arm shadow contract (apply_count stays 0).
    """
    ref = {'configuration': {'controller_version': CONTROLLER_VERSION,
                             'algorithm_revision': 'QF-4'},
           'c_ref_ms': {'FAST_FULL': 4.0, 'FAST_PREFIX': 2.0,
                        'FAST_SUFFIX': 3.0, 'ACCURATE': 8.0}}
    cost = {g: {k: {'mean_wall_ms': 5.0, 'gate_ms': 6.0, 'sample_n': 40,
                    'sample_revision': 40, 'cost_version': 1,
                    'measurement_source': WARMUP_SOURCE}
                for k in GPU_KINDS} for g in CARD_IDS}
    cfg = make_controller_inputs_qf4(ref, cost, 50_000_000, ['a', 'b'],
                                     DYNAMIC_POLICY)
    ctl = QueueFeedbackController(cfg)
    aF, aA = root_work_demand(100, 20, 40, 0.75, 0.25, ref['c_ref_ms'])
    # TASK section 3.4: the global cost version is monotone and is NOT the max
    # of the per-kind versions (a non-maximum kind moving 1 -> 2 still bumps).
    book = CostBook(cap=200)
    book.records[(0, 'FAST_FULL')].cost_version = 2
    book.records[(0, 'FAST_FULL')].mean_ns = 4.0 * 1_000_000
    book.records[(1, 'ACCURATE')].cost_version = 1
    book.records[(1, 'ACCURATE')].mean_ns = 8.0 * 1_000_000
    v_before = book.global_cost_version()
    max_before = max(r.cost_version for r in book.records.values())
    book.records[(1, 'ACCURATE')].cost_version = 2
    book.records[(1, 'ACCURATE')].mean_ns = 9.0 * 1_000_000
    v_after = book.global_cost_version()
    max_after = max(r.cost_version for r in book.records.values())
    cfg2 = make_controller_inputs_qf4(ref, cost, 50_000_000, ['a', 'b'],
                                      FIXED_POLICY, cost_book=book)
    ctl2 = QueueFeedbackController(cfg2)
    # TASK section 6: post_start = ceil((effective + settle) / 500 ms).
    eff = 150 * 1_000_000
    align = int(PROBE_POST_ALIGN_MS * 1_000_000)
    settle = int(SETTLE_MS * 1_000_000)
    post_start = ((eff + settle + align - 1) // align) * align
    eps0 = select_epsilon([], TPROFILE_S, SELECT_NOISE_MIN_PROFILES)
    out = {
        'controller_version': CONTROLLER_VERSION,
        'implementation_revision': IMPLEMENTATION_REVISION,
        'policy': ctl.policy,
        'state': ctl.state,
        'aF': aF, 'aA': aA,
        'phi_1_3': phi_cards(1.3),
        'grid_ok': bool(on_grid(1.3) and not on_grid(1.35)),
        'global_cost_version_increments': v_after - v_before,
        'cost_version_max_before': max_before,
        'cost_version_max_after': max_after,
        'shared_cost_book': bool(ctl2.cost_book is book and
                                 ctl2.observer.cost_book is book),
        'one_cost_book_per_arm': bool(ctl.cost_book is ctl.observer.cost_book),
        'new_command_count': int(getattr(ctl, 'new_command_count', -1)),
        'new_commands_this_tick_empty': bool(
            list(getattr(ctl, 'new_commands_this_tick', None) or []) == []),
        'pending_command_is_none': bool(ctl.pending_command is None),
        'apply_count': int(getattr(ctl, 'apply_count', -1)),
        'config_request_count': int(getattr(ctl, 'config_request_count', -1)),
        'probe_post_start_ms': post_start / 1_000_000.0,
        'probe_post_first_end_ms': (post_start +
                                    PROBE_FIRST_POST_MS * 1_000_000) / 1e6,
        'epsilon_floor': eps0['epsilon'],
        'epsilon_low_confidence': bool(eps0['low_confidence']),
        'max_unconfirmed_trials_per_direction':
            MAX_UNCONFIRMED_TRIALS_PER_DIRECTION,
        'fixed_arm_apply_count': int(getattr(ctl2, 'apply_count', -1)),
        'fixed_arm_effective_fast': ctl2.effective_fast,
        'fixed_arm_policy': ctl2.policy,
        'pressure_self_test': ctl._self_test_pressure(),
    }
    out['pass'] = bool(
        out['policy'] == DYNAMIC_POLICY and out['state'] == 'IDLE' and
        out['grid_ok'] and out['global_cost_version_increments'] == 1 and
        out['cost_version_max_before'] == 2 and
        out['cost_version_max_after'] == 2 and out['shared_cost_book'] and
        out['one_cost_book_per_arm'] and out['new_command_count'] == 0 and
        out['new_commands_this_tick_empty'] and
        out['pending_command_is_none'] and out['apply_count'] == 0 and
        out['config_request_count'] == 0 and
        out['probe_post_start_ms'] == 500.0 and
        out['probe_post_first_end_ms'] == 1000.0 and
        abs(out['epsilon_floor'] - 0.5) < 1e-12 and
        out['epsilon_low_confidence'] and
        out['max_unconfirmed_trials_per_direction'] == 2 and
        out['fixed_arm_policy'] == FIXED_POLICY and
        out['fixed_arm_apply_count'] == 0 and
        abs(out['fixed_arm_effective_fast'] - 1.0) < 1e-9 and
        out['pressure_self_test']['ok'])
    return out


if __name__ == '__main__':
    print(self_check())
