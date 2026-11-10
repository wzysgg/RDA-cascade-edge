"""Public ready-work deadline envelopes; no route oracle or ratio search.

An envelope is a fluid service requirement, not a schedulability guarantee.
Running and same-card continuation commitments reduce the available service
budget analytically. No counterfactual dispatch calendar is executed.
"""
import math


def _capacity(horizon, fast, role, blocked):
    return sum(max(0.,horizon-busy)*(min(1.,max(0.,fast-index))
        if role=='Fast' else 1-min(1.,max(0.,fast-index)))
        for index,busy in enumerate(blocked))


def _required_share(work, horizon, role, blocked):
    """Invert the existing one-fractional-card mapping, piecewise linearly."""
    order = list(range(len(blocked)))
    if role=='Accurate':order.reverse()
    remaining = float(work)
    for share,index in enumerate(order):
        available=max(0.,horizon-blocked[index])
        if available>0 and remaining<=available:
            return share+remaining/available
        remaining-=available
    return len(blocked)+max(0.,remaining)/max(1.,float(horizon))


def envelope(now, jobs, terminal_return_ns, blocked_ns=None):
    """jobs contain only published deadline, remaining service and role.

    Irrecoverable ready jobs stay in failure accounting but do not create an
    infinite rescue requirement. Running and pinned work is represented once in
    blocked_ns; jobs contains only uncommitted ready work in that mode.
    """
    result = dict(valid=True, basis='ready_work_cumulative_absolute_deadlines',
                  snapshot_ns=int(now), roles={})
    if blocked_ns is not None:
        if not blocked_ns or any(not math.isfinite(x) or x<0 for x in blocked_ns):
            raise ValueError('invalid public GPU commitments')
        result['blocked_ns']=list(blocked_ns)
    for role in ('Fast', 'Accurate'):
        points = []
        impossible = 0
        for job in jobs:
            if job['role'] != role:
                continue
            service = int(job['service_ns'])
            horizon = int(job['deadline_ns'])-int(now)-int(terminal_return_ns)
            if service <= 0 or horizon < service:
                impossible += 1
                continue
            points.append((horizon, service))
        grouped = {}
        for horizon, service in points:
            grouped[horizon] = grouped.get(horizon, 0)+service
        cumulative = 0
        requirement = 0.0
        curve = []
        # Evaluate equal-deadline batches together, including the endpoint.
        for horizon in sorted(grouped):
            cumulative += grouped[horizon]
            required = (cumulative/float(horizon) if blocked_ns is None else
                _required_share(cumulative,horizon,role,blocked_ns))
            requirement = max(requirement, required)
            curve.append(dict(horizon_ns=horizon, work_ns=cumulative,
                              required_gpu=required))
        result['roles'][role] = dict(required_gpu=requirement,
            ready_rescuable=len(points), zero_wait_infeasible=impossible,
            curve=curve)
    return result


def target(base, evidence, legal, cards=2.0, current=None):
    """Project workload target into deadline service bounds if feasible.

    If bounds conflict, equalise relative service shortfall. This is a declared
    fluid overload compromise, not a claim to maximise timely completions.
    Legal-grid evaluation is scalar arithmetic, never another inference replay.
    """
    if not evidence.get('valid'):
        raise ValueError('deadline evidence unavailable')
    uf = float(evidence['roles']['Fast']['required_gpu'])
    ua = float(evidence['roles']['Accurate']['required_gpu'])
    if not all(math.isfinite(v) and v >= 0 for v in (uf, ua)):
        raise ValueError('invalid deadline requirement')
    lower = max(min(legal), uf)
    upper = min(max(legal), cards-ua)
    feasible_grid = [g for g in legal if lower-1e-12 <= g <= upper+1e-12]
    if feasible_grid:
        chosen = min(feasible_grid, key=lambda g: (abs(g-base), g))
        mode = 'project_workload_target_into_deadline_bounds'
    else:
        def deficit(g):
            f=max(0.0,uf-g)/uf if uf else 0.0
            a=max(0.0,ua-(cards-g))/ua if ua else 0.0
            return max(f,a)
        if current is not None:
            # Equal-relative-shortfall solution, then only its two rounding
            # neighbours. No search over hypothetical task executions/ratios.
            balance=cards*uf/(uf+ua) if uf+ua>0 else base
            left=[g for g in legal if g<=balance]
            right=[g for g in legal if g>=balance]
            neighbours=set(([max(left)] if left else [])+([min(right)] if right else []))
            chosen=min(neighbours,key=lambda g:(deficit(g),abs(g-base),g))
        else:
            chosen = min(legal, key=lambda g: (deficit(g), abs(g-base), g))
        mode = 'minimax_relative_deadline_shortfall'
    selection=dict(mode=mode, workload_target_fast=base,
        deadline_lower_fast=lower, deadline_upper_fast=upper,
        bounds_feasible_on_grid=bool(feasible_grid),
        required_gpu=dict(Fast=uf, Accurate=ua),
        target_fast=chosen, target_accurate=cards-chosen,
        scope='fluid service shortfall, not an exact timely-task count or nonpreemptive scheduling guarantee; no replay veto')
    if current is not None:
        blocked=evidence.get('blocked_ns',[0.]*int(cards))
        effects={}
        for role in ('Fast','Accurate'):
            curve=evidence['roles'][role]['curve']
            def shortage(g):
                return max([max(0.,point['work_ns']-_capacity(
                    point['horizon_ns'],g,role,blocked)) for point in curve] or [0.])
            before,after=shortage(current),shortage(chosen)
            effects[role]=dict(before_shortfall_ns=before,after_shortfall_ns=after,
                gained_service_ns=max(0.,before-after),lost_service_ns=max(0.,after-before))
        selection['service_effect']=effects
        selection['net_service_gain_ns']=sum(v['gained_service_ns']-v['lost_service_ns'] for v in effects.values())
        selection['effect_scope']='change of worst cumulative deadline-work shortfall per role; not TCA prediction; no additional accept/reject gate'
    return chosen, selection
