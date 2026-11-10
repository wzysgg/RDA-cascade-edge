"""Execution-only guards; latent resource estimator remains unchanged."""
import math

GPU_ADMISSION_RULE = 'physical_deadline_and_capacity_v1'


def minimum_inference_ns(kind, costs, ipc, return_ns, affinity=None,
                         inline=False, head_ns=0, extra_ns=0, model_only=None):
    """Optimistic no-queue completion cost, independent of current shares.

    Prefix outcomes are not read: use the cheapest publicly possible terminal
    continuation. An owed Suffix keeps its actual activation affinity. The
    frozen simulation's service times are deterministic; this is not an ETA
    for real queues or a guarantee about future resource allocation.
    """
    devices = tuple(costs) if affinity is None else (int(affinity),)
    role = 'Accurate' if kind == 'ACCURATE' else 'Fast'
    if kind != 'FAST_PREFIX' and model_only and role != model_only:
        return math.inf
    options = []
    for g in devices:
        work = int(costs[g][kind])+int(ipc[g])
        if kind == 'FAST_PREFIX':
            tails = []
            if model_only in (None, 'Fast'):
                tails.append(int(costs[g]['FAST_SUFFIX'])+(0 if inline else int(ipc[g])))
            if model_only in (None, 'Accurate'):
                tails.append(min(int(costs[h]['ACCURATE'])+int(ipc[h]) for h in costs))
            if not tails:
                continue
            work += (0 if inline else int(head_ns))+min(tails)
        options.append(work)
    return min(options, default=math.inf)+int(return_ns)+max(0, int(extra_ns))
def prune_ledger(card,current_cycle):
    # Forecaster already ignores exactly these obsolete keys.
    # Preserve current, previous AND all future keys, including overruns.
    floor=current_cycle-1
    for key in list(card['cycle_charges']):
        if key<floor:del card['cycle_charges'][key]
def eta_horizon_ns(deadline_abs_ns,fresh_now_ns,delivery_budget_ns):
    """TASK 3.1: the ETA view is H = max(0, D - fresh_now - delivery_budget).

    The absolute deadline always comes from the original arrival + 50 ms; it is
    never re-derived from a GPU enqueue or a switch instant.  ``delivery_budget``
    is the arm's recent healthy actor-end -> parent-available P95 (bootstrap IPC
    when there are no samples); it is never silently treated as a measured 0.
    """
    return max(0,int(deadline_abs_ns)-int(fresh_now_ns)-max(0,int(delivery_budget_ns)))

def admits_after_prediction(estimate_ns,deadline_abs_ns,fresh_now_ns):
    """Admit only when the freshly re-read clock still leaves room.

    Replaces the old ``(estimate, arrival, decision, deadline=25 ms)`` form whose
    fixed 25 ms default silently rejected requests the real 50 ms deadline still
    admitted.  Callers must pass the absolute deadline and a ``fresh_now`` read
    *after* the prediction returned.
    """
    return (math.isfinite(estimate_ns) and estimate_ns>=0
            and estimate_ns<=int(deadline_abs_ns)-int(fresh_now_ns))
