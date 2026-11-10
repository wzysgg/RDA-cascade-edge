"""Reuse the actual dispatch choice in a causal known-ready p95 calendar.

Fractional sharing changes available budget/waiting, never CNN service duration.
Unknown Prefix outcomes conservatively reserve a same-card Suffix.
"""
import copy,heapq,math,time
from collections import Counter
PERIOD=50_000_000
# T014 s3.1.5: the prediction is a COOPERATIVE, bounded simulation.  At most
# this many calendar transitions may be processed for one request; the wall
# budget is checked at every transition.  ``LAST_FORECAST`` records, for the
# most recent call, the elapsed wall time, the transition count and the
# termination reason, so the caller can report a structured UNKNOWN instead of
# an ``inf`` that pretends the prediction was infeasible.
FORECAST_MAX_TRANSITIONS=128
LAST_FORECAST={}
SCORE_RULE = 'proportional_service_empty_turn_v2'

def enable_service_debt(s, balance=None):
    """Enable strict score scheduling and cumulative GPU-service accounting.

    Balances are service-score nanoseconds, not renewable start budgets.
    Existing earned/used counters describe score accounting, including empty
    turns; actual GPU occupancy comes from real dispatch/completion timings.
    Predictions are booked at dispatch and reconciled once at completion.
    Future Suffix promises remain in pending/pinned, not executed service.
    An empty winner charges a whole-task score quantum before comparing again.
    Previewing a card does not commit this deduction. Equal scores prefer the
    larger share, then Fast; neither age nor deadline changes that order.
    """
    if s.get('allow_idle_endpoint_loans', False):
        raise ValueError('strict score scheduling forbids idle endpoint loans')
    if '_service_debt' not in s:
        origin = {r: float((balance or {}).get(r, 0.0))
                  for r in ('Fast', 'Accurate')}
        s['_service_debt'] = {
            'score_rule': SCORE_RULE,
            'balance': dict(origin), 'origin': origin, 'inflight': {},
            'earned': {'Fast': 0.0, 'Accurate': 0.0},
            'used': {'Fast': 0.0, 'Accurate': 0.0},
            'idle_used': {'Fast': 0.0, 'Accurate': 0.0},
            'idle_repaid': {'Fast': 0.0, 'Accurate': 0.0},
            'endpoint_loan_used': {'Fast': 0.0, 'Accurate': 0.0},
            'endpoint_loans': {'Fast': 0, 'Accurate': 0}, 'settled': 0}
    return s['_service_debt']

def debt_copy(s):
    if '_service_debt' not in s:
        return None
    # The ledger schema has scalar counters, flat role maps, and one map of
    # flat in-flight records. Copy every mutable level explicitly. Generic
    # deepcopy traverses all scalar leaves on every 0.25 ms admission forecast.
    # Neither the forecast nor the candidate calendar may mutate the live maps
    # or the held dispatch records used for actual completion reconciliation.
    return {name: ({key: dict(held) for key, held in value.items()}
                   if name == 'inflight' else dict(value))
            if isinstance(value, dict) else value
            for name, value in s['_service_debt'].items()}

def _debt_service(d, role, duration, share, competing):
    # Every stage earns proportional scores and debits its full service cost.
    # Queue presence is diagnostic only; an absent flow retains its score.
    # A real stage or an empty turn uses the same proportional score rule.
    # Idle wall-clock time itself never refills scores.
    for r, p in (('Fast', share), ('Accurate', 1-share)):
        earned = p * duration
        d['earned'][r] += earned
        d['balance'][r] += earned
    d['used'][role] += duration
    d['balance'][role] -= duration
    return 0.0


def _score_owner(p, balance):
    shares = {'Fast': p, 'Accurate': 1-p}
    enabled = [r for r in shares if shares[r] > 0]
    return max(enabled, key=lambda r: (balance[r], shares[r], r == 'Fast'))


def score_owner(s):
    """Raw winner, before charging an empty turn."""
    p = float(s.get('effective', s['target']) if s['transition'] else s['target'])
    return _score_owner(p, s['_service_debt']['balance'])


def service_score_floors(cost):
    """Different role quanta, in ns of GPU service, from this card's costs.

    Fast includes the full same-card Prefix/Suffix GPU path. A GPU quality
    head already included in Prefix service stays included; a separate CPU
    head, IPC and queue waiting never become GPU service-score credits.
    These quanta are not a quota percentage or a hard deadline guarantee.
    """
    floors = {'Fast': max(float(cost['FAST_FULL']),
                          float(cost['FAST_PREFIX']) + float(cost['FAST_SUFFIX'])),
              'Accurate': float(cost['ACCURATE'])}
    if any(not math.isfinite(v) or v <= 0 for v in floors.values()):
        raise ValueError('empty-turn score floors require positive finite costs')
    return floors


def _service_turn(s, fast, accurate, cost):
    """Pure selection preview; only debt_dispatch commits the chosen card.

    If exactly one enabled role has work, repeatedly charge the empty
    winner's own floor with the original proportional rule until the ready
    role wins. Batch these identical
    comparisons arithmetically so a large historical balance cannot spin.
    Both-empty cards and zero-share roles never charge an empty turn.
    """
    p = float(s.get('effective', s['target']) if s['transition'] else s['target'])
    ready = {'Fast': p > 0 and bool(fast or s['pinned']),
             'Accurate': p < 1 and bool(accurate)}
    balance = s['_service_debt']['balance']
    winner = _score_owner(p, balance)
    if ready[winner]:
        return winner, None
    other = 'Accurate' if winner == 'Fast' else 'Fast'
    if not ready[other]:
        return None, None
    floor = service_score_floors(cost)[winner]
    winner_share = p if winner == 'Fast' else 1-p
    gap_step = 2 * (1-winner_share) * floor
    gap = max(0.0, balance[winner] - balance[other])
    turns = max(1, math.ceil(gap / gap_step))

    def after_empty_turns(n):
        total = n * floor
        return {'Fast': balance['Fast'] + p * total - (total if winner == 'Fast' else 0),
                'Accurate': balance['Accurate'] + (1-p) * total - (total if winner == 'Accurate' else 0)}

    preview = after_empty_turns(turns)
    # Exactly equal scores retain the documented share/Fast tie-breaker.
    if _score_owner(p, preview) != other:
        turns += 1
        preview = after_empty_turns(turns)
    if _score_owner(p, preview) != other:
        raise ValueError('empty-turn deduction did not produce a ready winner')
    return other, {'role': winner, 'charge_ns': turns * floor}

def debt_dispatch(s, j, role, cost, fast, accurate, thin_override=None):
    """Book one dispatch.  ``thin_override`` is the only narrow waiver of the
    score/empty-turn winner invariant: when it is supplied (rescue enabled) and
    the dispatched job is itself inside the thin window on this card, the role
    may differ from the empty-turn winner and NO empty-turn deduction is booked,
    because the thin job replaced those turns.  Everything else still raises."""
    if '_service_debt' not in s:
        return
    effective = s.get('effective', s['target']) if s['transition'] else s['target']
    share = effective if role == 'Fast' else 1-effective
    if share <= 0:
        raise ValueError('zero-share card cannot execute the other role')
    # Conversion must finish an already owed same-card suffix on its old share.
    owed_suffix = s['transition'] and j['kind'] == 'FAST_SUFFIX' and j in s['pinned']
    d = s['_service_debt']
    key = (j['task_id'], j['kind'])
    if key in d['inflight']:
        raise ValueError('GPU service was already booked for this stage')
    deduction = None
    if not owed_suffix:
        chosen, deduction = _service_turn(s, fast, accurate, cost)
        if role != chosen:
            allowed = False
            if thin_override is not None:
                tasks = thin_override.get('tasks') or {}
                if j['task_id'] in tasks:
                    allowed, _slack = _thin_slack(j, tasks, cost, thin_override.get('ipc', 0),
                                                  thin_override['now'],
                                                  thin_override.get('network_return_ns', 0),
                                                  thin_override['margin'])
            if not allowed:
                raise ValueError('dispatch bypassed the score/empty-turn winner')
            deduction = None
    if deduction is not None:
        _debt_service(d, deduction['role'], deduction['charge_ns'], effective, False)
    # A topology transition retains its old contract until owed suffixes end.
    p = float(s.get('effective', s['target']) if s['transition'] else s['target'])
    other_ready = bool(accurate) if role == 'Fast' else bool(fast or s['pinned'])
    other_allowed = p < 1 if role == 'Fast' else p > 0
    competing = other_ready and other_allowed
    predicted = float(cost[j['kind']])
    if j.get('inline_fast_continuation'):
        predicted += float(cost['FAST_SUFFIX'])
    scores_before = dict(d['balance'])
    repaid = _debt_service(d, role, predicted, p, competing)
    d['inflight'][key] = {'role': role, 'share': p,
                          'competing': competing, 'predicted': predicted,
                          'scores_before': scores_before,
                          'idle_repayment': repaid,
                          'endpoint_loan': False}

def debt_complete(s, j, actual_ns):
    if '_service_debt' not in s:
        return
    d = s['_service_debt']
    held = d['inflight'].pop((j['task_id'], j['kind']))
    _debt_service(d, held['role'], float(actual_ns)-held['predicted'],
                  held['share'], held['competing'])
    repaid = 0.0
    if held.get('endpoint_loan'):
        d['endpoint_loan_used'][held['role']] += float(actual_ns)
        d['endpoint_loans'][held['role']] += 1
    d['settled'] += 1
    return {'balance': dict(d['balance']), 'earned': dict(d['earned']),
            'used': dict(d['used']), 'idle_used': dict(d['idle_used']),
            'effective_share_at_dispatch': held['share'],
            'score_rule': SCORE_RULE, 'scores_before_dispatch': held['scores_before'],
            'competing': held['competing'], 'actual_ns': int(actual_ns),
            'idle_repayment_ns': repaid, 'idle_repaid': dict(d['idle_repaid']),
            'endpoint_loan': bool(held.get('endpoint_loan'))}

def eligible_service(cards,costs,ipc,kind,affinity=None):
    """Intrinsic lower bound must respect actual new-job role eligibility.

    Already owed Suffix is different: it retains its original card through
    a conversion. Queue/quota/transition waiting remains forecast's job.
    """
    if kind=='FAST_SUFFIX':
        if affinity is None:raise ValueError('Suffix requires saved-activation affinity')
        return costs[affinity][kind]+ipc[affinity]
    # This is only an intrinsic lower bound. The forecast calendar must also
    # respect the score winner after its explicit empty-turn deduction.
    allowed=[g for g,s in enumerate(cards) if
             (s['target']<1 if kind=='ACCURATE' else s['target']>0)]
    return min((costs[g][kind]+ipc[g]+(costs[g]['FAST_SUFFIX']
                if kind=='FAST_PREFIX' and cards[g].get('inline_fast_continuation') else 0)
                for g in allowed),default=math.inf)

def _deadline(job,tasks):
    """Absolute deadline owned by the task record, never the job's local copy."""
    task=tasks[job['task_id']]
    return task['deadline_abs_ns'] if 'deadline_abs_ns' in task else task['deadline_ns']

def _thin_slack(job,tasks,cost,ipc,now,network_return_ns,margin):
    """(is_thin, slack) for one job on one card's cost table."""
    from execution_guards import minimum_inference_ns
    minimum=minimum_inference_ns(job['kind'],{0:cost},[ipc],network_return_ns,
        affinity=0 if job['kind']=='FAST_SUFFIX' else None,
        inline=bool(job.get('inline_fast_continuation')))
    remaining=int(_deadline(job,tasks))-int(now)
    return minimum<=remaining<margin*minimum,remaining-minimum

def rescue_swap(s,fast,accurate,tasks,cost,gate,ipc,now,best,bestrole,
                rescue_margin,network_return_ns):
    """Cross-role thin-budget swap for one card; None keeps the normal choice.

    Opt-in only.  When the card's normally selected head is not urgent but the
    OTHER role's queue head is viable and thin (minimum <= remaining < margin x
    minimum on this card), the thin job runs first.  Eligibility, the per-period
    share check and the FIFO head rule mirror select_job exactly; pinned
    commitments are never displaced (callers skip the swap for them).  Only the
    role/job chosen for one dispatch changes: no admission, service, deadline or
    accounting rule is touched and running work is never preempted.
    """
    if not (rescue_margin and now is not None and best is not None):
        return None
    alt_role='Accurate' if bestrole=='Fast' else 'Fast'
    alt_jobs=accurate if alt_role=='Accurate' else fast
    if not alt_jobs:return None
    alt=(min(alt_jobs,key=lambda j:(j['enqueue_ns'],j['task_id'])) if alt_role=='Accurate'
         else min(alt_jobs,key=lambda j:(tasks[j['task_id']]['arrival_ns'],j['task_id'])))
    share=s['target'] if alt_role=='Fast' else 1-float(s['target'])
    if share<=0:return None
    expected=gate[alt['kind']]+(gate['FAST_SUFFIX'] if alt['kind']=='FAST_PREFIX' else 0)
    reserve=len(s['pending'])*gate['FAST_SUFFIX'] if alt_role=='Fast' else 0
    if 0<share<1 and s['spent'][alt_role]+reserve+expected>share*PERIOD:
        return None
    alt_thin,alt_slack=_thin_slack(alt,tasks,cost,ipc,now,network_return_ns,rescue_margin)
    if not alt_thin:return None
    best_thin,best_slack=_thin_slack(best,tasks,cost,ipc,now,network_return_ns,rescue_margin)
    if best_thin and best_slack<=alt_slack:return None
    return alt,alt_role

def select_job(s,fast,accurate,tasks,cost,gates=None,now=None,ipc=0,
               rescue_margin=None,network_return_ns=0):
    """``cost`` is the mean service prediction; ``gates`` is the start budget.

    TASK 3.4: the start gate is ``max(warmup_floor, P95 live service)`` while the
    predicted service charge uses the mean.  When ``gates`` is omitted the two
    coincide, which is exactly the frozen kernel behaviour the signed
    reference_checks fixtures pin.
    """
    gate=gates if gates is not None else cost
    pinned = min(s['pinned'],key=lambda j:(tasks[j['task_id']]['arrival_ns'],j['task_id'])) if s['pinned'] else None
    if pinned is not None and ('_service_debt' not in s or s['transition']):
        return pinned,'Fast'
    if s['transition']:return None,None
    if '_service_debt' in s:
        role, _deduction = _service_turn(s, fast, accurate, cost)
        if role is None:
            return None, None
        if role == 'Fast' and pinned is not None:
            return pinned, role
        jobs = fast if role == 'Fast' else accurate
        if not jobs:
            return None, None
        head = min(jobs, key=lambda j: (
            tasks[j['task_id']]['arrival_ns'] if role == 'Fast' else j['enqueue_ns'],
            j['task_id']))
        return rescue_swap(s,fast,accurate,tasks,cost,gate,ipc,now,head,role,
                           rescue_margin,network_return_ns) or (head, role)
    best=None;bestrole=None;fraction=s['target']
    for role,jobs,share in [('Fast',fast,fraction),('Accurate',accurate,1-fraction)]:
        if not jobs or share<=0:continue
        j=min(jobs,key=lambda j:(tasks[j['task_id']]['arrival_ns'],j['task_id'])) if role=='Fast' else min(jobs,key=lambda j:(j['enqueue_ns'],j['task_id']))
        expected=gate[j['kind']]+(gate['FAST_SUFFIX'] if j['kind']=='FAST_PREFIX' else 0)
        reserve=len(s['pending'])*gate['FAST_SUFFIX'] if role=='Fast' else 0
        if 0<share<1 and s['spent'][role]+reserve+expected>share*PERIOD:continue
        if best is None or s['spent'][role]/share<s['spent'][bestrole]/(fraction if bestrole=='Fast' else 1-fraction):best,bestrole=j,role
    return rescue_swap(s,fast,accurate,tasks,cost,gate,ipc,now,best,bestrole,
                       rescue_margin,network_return_ns) or (best,bestrole)

def rescue_select(candidates,tasks,now,margin,network_return_ns=0):
    """Least-slack choice among viable-but-thin candidates; None when none is thin.

    Opt-in only.  A candidate is thin when its remaining absolute budget is below
    ``margin`` times its own no-queue completion on its candidate card - the same
    service + ipc + return terms the pre-start budget guard uses.  Candidates that
    already cannot finish (remaining < minimum) are left to that guard instead of
    being rescued here.  This only reorders the already-legal candidate set: it
    admits no new job, changes no service, deadline or accounting rule, and never
    preempts running work.  ``margin`` must be a predeclared decimal (e.g. 1.5 or
    2.0) so the comparison stays exact.
    """
    thin=[]
    for work,arrival,card,job,role in candidates:
        remaining=int(_deadline(job,tasks))-int(now)
        minimum=int(work)+int(network_return_ns)
        if minimum<=remaining<margin*minimum:
            thin.append((remaining-minimum,work,arrival,card,job,role))
    if not thin:return None
    slack,work,arrival,card,job,role=min(thin,key=lambda t:(t[0],t[1],t[2],t[3]))
    return work,arrival,card,job,role

def dispatch_choice(cards,fast,accurate,tasks,costs,ipc,gates=None,now=None,
                    rescue_margin=None,network_return_ns=0):
    """Assign the selected ready job to the earliest completing idle card.

    Per-card quota/FIFO/owed-suffix rules remain select_job's responsibility.
    Evaluate all idle cards before removing a shared ready job from its queue.
    ``rescue_margin`` enables the opt-in thin-budget rescue (see rescue_select);
    it stays off unless a caller passes a positive predeclared margin, so every
    existing caller and replay fixture keeps its frozen behaviour.
    """
    candidates=[]
    gate_costs=gates if gates is not None else costs
    for g,s in enumerate(cards):
        if s['busy']:continue
        j,role=select_job(s,fast,accurate,tasks,costs[g],gate_costs[g],now=now,ipc=ipc[g],
                          rescue_margin=rescue_margin,network_return_ns=network_return_ns)
        if j is None:continue
        work=costs[g][j['kind']]+ipc[g]
        if j['kind']=='FAST_PREFIX':
            work+=costs[g]['FAST_SUFFIX']+(0 if j.get('inline_fast_continuation') else ipc[g])
        candidates.append((work,tasks[j['task_id']]['arrival_ns'],g,j,role))
    if not candidates:return None,None,None
    selected=min(candidates,key=lambda x:x[:3])
    if rescue_margin and now is not None:
        rescued=rescue_select(candidates,tasks,now,float(rescue_margin),network_return_ns)
        if rescued is not None:selected=rescued
    # Preserve the existing job and earliest-completion choice. Only break a
    # same-job, equal-service/equal-gate tie by keeping the shared card free
    # for its other role. Forecast and live dispatch call this same function.
    def gate_work(c):
        _,_,card,job,_=c
        return (gate_costs[card][job['kind']]+ipc[card]+
                (gate_costs[card]['FAST_SUFFIX']+
                 (0 if job.get('inline_fast_continuation') else ipc[card])
                 if job['kind']=='FAST_PREFIX' else 0))
    tied=[c for c in candidates
          if c[0]==selected[0] and c[3]['task_id']==selected[3]['task_id']
          and c[3]['kind']==selected[3]['kind']
          and gate_work(c)==gate_work(selected)]
    def preserve_other_role(c):
        _,_,card,_,role=c
        p=float(cards[card]['target'])
        dedicated=(p==1.0 if role=='Fast' else p==0.0)
        return (not dedicated,card)
    _,_,g,j,role=min(tied,key=preserve_other_role)
    return g,j,role

def _expired(job,now):
    """A waiting prediction entry carries its real absolute deadline.

    TASK 3.1: already-expired waiting entries are filtered out of the prediction;
    in-flight expired work still occupies the card and is charged real charge.
    Entries with no recorded deadline (frozen scalar fixtures) are never filtered.
    """
    d=job.get('deadline_ns')
    return d is not None and int(d)<=int(now)

def forecast(cards,ready,tasks,costs,ipc,now,kind,affinity=None,task_id=None,decision_ns=600_000,horizon_ns=None,gates=None,wall_budget_ns=None,candidate_job=None,prefix_terminal=False):
    """No evaluation results, no arrival plan, no service samples are read.

    TASK 3.1: the horizon is H = max(0, D - fresh_now - delivery_budget) and must
    be derived from the request's absolute deadline.  There is deliberately no
    default horizon: the old fixed 25 ms view silently rejected requests the real
    50 ms deadline still admitted.  ``gates`` carries the start budget while
    ``costs`` carries the mean used for the predicted service charge.

    T014 s3.1.9: ``wall_budget_ns`` makes the forecast wall budget REAL.  The
    prediction is a bounded event simulation whose cost grows with the waiting
    population, so the caller's budget is enforced inside the loop and an
    over-budget call returns ``None`` (an explicit partial/unknown result) with
    the delay already measured by the caller.  A budget of ``None`` keeps the
    unbounded frozen-kernel behaviour the signed fixtures pin.

    ``prefix_terminal`` follows a candidate Prefix through the public Head
    delay and its same-card Suffix calendar. It reserves the Fast continuation,
    not an unrevealed Accurate/Abstain outcome. Default callers still request
    the stage completion only; no extra service is booked before dispatch.
    """
    if horizon_ns is None:
        raise ValueError('forecast requires horizon_ns derived from the absolute deadline (TASK 3.1)')
    horizon_ns=max(0,int(horizon_ns))
    gate_costs=gates if gates is not None else costs
    budget=int(wall_budget_ns) if wall_budget_ns else 0
    t_start=time.perf_counter_ns() if budget else 0
    transitions=0
    expired_waiters=0
    def _over_budget(reason):
        # T014 s3.1.5: the wall budget is checked at EVERY calendar transition
        # (and at every outer step), not every N steps; a clock read costs
        # ~60 ns, far below the 0.25 ms budget.  The snapshot construction above
        # is inside the budget.  ``min(elapsed, limit)`` truncation is never
        # applied - the real elapsed time is recorded.
        if not budget:return False
        elapsed=time.perf_counter_ns()-t_start
        if elapsed<=budget:return False
        LAST_FORECAST.update({'terminated':reason,'elapsed_ns':elapsed,
                              'transitions':transitions,'budget_ns':budget,
                              'expired_waiters_removed':expired_waiters})
        return True
    def _terminate(value,reason):
        LAST_FORECAST.update({'terminated':reason,
                              'elapsed_ns':(time.perf_counter_ns()-t_start) if budget else None,
                              'transitions':transitions,'budget_ns':budget,
                              'expired_waiters_removed':expired_waiters})
        return value
    end=now+horizon_ns
    # T014 s3.1.9 targeted performance fix (semantically equivalent): the card
    # state was copied with ``copy.deepcopy``, which measured 2.1-3.0 ms per
    # forecast call on a realistic card (40 cycle buckets x 2 roles + 256
    # pending + 256 pinned) and pushed the real GPU_READY handler over its 5 ms
    # max gate.  ``forecast`` only mutates the four containers below plus
    # scalar/rebound keys, and the job record is read-only, so an explicit
    # non-mutating copy of exactly those containers is enough -- measured
    # 0.08-0.10 ms (about 25x cheaper) with ``self_check``'s
    # ``assert s == before`` (the input state is never mutated) still holding.
    state=[dict(s) for s in cards]
    for s in state:
        charges = s.get('cycle_charges') or {}
        s['cycle_charges'] = {k: Counter(charges[k])
                             for k in range(now//PERIOD-1, end//PERIOD+2)
                             if k in charges}
        s['pending']=set(s.get('pending') or ())
        s['pinned']=[dict(j) for j in (s.get('pinned') or ())]
        s['spent']=Counter(s.get('spent') or {})
        if '_service_debt' in s:
            # Candidate calendars must never spend the live card's service
            # balances or settle its outstanding prediction tokens.
            s['_service_debt']=debt_copy(s)
    # T014 targeted performance fix (semantically equivalent): the ready jobs
    # only carry scalar/str fields, so a per-job SHALLOW copy is enough.  The
    # previous whole-queue deepcopy made one forecast O(queue) in Python object
    # copying and pushed the real GPU_READY handler far over its 0.5 ms budget
    # once the observed waiting population reached the required 256 components.
    fast=[dict(j) for j in ready['Fast'] if not _expired(j,now)]
    accurate=[dict(j) for j in ready['Accurate'] if not _expired(j,now)]
    ids={j['task_id'] for j in fast+accurate}
    for s in state:
        ids.update(s['pending']);ids.update(j['task_id'] for j in s['pinned'])
        if s['job']:ids.add(s['job']['task_id'])
    if task_id is None:task_id=max(ids|set(tasks),default=-1)+1
    ages={i:{'arrival_ns':tasks[i]['arrival_ns'],
             'deadline_ns':tasks[i].get('deadline_ns',tasks[i].get('deadline_abs_ns'))}
          for i in ids if i in tasks}
    ages[task_id]={'arrival_ns':tasks.get(task_id,{}).get('arrival_ns',now),
                   'deadline_ns':tasks.get(task_id,{}).get('deadline_ns',tasks.get(task_id,{}).get('deadline_abs_ns'))}
    request={'task_id':task_id,'kind':kind,'enqueue_ns':now,'candidate':True,'affinity':affinity}
    if kind == 'FAST_PREFIX' and any(s.get('inline_fast_continuation') for s in state):
        request['inline_fast_continuation'] = True
    if candidate_job is not None:
        request.update(candidate_job)
        request['candidate']=True
    if ages[task_id]['deadline_ns'] is not None:
        request['deadline_ns']=ages[task_id]['deadline_ns']
    if kind=='FAST_SUFFIX':state[affinity]['pending'].discard(task_id);state[affinity]['pinned'].append(request)
    else:(accurate if kind=='ACCURATE' else fast).append(request)
    heap=[];serial=0
    def event(at,typ,g,j):
        nonlocal serial
        serial+=1;heapq.heappush(heap,(at,serial,typ,g,j))
    def charge(s,role,a,b):
        while a<b:
            cy=int(a//PERIOD);e=min(b,(cy+1)*PERIOD)
            s['cycle_charges'].setdefault(cy,Counter())[role]+=e-a;a=e
    for g,s in enumerate(state):
        s['cycle_charges']={int(k):Counter(v) for k,v in s['cycle_charges'].items() if k>=now//PERIOD-1}
        running=s['job']
        if s['busy'] and running:
            finish=max(now+100_000,s['budget_finish_ns']);role='Accurate' if running['kind']=='ACCURATE' else 'Fast'
            charge(s,role,max(running['dispatch_ns'],int(now//PERIOD)*PERIOD),finish)
            if '_service_debt' in s:
                # The runtime already booked this job at dispatch.  Settle
                # only the difference at simulated completion; budget_finish
                # includes delivery IPC, which is not GPU service.
                running=dict(running,_forecast_service_ns=max(0,finish-running['dispatch_ns']-ipc[g]))
            event(finish,'DONE',g,running)
        for i in list(s['pending']):
            if (running and running['kind']=='FAST_PREFIX' and running['task_id']==i) or i==task_id:continue
            # A pending Head retains activation until its decision arrives.
            # Even an expired root needs that release event in the calendar.
            event(now+decision_ns,'DECISION',g,{'task_id':i,'kind':'FAST_SUFFIX','enqueue_ns':now+decision_ns,'affinity':g,'deadline_ns':ages.get(i,{}).get('deadline_ns')})
    for s in state:
        s['pinned']=[j for j in s['pinned'] if not _expired(j,now)]
    at=now
    def waiting_expired(j):
        d=j.get('deadline_ns')
        return d is not None and int(d)<int(at)
    for _ in range(1000):
        if _over_budget('wall_budget_loop'):return None
        while heap and heap[0][0]<=at:
            if _over_budget('wall_budget_transition'):return None
            transitions+=1
            if transitions>FORECAST_MAX_TRANSITIONS:
                return _terminate(None,'max_transitions')
            _,_,typ,g,j=heapq.heappop(heap);s=state[g]
            if typ=='DECISION':
                s['pending'].discard(j['task_id'])
                if not _expired(j,at):s['pinned'].append(j)
                else:expired_waiters+=1
            else:
                if '_service_debt' in s:
                    debt_complete(s,j,j['_forecast_service_ns'])
                s['busy']=False;s['job']=None
                if j.get('candidate') and not (prefix_terminal and j['kind']=='FAST_PREFIX'
                                              and not j.get('inline_fast_continuation')):
                    return _terminate(at-now,'complete')
                if j['kind']=='FAST_PREFIX' and not j.get('inline_fast_continuation'):
                    continuation={'task_id':j['task_id'],'kind':'FAST_SUFFIX','enqueue_ns':at+decision_ns,'affinity':g,'deadline_ns':ages[j['task_id']].get('deadline_ns')}
                    if j.get('candidate') and prefix_terminal:
                        continuation['candidate']=True
                    event(at+decision_ns,'DECISION',g,continuation)
        if at>end:return _terminate(math.inf,'horizon_infeasible')
        # Runtime._expire_ready runs before EVERY dispatch, not just at the
        # admission snapshot. An old waiter that expires while a card is busy
        # cannot consume predicted GPU service after the card becomes free.
        # Keep the inclusive runtime boundary and never remove running jobs.
        for jobs in (fast,accurate):
            if not jobs:continue
            kept=[j for j in jobs if not waiting_expired(j)]
            expired_waiters+=len(jobs)-len(kept)
            jobs[:]=kept
        for s in state:
            if not s['pinned']:continue
            kept=[]
            for j in s['pinned']:
                if waiting_expired(j):
                    s['pending'].discard(j['task_id']);expired_waiters+=1
                else:kept.append(j)
            s['pinned']=kept
        for g,s in enumerate(state):
            if s['busy']:continue
            s['spent']=Counter(s['cycle_charges'].get(int(at//PERIOD),{}))
            interior_change=('_service_debt' in s and
                             0<s['effective']<1 and 0<s['target']<1)
            if s['transition'] and (interior_change or (not s['pending'] and not s['pinned'])):
                s['transition']=False;s['effective']=s['target']
        for _ in state:
            g,j,role=dispatch_choice(state,fast,accurate,ages,costs,ipc,gate_costs,now=at)
            if j is None:break
            s=state[g]
            if '_service_debt' in s:
                # Read both shared demand queues before removing the winner,
                # exactly as real dispatch does when crediting competition.
                debt_dispatch(s,j,role,costs[g],fast,accurate)
            (s['pinned'] if j in s['pinned'] else fast if role=='Fast' else accurate).remove(j)
            s['busy']=True;s['job']=j
            if j['kind']=='FAST_PREFIX' and not j.get('inline_fast_continuation'):
                s['pending'].add(j['task_id'])
            # Kernel/API service retains its measured duration at every share.
            duration=costs[g][j['kind']]
            if j.get('inline_fast_continuation'):duration+=costs[g]['FAST_SUFFIX']
            charge(s,role,at,at+duration)
            event(at+duration+ipc[g],'DONE',g,
                  dict(j,_forecast_service_ns=duration) if '_service_debt' in s else j)
        boundary=(int(at//PERIOD)+1)*PERIOD
        nextat=min(heap[0][0] if heap else math.inf,boundary)
        if nextat>end:return _terminate(math.inf,'horizon_infeasible')
        at=nextat
    return _terminate(math.inf,'step_limit')

def self_check():
    costs=[{'FAST_PREFIX':1_000_000,'FAST_SUFFIX':2_000_000,'FAST_FULL':2_000_000,'ACCURATE':4_000_000}]*2
    def card(share):return {'busy':False,'job':None,'pinned':[],'pending':set(),'target':share,'effective':share,'transition':False,
                          'spent':Counter(),'cycle_charges':{},'budget_finish_ns':0}
    s=[card(.3),card(0)];before=copy.deepcopy(s)
    free=forecast(s,{'Fast':[],'Accurate':[]},{0:{'arrival_ns':0}},costs,[0,0],0,'FAST_FULL',task_id=0,horizon_ns=25_000_000)
    assert free==2_000_000,free;assert s==before
    s[0]['cycle_charges']={0:Counter(Fast=14_000_000)}
    exhausted=forecast(s,{'Fast':[],'Accurate':[]},{0:{'arrival_ns':0}},costs,[0,0],0,'FAST_FULL',task_id=0,horizon_ns=25_000_000)
    assert math.isinf(exhausted)
    s=[card(0),card(0)];s[0]['transition']=True;s[0]['pending']={0}
    owed=forecast(s,{'Fast':[],'Accurate':[]},{0:{'arrival_ns':0}},costs,[0,0],0,'FAST_SUFFIX',affinity=0,task_id=0,horizon_ns=25_000_000)
    assert owed==2_000_000,owed
    # TASK 3.1: the horizon must come from the absolute deadline, not a 25 ms default.
    try:
        forecast(s,{'Fast':[],'Accurate':[]},{0:{'arrival_ns':0}},costs,[0,0],0,'FAST_FULL',task_id=0)
        raise AssertionError('forecast accepted a missing horizon_ns')
    except ValueError:pass
    # An already-expired waiting entry must be filtered out of the prediction.
    s_exp=card(.3);s_exp['pinned']=[{'task_id':7,'kind':'FAST_SUFFIX','enqueue_ns':0,'affinity':0,'deadline_ns':10}]
    assert forecast([s_exp,card(0)],{'Fast':[],'Accurate':[]},{7:{'arrival_ns':0,'deadline_ns':10}},costs,[0,0],20,'FAST_FULL',task_id=8,horizon_ns=25_000_000)==2_000_000
    # Start gate separated from mean service: a larger gate blocks the start while
    # the predicted service charge keeps using the mean.
    hot=card(1.);hot['cycle_charges']={0:Counter(Fast=0)}
    gate_only=[{'FAST_PREFIX':1_000_000,'FAST_SUFFIX':2_000_000,'FAST_FULL':2_000_000,'ACCURATE':4_000_000},
               {'FAST_PREFIX':1_000_000,'FAST_SUFFIX':2_000_000,'FAST_FULL':2_000_000,'ACCURATE':4_000_000}]
    gated=[{'FAST_PREFIX':30_000_000,'FAST_SUFFIX':30_000_000,'FAST_FULL':30_000_000,'ACCURATE':30_000_000},
           {'FAST_PREFIX':30_000_000,'FAST_SUFFIX':30_000_000,'FAST_FULL':30_000_000,'ACCURATE':30_000_000}]
    assert math.isinf(forecast([card(.3),card(0)],{'Fast':[],'Accurate':[]},{0:{'arrival_ns':0}},costs,[0,0],0,'FAST_FULL',task_id=0,horizon_ns=25_000_000,gates=gated))
    assert forecast([card(.3),card(0)],{'Fast':[],'Accurate':[]},{0:{'arrival_ns':0}},costs,[0,0],0,'FAST_FULL',task_id=0,horizon_ns=25_000_000,gates=gate_only)==2_000_000
    ss=card(.3);ss['pending']={1};ss['spent']['Fast']=14_000_000
    j,role=select_job(ss,[{'task_id':0,'kind':'FAST_PREFIX','enqueue_ns':0}],[],{0:{'arrival_ns':0}},costs[0])
    assert j is None
    ss['target']=1.;ss['spent']=Counter();j,role=select_job(ss,[{'task_id':0,'kind':'FAST_PREFIX','enqueue_ns':0}],[],{0:{'arrival_ns':0}},costs[0])
    assert j is not None and role=='Fast'
    heterogeneous=[{**costs[0],'FAST_FULL':8_000_000},{**costs[1],'FAST_FULL':4_000_000}]
    assert eligible_service([card(1),card(0)],heterogeneous,[0,0],'FAST_FULL')==8_000_000
    assert math.isinf(eligible_service([card(0),card(0)],costs,[0,0],'FAST_FULL'))
    assert math.isinf(eligible_service([card(1),card(1)],costs,[0,0],'ACCURATE'))
    assert eligible_service([card(0),card(0)],costs,[0,0],'FAST_SUFFIX',0)==2_000_000
    heterogeneous=[{**costs[0],'ACCURATE':14_000_000},{**costs[1],'ACCURATE':7_000_000}]
    dedicated=forecast([card(1),card(0)],{'Fast':[],'Accurate':[]},{0:{'arrival_ns':0}},heterogeneous,[0,0],0,'ACCURATE',task_id=0,horizon_ns=25_000_000)
    expanded=forecast([card(.7),card(0)],{'Fast':[],'Accurate':[]},{0:{'arrival_ns':0}},heterogeneous,[0,0],0,'ACCURATE',task_id=0,horizon_ns=25_000_000)
    assert dedicated==expanded==7_000_000,(dedicated,expanded)
    return {'pass':True,'single_shared_job_ms':free/1e6,'quota_exhausted_wait_exceeds25ms':math.isinf(exhausted),
            'owed_suffix_on_new_Accurate_card_ms':owed/1e6,'input_state_not_mutated':True,
            'stable_dedicated_Fast_accepts_new_prefix_while_CPU_pending':True,
            'faster_ineligible_card_not_used_as_continuation_cost':True,'no_role_capacity_returns_infinity':True}
if __name__=='__main__':print(self_check())
