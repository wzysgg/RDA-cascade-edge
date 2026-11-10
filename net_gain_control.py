"""Bounded two-allocation, two-observed-window deadline/latency forecast.

No labels, image cache, source plan, quality scores or future events are read.
Past Prefix reveals predict future routes; they do not reveal live Prefixes.
This is a forecast model, not the admission oracle: future offered jobs use
intrinsic deadline and finite-capacity checks, not per-arrival ETA recursion.
"""
import heapq
import math
from collections import Counter, deque

from resource_axis import card_shares
from quota_forecast_v8 import debt_copy, debt_dispatch, debt_complete, dispatch_choice
from execution_guards import minimum_inference_ns

TICK = 100000000
ROLE = {'FAST_FULL': 'Fast', 'FAST_SUFFIX': 'Fast', 'ACCURATE': 'Accurate'}


class Unknown(ValueError):
    pass


def clean_job(job):
    return {k: job[k] for k in ('task_id', 'kind', 'enqueue_ns', 'affinity',
        'inline_fast_continuation', 'dispatch_ns') if k in job}


class PublicHistory:
    def __init__(self, maximum=512):
        self.roots = {}
        self.order = deque()
        self.maximum = maximum
        self.dropped_until = -1

    def observe(self, root, now):
        publish = int(root['source_publish_ns'])
        if publish > now:
            raise ValueError('future net-gain observation')
        floor = now - 2*TICK
        while self.order and self.order[0][0] < floor:
            _, rid = self.order.popleft()
            self.roots.pop(rid, None)
        rid = root['root_id']
        if publish < floor:
            return
        if rid not in self.roots:
            self.order.append((publish, rid))
        # Whitelist only public fields; no quality, bank index, or prediction.
        self.roots[rid] = {k: root[k] for k in ('root_id', 'source_publish_ns',
            'original_arrival_ns', 'deadline_ns', 'kind', 'revealed_role') if k in root}
        while len(self.roots) > self.maximum:
            t, old = self.order.popleft()
            self.roots.pop(old, None)
            self.dropped_until = max(self.dropped_until, t)

    def windows(self, now):
        if now < 2*TICK or self.dropped_until >= now-2*TICK:
            raise Unknown('public_history_incomplete')
        return [[dict(r) for r in self.roots.values()
            if now-(2-i)*TICK <= r['source_publish_ns'] < now-(1-i)*TICK]
            for i in (0, 1)]


def calendar(snapshot, target, tasks, jobs, future, routes, max_events=8192):
    """One joint queue calendar; never changes an individual CNN duration."""
    now = snapshot['now']
    costs = snapshot['costs']; gates = snapshot['gates']; ipc = snapshot['ipc']
    terminal_return = snapshot['terminal_return']
    cards = []
    for source, share in zip(snapshot['cards'], card_shares(target)):
        card = dict(source, pending=set(source['pending']),
                    pinned=[dict(j) for j in source['pinned']],
                    _service_debt=debt_copy(source))
        card['target'] = share
        card['transition'] = abs(card['effective']-share) > 1e-12
        cards.append(card)
    ready = {r: [dict(j) for j in jobs[r]] for r in ('Fast', 'Accurate')}
    heap = []; serial = 0; finished = {}; operations = 0
    end = max([now]+[v['deadline_ns'] for v in tasks.values()])

    def event(at, kind, gpu, job):
        nonlocal serial
        serial += 1
        heapq.heappush(heap, (int(at), serial, kind, gpu, job))

    def route(job):
        return routes.get(job['task_id'], ROLE.get(job['kind']))

    def remaining_kind(job, at):
        kind = job['kind']
        role = route(job)
        duration = int(costs[0][kind])
        if kind == 'FAST_PREFIX' and role == 'Fast':
            if at+duration+int(gates[0]['FAST_SUFFIX'])+terminal_return > tasks[job['task_id']]['deadline_ns']:
                return duration, 'Abstain'
            duration += int(costs[0]['FAST_SUFFIX'])
        return duration, role

    # The two measured cards are homogeneous in this frozen experiment.
    # Check rather than silently using card0 timings on heterogeneous devices.
    if any(costs[g] != costs[0] or gates[g] != gates[0] for g in range(1,len(cards))):
        raise Unknown('heterogeneous_service_requires_card_specific_prefix_model')
    for gpu, card in enumerate(cards):
        if card['pending'] or card['pinned']:
            # Current production uses inline Prefix/Suffix. Do not invent a
            # CPU-Head promise or migrate saved activation in other modes.
            raise Unknown('non_inline_pending_not_supported')
        if card['busy'] and card['job']:
            job = dict(card['job'])
            # Public gate completion is conservative for an unrevealed running
            # Prefix. Never shorten a running GPU operation using hidden EARLY.
            finish = max(now, int(card['budget_finish_ns']))
            service = max(0, finish-int(job.get('dispatch_ns',now))-int(ipc[gpu]))
            job['_service'] = service;job['_route'] = route(job)
            event(finish, 'DONE', gpu, job)
    for job in future:
        event(job['enqueue_ns'], 'ARRIVE', -1, dict(job))
    at = now
    while at <= end:
        operations += 1
        if operations > max_events:
            raise Unknown('calendar_operation_limit')
        while heap and heap[0][0] <= at:
            operations += 1
            if operations > max_events:
                raise Unknown('calendar_operation_limit')
            _, _, typ, gpu, job = heapq.heappop(heap)
            tid = job['task_id']
            if typ == 'ARRIVE':
                if tasks[tid]['deadline_ns'] <= at:
                    continue
                waiting = {j['task_id'] for v in ready.values() for j in v}
                capacity = snapshot['capacity']
                if capacity is not None and len(waiting) >= capacity:
                    continue
                kind = job['kind']
                minimum = minimum_inference_ns(kind,costs,ipc,
                    snapshot.get('physical_return',terminal_return),
                    affinity=job.get('affinity') if kind=='FAST_SUFFIX' else None,
                    inline=bool(job.get('inline_fast_continuation')),
                    head_ns=snapshot.get('head_ns',0),model_only=snapshot.get('model_only'))
                if at+minimum > tasks[tid]['deadline_ns']:
                    continue
                ready['Accurate' if kind=='ACCURATE' else 'Fast'].append(job)
                continue
            card = cards[gpu]
            debt_complete(card, job, job['_service'])
            card['busy'] = False;card['job'] = None
            if job['kind'] == 'FAST_PREFIX' and job['_route'] == 'Accurate':
                # Inline GPU quality head is included in Prefix service.
                # Accurate gets no saved Fast activation or cross-card Suffix.
                event(at, 'ARRIVE', -1, dict(task_id=tid,kind='ACCURATE',enqueue_ns=at,affinity=None))
            elif job['_route'] != 'Abstain':
                finished[tid] = at+terminal_return
        for role in ready:
            ready[role][:] = [j for j in ready[role] if tasks[j['task_id']]['deadline_ns'] > at]
        for card in cards:
            if not card['busy'] and card['transition']:
                card['effective'] = card['target'];card['transition'] = False
        # Removing an impossible head must not consume a GPU slot or strand the
        # next feasible task. This matches the real before-booking guard.
        attempts = len(cards)+sum(len(q) for q in ready.values())
        for _ in range(attempts):
            operations += 1
            if operations > max_events:
                raise Unknown('calendar_operation_limit')
            gpu, job, role = dispatch_choice(cards,ready['Fast'],ready['Accurate'],tasks,
                                              costs,ipc,gates,now=at,
                                              rescue_margin=snapshot.get('rescue_margin',0),
                                              network_return_ns=snapshot.get('physical_return',terminal_return))
            if job is None:
                break
            minimum = minimum_inference_ns(job['kind'],costs,ipc,
                snapshot.get('physical_return',terminal_return),
                affinity=job.get('affinity') if job['kind']=='FAST_SUFFIX' else None,
                inline=bool(job.get('inline_fast_continuation')),
                head_ns=snapshot.get('head_ns',0),model_only=snapshot.get('model_only'))
            if at+minimum > tasks[job['task_id']]['deadline_ns']:
                ready[role].remove(job)
                continue
            card = cards[gpu]
            debt_dispatch(card,job,role,costs[gpu],ready['Fast'],ready['Accurate'],
                thin_override=(dict(tasks=tasks,ipc=ipc[gpu],now=at,
                    network_return_ns=snapshot.get('physical_return',terminal_return),
                    margin=snapshot['rescue_margin']) if snapshot.get('rescue_margin',0)>0 else None))
            ready[role].remove(job)
            duration, outcome = remaining_kind(job,at)
            running = dict(job,dispatch_ns=at,_service=duration,_route=outcome)
            card['busy'] = True;card['job'] = running
            event(at+duration+int(ipc[gpu]),'DONE',gpu,running)
        if not heap:
            break
        next_at = heap[0][0]
        if next_at <= at:
            # A completion can release another immediate queue arrival.
            continue
        at = next_at
    return finished, operations


def summarize_pair(tasks, routes, current, candidate):
    result = dict(gained=0,lost=0,delta_ontime=0,delay_saved_ns=0,
                  roles={r:dict(gained=0,lost=0,delta_ontime=0,delay_saved_ns=0)
                         for r in ('Fast','Accurate','Abstain')})
    for tid, task in tasks.items():
        old = current.get(tid, math.inf);new = candidate.get(tid, math.inf)
        deadline = task['deadline_ns'];arrival = task['arrival_ns']
        gain = int(new<=deadline and old>deadline)
        loss = int(old<=deadline and new>deadline)
        saved = min(old,deadline)-min(new,deadline)
        role = routes[tid]
        for row in (result,result['roles'][role]):
            row['gained'] += gain;row['lost'] += loss
            row['delta_ontime'] += gain-loss;row['delay_saved_ns'] += int(saved)
    return result


def decide(scenarios):
    counts = [r['delta_ontime'] for r in scenarios]
    delays = [r['delay_saved_ns'] for r in scenarios]
    lower_count = min(counts);lower_delay = min(delays)
    accept = lower_count>0 or (lower_count>=0 and lower_delay>0)
    return dict(valid=True,accept=accept,reason='robust_timely_gain' if lower_count>0 else
        'robust_nonharm_latency_gain' if accept else 'net_gain_not_supported',
        mean_delta_ontime=sum(counts)/2,epsilon_ontime=abs(counts[1]-counts[0])/2,
        mean_delay_saved_ns=sum(delays)/2,epsilon_delay_ns=abs(delays[1]-delays[0])/2,
        lower_delta_ontime=lower_count,lower_delay_saved_ns=lower_delay,
        ratios_evaluated=2,calendars_evaluated=4,scenarios=scenarios,
        scope='Public past-window forecast, not oracle; two-scenario spread is empirical, not a confidence bound')


def evaluate(snapshot, windows, current, target, maximum=512, max_events=8192):
    """Four bounded calendars: exactly two ratios x two public input windows."""
    now = snapshot['now']
    known = [r.get('revealed_role') for w in windows for r in w
             if r['kind']=='FAST_PREFIX' and r.get('revealed_role') in ('Fast','Accurate','Abstain')]
    prefix_ids = set()
    for jobs in snapshot['ready'].values():
        prefix_ids.update(j['task_id'] for j in jobs if j['kind']=='FAST_PREFIX')
    prefix_ids.update(c['job']['task_id'] for c in snapshot['cards']
                      if c['busy'] and c['job'] and c['job']['kind']=='FAST_PREFIX')
    if not known and (prefix_ids or any(r['kind']=='FAST_PREFIX' for w in windows for r in w)):
        return dict(valid=False,accept=False,reason='no_public_prefix_route_evidence')
    # Deterministic stratification uses only past publicly revealed frequencies.
    order = sorted(known,key=lambda r: ('Fast','Accurate','Abstain').index(r))
    scenarios = []
    try:
        for index, window in enumerate(windows):
            tasks = {i:dict(t) for i,t in snapshot['tasks'].items()}
            routes = dict(snapshot['routes']);future=[]
            unknown = sorted(prefix_ids)
            for j,tid in enumerate(unknown):
                routes[tid] = order[min(len(order)-1,(2*j+1)*len(order)//(2*len(unknown)))]
            origin = now-(2-index)*TICK
            for j,r in enumerate(sorted(window,key=lambda v:(v['source_publish_ns'],v['root_id']))):
                tid = -(j+1)
                release = now+r['source_publish_ns']-origin
                age = r['source_publish_ns']-r['original_arrival_ns']
                budget = r['deadline_ns']-r['source_publish_ns']
                tasks[tid] = dict(arrival_ns=release-age,deadline_ns=release+budget)
                if r['kind']=='FAST_PREFIX':
                    role = r.get('revealed_role')
                    if role not in ('Fast','Accurate','Abstain'):
                        role = order[min(len(order)-1,(2*j+1)*len(order)//(2*max(1,len(window))))]
                else:
                    role = ROLE[r['kind']]
                routes[tid] = role
                future.append(dict(task_id=tid,kind=r['kind'],enqueue_ns=release,
                    affinity=None,inline_fast_continuation=r['kind']=='FAST_PREFIX'))
            if len(tasks)>maximum:
                raise Unknown('cohort_population_limit')
            before,ops0 = calendar(snapshot,current,tasks,snapshot['ready'],future,routes,max_events)
            after,ops1 = calendar(snapshot,target,tasks,snapshot['ready'],future,routes,max_events)
            summary = summarize_pair(tasks,routes,before,after)
            scenarios.append(dict(summary,window_start_ns=origin,window_end_ns=origin+TICK,
                observed_roots=len(window),cohort_roots=len(tasks),operations=[ops0,ops1]))
    except Unknown as exc:
        return dict(valid=False,accept=False,reason=str(exc),scenarios=scenarios)
    return dict(decide(scenarios),current_fast=current,target_fast=target)


class LiveMarginalAllocator:
    """Bounded public-state batch capacity envelope, not a schedule replay.

    Unit reward is an on-time proxy, NOT a calibrated correctness probability.
    Actual TCA is evaluated offline. Unknown Prefix routing uses only published
    EARLY outcomes; absent evidence evaluates both endpoints conservatively.
    The fractional envelope is a prediction approximation, never GPU execution.
    """
    def __init__(self, runtime):
        self.rt=runtime;self.inflight={};self.inflight_expiry=[];self.outcomes=deque(maxlen=2048)
        self.routing=deque(maxlen=2048);self.revealed={};self.sequence=0
        self.command_delays=deque(maxlen=256);self.dispatch_overheads=deque(maxlen=256)
        self.last_epoch=None;self.last_snapshot=None;self.last_evidence=None
        # Already observed generation events, never a future source iterator.
        self.future_arrivals=deque(maxlen=8192)
        self.interval_cache=None
        self.cycle_samples=deque(maxlen=256)
        self.decision_delays=deque(maxlen=256)
        if runtime.cfg.get('marginal_interval_capacity',False):
            # Import/setup precedes the common execution epoch.
            self.np=__import__('numpy')

    def observe(self,frame,now):
        typ=frame.get('outcome');tid=frame.get('task_id')
        if tid is None:return
        if int(frame.get('source_publish_ns',now))>now:raise ValueError('Future marginal input')
        if typ=='SHARED_ARRIVAL':
            self.inflight[tid]={k:frame[k] for k in ('task_id','arrival_ns','deadline_ns')}
            heapq.heappush(self.inflight_expiry,(frame['deadline_ns'],tid))
            if self.rt.cfg.get('marginal_future_arrivals',False):
                self.future_arrivals.append((frame['arrival_ns'],frame['deadline_ns']-frame['arrival_ns']))
        elif typ=='SHARED_QUALITY_INTENT' and tid in self.inflight:
            role=frame.get('quality_role');stage=frame.get('stage')
            # Frozen Local protocol: MID1 Accurate uploads immediately; other
            # MID1 intents may still take MID2. Only final public decisions are
            # treated as announced routes, never cached/future quality.
            if role=='Accurate' or stage=='MID2':
                self.inflight[tid].update(announced_role=role,announced_ns=now)
        elif typ=='PREFIX_QUALITY_REVEALED':
            role=frame.get('quality_role');self.routing.append((now,role));self.revealed[tid]=role
        elif typ in ('GPU_READY','PREFIX_READY','COMMON_LOCAL_TERMINAL','COMMON_ABSTAIN'):
            role=('Prefix' if typ=='PREFIX_READY' else 'Local' if typ=='COMMON_LOCAL_TERMINAL'
                  else (frame.get('gpu_candidates') or ['Abstain'])[0])
            old=self.inflight.pop(tid,None)
            if old:self.outcomes.append((now,role,max(0,now-old['arrival_ns'])))
        # Incremental expiry avoids scanning every Local root on every event.
        while self.inflight_expiry and self.inflight_expiry[0][0]<=now:
            _,old_tid=heapq.heappop(self.inflight_expiry)
            self.inflight.pop(old_tid,None)

    def observe_cycle(self,role,dispatch_ns,received_ns,cuda_service_ns):
        if cuda_service_ns>0:
            self.cycle_samples.append((role,float(cuda_service_ns),
                max(0.,received_ns-dispatch_ns-cuda_service_ns)))

    @staticmethod
    def percentile(values,p=.95):
        values=sorted(values)
        return values[min(len(values)-1,int((len(values)-1)*p))] if values else 0.

    def card_availability(self,now):
        overhead=self.percentile(self.dispatch_overheads)
        result=[]
        for g,c in enumerate(self.rt.cards):
            batch=next((b for b in self.rt.batches.values() if b.get('gpu')==g),None)
            if not c['busy']:result.append(float(now));continue
            if batch is None:raise ValueError('Busy card missing public batch booking')
            predicted=batch['dispatch_ns']+self.rt.batch_mean(batch['jobs'],'p95_ms')+overhead
            # An overrun is still non-preemptible. At least one observed command
            # overhead remains; zero is not evidence that a busy card is free.
            result.append(max(float(now)+max(1.,overhead),predicted))
        return result

    @staticmethod
    def capacity(at,resources):
        return sum(p*max(0.,at-start) for p,start in resources)

    @staticmethod
    def finish_work(work,resources):
        active=0.;used=0.;last=0.
        for start,p in sorted((start,p) for p,start in resources if p>0):
            if active and used+active*(start-last)>=work:
                return last+(work-used)/active
            used+=active*(start-last);active+=p;last=start
        return last+(work-used)/active if active else math.inf

    def _role_curve(self,rows,role,resources,credit):
        if self.rt.cfg.get('marginal_unit_capacity_curve',False):
            return self._unit_capacity_curve(rows,resources,credit)
        if self.rt.cfg.get('marginal_deadline_feasible_curve',False):
            return self._deadline_role_curve(rows,resources,credit)
        rt=self.rt;pred={};used=max(0.,-credit)
        rows=sorted(rows,key=lambda j:(j['deadline_ns'],j.get('public_future',False),j['task_id']))
        for pos in range(0,len(rows),rt.cfg['batch_max']):
            batch=rows[pos:pos+rt.cfg['batch_max']]
            release=max(self.now,max(j.get('release_ns',self.now) for j in batch))
            # Existing ordinary/mixed tables include H2D, EARLY/conditional
            # same-card Suffix and D2H. No service duration is scaled by share.
            key=tuple((j['task_id'],j['kind']) for j in batch)
            if key not in self.duration_cache:self.duration_cache[key]=rt.batch_mean(batch,'p95_ms')
            duration=self.duration_cache[key]
            weight=sum(j.get('work_weight',1.) for j in batch)/len(batch)
            used=max(used,self.capacity(release,resources))+duration*weight
            done=max(release+duration,self.finish_work(used,resources))
            for j in batch:pred[j['task_id']]=done
        return pred

    def _unit_capacity_curve(self,rows,resources,credit):
        """R003: cheap deadline-capacity reward using observed batch work units.

        Work units are measured service/root, not a rescaling of real task
        duration by GPU share. Minimum elapsed service remains unscaled. The
        real executor retains its original batches, guards and non-preemption.
        This fluid relaxation is explicitly not a realizable schedule proof.
        """
        rt=self.rt;pred={};used=max(0.,-credit)
        capacity,inverse=self.two_resource_functions(resources)
        for j in sorted(rows,key=lambda j:(j['deadline_ns'],j.get('public_future',False),j['task_id'])):
            kind=j['kind'];key=('unit_capacity',kind)
            if key not in self.duration_cache:
                unit=rt.control_unit_ns(kind)
                physical=rt.batch_mean([j],'p95_ms')
                self.duration_cache[key]=(unit,physical)
            unit,physical=self.duration_cache[key]
            release=max(self.now,j.get('release_ns',self.now))
            next_used=max(used,capacity(release))+unit*j.get('work_weight',1.)
            limit=j['deadline_ns']-rt.cfg['network_return_ns']
            if release+physical<=limit and next_used<=capacity(limit):
                used=next_used;pred[j['task_id']]=max(release+physical,inverse(next_used))
            else:pred[j['task_id']]=math.inf
        return pred

    @staticmethod
    def two_resource_functions(resources):
        """Exact algebraic capacity/inverse; avoid sorting per forecast row."""
        live=sorted((start,p) for p,start in resources if p>0)
        if not live:return lambda t:0.,lambda w:math.inf
        a,p=live[0]
        if len(live)==1:return lambda t:p*max(0.,t-a),lambda w:a+w/p
        if len(live)!=2:raise ValueError('Research model requires exactly two GPUs')
        b,q=live[1];threshold=p*(b-a)
        return (lambda t:p*max(0.,t-a)+q*max(0.,t-b),
                lambda w:a+w/p if w<=threshold else b+(w-threshold)/(p+q))

    def deadline_curve_diagnostic(self,snapshot,candidates):
        """Small observed-deadline/profile knot table; not a schedulability proof."""
        now=snapshot['now_ns'];rt=self.rt
        groups={k:snapshot.get(k,[]) for k in ('waiting','inflight','future')}
        slacks=sorted(j['deadline_ns']-now for rows in groups.values() for j in rows if j['deadline_ns']>now)
        knots={self.percentile(slacks,q) for q in (0.,.5,.95,1.)} if slacks else set()
        for kind in ('FAST_FULL','ACCURATE'):
            sample=dict(task_id=-1,kind=kind,role='Fast' if kind=='FAST_FULL' else 'Accurate')
            knots.add(rt.batch_mean([sample],'p95_ms'))
        knots=sorted(k for k in knots if k>0)
        demand={};conditional=[];pf=snapshot['prefix_fast_probability']
        prices={kind:(self.duration_cache['unit_capacity',kind][0]
            if ('unit_capacity',kind) in self.duration_cache else rt.control_unit_ns(kind))
            for kind in ('FAST_FULL','FAST_PREFIX','ACCURATE')}
        for group,rows in groups.items():
            demand[group]={r:[sum(j.get('work_weight',1.)*prices[j['kind']] for j in rows
                    if j['role']==r and j['deadline_ns']<=now+h) for h in knots] for r in ('Fast','Accurate')}
            conditional.append([sum(j.get('work_weight',1.)*prices['ACCURATE'] for j in rows
                if j['kind']=='FAST_PREFIX' and j['deadline_ns']<=now+h) for h in knots])
        prefix_A=[sum(row[i] for row in conditional) for i in range(len(knots))]
        capacity={}
        for g in candidates:
            shares=card_shares(g)
            capacity[str(g)]={r:[self.capacity(now+h,[(p if r=='Fast' else 1-p,a)
                for p,a in zip(shares,snapshot['available'])]) for h in knots] for r in ('Fast','Accurate')}
        return dict(knots_ns=knots,demand_ns=demand,capacity_ns=capacity,
            conditional_prefix_A_work_lower_ns=[(1-pf)*x if pf is not None else 0. for x in prefix_A],
            conditional_prefix_A_work_upper_ns=[(1-pf)*x if pf is not None else x for x in prefix_A],
            scope='fluid demand/capacity diagnostic; conditional A release also requires Prefix; not a feasibility certificate')

    def _deadline_role_curve(self,rows,resources,credit):
        """R002: match the executor's shrink-before-service budget semantics.

        Infeasible work earns zero and consumes no predicted GPU service. A
        feasible batch keeps its measured unscaled duration; fluid shares only
        limit cumulative capacity. This remains an approximate role envelope,
        not a global counterfactual replay or a schedulability certificate.
        """
        rt=self.rt;pred={};used=max(0.,-credit);pos=0
        rows=sorted(rows,key=lambda j:(j['deadline_ns'],j.get('public_future',False),j['task_id']))
        while pos<len(rows):
            first=rows[pos]
            start=max(self.now,first.get('release_ns',self.now),self.finish_work(used,resources))
            batch=[]
            for job in rows[pos:pos+rt.cfg['batch_max']]:
                if job.get('release_ns',self.now)>start:break
                batch.append(job)
            if not batch:raise AssertionError('Head must be released at its predicted start')
            while batch:
                key=tuple((j['task_id'],j['kind']) for j in batch)
                if key not in self.duration_cache:
                    self.duration_cache[key]=rt.batch_mean(batch,'p95_ms')
                duration=self.duration_cache[key]
                weight=sum(j.get('work_weight',1.) for j in batch)/len(batch)
                next_used=max(used,self.capacity(start,resources))+duration*weight
                done=max(start+duration,self.finish_work(next_used,resources))
                if done+rt.cfg['network_return_ns']<=min(j['deadline_ns'] for j in batch):
                    for j in batch:pred[j['task_id']]=done
                    used=next_used;pos+=len(batch)
                    break
                if len(batch)>1:batch=batch[:-1]
                else:
                    # Exactly a forecast rejection, never remove a live job.
                    pred[first['task_id']]=math.inf;pos+=1;batch=[]
        return pred

    def snapshot(self,now):
        rt=self.rt;self.now=now;self.duration_cache={}
        maximum=rt.cfg.get('marginal_max_roots',512)
        waiting=sorted((dict(j,release_ns=now,reward=1.) for j in rt.queued.values()
                       if j['deadline_ns']>now),key=lambda j:(j['deadline_ns'],j['task_id']))
        truncated=max(0,len(waiting)-maximum);waiting=waiting[:maximum]
        recent=[x for x in self.outcomes if now-x[0]<=2*self.rt.cfg['deadline_ns']]
        counts=Counter(r for t,r,d in recent);n=sum(counts.values())
        inflight=[]
        delays_by_role={role:self.percentile([d for t,a,d in recent if a==role])
                        for role in ('Fast','Accurate','Prefix')}
        for tid,r in list(self.inflight.items())[:max(0,maximum-len(waiting))]:
            announced=r.get('announced_role')
            for role,kind in (('Fast','FAST_FULL'),('Accurate','ACCURATE'),('Prefix','FAST_PREFIX')):
                if announced is not None:
                    if announced!=role:continue
                    probability=1.
                    release=max(now,r['announced_ns']+rt.cfg.get('marginal_public_upload_ns',6_500_000))
                else:
                    if not n or not counts[role]:continue
                    probability=counts[role]/n
                    release=max(now,r['arrival_ns']+delays_by_role[role])
                if release>=r['deadline_ns']:continue
                inflight.append(dict(r,root_id=tid,task_id=-(tid*4+('Fast','Accurate','Prefix').index(role)+1),
                    kind=kind,role='Accurate' if role=='Accurate' else 'Fast',release_ns=release,
                    reward=probability,work_weight=probability,public_inflight=True))
        routes=[r for t,r in self.routing if now-t<=2_000_000_000 and r in ('Fast','Accurate')]
        pf=routes.count('Fast')/len(routes) if routes else None
        available=self.card_availability(now)
        credit={r:sum(c['_service_debt']['balance'][r] for c in rt.cards) for r in ('Fast','Accurate')}
        running=[dict(j,release_ns=available[b['gpu']],revealed_role=self.revealed.get(j['task_id']))
                 for b in rt.batches.values() for j in b['jobs'] if j['task_id'] not in rt.ended]
        self.last_snapshot=dict(now_ns=now,waiting=waiting,inflight=inflight,running=running,
            available=available,prefix_fast_probability=pf,credit=credit,truncated=truncated,
            inflight_basis='final_public_Local_intents_plus_empirical_unresolved_mix' if n else 'unknown_no_inflight_point_estimate',
            observed_terminal_count=n)
        if self.rt.cfg.get('marginal_future_arrivals',False):
            self.last_snapshot.update(self.future_snapshot(now,recent))
        if self.rt.cfg.get('marginal_cycle_capacity',False):
            ratios={};samples={}
            for role in ('Fast','Accurate'):
                rows=[row for row in self.cycle_samples if row[0]==role]
                service=sum(row[1] for row in rows)
                ratios[role]=sum(row[2] for row in rows)/service if service else 0.
                samples[role]=len(rows)
            self.last_snapshot.update(cycle_overhead_ratio=ratios,cycle_sample_counts=samples)
        return self.last_snapshot

    def future_snapshot(self,now,recent):
        """Expected ungenerated cohorts, disjoint from waiting/inflight IDs.

        Generation window is the observed intrinsic deadline. Rate/mix use
        already observed events in the existing two-deadline history. Releases
        include observed upstream delay; these rows never enter the executor.
        """
        window=2*self.rt.cfg['deadline_ns']
        while self.future_arrivals and self.future_arrivals[0][0]<now-window:
            self.future_arrivals.popleft()
        arrivals=[(a,d) for a,d in self.future_arrivals if a<=now and d>0]
        counts=Counter(role for t,role,delay in recent)
        total=sum(counts.values())
        if len(arrivals)<2 or not total:
            return dict(future=[],future_basis='insufficient_observed_history')
        horizon=self.percentile([d for a,d in arrivals],.5)
        span=min(window,max(1,now-arrivals[0][0]))
        rate=len(arrivals)/span
        rows=[];ordinal=0;mix={};delays={}
        for role,kind in (('Fast','FAST_FULL'),('Accurate','ACCURATE'),('Prefix','FAST_PREFIX')):
            probability=counts[role]/total
            if not probability:continue
            delay=self.percentile([d for t,r,d in recent if r==role])
            mix[role]=probability;delays[role]=delay
            expected=rate*horizon*probability
            number=max(1,math.ceil(expected))
            for i in range(number):
                generated=now+(i+.5)*horizon/number
                ordinal+=1
                # Cannot alias real integer roots or negative conditional IDs.
                tid='forecast:'+str(ordinal)
                rows.append(dict(task_id=tid,root_id=tid,kind=kind,
                    role='Accurate' if role=='Accurate' else 'Fast',
                    arrival_ns=generated,release_ns=generated+delay,
                    deadline_ns=generated+horizon,reward=expected/number,
                    work_weight=expected/number,public_future=True))
        return dict(future=rows,future_basis='past_public_generation_rate_and_terminal_mix',
            future_generation_horizon_ns=horizon,future_arrival_rate_per_ns=rate,
            future_history_span_ns=span,future_role_mix=mix,future_upstream_p95_ns=delays)

    def _prepare_intervals(self,snapshot):
        """R004 immutable demand matrices shared by every candidate share."""
        if self.interval_cache and self.interval_cache[0] is snapshot:return self.interval_cache[1]
        np=self.np;rt=self.rt;now=snapshot['now_ns'];pf=snapshot['prefix_fast_probability']
        source=snapshot['waiting']+snapshot['inflight']+snapshot.get('future',[])
        if pf is None and (any(j['kind']=='FAST_PREFIX' for j in source) or any(
                j['kind']=='FAST_PREFIX' and j['revealed_role'] is None for j in snapshot['running'])):
            # Same unknown-prior endpoint rule as C2, prepared separately below.
            priors=(0.,1.)
        else:priors=(1. if pf is None else pf,)
        prices={k:rt.control_unit_ns(k) for k in ('FAST_FULL','FAST_PREFIX','ACCURATE')}
        physical={k:rt.batch_mean([dict(kind=k,task_id=-1)],'p95_ms') for k in prices}
        prepared=[];earliest=min(snapshot['available'])
        for prior in priors:
            roles={'Fast':[],'Accurate':[]};rewards=[];running={'Fast':0.,'Accurate':0.}
            def append(j,kind,release,mass,group,terminal_role,reward,parent=False):
                release=max(now,release);deadline=j['deadline_ns']-rt.cfg['network_return_ns']
                eligible=max(release,earliest)+physical[kind]<=deadline
                if not eligible:return
                role='Accurate' if kind=='ACCURATE' else 'Fast'
                roles[role].append((deadline,release,mass*prices[kind],prices[kind],mass,group+':'+kind))
                rewards.append((j.get('root_id',j['task_id']),terminal_role,reward,parent,j.get('public_future',False)))
            for group in ('waiting','inflight','future'):
                for j in snapshot.get(group,[]):
                    mass=j.get('work_weight',1.);reward=j.get('reward',1.);release=j.get('release_ns',now)
                    if j['kind']=='FAST_PREFIX':
                        append(j,'FAST_PREFIX',release,mass,group,'Fast',reward*prior)
                        if prior<1:
                            append(j,'ACCURATE',release+physical['FAST_PREFIX'],mass*(1-prior),
                                   group+':prefix_child','Accurate',reward*(1-prior),True)
                    else:append(j,j['kind'],release,mass,group,j['role'],reward)
            queued_ids={j['task_id'] for j in source}
            for j in snapshot['running']:
                role='Accurate' if j['kind']=='ACCURATE' else 'Fast';weight=1.
                if j['kind']=='FAST_PREFIX':
                    if j['revealed_role']=='Accurate':continue
                    weight=1. if j['revealed_role']=='Fast' else prior
                    if j['revealed_role'] is None and prior<1 and j['task_id'] not in queued_ids:
                        append(j,'ACCURATE',j['release_ns'],1-prior,'running_prefix_child','Accurate',1-prior)
                if j['release_ns']+rt.cfg['network_return_ns']<=j['deadline_ns']:running[role]+=weight
            matrices={}
            for role,items in roles.items():
                items.sort(key=lambda row:row[0])
                if not items:matrices[role]=None;continue
                deadlines=np.array([x[0] for x in items]);release=np.array([x[1] for x in items])
                work=np.array([x[2] for x in items]);units=np.array([x[3] for x in items])
                origins={float(now)}
                groups={x[5] for x in items}
                for group in groups:
                    times=[x[1] for x in items if x[5]==group];origins.update((min(times),max(times)))
                origins=np.array(sorted(origins))
                selected=release[None,:]>=origins[:,None]
                demand=np.cumsum(np.where(selected,work[None,:],0.),axis=1)
                maximum_unit=np.maximum.accumulate(np.where(selected,units[None,:],0.),axis=1)
                owed=max(0.,-snapshot['credit'][role]);demand[0,:]+=owed
                matrices[role]=dict(deadlines=deadlines,origins=origins,demand=demand,
                    maximum_unit=maximum_unit,mass=sum(x[4] for x in items))
            prepared.append(dict(matrices=matrices,rewards=rewards,running=running,prior=prior))
        self.interval_cache=(snapshot,prepared)
        return prepared

    def _interval_evaluate(self,g,snapshot,include_root_predictions=True):
        np=self.np;shares=card_shares(g);scenarios=[]
        for prepared in self._prepare_intervals(snapshot):
            q={};risk={}
            for role,data in prepared['matrices'].items():
                if data is None:q[role]=1.;risk[role]=dict(mass=0.,unavoidable_loss_proxy=0.);continue
                origins=data['origins'];deadlines=data['deadlines']
                capacity=np.zeros_like(data['demand'])
                for p,start in zip(shares,snapshot['available']):
                    fraction=p if role=='Fast' else 1-p
                    if self.rt.cfg.get('marginal_cycle_capacity',False):
                        chi=snapshot.get('cycle_overhead_ratio',{'Fast':0.,'Accurate':0.})
                        fraction/=1+p*chi['Fast']+(1-p)*chi['Accurate']
                    capacity+=fraction*np.maximum(0.,np.maximum(0.,deadlines[None,:]-start)-np.maximum(0.,origins[:,None]-start))
                deficits=np.maximum(0.,data['demand']-capacity)
                losses=np.divide(deficits,data['maximum_unit'],out=np.zeros_like(deficits),where=data['maximum_unit']>0)
                index=np.unravel_index(int(np.argmax(losses)),losses.shape)
                loss=min(data['mass'],float(losses[index]));q[role]=max(0.,1-loss/data['mass']) if data['mass'] else 1.
                risk[role]=dict(mass=data['mass'],unavoidable_loss_proxy=loss,
                    worst_release_ns=float(origins[index[0]]),worst_deadline_ns=float(deadlines[index[1]]),
                    demand_ns=float(data['demand'][index]),capacity_ns=float(capacity[index]),
                    release_anchors=len(origins),deadline_knots=len(deadlines))
            result=dict(Fast=prepared['running']['Fast'],Accurate=prepared['running']['Accurate'],
                future_on_time=dict(Fast=0.,Accurate=0.),root_prediction={},fast_share=g,
                prefix_uncertainty_endpoints=len(self._prepare_intervals(snapshot)),interval_risk=risk,
                interval_coverage=q,prefix_fast_probability_assumed=prepared['prior'])
            # Only two scalar dot products per role; root tables are deferred
            # until the selected/current candidates need task-level evidence.
            for role in ('Fast','Accurate'):
                independent=sum(reward for tid,r,reward,parent,future in prepared['rewards'] if r==role and not parent)
                dependent=sum(reward for tid,r,reward,parent,future in prepared['rewards'] if r==role and parent)
                result[role]+=q[role]*(independent+q['Fast']*dependent)
                result['future_on_time'][role]=q[role]*sum(reward*(q['Fast'] if parent else 1.)
                    for tid,r,reward,parent,future in prepared['rewards'] if r==role and future)
            result['J']=result['Fast']+result['Accurate'];result['_interval_prepared']=prepared
            scenarios.append(result)
        result=min(scenarios,key=lambda row:row['J'])
        prepared=result.pop('_interval_prepared')
        if include_root_predictions:self._interval_roots(result,prepared)
        return result

    @staticmethod
    def _interval_roots(result,prepared):
        q=result['interval_coverage'];pred=Counter()
        for tid,role,reward,parent,future in prepared['rewards']:
            if not future:pred[tid]+=reward*q[role]*(q['Fast'] if parent else 1.)
        result['root_prediction']=dict(pred)

    def evaluate(self,g,snapshot,include_root_predictions=True):
        if self.rt.cfg.get('marginal_interval_capacity',False):
            return self._interval_evaluate(g,snapshot,include_root_predictions)
        now=snapshot['now_ns'];self.now=now
        ps=card_shares(g);avail=snapshot['available'];rt=self.rt
        resources={r:[((p if r=='Fast' else 1-p),a) for p,a in zip(ps,avail)] for r in ('Fast','Accurate')}
        source=snapshot['waiting']+snapshot['inflight']+snapshot.get('future',[])
        prefix=[j for j in source if j['kind']=='FAST_PREFIX']
        routes=([snapshot['prefix_fast_probability']] if snapshot['prefix_fast_probability'] is not None
                else ([0.,1.] if prefix or any(j['kind']=='FAST_PREFIX' and j['revealed_role'] is None for j in snapshot['running']) else [1.]))
        scenarios=[]
        for pf in routes:
            fast=[dict(j,reward=j.get('reward',1.)*(pf if j['kind']=='FAST_PREFIX' else 1.))
                  for j in source if j['role']=='Fast']
            fp=self._role_curve(fast,'Fast',resources['Fast'],snapshot['credit']['Fast'])
            accurate=[dict(j) for j in source if j['role']=='Accurate']
            for j in prefix:
                if pf<1 and math.isfinite(fp[j['task_id']]):
                    accurate.append(dict(j,kind='ACCURATE',role='Accurate',
                        release_ns=fp[j['task_id']],reward=j.get('reward',1.)*(1-pf),
                        work_weight=j.get('work_weight',1.)*(1-pf)))
            running_constant={'Fast':0.,'Accurate':0.}
            queued_ids={j['task_id'] for j in source}
            for j in snapshot['running']:
                role='Accurate' if j['kind']=='ACCURATE' else 'Fast'
                weight=1.
                if j['kind']=='FAST_PREFIX':
                    if j['revealed_role']=='Accurate':continue
                    weight=1. if j['revealed_role']=='Fast' else pf
                    if j['revealed_role'] is None and pf<1 and j['task_id'] not in queued_ids:
                        accurate.append(dict(j,kind='ACCURATE',role='Accurate',reward=1-pf,work_weight=1-pf))
                if j['release_ns']+rt.cfg['network_return_ns']<=j['deadline_ns']:
                    running_constant[role]+=weight
            ap=self._role_curve(accurate,'Accurate',resources['Accurate'],snapshot['credit']['Accurate'])
            root_prediction=Counter()
            for jobs,predictions in ((fast,fp),(accurate,ap)):
                for j in jobs:
                    if not j.get('public_future',False):
                        root_prediction[j.get('root_id',j['task_id'])]+=j.get('reward',1.)*int(predictions.get(j['task_id'],math.inf)+rt.cfg['network_return_ns']<=j['deadline_ns'])
            result={r:sum(j.get('reward',1.) for j in jobs
                if predictions.get(j['task_id'],math.inf)+rt.cfg['network_return_ns']<=j['deadline_ns'])
                for r,jobs,predictions in [('Fast',fast,fp),('Accurate',accurate,ap)]}
            for r in running_constant:result[r]+=running_constant[r]
            # Running-task rewards cancel between allocations; keep separate
            # from per-root queued/inflight comparison rather than fabricate it.
            result['root_prediction']=dict(root_prediction)
            if snapshot.get('future') is not None:
                result['future_on_time']={r:sum(j.get('reward',1.) for j in jobs
                    if j.get('public_future',False) and predictions.get(j['task_id'],math.inf)+rt.cfg['network_return_ns']<=j['deadline_ns'])
                    for r,jobs,predictions in [('Fast',fast,fp),('Accurate',accurate,ap)]}
            result['J']=result['Fast']+result['Accurate'];scenarios.append(result)
        # No Prefix prior: use the worst on-time envelope, not an oracle route.
        result=min(scenarios,key=lambda v:v['J']);result=dict(result,fast_share=g)
        result['prefix_uncertainty_endpoints']=len(scenarios)
        return result

    def choose(self,now,current,reference=None):
        from resource_axis import bounded_grid,CANDIDATE_GRID
        s=self.snapshot(now);rt=self.rt
        predicted_decision_delay=0.
        if rt.cfg.get('marginal_project_decision_time',False):
            predicted_decision_delay=self.percentile(self.decision_delays)
            activation=now+predicted_decision_delay
            # Project every candidate to the same estimated action timestamp.
            # Keep the raw snapshot for the actual post-search consistency check.
            s=dict(s,now_ns=activation,available=[max(activation,a) for a in s['available']])
        lo=rt.cfg['minimum_fast_share'];hi=2-rt.cfg['minimum_accurate_share']
        legal=[round(i*rt.cfg['allocation_grid'],10) for i in range(round(lo/rt.cfg['allocation_grid']),round(hi/rt.cfg['allocation_grid'])+1)]
        # Reuse the existing small public-demand grid search, not every .01
        # point in a .1-wide neighborhood. Applied targets still have .01
        # precision; current and the quantized public anchor remain candidates.
        coarse=[g for g in CANDIDATE_GRID if lo-1e-9<=g<=hi+1e-9]
        allowed=bounded_grid(current,reference,self.sequence,coarse)
        anchor=min(legal,key=lambda g:abs(g-reference)) if reference is not None else current
        allowed=sorted(set(allowed+[current,anchor,legal[0],legal[-1]]));self.sequence+=1
        bounded_hold=False;upper_bound=None
        if rt.cfg.get('marginal_no_gain_bound',False):
            if not rt.cfg.get('marginal_interval_capacity',False):
                raise ValueError('The no-gain proof requires the interval objective')
            initial=self.evaluate(current,s,include_root_predictions=False)
            upper_bound=min(sum(value[2] for value in p['rewards'])+sum(p['running'].values())
                for p in self._prepare_intervals(s))
            bounded_hold=upper_bound-initial['J']<1-1e-9
            values=([initial] if bounded_hold else [initial if abs(g-current)<1e-9 else
                self.evaluate(g,s,include_root_predictions=False) for g in allowed])
        else:
            values=[self.evaluate(g,s,include_root_predictions=False)
                if rt.cfg.get('marginal_interval_capacity',False) else self.evaluate(g,s) for g in allowed]
        baseline=next(v for v in values if abs(v['fast_share']-current)<1e-9)
        # Goal explicitly requires avoiding Fast sacrifice. This is a prediction
        # safeguard only; actual paired Fast results remain the final check.
        safe=[v for v in values if v['Fast']+1e-9>=baseline['Fast']]
        best=max(safe,key=lambda v:(v['J'],-abs(v['fast_share']-current)))
        if rt.cfg.get('marginal_interval_capacity',False):
            for value in (baseline,best):
                prepared=next(p for p in self._prepare_intervals(s) if p['prior']==value['prefix_fast_probability_assumed'])
                self._interval_roots(value,prepared)
        evidence=dict(now_ns=now,current=baseline,best=best,delta_J=best['J']-baseline['J'],
            candidates=[{k:x for k,x in v.items() if k!='root_prediction'} for v in values],ready_roots=len(s['waiting']),inflight_components=len(s['inflight']),
            available_ns=s['available'],truncated_roots=s['truncated'],
            prefix_fast_probability=s['prefix_fast_probability'],inflight_basis=s['inflight_basis'],
            observed_terminal_count=s['observed_terminal_count'],
            quality_basis='unit_reward_on_time_proxy_no_calibrated_correctness_predictor',
            model=('aggregate_release_deadline_interval_loss_proxy_not_exact_schedule'
                if rt.cfg.get('marginal_interval_capacity',False) else 'observed_batch_unit_deadline_capacity_relaxation_not_exact_schedule'
                if rt.cfg.get('marginal_unit_capacity_curve',False)
                else 'bounded_batch_fluid_capacity_envelope_not_exact_schedule'),
            predicted_task_ids=sorted(set(j.get('root_id',j['task_id']) for j in s['waiting']+s['inflight'])),prediction_horizon_ns=max([now]+[j['deadline_ns'] for j in s['waiting']+s['inflight']]))
        self.last_evidence=evidence
        if rt.cfg.get('marginal_no_gain_bound',False):
            evidence.update(model_reward_upper_bound=upper_bound,model_no_gain_proved=bounded_hold,
                skipped_candidate_evaluations=len(allowed)-1 if bounded_hold else 0,
                upper_bound_scope='Only this causal forecast at its projected action epoch; not an upper bound on actual achievable system TCA')
        if rt.cfg.get('marginal_project_decision_time',False):
            evidence.update(predicted_decision_delay_ns=predicted_decision_delay,
                decision_delay_sample_count=len(self.decision_delays),evaluation_now_ns=s['now_ns'])
        if rt.cfg.get('marginal_cycle_capacity',False):
            evidence.update(cycle_overhead_ratio=s.get('cycle_overhead_ratio'),
                cycle_sample_counts=s.get('cycle_sample_counts'),
                capacity_scope='GPU service opportunity limited by observed dispatch-to-receive actor cycles; real scorecard still charges CUDA only')
        if rt.cfg.get('marginal_unit_capacity_curve',False):
            evidence['deadline_curves']=self.deadline_curve_diagnostic(s,allowed)
        if 'future' in s:
            evidence.update(future_components=len(s['future']),
                future_expected_roots=sum(j['reward'] for j in s['future']),
                future_basis=s['future_basis'],
                future_generation_horizon_ns=s.get('future_generation_horizon_ns'),
                future_arrival_rate_per_ns=s.get('future_arrival_rate_per_ns'),
                future_role_mix=s.get('future_role_mix'),
                future_upstream_p95_ns=s.get('future_upstream_p95_ns'))
        return best['fast_share'],evidence
