"""Causal policy adapters for the common real-GPU batch executor.

Neither policy receives tensors, labels, cached EARLY output, or future frames.
The QF target formula is reused from QueueFeedbackController, with the
selected no-deadline profile. Batch execution/accounting is owned by the runtime.
"""
from collections import Counter
import math
import time
from bisect import bisect_right
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from queue_feedback_controller import QueueFeedbackController
from dual_clock_state import DualClockGovernor
from resource_axis import card_shares
from quota_forecast_v8 import _service_turn, _debt_service
from mixed_fast_batch import batch_kind
from literature_baselines.edge_serving_adapter import EdgeServingAdapter

ROOT = Path(__file__).resolve().parent
KINDS = ('FAST_FULL', 'FAST_PREFIX', 'FAST_SUFFIX', 'ACCURATE')


def _role_share(card, role):
    return card['effective'] if role == 'Fast' else 1 - card['effective']


class ClippedPressure:
    """Exact clipped exponential aggregate, including saturation at C.

    Equal relative deadlines use sorted weights/prefix sums. Heterogeneous
    deadlines fall back to the direct definition; no untruncated shortcut.
    """
    def __init__(self,jobs,now,clip):
        self.jobs=jobs;self.now=now;self.clip=clip
        spans={j['deadline_ns']-j['arrival_ns'] for j in jobs}
        self.tau=next(iter(spans)) if len(spans)==1 and min(spans)>0 else None
        self.weights=sorted(math.exp(min(700.,max(0,now-j['arrival_ns'])/self.tau-1))
                            for j in jobs) if self.tau else []
        self.prefix=[0.]
        for w in self.weights:self.prefix.append(self.prefix[-1]+w)

    def term(self,job,delay=0):
        tau=job['deadline_ns']-job['arrival_ns']
        x=(max(0,self.now-job['arrival_ns'])+delay)/max(1,tau)-1
        return math.exp(min(x,math.log(self.clip)))

    def total(self,delay):
        if not self.tau:return sum(self.term(j,delay) for j in self.jobs)
        z=delay/self.tau
        if z>=700:return len(self.jobs)*self.clip
        factor=math.exp(z);i=bisect_right(self.weights,self.clip/factor)
        return factor*self.prefix[i]+(len(self.weights)-i)*self.clip


def opportunity_score(runtime,g,batch,now,availability,aggregates=None):
    """Score public service-opportunity blocking, respecting parallel cards.

    Independent role queues have zero extra blocking from this action. If an
    alternate legal GPU is busy, only min(L,its remaining busy time) is charged.
    This is a one-action opportunity projection, not an end-to-end simulator.
    Unknown Prefix members remain unfinished in the score.
    """
    duration=runtime.batch_mean(batch,'p95_ms');clip=runtime.cfg['edge_serving_clip_C']
    selected={j['task_id']:j for j in batch};score=0.
    for role in ('Fast','Accurate'):
        agg=(aggregates[role] if aggregates is not None else
             ClippedPressure(runtime.public_jobs(role),now,clip))
        def enabled(card):
            # A busy alternate card adopts its already requested target when
            # it becomes available. Old positive share is not future capacity.
            p=card['target'] if card.get('transition') else card['effective']
            return p>0 if role=='Fast' else p<1
        delay=0.
        if enabled(runtime.cards[g]):
            alternate=[max(0.,a-now) for h,(c,a) in enumerate(zip(runtime.cards,availability))
                       if h!=g and enabled(c)]
            delay=min([duration]+alternate)
        # Subtract the action-independent current pressure. An independent
        # GPU queue contributes exactly zero blocking, not serial waiting.
        score+=agg.total(delay)-agg.total(0.)
        for j in batch:
            if j['role']!=role:continue
            completes=j['kind']!='FAST_PREFIX' and now+duration+runtime.cfg['network_return_ns']<=j['deadline_ns']
            if completes:score-=agg.term(j,delay)
            elif j['kind']=='FAST_PREFIX':
                # It is this batch's unknown work; another GPU cannot complete
                # this Prefix for it while its own non-preemptible stage runs.
                score+=agg.term(j,duration)-agg.term(j,delay)
    return score


class LiveDemand:
    def __init__(self, runtime):
        self.runtime = runtime
        self.buckets = {}
        self.wait_counts = {r:Counter() for r in ('Fast','Accurate')}
        self.resource_loss_counts = {r:Counter() for r in ('Fast','Accurate')}

    def c_ref_ms(self, kind):
        return self.runtime.control_unit_ns(kind)/1e6

    def offered(self, kind, now):
        bucket = int(now)//100000000
        self.buckets.setdefault(bucket, Counter())[kind] += 1
        for k in list(self.buckets):
            if k < bucket-50:del self.buckets[k]

    def read(self, end, span):
        if span <= 0:return dict(valid=False)
        if end%100000000 or span%100000000:
            raise ValueError('Unaligned live work buckets')
        counts = Counter()
        for k in range((end-span)//100000000, end//100000000):
            counts.update(self.buckets.get(k,{}))
        seconds = span/1e9
        return dict(valid=True, D_F_ms_s=sum(counts[k]*self.c_ref_ms(k)
                    for k in ('FAST_FULL','FAST_PREFIX'))/seconds,
                    D_A_ms_s=counts['ACCURATE']*self.c_ref_ms('ACCURATE')/seconds,
                    basis='all_published_requests_times_observed_batch_unit_service')


class OursBatchPolicy:
    def __init__(self, runtime):
        self.rt = runtime
        cfg = runtime.cfg
        prices = {k:max(.000001,runtime.profile.mean(k,1)/1e6) for k in KINDS}
        self.controller = QueueFeedbackController(dict(policy='qf4r_feedback',
            initial_fast=1., axis_min_fast=0., axis_max_fast=2., grid=.01,
            minimum_fast_share=cfg['minimum_fast_share'],
            minimum_accurate_share=cfg['minimum_accurate_share'],
            allocation_hysteresis_floor=cfg['allocation_hysteresis_floor'],
            c_ref_ms=prices, inline_fast_continuation=False))
        dc = json.loads((ROOT/'config/dual_clock_state.json').read_text())
        dc.update(deadline_target_enabled=False, net_gain_enabled=False,
                  strict_first_reverse_enabled=True)
        # Snapshot before enqueue: subsequent ticks mutate nested observations.
        # Serialization and disk I/O belong to the independent logger process.
        def trace_sink(name,row):
            runtime.log('control/'+name,copy.deepcopy(row))
        self.governor = DualClockGovernor('FULL', runtime.output/'control', config=dc,
                                         trace_sink=trace_sink)
        self.controller.dual_clock = self.governor
        self.demand = LiveDemand(runtime)
        self.controller.observer = self.demand
        self.controller._issue = self.issue
        self.pending = None
        self.variant=cfg.get('clra_variant','C0')
        self.controller.capacity_confirmed_window_enabled=bool(cfg.get('capacity_confirmed_window',False))
        if self.controller.capacity_confirmed_window_enabled and self.variant!='C0':
            raise ValueError('R019 evidence-span switch is isolated to the original C0 target path')
        self.controller.overload_unit_cost_allocation_enabled=bool(cfg.get('overload_unit_cost_allocation',False))
        self.urgency_swap_enabled=bool(cfg.get('urgency_swap',False))
        self.sbp_roles=('Accurate',) if cfg.get('density_packing',False) else ()
        self.sbp_frontier=int(cfg.get('density_packing_frontier',64))
        if self.sbp_roles and not (cfg.get('realizable_admission') and cfg.get('overload_unit_cost_allocation')):
            raise ValueError('RA-SBP requires M1 observed need and M3 overload regime')
        self.priority_deferral_enabled=bool(cfg.get('priority_deferral',False))
        if self.priority_deferral_enabled and not self.sbp_roles:
            raise ValueError('Priority deferral requires RA-SBP (M1 need and M3 overload regime)')
        self.maximal_filler_enabled=bool(cfg.get('maximal_filler',False))
        if self.maximal_filler_enabled and not self.priority_deferral_enabled:
            raise ValueError('MFC-SP replaces the PD turn and requires PD (turnaround, priority role)')
        # Safe-optimistic need (13b): overload paths read U for safety and D for choice.
        self.son_enabled=bool(cfg.get('safe_optimistic_need',False))
        if self.son_enabled and not self.sbp_roles:
            raise ValueError('Safe-optimistic need acts on the RA-SBP overload paths')
        if self.urgency_swap_enabled and not cfg.get('realizable_admission',False):
            raise ValueError('PRAA M6 requires the M1 observed need table')
        if self.controller.overload_unit_cost_allocation_enabled and (
                self.variant!='C0' or self.controller.capacity_confirmed_window_enabled):
            raise ValueError('PRAA M3 is isolated to the original C0 target path without R019')
        self.scoring_enabled=self.variant in ('C1','C4')
        self.marginal_enabled=self.variant in ('C2','C3','C4')
        self.emergency_enabled=self.variant in ('C3','C4')
        if self.variant not in ('C0','C1','C2','C3','C4'):raise ValueError('Unknown CLRA variant')
        self.allocator=None
        if self.marginal_enabled or self.scoring_enabled:
            from net_gain_control import LiveMarginalAllocator
            self.allocator=LiveMarginalAllocator(runtime)
        self.emergency_next_ns=0;self.emergency_episode=None;self.emergency_roots=set()

    def observe(self, frame, now):
        before = __import__('time').process_time_ns()
        self.governor.source.observe(frame, now)
        if self.allocator is not None:self.allocator.observe(frame,now)
        self.governor.source_event_cpu_ns += __import__('time').process_time_ns()-before
        self.governor.source_event_calls += 1

    def issue(self, now, view, apply, target, evidence, channel):
        if self.pending is not None:return None
        self.pending = dict(target_fast=target, issued_ns=now, effective_ns=None)
        self.controller.pending = self.pending
        self.controller.target_fast = target
        self.controller.state = 'REQUESTED'
        self.controller.counters['commands_issued'] += 1
        self.controller.command_seq += 1
        self.rt.request_shares(target, now)
        self.rt.log('control', dict(event='CONFIG_REQUEST',now_ns=now,
                    target_fast=target,evidence=copy.deepcopy(evidence)))
        self.effective(now)
        return dict(command_id=self.controller.command_seq)

    def effective(self, now):
        if self.pending is None or self.pending['effective_ns'] is not None:return
        target=self.pending['target_fast']
        targets=self.rt.card_targets(target) if hasattr(self.rt,'card_targets') else card_shares(target)
        if any(abs(c['effective']-p)>1e-9 for c,p in zip(self.rt.cards,targets)):return
        old=self.controller.effective_fast
        self.controller.effective_fast=target
        self.governor.command_applied(old,target)
        if self.allocator is not None:self.allocator.command_delays.append(now-self.pending['issued_ns'])
        self.pending['effective_ns']=now
        self.controller.state='EFFECTIVE'
        self.controller.epoch += 1
        self.rt.log('control',dict(event='TARGET_EFFECTIVE',now_ns=now,
                    from_fast=old,target_fast=target))

    def tick(self, now):
        c=self.controller
        self.effective(now)
        if self.rt.cfg.get('fixed_resource_shares',False):return
        if self.pending and self.pending['effective_ns'] is not None:
            if now>=self.pending['effective_ns']+c.settle_ns:
                c.direct_stable_ns=self.pending['effective_ns']+c.settle_ns
                c.direct_samples.clear()
                c.counters['resource_allocations_applied']+=1
                self.pending=None;c.pending=None;c.state='IDLE'
        view=dict(cards=self.rt.cards, direct_demand=self.demand.read,
                  direct_budget_horizon_ns=self.governor.c['base_ns'],
                  direct_inline_fast_continuation=False,
                  frame_gap_seen=False,log_backpressure=False)
        efficiency=getattr(self.rt,'cuda_efficiency',None)
        if efficiency is not None:view['cuda_efficiency']=efficiency.value
        self.governor.safe_monitor(now,view,self.demand,c.effective_fast,self.pending)
        if self.governor.disabled:
            raise RuntimeError(self.governor.disabled_reason)
        if self.marginal_enabled:self.marginal_tick(now,view)
        else:c._direct_demand_tick(now,view,lambda *a:None)
        self.rt.log('control',dict(event='TICK',now_ns=now,
                    effective_fast=c.effective_fast,counters=dict(c.counters),
                    unit_service_ns={k:self.rt.control_unit_ns(k) for k in KINDS},
                    timing=copy.deepcopy(c.direct_timing),reason=c.last_reason))

    def choose(self, now):
        if getattr(self,'scoring_enabled',False):return self.choose_scored(now)
        candidates=[]
        fast=self.rt.public_jobs('Fast');accurate=self.rt.public_jobs('Accurate')
        for g,card in enumerate(self.rt.cards):
            if card['busy'] or card['transition']:continue
            turn=self.priority_turn(g,card,now) if getattr(self,'priority_deferral_enabled',False) else None
            if turn is not None:
                if not turn:continue
                (role,batch),deduction,timed=turn,None,True
            else:
                timed=False
                cost=self.rt.single_costs()
                role,deduction=_service_turn(card,fast,accurate,cost)
                if role is None:continue
                batch=self.rt.peek_batch(role)
                if not batch:continue
                if deduction is None and getattr(self,'urgency_swap_enabled',False):
                    swapped=self.urgency_swap(g,card,role,batch,now)
                    if swapped is not None:role,batch=swapped
                if role in getattr(self,'sbp_roles',()):
                    packed=self.density_pack(g,card,role,now)
                    if packed is not None:batch=packed
            kind=batch[0]['kind'];p=card['effective']
            duration=self.rt.batch_mean(batch)
            other_p=p if role=='Accurate' else 1-p
            dedicated=int(other_p>0)
            candidates.append(((duration,dedicated,batch[0]['arrival_ns'],g),g,batch,deduction,timed))
        if not candidates:return None
        _,g,batch,deduction,timed=min(candidates,key=lambda c:c[0])
        evidence=dict(empty_deduction=deduction)
        if timed:evidence['priority_turn']=True
        return g,batch,evidence

    def priority_turn(self,g,card,now):
        """PRAA latest-feasible priority deferral, only in the M3 overload regime.

        M3 names the role with the lower unit cost as the priority role p; the
        other role o fills the cards. Serving p in small batches whenever its
        turn comes wastes the fixed per-batch cost (Deb-Serfozo control-limit
        argument). Here p is served only when it is due: its batch is full, or
        its EDF head would no longer be realizable at the next card
        opportunity (this card after o's batch, or the other card's predicted
        finish, plus the observed turnaround). Otherwise o's batch runs and p
        keeps accumulating. All times are M1 observed medians; no new threshold.
        Zero-share roles are never dispatched. Returns None outside overload,
        () when this card has nothing to dispatch, else (role, batch).
        """
        need=getattr(self.rt,'need',None)
        evidence=getattr(self.controller,'direct_demand_evidence',None) or {}
        info=evidence.get('overload_unit_cost_allocation') or {}
        p=info.get('priority_role')
        if need is None or not info.get('active') or p not in ('Fast','Accurate'):return None
        o='Accurate' if p=='Fast' else 'Fast'
        pb=self.rt.peek_batch(p) if _role_share(card,p)>0 else []
        ob=self.rt.peek_batch(o) if _role_share(card,o)>0 else []
        if getattr(self,'maximal_filler_enabled',False):
            return self.maximal_filler_turn(g,card,now,p,o,pb,ob)
        if ob and o in self.sbp_roles:ob=self.density_pack(g,card,o,now) or ob
        if pb and (not ob or self.priority_due(g,now,pb,ob,p)):
            if p in self.sbp_roles:pb=self.density_pack(g,card,p,now) or pb
            return p,pb
        if not ob:return ()
        if pb:self.rt.counts['praa_priority_deferrals']+=1
        return o,ob

    def safe_need(self,kind,size,key):
        """Need for a realizability test: U (13b, Lemma 6) with SON, else the M1 value."""
        need=self.rt.need
        if not getattr(self,'son_enabled',False):return need.median(kind,size,key)
        value=need.upper(kind,size,key)
        return value if value is not None else self.startup_need(kind,size,key)

    def choice_need(self,kind,size,key):
        """Need for ranking or pricing a choice: D (13b, Proposition 7) with SON, else the M1 value."""
        need=self.rt.need
        if not getattr(self,'son_enabled',False):return need.median(kind,size,key)
        value=need.lower(kind,size,key)
        return value if value is not None else self.startup_need(kind,size,key)

    def startup_need(self,kind,size,key):
        """SON startup (13b/13c): while no size of this kind is warm on this card, the frozen
        calibration plus the return, the rule the executor itself admits cold batches with.
        Without it a cold filler has no estimate, is never chosen and never warms."""
        if self.rt.need._warm(kind,key):return None
        anchor=getattr(self.rt,'need_anchor',None)
        value=anchor(kind,size) if anchor is not None else None
        return None if value is None else value+self.rt.cfg['network_return_ns']

    def priority_batch(self,g,p,now,pb):
        """The p batch MFC serves: packed like PD packs p when p is an RA-SBP role (Accurate
        priority), with MFC's uncapped filler packing; else the EDF batch."""
        if p in self.sbp_roles:
            packed=self.filler_batch(g,p,now,None)
            if packed:return packed
        return pb

    def priority_due(self,g,now,pb,ob,p):
        need=self.rt.need;key=self.rt.need_card(g)
        if len(pb)>=self.rt.cfg['batch_max']:return True  # waiting adds no batch density
        p_need=self.safe_need(batch_kind(pb),len(pb),key)
        o_need=self.safe_need(batch_kind(ob),len(ob),key)
        turn=self.rt.turnaround_estimate()
        if p_need is None or o_need is None or turn is None:return True  # cold: serve now
        # need ends at the return; the configured return is <= the measured one,
        # so these finish predictions are late, i.e. the deferral is conservative.
        return min(j['deadline_ns'] for j in pb)-self.next_opportunity(g,now,o_need,turn,p)<p_need

    def next_opportunity(self,g,now,o_need,turn,p):
        """Earliest predicted dispatch time for role p: this card after a batch of
        need o_need, or another card with a p share (now if idle, else its flight)."""
        need=self.rt.need
        ret=self.rt.cfg['network_return_ns']
        nxt=now+o_need-ret+turn
        for h,other in enumerate(self.rt.cards):
            if h==g or other['transition'] or _role_share(other,p)<=0:continue
            if not other['busy']:nxt=min(nxt,now);continue
            flight=next((b for b in self.rt.batches.values() if b['gpu']==h),None)
            if flight is None:continue
            n=self.safe_need(batch_kind(flight['jobs']),len(flight['jobs']),self.rt.need_card(h))
            if n is not None:nxt=min(nxt,max(now,flight['dispatch_ns']+n-ret)+turn)
        return nxt

    def maximal_filler_turn(self,g,card,now,p,o,pb,ob):
        """MFC-SP: maximal filler with a shadow-price exchange, M3 overload only.

        1. A full p batch is served (waiting adds no batch density).
        2. o* is the density-optimal realizable o batch with no single-card cap.
           It runs if p is empty or p's EDF head stays realizable at its next
           opportunity (two cards, need of a full p batch: Lemma 3).
        3. Otherwise o' is the best o batch the guard admits. It runs only if its
           density b'/(tau+L(b')) is at least the cycle price b*/(2 tau+L(b*)+a_p),
           since an inserted batch absorbs p arrivals without a p batch overhead;
           else p is served now.
        All times are M1 needs (anchored when enabled) and the observed turnaround;
        cold estimates serve p now, as PD does. With SON the guard and cap read U and
        the densities read D (13b). No new parameter.
        """
        B=self.rt.cfg['batch_max'];counts=self.rt.counts
        if pb and len(pb)>=B:
            counts['praa_mfc_full_priority']+=1
            return p,self.priority_batch(g,p,now,pb)
        best=self.filler_batch(g,o,now,None) if ob else []
        if not best:
            if ob:counts['praa_mfc_no_filler']+=1  # o queued, none realizable (or no estimate)
            return (p,self.priority_batch(g,p,now,pb)) if pb else ()
        if not pb:
            counts['praa_mfc_filler']+=1
            return o,best
        need=self.rt.need;key=self.rt.need_card(g);ret=self.rt.cfg['network_return_ns']
        turn=self.rt.turnaround_estimate()
        p_need=self.full_need(p,key)
        o_need=self.safe_need(batch_kind(best),len(best),key)
        if p_need is None or o_need is None or turn is None:
            counts['praa_mfc_cold_priority']+=1
            return p,self.priority_batch(g,p,now,pb)
        head=min(j['deadline_ns'] for j in pb)
        if head-self.next_opportunity(g,now,o_need,turn,p)>=p_need:
            counts['praa_mfc_filler']+=1
            return o,best
        # The guard failed for both opportunities, so a smaller o batch can pass only
        # through this card's term: it caps o's need on this card.
        cap=head-p_need-turn+ret-now
        cut=self.filler_batch(g,o,now,cap) if cap>0 else []
        if cut:
            a_p=self.batch_intercept(p,key)
            cut_need=self.choice_need(batch_kind(cut),len(cut),key)
            o_value=self.choice_need(batch_kind(best),len(best),key)
            if cut_need is not None and a_p is not None:
                inserted=len(cut)/(turn+cut_need-ret)
                cycle=len(best)/(2*turn+o_value-ret+a_p)
                if inserted>=cycle:
                    counts['praa_mfc_inserted']+=1
                    return o,cut
        counts['praa_mfc_priority']+=1
        return p,self.priority_batch(g,p,now,pb)

    def filler_batch(self,g,role,now,cap):
        """Density-optimal realizable batch of role with need <= cap if given, or its
        longest realizable EDF prefix under the same cap when that is larger. [] if none.
        With SON, feasibility and the cap read U and the density reads D."""
        key=self.rt.need_card(g);B=self.rt.cfg['batch_max']
        son=getattr(self,'son_enabled',False)
        front=(self.rt._kind_jobs('ACCURATE',self.sbp_frontier) if role=='Accurate'
               else self.rt.peek_batch(role))
        best=None
        if role=='Accurate':
            for size in range(1,B+1):
                n=self.safe_need('ACCURATE',size,key)
                if n is None:continue
                if cap is not None and n>cap:
                    if son:continue  # U need not be monotone in b
                    break
                feasible=[j for j in front if j['deadline_ns']-now>=n]
                if len(feasible)<size:continue
                d=self.choice_need('ACCURATE',size,key) if son else n
                if best is None or size/d>=best[0]:best=(size/d,feasible[:size])
        edf=0
        for size in range(1,min(len(front),B)+1):
            batch=front[:size];n=self.safe_need(batch_kind(batch),size,key)
            if n is None or (cap is not None and n>cap):continue
            if all(j['deadline_ns']-now>=n for j in batch):edf=size
        if best is not None and len(best[1])>edf:return best[1]
        return front[:edf]

    def full_need(self,role,key):
        """Need of a full batch of role (largest over its batch kinds); None if cold."""
        B=self.rt.cfg['batch_max']
        kinds=('ACCURATE',) if role=='Accurate' else ('FAST_FULL','FAST_PREFIX','FAST_MIXED')
        values=[v for v in (self.safe_need(k,B,key) for k in kinds) if v is not None]
        return max(values) if values else None

    def batch_intercept(self,role,key):
        """Fixed card time a_p of one batch of role: the affine intercept of its needs
        between the smallest available size and a full batch, minus the return."""
        B=self.rt.cfg['batch_max'];ret=self.rt.cfg['network_return_ns']
        kinds=('ACCURATE',) if role=='Accurate' else ('FAST_MIXED','FAST_FULL')
        if getattr(self,'son_enabled',False):
            # An estimate for pricing, not a bound: observed values only, the line through
            # the smallest and largest warm sizes of the first kind with two warm sizes.
            for k in kinds:
                warm=self.rt.need._warm(k,key)
                if len(warm)>=2:
                    lo,hi=min(warm),max(warm)
                    return max(0.,warm[lo]-lo*(warm[hi]-warm[lo])/(hi-lo)-ret)
            return None
        for k in kinds:
            top=self.rt.need.median(k,B,key)
            for lo in (1,2):
                low=self.rt.need.median(k,lo,key)
                if top is not None and low is not None and B>lo:
                    return max(0.,low-lo*(top-low)/(B-lo)-ret)
        return None

    def density_pack(self,g,card,role,now):
        """RA-SBP: throughput-density batch packing, only in the overload regime.

        Under persistent overload EDF lets the most urgent head fix a small
        feasible batch. Among the queued jobs whose remaining budget covers the
        observed median need L(b) of a b-batch on this card, take the b
        smallest-slack ones maximizing b/L(b). A non-blocking constraint keeps
        the other role's currently realizable head realizable after this batch.
        Underload (no M3 overload) keeps the EDF batch unchanged.
        """
        need=getattr(self.rt,'need',None)
        if need is None or role!='Accurate':return None
        evidence=getattr(self.controller,'direct_demand_evidence',None) or {}
        if not (evidence.get('overload_unit_cost_allocation') or {}).get('active'):return None
        key=self.rt.need_card(g)
        front=self.rt._kind_jobs('ACCURATE',self.sbp_frontier)
        if len(front)<2:return None
        other_cap=None
        if card['effective']>0:
            other=self.rt.peek_batch('Fast')
            if other:
                head_need=self.safe_need(batch_kind(other[:1]),1,key)
                head_slack=other[0]['deadline_ns']-now
                if head_need is not None and head_slack>=head_need:other_cap=head_slack-head_need
        son=getattr(self,'son_enabled',False)
        best=None
        for size in range(1,self.rt.cfg['batch_max']+1):
            n=self.safe_need('ACCURATE',size,key)
            if n is None:continue
            if other_cap is not None and n>other_cap:
                if son:continue  # U need not be monotone in b
                break
            feasible=[j for j in front if j['deadline_ns']-now>=n]
            if len(feasible)<size:continue
            density=size/(self.choice_need('ACCURATE',size,key) if son else n)
            if best is None or density>=best[0]:best=(density,feasible[:size])
        # The EDF batch the executor would actually start: its longest realizable prefix.
        edf=0
        for size in range(1,min(len(front),self.rt.cfg['batch_max'])+1):
            n=self.safe_need('ACCURATE',size,key)
            if n is not None and all(j['deadline_ns']-now>=n for j in front[:size]):edf=size
        if best is None or len(best[1])<=edf:return None
        self.rt.counts['praa_density_packed_batches']+=1
        self.rt.counts['praa_density_packed_members']+=len(best[1])
        return best[1]

    def urgency_swap(self,g,card,role,batch,now):
        """PRAA M6: feasibility-preserving order swap between roles on one card.

        The scorecard winner keeps its turn unless (1) the other role's head
        would become unrealizable if it waited for the winner's batch, and (2)
        the winner's batch still finishes on time after the other role's batch.
        All times are M1 observed medians for this card; no new threshold.
        Both roles need a positive share here (no zero-share execution); the
        actual service is charged by the unchanged scorecard. Disabled while M3
        reports an overload, where the unit-cost priority governs.
        """
        need=getattr(self.rt,'need',None)
        if need is None:return None
        other='Accurate' if role=='Fast' else 'Fast'
        share=card['effective'] if other=='Fast' else 1-card['effective']
        if share<=0:return None
        evidence=getattr(self.controller,'direct_demand_evidence',None) or {}
        if (evidence.get('overload_unit_cost_allocation') or {}).get('active'):return None
        alt=self.rt.peek_batch(other)
        if not alt:return None
        key=self.rt.need_card(g)
        winner_need=need.median(batch_kind(batch),len(batch),key)
        head_need=need.median(batch_kind(alt[:1]),1,key)
        if winner_need is None or head_need is None:return None
        head_slack=alt[0]['deadline_ns']-now
        winner_slack=min(j['deadline_ns'] for j in batch)-now
        if not head_need<=head_slack<winner_need+head_need:return None
        # Largest prefix of the other role's batch that keeps the winner feasible.
        chosen=None
        for size in range(1,len(alt)+1):
            alt_need=need.median(batch_kind(alt[:size]),size,key)
            if alt_need is None or winner_slack-alt_need<winner_need:break
            if all(j['deadline_ns']-now>=alt_need for j in alt[:size]):chosen=alt[:size]
        if chosen is None:return None
        self.rt.counts['praa_urgency_swaps']+=1
        return other,chosen

    def marginal_tick(self,now,view,emergency=False):
        c=self.controller;dc=self.governor
        if self.pending is not None or c.state!='IDLE':return
        if not emergency and not dc.context.get('joint'):return
        started=time.process_time_ns()
        # The old workload estimate is only a search anchor, never the reward.
        d=dc.context.get('D',{});df=d.get('Fast') or 0.;da=d.get('Accurate') or 0.
        reference=2*df/(df+da) if df+da else c.effective_fast
        decision_started_ns=now
        target,e=self.allocator.choose(now,c.effective_fast,reference)
        # Charge actual search time against the SAME public snapshot before
        # issuing. This is a two-envelope consistency check, not task replay.
        ready=self.rt.now();snapshot=dict(self.allocator.last_snapshot,now_ns=ready,
                                        available=self.allocator.card_availability(ready))
        e['search_predicted_delta_J']=e['delta_J']
        if not e.get('model_no_gain_proved',False):
            e['current']=self.allocator.evaluate(c.effective_fast,snapshot)
            e['best']=self.allocator.evaluate(target,snapshot)
            e['delta_J']=e['best']['J']-e['current']['J']
        else:
            e['postcheck_skipped_for_no_action']=True
        now=self.rt.now()
        e.update(now_ns=now,decision_started_ns=decision_started_ns,decision_ready_ns=now,
                 target_fast=target,from_fast=c.effective_fast,emergency=emergency,
                 decision_cpu_ns=time.process_time_ns()-started)
        if self.rt.cfg.get('marginal_project_decision_time',False):
            self.allocator.decision_delays.append(now-decision_started_ns)
        self.rt.cpu['marginal_allocation_ns']+=e['decision_cpu_ns']
        if not emergency:dc.joint_evaluated(now)
        reason=None
        if e['best']['Fast']+1e-9<e['current']['Fast']:reason='predicted_F_loss_after_search'
        elif e.get('model_no_gain_proved',False):reason='model_upper_bound_less_than_one_gain'
        elif e['delta_J']<1-1e-9:reason='less_than_one_predicted_saved_task'
        elif not emergency:
            tolerance,_=dc.allocation_tolerance(target-c.effective_fast,self.rt.cfg['allocation_hysteresis_floor'],0.)
            if abs(target-c.effective_fast)<tolerance-1e-9:reason='ordinary_amplitude_hysteresis'
            elif dc.filter_target(now,e,c.effective_fast):reason='ordinary_reverse_confirmation'
        if reason is None:
            issued=self.issue(now,view,None,target,e,'clra_emergency' if emergency else 'clra_marginal')
            e['command_issued']=issued is not None
            if emergency and issued is not None:
                # Consume the triggering evidence; regular clocks restart from
                # this joint decision, with independent per-role intervals.
                dc.joint_evaluated(now)
        else:e.update(command_issued=False,hold_reason=reason)
        c.last_reason=reason or ('clra_emergency_issued' if emergency else 'clra_marginal_issued')
        self.rt.log('marginal',e)
        return e

    def maybe_emergency(self,now):
        if not self.emergency_enabled or now<self.emergency_next_ns or self.pending is not None:return
        # Cheap frontiers only. Evaluate before the slow state window closes
        # when already observable work cannot fit its current service envelope.
        heads={r:[j for j in self.rt.peek_batch(r) if j['deadline_ns']>now] for r in ('Fast','Accurate')}
        for role in ('Fast','Accurate'):
            announced=[dict(r,kind='ACCURATE' if role=='Accurate' else 'FAST_FULL',role=role)
                       for r in self.allocator.inflight.values()
                       if r.get('announced_role')==role and r['deadline_ns']>now]
            heads[role]=sorted(heads[role]+announced,key=lambda j:(j['deadline_ns'],j['task_id']))[:self.rt.cfg['batch_max']]
        availability=self.allocator.card_availability(now);deficits=[]
        for role,batch in heads.items():
            if not batch:continue
            deadline=min(j['deadline_ns'] for j in batch)-self.rt.cfg['network_return_ns']
            duration=self.rt.batch_mean(batch,'p95_ms')
            resources=[((c['effective'] if role=='Fast' else 1-c['effective']),a)
                       for c,a in zip(self.rt.cards,availability)]
            capacity=self.allocator.capacity(deadline,resources)
            if capacity+1e-9<duration or min(availability)+duration>deadline:
                deficits.append((role,tuple(j['task_id'] for j in batch),deadline))
        if not deficits:
            self.emergency_episode=None;self.emergency_roots.clear();return
        episode=tuple((r,ids) for r,ids,d in deficits)
        roots={tid for role,ids,deadline in deficits for tid in ids}
        if roots & self.emergency_roots:return
        self.emergency_episode=episode;self.emergency_roots=roots
        # At most once per observed batch service quantum; this is not a new
        # fixed load/ratio threshold. Same urgent roots cannot retrigger forever.
        quantum=min(self.rt.batch_mean(b,'p95_ms') for b in heads.values() if b)
        # Reuse the existing minimum detector interval as the reevaluation
        # budget. The first deficit is still evaluated immediately between
        # normal ticks; do not run the allocation search at batch frequency.
        self.emergency_next_ns=now+max(1,int(quantum),self.governor.c['min_check_ns'])
        started=time.process_time_ns();e=self.marginal_tick(now,{},emergency=True)
        self.rt.cpu['emergency_check_ns']+=time.process_time_ns()-started
        if e is not None:self.rt.log('emergency',dict(now_ns=now,deficits=deficits,
            target_fast=e['target_fast'],delta_J=e['delta_J'],command_issued=e.get('command_issued',False)))

    def choose_scored(self,now):
        candidates=[];available=self.allocator.card_availability(now)
        fast=self.rt.public_jobs('Fast');accurate=self.rt.public_jobs('Accurate')
        aggregates={r:ClippedPressure(jobs,now,self.rt.cfg['edge_serving_clip_C'])
                    for r,jobs in [('Fast',fast),('Accurate',accurate)]}
        seen=set()
        for g,card in enumerate(self.rt.cards):
            if card['busy'] or card['transition']:continue
            role,deduction=_service_turn(card,fast,accurate,self.rt.single_costs())
            if role is None:continue
            batches=[self.rt.peek_batch(role)]
            if role=='Fast':batches.extend(self.rt.peek_kind_batch(k) for k in ('FAST_FULL','FAST_PREFIX'))
            for batch in batches:
                if not batch:continue
                key=(g,tuple(j['task_id'] for j in batch))
                if key in seen:continue
                seen.add(key);duration=self.rt.batch_mean(batch,'p95_ms')
                feasible=all(now+duration+self.rt.cfg['network_return_ns']<=j['deadline_ns'] for j in batch)
                score=opportunity_score(self.rt,g,batch,now,available,aggregates)
                other=card['effective'] if role=='Accurate' else 1-card['effective']
                candidates.append(((not feasible,score,duration,int(other>0),batch[0]['arrival_ns'],g),g,batch,deduction))
        if not candidates:return None
        rank,g,batch,deduction=min(candidates,key=lambda x:x[0])
        return g,batch,dict(empty_deduction=deduction,opportunity_score=rank[1],
            p95_feasible=not rank[0],parallel_aware=True,unknown_prefix_not_removed=True)

    def book(self, g, batch, evidence):
        card=self.rt.cards[g];p=card['effective'];role=batch[0]['role']
        deduction=evidence.get('empty_deduction')
        if deduction:
            _debt_service(card['_service_debt'],deduction['role'],deduction['charge_ns'],p,False)
        booking=dict(share=p,role=role,score_before=dict(card['_service_debt']['balance']))
        # Overload priority turns bypass the scorecard; it stays frozen until underload.
        if evidence.get('priority_turn'):booking['scored']=False
        return booking

    def complete(self,g,booking,actual_ns):
        if booking.get('scored',True) is False:return
        # One atomic batch gets exactly one real charge; no N*batch overcharge.
        _debt_service(self.rt.cards[g]['_service_debt'],booking['role'],actual_ns,
                      booking['share'],True)

    def close(self):
        self.governor.close()


class EdgeServingBatchPolicy:
    """Paper stability rule with B*=min(queue,10), fixed existing paths.

    Two-GPU placement, fixed quality routing, and lack of selectable trained
    classifier exits are explicit adaptations. Every batch member is removed
    from the candidate's projected queue state, not just the queue head.
    """
    def __init__(self,runtime):
        self.rt=runtime
        self.reference=EdgeServingAdapter(dict(batch_max=10,
            clip_C=runtime.cfg['edge_serving_clip_C']))
        self.calls=0

    def tick(self,now):pass
    def observe(self,frame,now):pass
    def effective(self,now):pass

    def choose(self,now):
        free=[g for g,c in enumerate(self.rt.cards) if not c['busy']]
        if not free:return None
        pending=self.rt.public_jobs('Fast')+self.rt.public_jobs('Accurate')
        if not pending:return None
        self.calls+=1
        candidates=[]
        for kind in ('FAST_FULL','FAST_PREFIX','ACCURATE'):
            batch=self.rt.peek_kind_batch(kind)
            if not batch:continue
            duration=self.rt.profile.budget(kind,len(batch))
            ids={j['task_id'] for j in batch}
            score=sum(self.reference._f(max(0,now-j['arrival_ns'])+duration,
                        j['deadline_ns']-j['arrival_ns']) for j in pending if j['task_id'] not in ids)
            feasible=all(now+duration+self.rt.cfg['network_return_ns']<=j['deadline_ns'] for j in batch)
            for g in free:
                candidates.append((not feasible,score,duration,batch[0]['enqueue_ns'],g,batch))
        if not candidates:return None
        feasible=[c for c in candidates if not c[0]]
        pool=feasible or candidates
        best=min(pool,key=lambda c:c[1:5])
        return best[4],best[5],dict(score=best[1],eq6_feasible=not best[0],
                    fallback=not bool(feasible),batch_rule='min(current_kind_queue,10)')

    def book(self,g,batch,evidence):return None
    def complete(self,g,booking,actual_ns):pass
    def close(self):
        self.rt.log('control',dict(event='EDGE_SERVING_FINAL',choose_action_calls=self.calls,
                    max_batch=10,original_paper_reproduction=False,
                    stability_helper_counters=dict(self.reference._counters)))
