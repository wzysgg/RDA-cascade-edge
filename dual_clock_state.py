"""Causal public source state and independent role clocks. No model/label access.

OFF never instantiates these classes. All work units used here are ms and ms/s.
"""
import json
import math
from collections import deque
from pathlib import Path

NS = 1000000000
ROLES = ('Fast', 'Accurate')
MODES = ('OFF', 'SHADOW', 'DUAL_CLOCK', 'FULL', 'UNIFORM_100MS', 'HYSTERESIS')


def clip(x):
    return max(0., min(1., x))


class SourceStateEstimator:
    def __init__(self, config):
        self.c = config
        self.blocks = {}
        self.block_counts = {}
        self.last_source_update_ns = None
        self.last_source_evidence_id = -1
        self.load_strength = None
        self.degradation_strength = None
        self.degradation_lower = 0.
        self.degradation_upper = 1.
        self.state_valid = False
        self.load_valid = False
        self.composition_valid = False
        self.smoothed_degradation_lower = None
        self.smoothed_degradation_upper = None
        self.role_state_references = {'load': None, 'composition': None}
        self.role_state_candidates = {'load': None, 'composition': None}
        self.role_state_confirmations = {'load': 0, 'composition': 0}
        self.confirmed_state_reference = None
        self.candidate = None
        self.confirm_count = 0
        self.confirmed_changes = 0
        self.duplicate_arrivals = 0
        self.public_cutoff_ns = 0
        self.ended = False
        self.latest = {}
        self.total_original_arrivals = 0
        self.retained_roots = 0
        self.peak_retained_roots = 0
        self.prefix_reveals = 0
        self.late_prefix_reveals = 0

    def _confirm_role_state(self, name, value, valid):
        if not valid:
            self.role_state_candidates[name] = None
            self.role_state_confirmations[name] = 0
            return
        reference = self.role_state_references[name]
        threshold = self.c['state_change_threshold']
        if reference is not None and abs(value-reference) < threshold:
            self.role_state_candidates[name] = None
            self.role_state_confirmations[name] = 0
            return
        previous = self.role_state_candidates[name]
        self.role_state_confirmations[name] = (self.role_state_confirmations[name]+1
            if previous is not None and abs(previous-value)<threshold else 1)
        self.role_state_candidates[name] = value
        if self.role_state_confirmations[name] >= self.c['confirmation_blocks']:
            self.role_state_references[name] = value
            self.role_state_confirmations[name] = 0

    def observe(self, frame, now):
        published = int(frame.get('source_publish_ns', now))
        if published > now:
            raise ValueError('future public source event')
        self.public_cutoff_ns = max(self.public_cutoff_ns, published)
        outcome = frame.get('outcome')
        if outcome == 'SOURCE_END':
            self.ended = True
            return
        if 'arrival_ns' not in frame or 'task_id' not in frame:
            return
        block = int(frame['arrival_ns']) // self.c['state_window_ns']
        if block <= self.last_source_evidence_id:
            # Causal maturation has already consumed this source block.
            if outcome == 'PREFIX_QUALITY_REVEALED':
                self.late_prefix_reveals += 1
            return
        roots = self.blocks.setdefault(block, {})
        counts = self.block_counts.setdefault(block,
            {r:0 for r in ('Local','Fast','Accurate','Abstain','Unknown')})
        tid = frame['task_id']
        if outcome == 'SHARED_ARRIVAL':
            if tid in roots:
                self.duplicate_arrivals += 1
            else:
                roots[tid] = {'role': 'Unknown', 'prefix': False}
                counts['Unknown'] += 1
                self.total_original_arrivals += 1
                self.retained_roots += 1
                self.peak_retained_roots = max(self.peak_retained_roots, self.retained_roots)
            return
        root = roots.get(tid)
        if root is None:
            return
        old_role = root['role']
        if outcome == 'PREFIX_READY':
            root.update(role='Unknown', prefix=True)
        elif not root['prefix'] and outcome == 'COMMON_LOCAL_TERMINAL':
            root['role'] = 'Local'
        elif not root['prefix'] and outcome == 'GPU_READY':
            roles = frame.get('gpu_candidates') or []
            if roles and roles[0] in ROLES:
                root['role'] = roles[0]
        elif outcome == 'PREFIX_QUALITY_REVEALED' and root['prefix']:
            role = frame.get('quality_role')
            if role in ('Fast', 'Accurate', 'Abstain'):
                if old_role != 'Unknown' and old_role != role:
                    raise ValueError('conflicting public Prefix quality evidence')
                root['role'] = role
                self.prefix_reveals += int(old_role == 'Unknown')
        if old_role != root['role']:
            counts[old_role] -= 1
            counts[root['role']] += 1
        # MID intentions never prove a latent role. Prefix classification is
        # resolved only by an actually published EARLY quality decision.

    def advance(self, now):
        if now < self.public_cutoff_ns:
            raise ValueError('state clock precedes handled public source events')
        changed = False
        w = self.c['state_window_ns']
        for bid in sorted(list(self.blocks)):
            if (bid + 1)*w + self.c['maturation_ns'] > now:
                break
            roots = self.blocks.pop(bid)
            counts = self.block_counts.pop(bid)
            self.retained_roots -= len(roots)
            if bid <= self.last_source_evidence_id:
                continue
            n = sum(counts.values())
            elapsed = (w if self.last_source_update_ns is None else
                       (bid-self.last_source_evidence_id)*w) / NS
            rate = n/(w/NS)
            raw_l = clip((rate-self.c['lambda_normal']) /
                         (self.c['lambda_stress']-self.c['lambda_normal']))
            beta = 1-math.exp(-elapsed/self.c['tau_load_s'])
            self.load_strength = (raw_l if self.load_strength is None else
                                  (1-beta)*self.load_strength+beta*raw_l)
            lower = counts['Accurate']/n if n else 0.
            upper = (counts['Accurate']+counts['Unknown'])/n if n else 1.
            scale = self.c['pA_stress']-self.c['pA_normal']
            self.degradation_lower = clip((lower-self.c['pA_normal'])/scale)
            self.degradation_upper = clip((upper-self.c['pA_normal'])/scale)
            valid = (n >= self.c['minimum_roots'] and
                counts['Unknown']/max(1,n) <= self.c['max_unknown_fraction'] and
                (not self.c.get('independent_checks_enabled') or
                 self.degradation_upper-self.degradation_lower<=self.c['state_change_threshold']))
            self.state_valid = valid
            self.load_valid = n >= self.c['minimum_roots']
            self.composition_valid = valid
            bq = 1-math.exp(-elapsed/self.c['tau_composition_s'])
            self.smoothed_degradation_lower = (self.degradation_lower
                if self.smoothed_degradation_lower is None else
                (1-bq)*self.smoothed_degradation_lower+bq*self.degradation_lower)
            self.smoothed_degradation_upper = (self.degradation_upper
                if self.smoothed_degradation_upper is None else
                (1-bq)*self.smoothed_degradation_upper+bq*self.degradation_upper)
            if valid:
                # Midpoint is only a point estimate within explicitly logged bounds.
                qraw = (self.degradation_lower+self.degradation_upper)/2
                self.degradation_strength = (qraw if self.degradation_strength is None else
                    (1-bq)*self.degradation_strength+bq*qraw)
            self._confirm_role_state('load', self.load_strength, self.load_valid)
            self._confirm_role_state('composition', self.degradation_strength, valid)
            evidence = (self.load_strength, self.degradation_strength) if valid else None
            if evidence is not None:
                reference = self.confirmed_state_reference
                is_change = reference is None or max(abs(evidence[i]-reference[i]) for i in (0,1)) >= self.c['state_change_threshold']
                if is_change:
                    consistent = self.candidate is not None and max(abs(evidence[i]-self.candidate[i]) for i in (0,1)) < self.c['state_change_threshold']
                    self.confirm_count = self.confirm_count+1 if consistent else 1
                    self.candidate = evidence
                    if self.confirm_count >= self.c['confirmation_blocks']:
                        self.confirmed_state_reference = evidence
                        self.confirmed_changes += int(reference is not None)
                        self.confirm_count = 0
                else:
                    self.candidate, self.confirm_count = None, 0
            else:
                self.candidate, self.confirm_count = None, 0
            self.last_source_update_ns = int(now)
            self.last_source_evidence_id = bid
            self.latest = dict(counts=counts, roots=n, input_rate=rate,
                early_offload=sum(r['prefix'] for r in roots.values()),
                unknown_fraction=counts['Unknown']/max(1,n), pA_lower=lower, pA_upper=upper,
                block_start_ns=bid*w, block_end_ns=(bid+1)*w,
                public_cutoff_ns=self.public_cutoff_ns)
            changed = True
        return changed

    def snapshot(self):
        return dict(self.latest, load_strength=self.load_strength,
            degradation_strength=self.degradation_strength,
            degradation_lower=self.degradation_lower, degradation_upper=self.degradation_upper,
            smoothed_degradation_lower=self.smoothed_degradation_lower,
            smoothed_degradation_upper=self.smoothed_degradation_upper,
            load_valid=self.load_valid, composition_valid=self.composition_valid,
            confirmed_role_states=dict(self.role_state_references),
            prefix_reveals=self.prefix_reveals, late_prefix_reveals=self.late_prefix_reveals,
            state_valid=self.state_valid, last_source_update_ns=self.last_source_update_ns,
            last_source_evidence_id=self.last_source_evidence_id,
            confirmed_state_reference=self.confirmed_state_reference,
            confirmed_changes=self.confirmed_changes,
            duplicate_arrivals=self.duplicate_arrivals, source_end=self.ended,
            retained_roots=self.retained_roots,peak_retained_roots=self.peak_retained_roots,
            total_original_arrivals=self.total_original_arrivals)


class DualClockGovernor:
    def __init__(self, mode, folder=None, config=None, trace_sink=None):
        if mode not in MODES or mode == 'OFF':
            raise ValueError('explicit enabled mode required')
        self.mode = mode
        self.c = config or json.loads((Path(__file__).parent/'config/dual_clock_state.json').read_text())
        if not (0 < self.c['base_ns'] <= self.c['min_check_ns'] <= self.c['max_check_ns']
                and 0 < self.c['state_window_ns'] <= self.c['state_window_max_ns']
                and self.c['max_joint_ns'] >= self.c['base_ns']
                and self.c['maturation_ns'] >= 0
                and self.c['state_window_ns'] % self.c['base_ns'] == 0
                and 0 <= self.c['max_unknown_fraction'] <= 1
                and self.c['confirmation_blocks'] >= 2
                and 0 < self.c['lambda_normal'] < self.c['lambda_stress']
                and 0 <= self.c['pA_normal'] < self.c['pA_stress'] <= 1
                and self.c['tau_load_s'] > 0 and self.c['tau_composition_s'] > 0):
            raise ValueError('invalid state/clock configuration')
        self.source = SourceStateEstimator(self.c)
        for key in ('full_fast_increase_hysteresis_floor', 'full_accurate_increase_hysteresis_floor'):
            if key in self.c and not (0 < float(self.c[key]) <= 2):
                raise ValueError('invalid directional allocation hysteresis: '+key)
        if self.c.get('independent_checks_enabled') and not (
                self.c.get('role_observation_ns',0)>=2*self.c['base_ns'] and
                self.c['role_observation_ns']%(2*self.c['base_ns'])==0 and
                0<float(self.c.get('demand_change_floor_gpu',0))<=2):
            raise ValueError('invalid independent-role observation configuration')
        self.disabled = False
        self.disabled_reason = None
        self.source_event_cpu_ns = 0
        self.source_event_calls = 0
        self.folder = Path(folder) if folder else None
        self.last_loss_counts = {r: 0. for r in ROLES}
        self.last_loss_time_ns = None
        self.loss_history = deque(maxlen=3)
        self.next_fast_check_ns = self.c['min_check_ns']
        self.next_accurate_check_ns = self.c['min_check_ns']
        self.next_forced_joint_ns = self.c['max_joint_ns']
        self.reference = None
        self.reference_fast_share = None
        self.reference_history = deque(maxlen=2)
        self.last_reference_evidence_id = -1
        self.last_applied_command_direction = None
        self.current_proposal_direction = None
        self.state_reference_direction = None
        self.reverse_candidate = None
        self.reverse_evidence = None
        self.proposals = deque(maxlen=4)
        self.recent = deque(maxlen=3)
        self.last_now = -1
        self.context = {}
        self.stats = dict(cheap_checks=0, fast_detailed=0, accurate_detailed=0,
                          joint_checks=0, held_reversals=0, potential_oscillations=0)
        self.role_reference = {r:None for r in ROLES}
        self.role_previous = {r:None for r in ROLES}
        self.reverse_window_end_ns = None
        self.streams = {}
        self.trace_sink = trace_sink
        if folder and trace_sink is None:
            p = Path(folder)
            p.mkdir(parents=True, exist_ok=True)
            for name in ('STATE_TRACE', 'DETECTION_TRACE', 'JOINT_DECISIONS'):
                self.streams[name] = (p/(name+'.jsonl')).open('w', buffering=1)

    def log(self, stream, row):
        if self.trace_sink is not None:
            self.trace_sink(stream, row)
            return
        if stream in self.streams:
            self.streams[stream].write(json.dumps(row, sort_keys=True)+'\n')

    def monitor(self, now, view, observer, effective, pending):
        if self.mode == 'FULL' and self.c.get('independent_checks_enabled'):
            return self._independent_monitor(now, view, observer, effective, pending)
        if now < self.last_now:
            raise ValueError('nonmonotonic governor clock')
        if now == self.last_now:
            return self.context
        self.last_now = now
        self.stats['cheap_checks'] += 1
        self.source.advance(now)
        state = self.source.snapshot()
        span = min(200000000, now)
        read = view.get('direct_demand')
        quick = read(now, span) if callable(read) and span else None
        prices = {k: observer.c_ref_ms(k) for k in ('FAST_FULL','FAST_PREFIX','FAST_SUFFIX','ACCURATE')}
        units = dict(Fast=max(prices['FAST_FULL'],prices['FAST_PREFIX']+prices['FAST_SUFFIX']), Accurate=prices['ACCURATE'])
        backlog = {r: sum(float(observer.wait_counts[r].get(k,0))*float(v) for k,v in prices.items()) for r in ROLES}
        loss = {r: sum(float(observer.resource_loss_counts[r].get(k,0))*float(v) for k,v in prices.items()) for r in ROLES}
        sample = dict(now=now, backlog=backlog, loss=loss)
        old = self.recent[0] if self.recent else sample
        dt = max(1,now-old['now'])/NS
        growth = {r:(backlog[r]-old['backlog'][r])/dt for r in ROLES}
        loss_base = self.loss_history[0] if self.loss_history else None
        loss_dt = max(.1, now/NS) if loss_base is None else max(1,now-loss_base['now'])/NS
        loss_rate = {r:max(0.,loss[r]-(loss_base['loss'][r] if loss_base else 0.))/loss_dt for r in ROLES}
        new_loss = {r:loss[r]>self.last_loss_counts[r] for r in ROLES}
        self.last_loss_counts=dict(loss)
        self.last_loss_time_ns=now
        self.loss_history.append(dict(now=now,loss=dict(loss)))
        self.recent.append(sample)
        demand = ({'Fast':quick['D_F_ms_s'],'Accurate':quick['D_A_ms_s']} if quick and quick.get('valid') else None)
        if demand and not all(math.isfinite(float(v)) and v>=0 for v in demand.values()):
            raise ValueError('nonfinite or negative public work demand')
        slack = view.get('direct_role_slack_ns') or {r:view.get('direct_min_queue_slack_ns') for r in ROLES}
        intervals, triggers, urgent = {}, {}, False
        for r in ROLES:
            s = state['load_strength'] if r=='Fast' else state['degradation_strength']
            valid = s is not None and (r=='Fast' or state['state_valid'])
            state_interval = self.c['min_check_ns']+(self.c['max_check_ns']-self.c['min_check_ns'])*(1-s) if valid else self.c['min_check_ns']
            pressure = max(0.,growth[r])+loss_rate[r]
            risk = units[r]/pressure*NS if pressure>0 else self.c['max_check_ns']
            deadline = max(0,slack[r]) if slack.get(r) is not None else self.c['max_check_ns']
            raw = min(state_interval,risk,deadline)
            if new_loss[r]:raw=min(raw,self.c['base_ns'])
            intervals[r] = max(self.c['min_check_ns'],min(self.c['max_check_ns'],int(raw//self.c['base_ns'])*self.c['base_ns']))
            reasons=[]
            if demand and self.reference:
                threshold = .1*sum(demand.values())/2 + units[r]/max(.1,span/NS)
                if abs(demand[r]-self.reference[r]) > threshold:
                    reasons.append('demand_shift_'+r)
            if pressure>0 and risk<=self.c['base_ns']:
                reasons.append('backlog_risk_'+r)
            if new_loss[r]:
                reasons.append('resource_loss_'+r)
            if slack.get(r) is not None and slack[r]<=self.c['base_ns']:
                reasons.append('deadline_risk_'+r)
            urgent = urgent or new_loss[r] or bool(backlog[r] and deadline<=self.c['base_ns'])
            triggers[r]=reasons
        self.next_fast_check_ns=min(self.next_fast_check_ns,now+intervals['Fast'])
        self.next_accurate_check_ns=min(self.next_accurate_check_ns,now+intervals['Accurate'])
        due_f,due_a=now>=self.next_fast_check_ns,now>=self.next_accurate_check_ns
        reasons=triggers['Fast']+triggers['Accurate']
        if now>=self.next_forced_joint_ns:
            reasons.append('forced_review')
        if self.reference is None and demand:
            reasons.append('initial_public_demand')
        if self.mode=='UNIFORM_100MS':
            reasons.append('uniform_100ms')
        if view.get('frame_gap_seen') or view.get('log_backpressure') or not demand:
            fallback='missing_or_incomplete_public_evidence'
        else:
            fallback=None
        joint=bool(reasons) and now>=2*self.c['base_ns']
        self.stats['fast_detailed']+=int(due_f or joint)
        self.stats['accurate_detailed']+=int(due_a or joint)
        if joint:
            self.stats['joint_checks']+=1
            self.next_fast_check_ns=now+intervals['Fast']
            self.next_accurate_check_ns=now+intervals['Accurate']
            self.next_forced_joint_ns=now+self.c['max_joint_ns']
            if demand:
                self.reference=dict(demand)
                if (state['state_valid'] and not urgent and sum(demand.values())>0
                        and self.source.last_source_evidence_id>self.last_reference_evidence_id):
                    target=2*demand['Fast']/sum(demand.values())
                    self.last_reference_evidence_id=self.source.last_source_evidence_id
                    self.reference_history.append(target)
                    if len(self.reference_history)==2 and abs(self.reference_history[1]-self.reference_history[0])<=.1:
                        self.reference_fast_share=sum(self.reference_history)/2
        else:
            if due_f:self.next_fast_check_ns=now+intervals['Fast']
            if due_a:self.next_accurate_check_ns=now+intervals['Accurate']
        self.context=dict(tick_ns=now, public_event_cutoff_ns=self.source.public_cutoff_ns,
            state=state, D=demand, B=backlog, v=growth, loss=loss_rate, slack_ns=slack,
            intervals_ns=intervals, next_fast_check_ns=self.next_fast_check_ns,
            next_accurate_check_ns=self.next_accurate_check_ns,
            next_forced_joint_ns=self.next_forced_joint_ns, trigger_reasons=reasons,
            joint=joint, urgent=urgent, pending=bool(pending), effective_fast=effective,
            fallback_reason=fallback, stats=dict(self.stats), reference_fast_share=self.reference_fast_share,
            quick=quick, quick_span_ns=span)
        row={k:v for k,v in self.context.items() if k!='quick'}
        self.log('STATE_TRACE',row)
        self.log('DETECTION_TRACE',dict(row,fast_due=due_f,accurate_due=due_a))
        return self.context

    def _role_observation(self, role, now, view, observer, interval, effective):
        """One role only: bucket sums and its public deadline envelope, no replay."""
        span = int(self.c['role_observation_ns'])
        aligned=(int(now)//self.c['base_ns'])*self.c['base_ns']
        if aligned < span:
            return None
        read = view.get('direct_demand')
        half = span//2
        halves = [read(aligned-half, half), read(aligned, half)] if callable(read) else []
        if len(halves)!=2 or not all(d.get('valid') for d in halves):
            return None
        key = 'D_F_ms_s' if role=='Fast' else 'D_A_ms_s'
        values = [float(d[key]) for d in halves]
        if not all(math.isfinite(v) and v>=0 for v in values):
            raise ValueError('invalid single-role public work')
        prices = {k:observer.c_ref_ms(k) for k in ('FAST_FULL','FAST_PREFIX','FAST_SUFFIX','ACCURATE')}
        inline = bool(view.get('direct_inline_fast_continuation'))
        def work(counts):
            return sum(float(n)*(prices[k]+(prices['FAST_SUFFIX']
                if inline and k=='FAST_PREFIX' else 0.)) for k,n in counts.items())
        backlog = work(observer.wait_counts[role])
        loss = work(observer.resource_loss_counts[role])
        previous = self.role_previous[role]
        dt = (now-previous['now_ns'])/NS if previous is not None else 0.
        growth = (backlog-previous['backlog_ms'])/dt if dt>0 else 0.
        loss_rate = max(0.,(loss-previous['loss_work_ms'])/dt) if dt>0 else 0.
        horizon = max(1, int(view.get('direct_budget_horizon_ns', self.c['base_ns'])
            if self.c.get('deadline_target_enabled', True) else interval))/NS
        offered = sum(values)/2
        required = 0.
        if self.c.get('deadline_target_enabled', True):
            deadline_read = view.get('deadline_role_envelope')
            deadline = deadline_read(role) if callable(deadline_read) else None
            if not deadline or not deadline.get('valid'):
                return None
            required = float(deadline['roles'][role]['required_gpu'])
        need = max((offered+backlog/horizon+max(0.,growth))/1000., required)
        unit = max(prices['FAST_FULL'],prices['FAST_PREFIX']+prices['FAST_SUFFIX']) if role=='Fast' else prices['ACCURATE']
        threshold = max(float(self.c['demand_change_floor_gpu']),
                        (abs(values[1]-values[0])/2+unit/(span/NS))/1000.)
        capacity = float(effective) if role=='Fast' else float(len(view.get('cards',[])))-float(effective)
        observation = dict(now_ns=int(now), window_start_ns=aligned-span, window_end_ns=aligned,
            offered_ms_s=offered, backlog_ms=backlog, growth_ms_s=growth,
            growth_known=dt>0, loss_work_ms=loss, loss_rate_ms_s=loss_rate,
            deadline_required_gpu=required, deadline_excess_gpu=max(0.,required-capacity),
            need_gpu=need, change_threshold_gpu=threshold, half_offered_ms_s=values,
            growth_scatter_known=bool(previous is not None and previous['growth_known']),
            growth_scatter_ms_s=(abs(growth-previous['growth_ms_s'])/2
                if previous is not None and previous['growth_known'] else 0.),
            interval_ns=int(interval))
        self.role_previous[role] = observation
        return observation

    def _independent_monitor(self, now, view, observer, effective, pending):
        if now < self.last_now:
            raise ValueError('nonmonotonic governor clock')
        if now == self.last_now:
            return self.context
        self.last_now = now
        self.stats['cheap_checks'] += 1
        self.source.advance(now)
        clock_now=(int(now)//self.c['base_ns'])*self.c['base_ns']
        state = self.source.snapshot()
        intervals = {}
        for role in ROLES:
            if role=='Fast':
                strength = state['load_strength'] if state.get('load_valid') else None
            else:
                strength = (state['degradation_strength'] if state.get('composition_valid')
                    else state.get('smoothed_degradation_upper'))
            raw = (self.c['min_check_ns'] if strength is None else
                self.c['min_check_ns']+(self.c['max_check_ns']-self.c['min_check_ns'])*(1-strength))
            intervals[role] = max(self.c['min_check_ns'], min(self.c['max_check_ns'],
                int(raw//self.c['base_ns'])*self.c['base_ns']))
        # A state update may advance a role's due time, never run its check early.
        self.next_fast_check_ns = min(self.next_fast_check_ns,clock_now+intervals['Fast'])
        self.next_accurate_check_ns = min(self.next_accurate_check_ns,clock_now+intervals['Accurate'])
        due = dict(Fast=clock_now>=self.next_fast_check_ns, Accurate=clock_now>=self.next_accurate_check_ns)
        observations, reasons = {}, []
        urgent = False
        incomplete = bool(view.get('frame_gap_seen') or view.get('log_backpressure'))
        for role in ROLES:
            if not due[role]:
                continue
            self.stats['fast_detailed' if role=='Fast' else 'accurate_detailed'] += 1
            item = (None if incomplete else
                self._role_observation(role,now,view,observer,intervals[role],effective))
            observations[role] = item
            if item is not None:
                reference = self.role_reference[role]
                if reference is None:
                    reasons.append('initial_demand_'+role)
                else:
                    threshold = item['change_threshold_gpu']
                    rate_change = abs(item['offered_ms_s']-reference['offered_ms_s'])/1000.
                    need_change = abs(item['need_gpu']-reference['need_gpu'])
                    deadline_change = abs(item['deadline_excess_gpu']-reference['deadline_excess_gpu'])
                    loss_change = (item['loss_rate_ms_s']-reference['loss_rate_ms_s'])/1000.
                    if max(rate_change,need_change,deadline_change,abs(loss_change))>=threshold:
                        reasons.append('demand_change_'+role)
                    urgent = urgent or (item['deadline_excess_gpu']>0 and
                        item['deadline_excess_gpu']-reference['deadline_excess_gpu']>=threshold) or loss_change>=threshold
                # An unresolved reverse is reconsidered only with a new,
                # nonoverlapping public observation, on the receiving role clock.
                direction = 'F' if role=='Fast' else 'A'
                fresh_reverse = (item['window_end_ns']>self.reverse_window_end_ns
                    if self.reverse_window_end_ns is not None and self.c.get('strict_first_reverse_enabled')
                    else self.reverse_window_end_ns is not None and item['window_start_ns']>=self.reverse_window_end_ns)
                if self.reverse_candidate==direction and fresh_reverse:
                    reasons.append('fresh_reverse_evidence_'+role)
            next_at = clock_now+(intervals[role] if item is not None else self.c['min_check_ns'])
            if role=='Fast':self.next_fast_check_ns=next_at
            else:self.next_accurate_check_ns=next_at
        joint = bool(reasons) and not incomplete and not pending and not any(
            c.get('transition') for c in view.get('cards',[]))
        if joint:
            # Only now refresh the other role for the ONE joint calculation.
            for role in ROLES:
                if observations.get(role) is None:
                    observations[role]=self._role_observation(role,now,view,observer,intervals[role],effective)
            joint = all(observations.get(r) is not None for r in ROLES)
        if joint:self.stats['joint_checks'] += 1
        current = {r:observations.get(r) or self.role_previous[r] for r in ROLES}
        self.context = dict(tick_ns=now, public_event_cutoff_ns=self.source.public_cutoff_ns,
            state=state, D={r:v['offered_ms_s'] if v else None for r,v in current.items()},
            B={r:v['backlog_ms'] if v else 0. for r,v in current.items()},
            v={r:v['growth_ms_s'] if v else 0. for r,v in current.items()},
            loss={r:v['loss_rate_ms_s'] if v else 0. for r,v in current.items()},
            intervals_ns=intervals, next_fast_check_ns=self.next_fast_check_ns,
            next_accurate_check_ns=self.next_accurate_check_ns, next_forced_joint_ns=None,
            trigger_reasons=reasons, joint=joint, urgent=bool(urgent), pending=bool(pending),
            effective_fast=effective, fallback_reason='incomplete_public_evidence' if incomplete else None,
            card_count=len(view.get('cards',[])),
            role_checks=observations, role_references=dict(self.role_reference),
            fast_due=due['Fast'], accurate_due=due['Accurate'], stats=dict(self.stats),
            reference_fast_share=self.reference_fast_share)
        self.log('STATE_TRACE',self.context)
        self.log('DETECTION_TRACE',self.context)
        return self.context

    def joint_evaluated(self, now):
        """Acknowledge a calculated target, including a legitimate no-change result."""
        if not self.c.get('independent_checks_enabled') or self.mode!='FULL':
            return
        for role in ROLES:
            item=self.context.get('role_checks',{}).get(role)
            if item is not None:self.role_reference[role]=dict(item)
        clock_now=(int(now)//self.c['base_ns'])*self.c['base_ns']
        self.next_fast_check_ns=clock_now+self.context['intervals_ns']['Fast']
        self.next_accurate_check_ns=clock_now+self.context['intervals_ns']['Accurate']
        self.context.update(next_fast_check_ns=self.next_fast_check_ns,
                            next_accurate_check_ns=self.next_accurate_check_ns)
        demand=self.context.get('D',{})
        if all(demand.get(r) is not None for r in ROLES):
            self.reference=dict(demand)
            total=sum(demand.values())
            eid=self.source.last_source_evidence_id
            if total>0 and self.source.state_valid and eid>self.last_reference_evidence_id:
                self.last_reference_evidence_id=eid
                self.reference_history.append(self.context['card_count']*demand['Fast']/total)
                if len(self.reference_history)==2 and abs(self.reference_history[1]-self.reference_history[0])<=.1:
                    self.reference_fast_share=sum(self.reference_history)/2
                else:
                    self.reference_fast_share=None

    def safe_monitor(self, now, view, observer, effective, pending):
        if self.disabled:
            return self.context
        try:
            return self.monitor(now,view,observer,effective,pending)
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            self.disabled=True
            self.disabled_reason='state_monitor_fallback:'+type(exc).__name__+':'+str(exc)
            self.context=dict(self.context,tick_ns=now,joint=False,
                fallback_reason=self.disabled_reason,effective_fast=effective)
            self.log('DETECTION_TRACE',dict(self.context,disabled=True))
            return self.context

    def allocation_tolerance(self, delta_fast, legacy_floor, noise):
        """FULL-only command deadband; direction names the receiving role.

        Missing keys preserve the frozen controller. This never quantizes a
        target, changes resource minima, or bypasses empirical noise.
        """
        floor = legacy_floor
        if self.mode == 'FULL' and not self.disabled:
            key = ('full_fast_increase_hysteresis_floor' if delta_fast > 0 else
                   'full_accurate_increase_hysteresis_floor')
            floor = max(legacy_floor, float(self.c.get(key, legacy_floor)))
        return max(floor, noise), floor

    def filter_target(self, now, evidence, effective):
        if self.mode=='FULL' and self.c.get('independent_checks_enabled'):
            return self._filter_independent_target(now,evidence,effective)
        target=evidence['target_fast']
        direction='F' if target>effective else 'A'
        self.current_proposal_direction=direction
        self.state_reference_direction=(None if self.reference_fast_share is None else
            'F' if self.reference_fast_share>effective else 'A')
        c=self.context
        reason=None
        # Only nonurgent FULL actions are filtered, never replace the QF target.
        reverse=self.last_applied_command_direction is not None and direction!=self.last_applied_command_direction
        eid=self.source.last_source_evidence_id
        trusted = self.source.state_valid and self.reference_fast_share is not None
        persistent_support=(trusted and self.state_reference_direction==direction and
            len(self.reference_history)==2 and abs(self.reference_history[1]-self.reference_history[0])<=.1
            and self.last_reference_evidence_id==eid and self.source.confirmed_state_reference is not None)
        if self.mode=='FULL' and trusted and reverse and not persistent_support and not c.get('urgent') and not c.get('fallback_reason'):
            if self.reverse_candidate!=direction:
                self.reverse_candidate,self.reverse_evidence=direction,eid
                reason='first_nonurgent_reverse_await_independent_source_block'
            elif eid<=self.reverse_evidence:
                reason='reverse_needs_new_independent_source_block'
            else:
                self.reverse_candidate=None
        elif not reverse:
            self.reverse_candidate=None
        self.proposals.append(direction)
        alternating=len(self.proposals)==4 and all(self.proposals[i]!=self.proposals[i+1] for i in range(3))
        stable_state=(self.source.confirmed_state_reference is not None and trusted and
            max(abs(a-b) for a,b in zip(self.source.confirmed_state_reference,
                (self.source.load_strength,self.source.degradation_strength)))<self.c['state_change_threshold'])
        if alternating and stable_state:
            self.stats['potential_oscillations']+=1
        if reason:
            self.stats['held_reversals']+=1
            self.next_fast_check_ns=min(self.next_fast_check_ns,now+self.c['base_ns'])
            self.next_accurate_check_ns=min(self.next_accurate_check_ns,now+self.c['base_ns'])
        row=dict(self.context, qf_target_fast=target, final_target_fast=effective if reason else target,
            record_kind='filtered_candidate',command_issued=False if reason else None,
            filter_reason=reason, current_proposal_direction=direction,
            last_applied_command_direction=self.last_applied_command_direction,
            state_reference_direction=self.state_reference_direction, potential_oscillation=alternating,
            qf_evidence=evidence)
        row['state_filter_valid'] = trusted
        row['persistent_reference_support'] = persistent_support
        row['state_filter_fallback'] = None if trusted else 'no_trusted_original_composition_reference_use_original_QF_target'
        row.pop('quick',None)
        self.log('JOINT_DECISIONS',row)
        return reason

    def _filter_independent_target(self, now, evidence, effective):
        direction='F' if evidence['target_fast']>effective else 'A'
        self.current_proposal_direction=direction
        reverse=(self.last_applied_command_direction is not None and
                 direction!=self.last_applied_command_direction)
        self.state_reference_direction=(None if self.reference_fast_share is None else
            'F' if self.reference_fast_share>effective else 'A')
        trusted=(self.source.state_valid and self.reference_fast_share is not None and
                 self.last_reference_evidence_id==self.source.last_source_evidence_id)
        supported=trusted and self.state_reference_direction==direction
        role='Fast' if direction=='F' else 'Accurate'
        sample=self.context['role_checks'][role]
        self.proposals.append(direction)
        alternating=len(self.proposals)==4 and all(self.proposals[i]!=self.proposals[i+1] for i in range(3))
        if alternating:self.stats['potential_oscillations']+=1
        reason=None
        strict = bool(self.c.get('strict_first_reverse_enabled'))
        if reverse and (strict or (not supported and not self.context.get('urgent'))):
            if self.reverse_candidate!=direction:
                self.reverse_candidate=direction
                self.reverse_evidence=self.source.last_source_evidence_id
                self.reverse_window_end_ns=sample['window_end_ns']
                reason=('first_reverse_hold_recheck_at_minimum_interval' if strict else
                        'nonurgent_reverse_wait_independent_role_observation')
            elif self.reverse_window_end_ns is None or (
                    sample['window_end_ns']<=self.reverse_window_end_ns if strict else
                    sample['window_start_ns']<self.reverse_window_end_ns):
                reason='reverse_role_observations_overlap'
            else:
                self.reverse_candidate=self.reverse_evidence=self.reverse_window_end_ns=None
        else:
            self.reverse_candidate=self.reverse_evidence=self.reverse_window_end_ns=None
        if reason:
            self.stats['held_reversals']+=1
            # Wake only the receiver. This is a pending evidence check, not a
            # forced joint calculation or a lock on the allocation direction.
            next_at=(int(now)//self.c['base_ns'])*self.c['base_ns']+(
                self.c['min_check_ns'] if strict else self.c['role_observation_ns'])
            if role=='Fast':self.next_fast_check_ns=min(self.next_fast_check_ns,next_at)
            else:self.next_accurate_check_ns=min(self.next_accurate_check_ns,next_at)
        self.context.update(next_fast_check_ns=self.next_fast_check_ns,
                            next_accurate_check_ns=self.next_accurate_check_ns)
        self.log('JOINT_DECISIONS',dict(tick_ns=now,record_kind='filtered_candidate',
            qf_target_fast=evidence['target_fast'],final_target_fast=effective if reason else evidence['target_fast'],
            filter_reason=reason,state_filter_valid=trusted,persistent_reference_support=supported,
            last_applied_command_direction=self.last_applied_command_direction,
            current_proposal_direction=direction,reverse_window_end_ns=self.reverse_window_end_ns,
            potential_oscillation=alternating,
            qf_evidence=evidence))
        return reason

    def command_applied(self, old, new):
        if old!=new:
            self.last_applied_command_direction='F' if new>old else 'A'
            self.recent.clear()  # invalidate allocation-specific queue slopes only
            self.role_previous = {r:None for r in ROLES}
            for role in ROLES:
                reference=self.role_reference[role]
                if reference is not None:
                    capacity=new if role=='Fast' else self.context.get('card_count',2)-new
                    reference['deadline_excess_gpu']=max(0.,reference['deadline_required_gpu']-capacity)
            self.reverse_candidate=self.reverse_evidence=None
            self.reverse_window_end_ns=None

    def close(self):
        for stream in self.streams.values():stream.close()
        if self.folder:
            metrics=dict(mode=self.mode,
                disabled=self.disabled,disabled_reason=self.disabled_reason,stats=self.stats,
                source_event_cpu_ns=self.source_event_cpu_ns,source_event_calls=self.source_event_calls,
                source_final=self.source.snapshot())
            if self.trace_sink is not None:
                self.trace_sink('GOVERNOR_METRICS', metrics)
            else:
                (self.folder/'GOVERNOR_METRICS.json').write_text(json.dumps(metrics,indent=2))
