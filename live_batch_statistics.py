"""Offline statistics only. This module never imports Torch or runs a policy."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import time

ROOT=Path(__file__).resolve().parent

def rows(path):
    if not path.exists():return
    with path.open() as stream:
        for line in stream:yield json.loads(line)

def save(path,value):
    tmp=Path(str(path)+'.partial');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def distribution(values):
    values.sort();n=len(values)
    return dict(n=n,mean=sum(values)/n if n else None,
        p95=values[int(.95*(n-1))] if n else None,p99=values[int(.99*(n-1))] if n else None)

def analyze(folder,cpu):
    os.sched_setaffinity(0,{cpu});before=time.process_time_ns();wall=time.monotonic()
    folder=Path(folder);execution=json.loads((folder/'execution.json').read_text())
    if not execution['online_processes_stopped']:raise ValueError('Statistics require stopped online processes')
    identity=json.loads((folder/'identity.json').read_text())
    manifest_path=Path(identity['manifest_path'])
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=identity['manifest_sha256']:
        raise ValueError('Manifest changed before offline evaluation')
    manifest=json.loads(manifest_path.read_text())
    labels=json.loads((ROOT/'evidence/evaluation_private_labels.json').read_text())
    totals=Counter();roles={r:Counter() for r in ('Local','Fast','Accurate','Abstain')}
    seen=set();delays=[];phases={};edge=0;reasons=Counter();returned_tasks={}
    for row in rows(folder/'tasks.jsonl'):
        tid=row['task_id']
        if tid in seen:raise ValueError('Duplicate terminal')
        seen.add(tid)
        if row['reason']=='gpu_completed':
            returned_tasks[tid]=(row['terminal_ns'],row['role'],row['prediction'])
        if row['deadline_ns']-row['arrival_ns']!=manifest['deadline_ns']:raise ValueError('Deadline mutation')
        correct=row['prediction'] is not None and row['prediction']==labels[str(row['bank_index'])]
        good=correct and row['ontime'];totals['correct_ontime']+=good
        roles[row['role']]['tasks']+=1;roles[row['role']]['correct_ontime']+=good
        edge+=bool(row['edge_published']);reasons[row['reason']]+=1
        if row['prediction'] is None:totals['no_prediction']+=1
        elif not correct:totals['classification_error']+=1
        if row['prediction'] is not None and not row['ontime']:totals['late_completion']+=1
        if row['prediction'] is not None:delays.append((row['terminal_ns']-row['arrival_ns'])/1e6)
        t=row['arrival_ns']/1e9
        phase=(('normal' if t<50 else 'accurate_increase' if t<125 else 'fast_increase' if t<200 else 'recovery')
               if 'demand_transfer' in manifest.get('scene','').lower() else 'normal_composition_high_load')
        if manifest.get('analysis_phases'):
            phase=next(x['name'] for x in manifest['analysis_phases'] if x['start_s']<=t<x['end_s'])
        p=phases.setdefault(phase,Counter());p['tasks']+=1;p['correct_ontime']+=good
        p[row['role']+'_correct_ontime']+=good
    if len(seen)!=execution['arrivals']:raise ValueError('Missing terminal events')
    waits=[];sizes=Counter();last={};gpu_intervals=[[],[]];actual_member_count=0
    prefix_batches=0;early_a=0;executed=set();stages=set();deferred_count=0
    for b in rows(folder/'batches.jsonl'):
        n=len(b['members'])
        if not 1<=n<=10 or len(set(b['members']))!=n:raise ValueError('Invalid batch membership')
        if b['actual_batch']>n:raise ValueError('Batch grew inside actor')
        sizes[n]+=1;actual_member_count+=b['actual_batch']
        ids=b.get('executed_task_ids',[i for i in b['members'] if i not in b['prestart_refused']])
        if len(ids)!=b['actual_batch']:raise ValueError('Actor execution count mismatch')
        if set(ids)&set(b.get('deferred_task_ids',[])):raise ValueError('Executed task was deferred')
        for i in ids:
            key=(i,b.get('job_kinds',{}).get(str(i),b['kind']))
            if key in stages:raise ValueError('Repeated actual GPU stage')
            stages.add(key);executed.add((b['batch_id'],i))
        deferred_count+=len(b.get('deferred_task_ids',[]))
        if b['kind'] in ('FAST_PREFIX','FAST_MIXED'):
            prefix_batches+=1
            f=set(b['fast_suffix_task_ids']);a=set(b['prefix_accurate_task_ids'])
            if f&a or f!=set(map(int,b['predictions'])):raise ValueError('Accurate entered Fast Suffix')
            groups=b.get('suffix_groups',[list(f)] if f else [])
            flat=[i for group in groups for i in group]
            if len(flat)!=len(set(flat)) or set(flat)!=f:raise ValueError('Invalid same-card Suffix groups')
            early_a+=len(a)
            if b['kind']=='FAST_MIXED':
                known=set(b['known_fast_task_ids']);probe=set(b['probe_task_ids'])
                if not known<=f or known&probe or a&known:raise ValueError('Mixed mask/known-route violation')
                expected={i for i in ids if b['job_kinds'][str(i)]=='FAST_PREFIX'}
                if probe!=expected:raise ValueError('Unknown rows not probed exactly once')
        if b['cuda_start_ns'] is not None:
            g=b['gpu'];a=b['cuda_start_ns'];z=b['cuda_end_ns']
            if a<last.get(g,-1) or z<a:raise ValueError('Overlapping/nonmonotonic GPU batches')
            last[g]=z;gpu_intervals[g].append((a,z))
    for d in rows(folder/'dispatch.jsonl'):
        keys=[(j['deadline_ns'],j['task_id']) for j in d['public_jobs']]
        order=identity.get('batch_order','absolute_deadline' if identity.get('batch_rule','').startswith('EDF') else 'original_fifo')
        if order=='absolute_deadline' and keys!=sorted(keys):raise ValueError('Non-EDF batch')
        for j in d['public_jobs']:
            if (d['batch_id'],j['task_id']) in executed:waits.append((d['now_ns']-j['enqueue_ns'])/1e6)
        if execution['arm']=='ours':
            p=d['fast_share'];role=d['public_jobs'][0]['role']
            if (role=='Fast' and p<=0) or (role=='Accurate' and p>=1):raise ValueError('Zero share dispatch')
    return_stats={}
    if identity.get('response_mode')=='independent_actual_timer':
        transport=[];timer=[];return_ids=set()
        for r in rows(folder/'returns.jsonl'):
            if not r['ready_ns']<=r['received_ns']<=r['due_ns']<=r['returned_ns']:
                raise ValueError('Noncausal response event')
            if r['due_ns']-r['received_ns']!=500000:raise ValueError('Return delay altered')
            for tid,pred in r['predictions'].items():
                tid=int(tid)
                if tid in return_ids:raise ValueError('Repeated actual response')
                return_ids.add(tid)
                if returned_tasks.pop(tid,None)!=(r['returned_ns'],r['role'],pred):
                    raise ValueError('Terminal is not the actual independent return event')
            transport.append((r['received_ns']-r['ready_ns'])/1e6)
            timer.append((r['returned_ns']-r['due_ns'])/1e6)
        if returned_tasks or len(return_ids)!=execution['independent_return_count']:
            raise ValueError('Incomplete response accounting')
        return_stats=dict(actual_returns=len(return_ids),causal_timestamps=True,
            actor_to_responder_ms=distribution(transport),timer_overshoot_ms=distribution(timer))
    lags=[(r['received_ns']-r['planned_ns'])/1e6 for r in rows(folder/'receipts.jsonl')]
    metrics=dict(execution,schema='real_gpu_batch_results_v2',totals=dict(totals),
        roles={r:dict(v) for r,v in roles.items()},tca=totals['correct_ontime']/len(seen),
        phases={k:dict(v) for k,v in phases.items()},edge_published=edge,
        edge_correct_ontime=totals['correct_ontime']-roles['Local']['correct_ontime'],
        edge_tca=(totals['correct_ontime']-roles['Local']['correct_ontime'])/edge if edge else None,
        completed_latency_ms=distribution(delays),gpu_wait_ms=distribution(waits),
        ingress_lag_ms=distribution(lags),batch_size_counts=dict(sizes),
        actual_gpu_component_members=actual_member_count,gpu_nonoverlap=True,
        no_repeated_gpu_stage=True,actor_deferred_requeues=deferred_count,
        gpu_wait_scope='Original ready enqueue to dispatch for actually executed stages only; unstarted deferred attempts excluded.',
        prefix_batches=prefix_batches,prefix_early_accurate=early_a,accurate_never_enters_fast_suffix=True,
        failure_reasons=dict(reasons),statistics_separate_process=True,
        independent_response_audit=return_stats,
        statistics_cpu_affinity=list(os.sched_getaffinity(0)),
        offline_statistics_cpu_s=(time.process_time_ns()-before)/1e9,
        offline_statistics_wall_s=time.monotonic()-wall)
    save(folder/'metrics.json',metrics)
    return metrics

def pair(output,n,cpu):
    os.sched_setaffinity(0,{cpu});output=Path(output);started=time.process_time_ns()
    labels=json.loads((ROOT/'evidence/evaluation_private_labels.json').read_text());arms={}
    for arm in ('ours','edge_serving'):
        good=bytearray(n);seen=bytearray(n)
        for row in rows(output/arm/'tasks.jsonl'):
            i=row['task_id']
            if seen[i]:raise ValueError('Repeated paired task')
            seen[i]=1;good[i]=int(row['prediction'] is not None and row['ontime'] and
                                row['prediction']==labels[str(row['bank_index'])])
        if sum(seen)!=n:raise ValueError('Incomplete paired source')
        arms[arm]=good
    gains=losses=0
    with (output/'paired_differences.jsonl').open('w') as f:
        for i,(a,b) in enumerate(zip(arms['ours'],arms['edge_serving'])):
            if a==b:continue
            gains+=int(a);losses+=int(b)
            f.write(json.dumps(dict(task_id=i,ours_correct_ontime=bool(a),edge_serving_correct_ontime=bool(b)))+'\n')
    result=dict(ours_gained=gains,ours_lost=losses,ours_net=gains-losses,
                offline_pairing_cpu_s=(time.process_time_ns()-started)/1e9,cpu_affinity=[cpu])
    save(output/'paired_summary.json',result);return result
