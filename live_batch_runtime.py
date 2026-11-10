"""Wall-clock Local-trace replay with real nonpreemptive CUDA batch actors.

Local inference/upload are frozen external events. Every edge prediction is
computed by the actual CNN. No cached GPU prediction or virtual GPU completion
can be used by this executor. All software/IPC delays remain on the clock.
"""
from collections import Counter, deque
import hashlib
import heapq
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import resource
import subprocess
import time
import traceback
from live_response import response_worker
from mixed_fast_batch import MixedFastProfile,batch_kind,fast_composition

ROOT = Path(__file__).resolve().parent
NS = 1000000000
KINDS = ('FAST_FULL','FAST_PREFIX','ACCURATE')


class DenseTerminalTracker:
    """Exact deduplication for frozen integer task IDs in [0, arrivals).

    Allocate once before replay. No labels, route information or future
    outcomes are stored; only whether a terminal has already been recorded.
    """
    def __init__(self,arrivals):
        if type(arrivals) is not int or arrivals<=0:
            raise ValueError('Dense terminal tracking requires positive integer arrivals')
        self._bits=bytearray(arrivals)
        self._count=0

    def _index(self,tid):
        if type(tid) is not int or not 0<=tid<len(self._bits):
            raise ValueError('Frozen task ID outside dense terminal domain')
        return tid

    def __contains__(self,tid):
        return bool(self._bits[self._index(tid)])

    def add(self,tid):
        index=self._index(tid)
        if not self._bits[index]:
            self._bits[index]=1
            self._count+=1

    def __len__(self):
        return self._count


# Literature arms (edge-side adaptations) and those with a separate CPU planner process.
LITERATURE_ARMS=('wma_node','jellyfish_edge','edgeboost_edge','clockwork_edge','symphony_edge','proteus_edge','slackfit_edge',
                 'pard_edge','ramsis_edge')
PLANNER_ARMS=('jellyfish_edge','proteus_edge')

def configure_replay_gc(gc_module,defer):
    previously_enabled=gc_module.isenabled()
    if defer:gc_module.disable()
    return previously_enabled


def restore_replay_gc(gc_module,previously_enabled):
    if previously_enabled:gc_module.enable()
    else:gc_module.disable()


def gather_host_inputs(bank,ids,host,direct=False):
    """Preserve order/duplicates while optionally avoiding a temporary copy.

    The output is the existing reusable pinned host buffer. Its lifetime and
    GPU synchronization are unchanged; no model output or route is inspected.
    """
    if direct:
        import torch
        torch.index_select(bank,0,torch.tensor(ids,dtype=torch.long),out=host[:len(ids)])
    else:host[:len(ids)].copy_(bank[ids])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    temporary=Path(str(path)+'.partial')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def cpu_snapshot():
    """Process user+system CPU, including its queue feeder threads."""
    r=resource.getrusage(resource.RUSAGE_SELF)
    return dict(user_s=r.ru_utime,system_s=r.ru_stime)


def save_process_cpu(folder,name,baseline):
    end=cpu_snapshot()
    threads={}
    for entry in Path('/proc/self/task').iterdir():
        try:threads[entry.name]=list(os.sched_getaffinity(int(entry.name)))
        except ProcessLookupError:pass
    write_json(Path(folder)/(name+'_process_cpu.json'),dict(pid=os.getpid(),
        lifetime=end,baseline=baseline,
        thread_cpu_affinity=threads,
        replay_and_drain_s=sum(end.values())-sum(baseline.values()),
        scope='Process user+system, all threads; after producer queue flush/join. Includes drain, excludes setup.'))


class BatchProfile:
    def __init__(self,path):
        self.data=json.loads(Path(path).read_text())
        if self.data['batch_size_limit']!=10 or self.data['cuda_graph_used']:
            raise ValueError('Expected measured eager batch1..10 profile')
        self.rows={(s['batch'],s['kind'],s['fast_survivors']):s for s in self.data['summary']}

    def stage(self,b,kind,clock='mean_ms',survivors=None):
        return self.rows[b,kind,survivors]['cuda_ms'][clock]*1e6

    def latency(self,kind,b,clock='mean_ms',survivors=None):
        if kind=='FAST_SUFFIX':
            return self.stage(b,'FAST_SUFFIX_ISOLATED',clock)+self.stage(b,'D2H_RESULT',clock)
        upload=self.stage(b,'H2D_INPUT',clock)
        if kind=='FAST_PREFIX':
            values=[self.stage(b,'FAST_SELECTIVE_EARLY_EXIT',clock,k)+
                    (self.stage(k,'D2H_RESULT',clock) if k else 0)
                    for k in ([survivors] if survivors is not None else range(b+1))]
            return upload+max(values)
        name='FAST_FULL' if kind=='FAST_FULL' else 'ACCURATE'
        return upload+self.stage(b,name,clock)+self.stage(b,'D2H_RESULT',clock)

    def mean(self,kind,b):return self.latency(kind,b)
    def budget(self,kind,b):return self.latency(kind,b,'p95_ms')


def source_worker(trace,output,epoch,cpu,folder):
    os.sched_setaffinity(0,{cpu})
    digest=hashlib.sha256();count=0;pending=[];previous=-1
    try:
        while not epoch.value:time.sleep(.001)
        while time.monotonic_ns()<epoch.value:time.sleep(.0005)
        baseline=cpu_snapshot()
        cpu_start=time.process_time_ns()
        with Path(trace).open('rb') as source:
            for raw in source:
                frame=json.loads(raw)
                at=int(frame['source_publish_ns'])
                if at<previous:raise ValueError('Frozen source publication times are not ordered')
                previous=at
                due=epoch.value+at
                while time.monotonic_ns()<due:
                    if pending:output.put(('FRAMES',pending));pending=[]
                    time.sleep(min(.001,max(0,(due-time.monotonic_ns())/NS)))
                digest.update(raw);count+=1
                pending.append((raw,time.monotonic_ns()-epoch.value))
                if len(pending)>=64:output.put(('FRAMES',pending));pending=[]
        if pending:output.put(('FRAMES',pending))
        output.put(('END',dict(sha256=digest.hexdigest(),frames=count,
                    steady_cpu_ns=time.process_time_ns()-cpu_start)))
    except Exception:
        output.put(('ERROR',traceback.format_exc()))
        raise
    finally:
        output.close();output.join_thread()
        if 'baseline' in locals():save_process_cpu(folder,'source',baseline)


def log_worker(folder,records,cpu,epoch):
    os.sched_setaffinity(0,{cpu})
    streams={};counts=Counter();before=time.process_time_ns();baseline=None
    try:
        while not epoch.value:time.sleep(.001)
        while time.monotonic_ns()<epoch.value:time.sleep(.0005)
        baseline=cpu_snapshot()
        while True:
            item=records.get()
            if item is None:break
            name,record=item
            path=Path(folder)/name
            if name in ('progress','control/GOVERNOR_METRICS'):
                path.parent.mkdir(parents=True,exist_ok=True)
                write_json(path.with_suffix('.json'),record);counts[name]+=1
                continue
            if name not in streams:
                path.parent.mkdir(parents=True,exist_ok=True)
                streams[name]=path.with_suffix('.jsonl').open('w',buffering=1024*1024)
            streams[name].write(json.dumps(record,separators=(',',':'))+'\n');counts[name]+=1
    finally:
        for stream in streams.values():stream.close()
        write_json(Path(folder)/'logger_metrics.json',dict(rows=dict(counts),cpu_ns=time.process_time_ns()-before))
        save_process_cpu(folder,'logger',baseline or cpu_snapshot())


def gpu_worker(g,bank,commands,results,responses,epoch,cfg,profile_path,folder):
    try:
        import torch
        from physical_probe import configure,model
        from batch_inference import BatchedEarlyHeads,FastBatchEarlyExit,MixedFastBatch
        configure(cfg['gpu_cpus'][g])
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_device(g)
        device=torch.device('cuda',g)
        fast,accurate=model('Fast',device),model('Accurate',device)
        heads=BatchedEarlyHeads(device);route=FastBatchEarlyExit(fast,heads)
        mixed_route=MixedFastBatch(fast,heads)
        profile=BatchProfile(profile_path)
        literature_calibration=(json.loads(Path(cfg['literature_calibration']).read_text())
            if cfg.get('literature_method')=='edgeboost_edge' else None)
        if literature_calibration is not None:
            from literature_baselines.current_literature import confidence_margin
        host=torch.empty((10,3,224,224),dtype=torch.float32,pin_memory=True)
        with torch.inference_mode():
            for b in range(1,11):
                x=bank[:b].to(device)
                for _ in range(3):fast(x);accurate(x);route(x)
            torch.cuda.synchronize()
        results.put(dict(type='READY',gpu=g,pid=os.getpid(),uuid=cfg['gpu_uuids'][g],
                    cpu_affinity=list(os.sched_getaffinity(0)),allocated_bytes=torch.cuda.memory_allocated(g)))
        while not epoch.value:time.sleep(.001)
        while time.monotonic_ns()<epoch.value:time.sleep(.0005)
        baseline=cpu_snapshot()
        cpu_start=time.process_time_ns()
        with torch.inference_mode():
            while True:
                command=commands.get()
                if command is None:break
                jobs=command['jobs'];kind=batch_kind(jobs);batch_id=command['batch_id']
                before=time.monotonic_ns()-epoch.value
                # Recheck after actual IPC, immediately before GPU execution.
                # This is a measured inference-budget check, no queue ETA.
                allowed=[];refused=[]
                for job in jobs:
                    unit=profile.mean(job['kind'],1)
                    if before+unit+cfg['network_return_ns']>job['deadline_ns']:
                        refused.append(job['task_id'])
                    else:allowed.append(job)
                first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
                elapsed=0.;predictions={};expired=[];suffix_ids=[];accurate_ids=[]
                cuda_start=None;cuda_end=None;probe_ids=[]
                if allowed:
                    ids=[j['bank_index'] for j in allowed];b=len(ids)
                    gather_host_inputs(bank,ids,host,cfg.get('direct_host_gather',False))
                    cuda_start=time.monotonic_ns()-epoch.value
                    first.record()
                    x=host[:b].to(device,non_blocking=True)
                    if kind in ('FAST_PREFIX','FAST_MIXED'):
                        def publish(public):
                            stamp=time.monotonic_ns()-epoch.value
                            rows=[]
                            positions=public.get('probe_indices',range(len(allowed)))
                            for pos,i in enumerate(positions):
                                j=allowed[i];probe_ids.append(j['task_id'])
                                values=public['public_decisions'][pos]
                                rows.append(dict(task_id=j['task_id'],role=public['roles'][pos],
                                                 probabilities=dict(zip(heads.roles,values[:-1])),need=values[-1]))
                            results.put(dict(type='PREFIX_ROUTES',gpu=g,batch_id=batch_id,
                                             published_ns=stamp,rows=rows))
                        def keep_suffix(indices):
                            now=time.monotonic_ns()-epoch.value
                            suffix=profile.mean('FAST_SUFFIX',1)
                            return [i for i in indices if allowed[i]['kind']=='FAST_FULL' or
                                now+suffix+cfg['network_return_ns']<=allowed[i]['deadline_ns']]
                        result=(mixed_route(x,[j['kind']=='FAST_PREFIX' for j in allowed],
                            on_routes=publish,suffix_filter=keep_suffix) if kind=='FAST_MIXED' else
                            route(x,on_routes=publish,suffix_filter=keep_suffix))
                        suffix_ids=[allowed[i]['task_id'] for i in result['fast_indices']]
                        accurate_ids=[allowed[i]['task_id'] for i in result['accurate_indices']]
                        predicted=result['fast_predictions'].cpu().tolist()
                        predictions={allowed[i]['task_id']:int(p) for i,p in zip(result['fast_indices'],predicted)}
                        expired=[allowed[i]['task_id'] for i in result['expired_fast_indices']]
                    else:
                        net=fast if kind=='FAST_FULL' else accurate
                        logits=net(x)
                        if cfg.get('literature_method')=='edgeboost_edge' and kind=='FAST_FULL':
                            calibration=literature_calibration
                            margins=confidence_margin(logits,calibration['temperature_fast']).cpu().tolist()
                            predicted=logits.argmax(1).cpu().tolist()
                            rows=[]
                            for job,pred,margin in zip(allowed,predicted,margins):
                                role='Fast' if margin>=calibration['threshold_margin'] else 'Accurate'
                                rows.append(dict(task_id=job['task_id'],role=role,margin=margin,
                                                 fast_prediction=int(pred),criterion='calibrated_top1_minus_top2'))
                                if role=='Fast':predictions[job['task_id']]=int(pred)
                            # Classification has actually completed before this
                            # public decision; uncertain rows do not return a
                            # provisional Fast answer and cannot terminate twice.
                            results.put(dict(type='EDGEBOOST_ROUTES',gpu=g,batch_id=batch_id,
                                published_ns=time.monotonic_ns()-epoch.value,rows=rows))
                        else:
                            predicted=logits.argmax(1).cpu().tolist()
                            predictions={j['task_id']:int(p) for j,p in zip(allowed,predicted)}
                    last.record();last.synchronize()
                    elapsed=first.elapsed_time(last)*1e6
                    cuda_end=time.monotonic_ns()-epoch.value
                ready=time.monotonic_ns()-epoch.value
                if predictions:
                    responses.put(dict(batch_id=batch_id,gpu=g,ready_ns=ready,
                        role='Accurate' if kind=='ACCURATE' else 'Fast',predictions=predictions))
                results.put(dict(type='BATCH_DONE',gpu=g,batch_id=batch_id,kind=kind,
                    actual_batch=len(allowed),started_ns=before,
                    job_kinds={j['task_id']:j['kind'] for j in jobs},
                    executed_task_ids=[j['task_id'] for j in allowed],probe_task_ids=probe_ids,
                    known_fast_task_ids=[j['task_id'] for j in allowed if j['kind']=='FAST_FULL'],
                    mixed_unknown=sum(j['kind']=='FAST_PREFIX' for j in allowed),
                    mixed_unknown_fast=sum(j['kind']=='FAST_PREFIX' and j['task_id'] in suffix_ids for j in allowed),
                    finished_ns=ready,
                    cuda_start_ns=cuda_start,cuda_end_ns=cuda_end,cuda_service_ns=elapsed,
                    fast_suffix_task_ids=suffix_ids,prefix_accurate_task_ids=accurate_ids,
                    predictions=predictions,prestart_refused=refused,suffix_refused=expired))
        results.put(dict(type='STOPPED',gpu=g,steady_cpu_ns=time.process_time_ns()-cpu_start))
    except Exception:
        results.put(dict(type='ERROR',gpu=g,error=traceback.format_exc()))
        raise
    finally:
        results.close();results.join_thread()
        responses.close();responses.join_thread()
        if 'baseline' in locals():save_process_cpu(folder,'gpu'+str(g),baseline)


class LiveBatchRuntime:
    def __init__(self,manifest_path,output,arm,cfg_path,allow_idle_resident_peers=False):
        self.manifest_path=Path(manifest_path);self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.cfg_path=Path(cfg_path);self.cfg=json.loads(self.cfg_path.read_text())
        cpus=self.cfg['gpu_cpus']+[self.cfg[k] for k in
                                  ('controller_cpu','source_cpu','logger_cpu','statistics_cpu','response_cpu')]
        if arm in PLANNER_ARMS:cpus.append(self.cfg['literature_planner_cpu'])
        if len(cpus)!=len(set(cpus)):raise ValueError('Execution and statistics cores must be disjoint')
        physical=[]
        for cpu in cpus:
            topology=Path('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology'
            physical.append(tuple((topology/name).read_text().strip() for name in
                                  ('physical_package_id','core_id')))
        if len(physical)!=len(set(physical)):raise ValueError('CPU assignment shares physical cores')
        os.sched_setaffinity(0,{self.cfg['controller_cpu']})
        self.arm=arm
        if self.cfg['gpu_cards']!=2 or self.cfg['batch_max']!=10:
            raise ValueError('This entry requires two GPUs and maximum batch ten')
        if self.cfg['deadline_in_resource_control']:
            raise ValueError('Live migration was authorized with deadline-free QF targets')
        required=dict(initial_fast=1.0,allocation_grid=.01,batch_formation_wait_ns=0,
            response_mode='independent_actual_timer',controller_trace_mode='logger_process',
            strict_first_reverse_enabled=True,software_delay_mode='actual_wall_clock',
            cuda_graph=False,cpu_threads_per_process=1,source_mode='frozen_local_trace_replay',
            edge_serving_eq6_fallback_serve=True)
        for key,value in required.items():
            if self.cfg[key]!=value:raise ValueError('Unsupported live configuration: '+key)
        self.manifest=json.loads(self.manifest_path.read_text())
        if self.manifest['deadline_ns']!=self.cfg['deadline_ns']:raise ValueError('Deadline mismatch')
        if arm in LITERATURE_ARMS:
            from literature_baselines.current_literature import load_calibration
            literature=load_calibration(self.cfg['literature_calibration'])
            if literature['models_sha256']!=self.manifest['models_sha256']:
                raise ValueError('Calibration/model identity mismatch')
        os.environ['CUDA_VISIBLE_DEVICES']=','.join(self.cfg['gpu_uuids'])
        os.environ['EDGE_GPU_CARD_COUNT']='2'
        os.environ['EDGE_DUAL_CLOCK_MODE']='OFF'
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
        self.allow_peers=allow_idle_resident_peers
        self.profile_path=ROOT/self.cfg['batch_profile'];self.profile=BatchProfile(self.profile_path)
        self.mixed_profile=MixedFastProfile(ROOT/self.cfg['mixed_fast_profile']) if self.cfg.get('mixed_fast_batching') else None
        self.tasks={}
        self.ended=(DenseTerminalTracker(self.manifest['arrivals'])
                    if self.cfg.get('dense_terminal_tracking',False) else set())
        self.queued={};self.heaps={k:[] for k in KINDS}
        self.expiry_heap=[] # Used only for legacy FIFO; EDF reuses its existing heaps.
        self.batch_seq=0
        self.batches={};self.counts=Counter();self.history=deque();self.prefix_mix=deque();self.service_observations=deque()
        self.gpu_busy=[0.,0.]
        self.source_done=False;self.source_digest=hashlib.sha256();self.source_frames=0
        self.source_metrics={};self.worker_metrics=[];self.cpu=Counter()
        self.price_cache={};self.price_cache_stamp=None
        self.section_metrics={};self.response_batches=set();self.expected_returns=set();self.actual_returns=set()
        self.return_notifications={}
        self.cards=[dict(busy=False,target=p,effective=p,transition=False,pending=set(),pinned=[])
                    for p in ([1.,0.] if arm=='ours' else [.5,.5])]
        from quota_forecast_v8 import enable_service_debt
        for card in self.cards:enable_service_debt(card)
        self.configure_praa()

    def configure_praa(self):
        """Opt-in PRAA switches; both off reproduce the frozen executor exactly."""
        from realizable_admission import NeedTable,CudaEfficiency
        self.realizable_admission=bool(self.cfg.get('realizable_admission',False))
        if self.realizable_admission and self.cfg.get('batch_order')!='absolute_deadline':
            raise ValueError('Realizable admission requires per-kind absolute-deadline heaps')
        # Sensitivity knobs (opt-in; defaults reproduce the frozen windows and the median).
        scale=float(self.cfg.get('estimation_window_scale',1.))
        quantile=float(self.cfg.get('need_quantile',.5))
        if not scale>0:raise ValueError('estimation_window_scale must be positive')
        # Anchored need (opt-in): a cold batch size gets a conservative bound from the warm
        # sizes and the frozen calibration profile instead of no estimate (lock-in, Lemma 4/5).
        anchored=bool(self.cfg.get('anchored_need',False))
        if anchored and not self.realizable_admission:raise ValueError('Anchored need requires the M1 need table')
        # Safe-optimistic need (opt-in, 13b): subadditive upper value for safety, observed lower
        # value for choice; read only by the overload policy paths, median() stays observed-only.
        son=bool(self.cfg.get('safe_optimistic_need',False))
        if son and not self.realizable_admission:raise ValueError('Safe-optimistic need requires the M1 need table')
        if son and anchored:raise ValueError('Anchored need and safe-optimistic need are alternatives')
        self.need=(NeedTable(window=round(256*scale),quantile=quantile,anchor=self.need_anchor if anchored else None,
                             son_return_ns=self.cfg['network_return_ns'] if son else None)
                   if self.realizable_admission else None)
        self.need_pending={}
        if self.cfg.get('need_from_terminal',False) and self.need is None:
            raise ValueError('Terminal-measured need requires realizable admission')
        self.cuda_efficiency=(CudaEfficiency(window=round(512*scale)) if self.cfg.get('overload_unit_cost_allocation',False)
                              else None)
        # PRAA priority deferral: observed card turnaround (actor finish -> next dispatch).
        self.turnaround=deque(maxlen=round(256*scale)) if self.cfg.get('priority_deferral',False) else None
        self.last_finished=[None,None]
        if self.turnaround is not None and self.need is None:
            raise ValueError('Priority deferral requires the M1 observed need table')
        self.balanced_card_shares=bool(self.cfg.get('balanced_card_shares',False))
        if self.balanced_card_shares and getattr(self,'arm',None)=='ours':
            # Same initial total 1F/1A, spread over both cards.
            for card,p in zip(self.cards,self.card_targets(self.cfg['initial_fast'])):
                card.update(target=p,effective=p,transition=False)

    def need_anchor(self,kind,size):
        """Frozen calibration latency of a (kind, size) batch, the anchor's shape; None if absent."""
        try:
            if kind=='FAST_MIXED':
                mixed=getattr(self,'mixed_profile',None)
                if mixed is None:return None
                rows=[r['cuda_ms']['mean_ms'] for (b,u,m),r in mixed.rows.items() if b==size]
                return max(rows)*1e6 if rows else None
            return self.profile.latency(kind,size)
        except (KeyError,AttributeError):return None

    def record_need_warming(self,row):
        if row is None:return
        if 'lower_ns' in row:
            # SON: each warming tests S/M (upper >= observed) and M (lower <= observed).
            self.counts['praa_son_warmings']+=1
            if row['bound_ns'] is None:self.counts['praa_son_unbounded_warmings']+=1
            elif row['observed_ns']>row['bound_ns']:self.counts['praa_son_upper_violations']+=1
            if row['lower_ns'] is not None and row['observed_ns']<row['lower_ns']:
                self.counts['praa_son_lower_violations']+=1
            self.log('need_warmings',dict(row,now_ns=self.now()))
            return
        self.counts['praa_anchor_warmings']+=1
        if row['bound_ns'] is None:self.counts['praa_anchor_unbounded_warmings']+=1
        elif row['observed_ns']>row['bound_ns']:self.counts['praa_anchor_violations']+=1
        self.log('need_warmings',dict(row,now_ns=self.now()))

    def card_targets(self,fast):
        from resource_axis import card_shares,balanced_card_shares
        return (balanced_card_shares if getattr(self,'balanced_card_shares',False) else card_shares)(fast)

    def turnaround_estimate(self):
        """Median observed actor-finish -> next-dispatch gap; None while cold."""
        t=getattr(self,'turnaround',None)
        if t is None or len(t)<30:return None
        return sorted(t)[len(t)//2]

    def need_card(self,g):
        """Card key for the need table only when per-card observation is enabled."""
        return g if getattr(self,'cfg',{}).get('per_card_need',False) else None

    def batch_unrealizable(self,jobs,g=None):
        """Observed median dispatch->result+return; the frozen mean rule while cold."""
        need=getattr(self,'need',None)
        value=need.median(batch_kind(jobs),len(jobs),self.need_card(g)) if need is not None else None
        if value is None:
            return any(self.now()+self.batch_mean(jobs)+
                       self.cfg['network_return_ns']>j['deadline_ns'] for j in jobs)
        self.counts['praa_observed_need_checks']+=1
        return any(self.now()+value>j['deadline_ns'] for j in jobs)

    def now(self):return time.monotonic_ns()-self.epoch.value
    def log(self,name,row):self.logs.put((name,row))

    def measure(self,name,wall_start,cpu_start):
        wall=time.monotonic_ns()-wall_start;cpu=time.process_time_ns()-cpu_start
        row=self.section_metrics.setdefault(name,dict(calls=0,wall_ns=0,cpu_ns=0,max_wall_ns=0,
                                                      over_1ms=0,over_10ms=0))
        row['calls']+=1;row['wall_ns']+=wall;row['cpu_ns']+=cpu
        row['max_wall_ns']=max(row['max_wall_ns'],wall)
        row['over_1ms']+=wall>1000000;row['over_10ms']+=wall>10000000

    def receive_return(self,message):
        if message['type']=='ERROR':raise RuntimeError(message['error'])
        if message['type']=='RESPONSE_STOPPED':return
        if message['type']!='DELIVERED':raise ValueError('Unexpected response event')
        bid=message['batch_id']
        if bid in self.batches:
            if bid in self.return_notifications:raise ValueError('Repeated pending response')
            self.return_notifications[bid]=message
            return
        if bid in self.response_batches:raise ValueError('Repeated response batch')
        self.response_batches.add(bid)
        if not (message['ready_ns']<=message['received_ns']<=message['due_ns']<=message['returned_ns']):
            raise ValueError('Noncausal independent return')
        if message['due_ns']-message['received_ns']!=self.cfg['network_return_ns']:
            raise ValueError('Return delay changed')
        pending=getattr(self,'need_pending',{}).pop(bid,None)
        if pending is not None:
            kind,size,dispatched,card=pending
            self.record_need_warming(self.need.add(kind,size,message['returned_ns']-dispatched,card))
        for tid,pred in message['predictions'].items():
            tid=int(tid)
            if tid in self.actual_returns:raise ValueError('Repeated GPU prediction return')
            self.actual_returns.add(tid)
            self.finish(tid,message['role'],pred,message['returned_ns'],'gpu_completed')

    def _preflight(self):
        for name in ('task_plan','source_trace','bank_manifest','environment'):
            item=self.manifest[name]
            if sha(item['path'])!=item['sha256']:raise ValueError('Frozen dependency changed: '+name)
        status=[]
        for uuid in self.cfg['gpu_uuids']:
            line=subprocess.check_output(['nvidia-smi','--id='+uuid,
                '--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
            if int(line.rsplit(',',1)[1].strip().split()[0])!=0:
                raise RuntimeError('Selected GPU is actively executing work: '+line)
            status.append(line)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                      '--format=csv,noheader'],text=True)
        peers=[line for line in apps.splitlines() if any(uuid in line for uuid in self.cfg['gpu_uuids'])]
        if peers and not self.allow_peers:raise RuntimeError('Resident peers require --allow-idle-resident-peers')
        identity=dict(schema='real_gpu_batch_arm_v1',arm=self.arm,config=self.cfg,
            manifest_path=str(self.manifest_path.resolve()),
            config_sha256=sha(self.cfg_path),manifest_sha256=sha(self.manifest_path),
            frozen_dependencies={k:self.manifest[k] for k in ('task_plan','source_trace','bank_manifest','environment')},
            calibration_sha256=sha(self.profile_path),gpu_preflight=status,resident_peers=peers,
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_sha256={p:sha(ROOT/p) for p in ('live_batch_runtime.py','live_response.py','live_batch_policies.py','batch_inference.py','dual_clock_state.py','queue_feedback_controller.py','live_batch_statistics.py','mixed_fast_batch.py','net_gain_control.py','tools/run_live_batch.py','tools/analyze_live_batch.py')},
            mixed_calibration_sha256=sha(ROOT/self.cfg['mixed_fast_profile']) if self.mixed_profile else None,
            batch_order=self.cfg.get('batch_order','original_fifo'),response_mode='independent_actual_timer',
            host_gather_mode='direct_index_select_into_existing_pinned_buffer' if self.cfg.get('direct_host_gather',False) else 'temporary_index_then_copy',
            batch_rule=('Absolute deadline order within each role, merged Fast Prefix; original guards'
                        if self.cfg.get('batch_order')=='absolute_deadline' else
                        'Fast generation FIFO across known/unknown rows, shared Prefix; Accurate enqueue FIFO; original guards'),
            actual_gpu_execution=True,cached_gpu_predictions_used=False,labels_visible_online=False,
            cpu_delays_zeroed=False,cuda_graph=False,source='frozen Local execution and upload trace',
            original_edge_serving_reproduction=False)
        if self.arm in LITERATURE_ARMS:
            identity.update(literature_method=self.arm,
                comparison_scope='Conditional edge-side method; common frozen Local outputs, not full-architecture reproduction',
                resource_geometry='Two physical GPUs, both CNNs resident; no CLRA role scorecard for literature methods',
                literature_calibration_sha256=sha(self.cfg['literature_calibration']),
                literature_adapter_sha256=sha(ROOT/'literature_baselines/current_literature.py'))
        write_json(self.output/'identity.json',identity)

    def public_jobs(self,role):
        return [j for j in self.queued.values() if j['role']==role]

    def batch_order_key(self,job):
        # Absolute deadlines are public original-generation deadlines. No
        # controller/share override, future quality lookup or aging clock.
        if self.cfg.get('batch_order')=='absolute_deadline':
            return job['deadline_ns'],job['task_id']
        return (job['enqueue_ns'] if job['role']=='Accurate' else job['arrival_ns']),job['task_id']

    def _kind_jobs(self,kind,limit=None):
        # Peek a bounded live frontier, never sort the complete waiting queue.
        heap=self.heaps[kind];front=[];limit=limit or self.cfg['batch_max']
        while heap and len(front)<limit:
            item=heapq.heappop(heap)
            if self.queued.get(item[1]) is item[2]:front.append(item)
        for item in front:heapq.heappush(heap,item)
        return [item[2] for item in front]

    def peek_kind_batch(self,kind):return self._kind_jobs(kind)

    def peek_batch(self,role):
        if role=='Accurate':return self._kind_jobs('ACCURATE')
        groups=[self._kind_jobs(k) for k in ('FAST_FULL','FAST_PREFIX')]
        if self.cfg.get('mixed_fast_batching'):
            return sorted(groups[0]+groups[1],key=self.batch_order_key)[:self.cfg['batch_max']]
        groups=[jobs for jobs in groups if jobs]
        if not groups:return []
        jobs=min(groups,key=lambda a:self.batch_order_key(a[0]))
        other=[a[0] for a in groups if a is not jobs]
        limit=min(self.batch_order_key(j) for j in other) if other else (math.inf,math.inf)
        return [j for j in jobs if self.batch_order_key(j)<limit]

    def batch_mean(self,jobs,clock='mean_ms'):
        kind=batch_kind(jobs)
        if kind=='FAST_MIXED':return self.mixed_profile.mean(jobs,clock)
        return self.profile.latency(kind,len(jobs),clock)

    def control_unit_ns(self,kind):
        if kind=='FAST_SUFFIX':return 0. # Prefix is one atomic selective batch contract.
        stamp=self.now()//100000000 if hasattr(self,'epoch') and self.epoch.value else -1
        if stamp!=self.price_cache_stamp:
            self.price_cache_stamp=stamp;self.price_cache={}
        if kind in self.price_cache:return self.price_cache[kind]
        observed=[(n,unit,mixed) for t,k,n,unit,mixed in self.service_observations if k==kind]
        if observed and (self.cfg.get('batch_work_conserving_prices',False) or any(mixed for n,unit,mixed in observed)):
            value=sum(n*unit for n,unit,mixed in observed)/sum(n for n,unit,mixed in observed)
            self.price_cache[kind]=value
            return value
        recent=[(b,n) for t,k,b,n in self.history if k==kind]
        b=max(1,min(10,round(sum(n for b,n in recent)/len(recent)))) if recent else 1
        if kind=='FAST_PREFIX':
            known=[r for t,r in self.prefix_mix]
            pf=known.count('Fast')/len(known) if known else 1.
            value=self.profile.latency(kind,b,survivors=round(pf*b))/b
        else:value=self.profile.mean(kind,b)/b
        self.price_cache[kind]=value
        return value

    def single_costs(self):
        return {k:(0. if k=='FAST_SUFFIX' else self.profile.mean(k,1))
                for k in ('FAST_FULL','FAST_PREFIX','FAST_SUFFIX','ACCURATE')}

    def enqueue(self,tid,kind,now):
        if tid in self.ended or tid in self.queued:raise ValueError('Duplicate enqueue')
        task=self.tasks[tid];role='Accurate' if kind=='ACCURATE' else 'Fast'
        task['edge_published']=True
        task['intended_role']='Prefix_unknown' if kind=='FAST_PREFIX' else role
        job=dict(task_id=tid,kind=kind,role=role,arrival_ns=task['arrival_ns'],
                 deadline_ns=task['deadline_ns'],bank_index=task['bank_index'],enqueue_ns=now)
        self.queued[tid]=job
        order=self.batch_order_key(job)[0]
        heapq.heappush(self.heaps[kind],(order,tid,job))
        if self.cfg.get('batch_order')!='absolute_deadline':
            heapq.heappush(self.expiry_heap,(job['deadline_ns'],tid,job))
        self.counts['enqueued_'+kind]+=1
        if self.arm=='ours':
            self.policy.demand.wait_counts[role][kind]+=1
            self.policy.demand.offered(kind,now)

    def remove(self,job,loss=False,prune_heap=True):
        tid=job['task_id'];self.queued.pop(tid)
        heap=self.heaps[job['kind']]
        while prune_heap and heap and self.queued.get(heap[0][1]) is not heap[0][2]:heapq.heappop(heap)
        if self.arm=='ours':
            self.policy.demand.wait_counts[job['role']][job['kind']]-=1
            if loss:self.policy.demand.resource_loss_counts[job['role']][job['kind']]+=1

    def finish(self,tid,role,prediction,at,reason):
        if tid in self.ended:raise ValueError('Repeated terminal result')
        task=self.tasks[tid];self.ended.add(tid)
        if getattr(getattr(self,'policy',None),'allocator',None) is not None:self.policy.allocator.revealed.pop(tid,None)
        row=dict(task_id=tid,bank_index=task['bank_index'],origin_node=task['origin_node'],
            arrival_ns=task['arrival_ns'],deadline_ns=task['deadline_ns'],
            terminal_ns=int(at),role=role,prediction=prediction,reason=reason,
            intended_role=task.get('intended_role',role),edge_published=task.get('edge_published',False),
            ontime=prediction is not None and at<=task['deadline_ns'])
        self.counts['terminal_'+reason]+=1
        self.log('tasks',row)

    def request_shares(self,fast,now):
        for g,p in enumerate(self.card_targets(fast)):
            c=self.cards[g];c['target']=p;c['transition']=abs(c['effective']-p)>1e-9
            if not c['busy']:
                c['effective']=p;c['transition']=False
                self.log('control',dict(event='CARD_EFFECTIVE',gpu=g,now_ns=now,fast_share=p))

    def receive_source(self,raw,sent):
        now=self.now();frame=json.loads(raw);self.source_digest.update(raw);self.source_frames+=1
        # Detailed lag statistics are computed by the separate offline program.
        self.log('receipts',dict(frame_id=frame.get('frame_id'),task_id=frame.get('task_id'),
            outcome=frame['outcome'],planned_ns=frame['source_publish_ns'],sent_ns=sent,received_ns=now))
        self.policy.observe(frame,now)
        outcome=frame['outcome'];tid=frame.get('task_id')
        if outcome=='SHARED_ARRIVAL':
            if tid in self.tasks:raise ValueError('Repeated original task')
            self.tasks[tid]={k:frame[k] for k in ('arrival_ns','deadline_ns','bank_index','origin_node')}
        elif outcome=='COMMON_LOCAL_TERMINAL':
            self.finish(tid,'Local',frame['prediction'],frame['terminal_available_ns'],'frozen_local_completion')
        elif outcome=='COMMON_ABSTAIN':
            self.finish(tid,'Abstain',None,frame['source_publish_ns'],'source_quality_reject')
        elif outcome=='GPU_READY':
            roles=frame.get('gpu_candidates') or []
            if not roles:raise ValueError('GPU_READY lacks public route')
            kind=(self.policy.input_kind(frame,now) if hasattr(self.policy,'input_kind')
                  else frame['gpu_candidate_kinds'][roles[0]])
            if kind is None:self.finish(tid,'Abstain',None,now,'literature_mapping_unassigned')
            else:self.enqueue(tid,kind,now)
        elif outcome=='PREFIX_READY':
            kind=self.policy.input_kind(frame,now) if hasattr(self.policy,'input_kind') else 'FAST_PREFIX'
            if kind is None:self.finish(tid,'Abstain',None,now,'literature_mapping_unassigned')
            else:self.enqueue(tid,kind,now)

    def retarget_public_job(self,job,kind,now):
        """Literature client remapping of waiting work only; preserve its age."""
        if self.arm not in ('jellyfish_edge','slackfit_edge','ramsis_edge'):
            raise ValueError('Waiting-job remapping is Jellyfish/SlackFit/RAMSIS-only')
        if kind==job['kind']:return
        tid=job['task_id'];self.remove(job)
        # Repeated A->F->A remapping can otherwise leave equal (age,task_id)
        # tombstones that make heapq compare two dict payloads. This optional
        # adapter path removes that task's old tombstones explicitly.
        for heap_kind in set((job['kind'],kind))-{None}:
            self.heaps[heap_kind]=[entry for entry in self.heaps[heap_kind] if entry[1]!=tid]
            heapq.heapify(self.heaps[heap_kind])
        if kind is None:
            self.finish(tid,'Abstain',None,now,'literature_mapping_unassigned');return
        replacement=dict(job,kind=kind,role='Fast' if kind=='FAST_FULL' else 'Accurate')
        self.queued[tid]=replacement
        heapq.heappush(self.heaps[kind],(self.batch_order_key(replacement)[0],tid,replacement))
        self.log('literature',dict(event='WAITING_JOB_REMAPPED',task_id=tid,
            now_ns=now,from_kind=job['kind'],to_kind=kind,original_enqueue_ns=job['enqueue_ns']))

    def receive_gpu(self,message):
        now=self.now();typ=message['type']
        if typ=='ERROR':raise RuntimeError(message['error'])
        if typ=='STOPPED':self.worker_metrics.append(message);return
        g=message['gpu'];batch=self.batches[message['batch_id']]
        if typ=='EDGEBOOST_ROUTES':
            if self.arm!='edgeboost_edge':raise ValueError('Unexpected margin-cascade message')
            for row in message['rows']:
                tid=row['task_id'];self.tasks[tid]['intended_role']=row['role']
                self.log('quality',dict(row,now_ns=now,gpu=g,batch_id=message['batch_id'],
                    published_ns=message['published_ns']))
                if row['role']=='Accurate':self.enqueue(tid,'ACCURATE',now)
            return
        if typ=='PREFIX_ROUTES':
            for row in message['rows']:
                tid=row['task_id'];role=row['role'];task=self.tasks[tid]
                task['intended_role']=role
                self.log('quality',dict(row,now_ns=now,gpu=g,batch_id=message['batch_id'],
                                       published_ns=message['published_ns']))
                self.prefix_mix.append((now,role))
                if self.arm in ('ours','wma_node'):
                    self.policy.observe(dict(task_id=tid,arrival_ns=task['arrival_ns'],
                        outcome='PREFIX_QUALITY_REVEALED',quality_role=role,source_publish_ns=now),now)
                if role=='Accurate':self.enqueue(tid,'ACCURATE',now)
                elif role=='Abstain':self.finish(tid,'Abstain',None,now,'gpu_quality_reject')
            return
        if typ!='BATCH_DONE':raise ValueError('Unexpected GPU actor message')
        # Read-only for policies that need actor stamps (PARD batch wait); no other use.
        self.last_gpu_message=message
        self.policy.complete(g,batch['booking'],message['cuda_service_ns'])
        self.cards[g]['busy']=False
        if getattr(self,'turnaround',None) is not None:self.last_finished[g]=message['finished_ns']
        if self.cards[g]['transition']:
            self.cards[g]['effective']=self.cards[g]['target'];self.cards[g]['transition']=False
        self.policy.effective(now)
        self.gpu_busy[g]+=message['cuda_service_ns']
        if message['actual_batch']:
            # Public executor clocks only: dispatch, actual GPU result, configured return.
            need=getattr(self,'need',None);efficiency=getattr(self,'cuda_efficiency',None)
            if need is not None and self.cfg.get('need_from_terminal',False):
                # Measured at the actual independent return (the on-time clock), so the
                # return-path overshoot beyond the configured delay is not omitted.
                if message['predictions']:
                    self.need_pending[message['batch_id']]=(message['kind'],message['actual_batch'],
                                                             batch['dispatch_ns'],self.need_card(g))
            elif need is not None:
                self.record_need_warming(need.add(message['kind'],message['actual_batch'],
                         message['finished_ns']-batch['dispatch_ns']+self.cfg['network_return_ns'],
                         self.need_card(g)))
            if efficiency is not None:
                efficiency.add(message['cuda_service_ns'],now-batch['dispatch_ns'])
        if getattr(self.policy,'allocator',None) is not None:
            overhead=max(0,message['finished_ns']-batch['dispatch_ns']-message['cuda_service_ns'])
            self.policy.allocator.dispatch_overheads.append(overhead)
            if self.cfg.get('marginal_cycle_capacity',False):
                self.policy.allocator.observe_cycle(batch['jobs'][0]['role'],
                    batch['dispatch_ns'],now,message['cuda_service_ns'])
        self.history.append((now,message['kind'],len(batch['jobs']),message['actual_batch']))
        b=message['actual_batch']
        if b:
            if message['kind']=='FAST_MIXED':
                u=message['mixed_unknown'];m=message['mixed_unknown_fast']
                # Prestart refusals can leave a homogeneous sub-batch.
                if 0<u<b:
                    known,unknown=self.mixed_profile.units(b,u,m)
                    self.service_observations.extend(((now,'FAST_FULL',b-u,known,True),
                                                     (now,'FAST_PREFIX',u,unknown,True)))
                else:
                    k='FAST_PREFIX' if u else 'FAST_FULL'
                    unit=self.profile.latency(k,b,survivors=m if u else None)/b
                    self.service_observations.append((now,k,b,unit,False))
            else:
                k=message['kind'];survivors=len(message['fast_suffix_task_ids']) if k=='FAST_PREFIX' else None
                self.service_observations.append((now,k,b,self.profile.latency(k,b,survivors=survivors)/b,False))
        for tid in message['prestart_refused']+message['suffix_refused']:
            self.finish(tid,'Abstain',None,now,'prestart_inference_budget')
            if self.arm=='ours':
                job=next(j for j in batch['jobs'] if j['task_id']==tid)
                self.policy.demand.resource_loss_counts[job['role']][job['kind']]+=1
        ids=set(map(int,message['predictions']))
        if self.expected_returns & ids:raise ValueError('Repeated prediction generated')
        self.expected_returns.update(ids)
        self.log('batches',dict(message,received_ns=now,dispatch_ns=batch['dispatch_ns'],
            members=[j['task_id'] for j in batch['jobs']],selection=batch['selection'],
            service_booking=batch['booking'],
            score_after=dict(self.cards[g]['_service_debt']['balance']) if self.arm=='ours' else None))
        del self.batches[message['batch_id']]
        returned=self.return_notifications.pop(message['batch_id'],None)
        if returned is not None:self.receive_return(returned)

    def purge_expired(self,stop):
        """Drain actually expired waiting jobs before policy scoring, in a slice.

        No queue ETA, service estimate, quality or future-route information is
        consulted. Offered demand is retained; these are real queue losses.
        Return False if housekeeping exhausted this slice: do not score yet.
        """
        start=time.process_time_ns()
        try:
            heaps=(self.heaps.values() if self.cfg.get('batch_order')=='absolute_deadline'
                   else (self.expiry_heap,))
            for kind,heap in (self.heaps.items() if self.cfg.get('batch_order')=='absolute_deadline'
                              else ((None,h) for h in heaps)):
                # PRAA M1: a job that cannot finish even in the fastest observed
                # compatible batch is released now instead of consuming a GPU.
                need=getattr(self,'need',None)
                floor=need.floor(kind) if need is not None else None
                while heap:
                    if time.monotonic_ns()>=stop:return False
                    _,tid,job=heap[0]
                    if self.queued.get(tid) is not job:
                        heapq.heappop(heap)
                        continue
                    now=self.now()
                    if job['deadline_ns']>now:
                        if floor is None or job['deadline_ns']>=now+floor:break
                        reason='queue_unrealizable_budget'
                    else:reason='queue_deadline_expired'
                    heapq.heappop(heap)
                    self.remove(job,loss=True,prune_heap=False)
                    self.finish(tid,'Abstain',None,now,reason)
            return True
        finally:
            self.cpu['expired_cleanup_ns']+=time.process_time_ns()-start

    def schedule(self):
        if not self.queued:return
        stop=time.monotonic_ns()+self.cfg['main_slice_ns']
        if not self.purge_expired(stop):return
        if all(c['busy'] for c in self.cards):return
        for _ in range(100):
            if time.monotonic_ns()>=stop:return
            if not self.purge_expired(stop):return
            now=self.now();start=time.process_time_ns();choice=self.policy.choose(now)
            self.cpu['dispatch_policy_ns']+=time.process_time_ns()-start
            if choice is None:return
            g,batch,evidence=choice
            # Only a selected batch is checked. Shrinking leaves the rest queued.
            while batch and self.batch_unrealizable(batch,g):
                if len(batch)>1:batch=batch[:-1]
                else:
                    job=batch[0];self.remove(job,loss=True)
                    self.finish(job['task_id'],'Abstain',None,self.now(),'prestart_inference_budget')
                    batch=[]
            if not batch:continue
            booking=self.policy.book(g,batch,evidence)
            for job in batch:
                self.remove(job)
            self.batch_seq+=1;bid=self.batch_seq;now=self.now()
            self.batches[bid]=dict(jobs=batch,booking=booking,dispatch_ns=now,selection=evidence,gpu=g)
            if getattr(self,'turnaround',None) is not None and self.last_finished[g] is not None:
                self.turnaround.append(now-self.last_finished[g]);self.last_finished[g]=None
            self.cards[g]['busy']=True
            self.commands[g].put(dict(batch_id=bid,jobs=batch))
            self.log('dispatch',dict(batch_id=bid,gpu=g,now_ns=now,
                kind=batch_kind(batch),members=[j['task_id'] for j in batch],
                public_jobs=batch,
                fast_share=self.cards[g]['effective'],selection=evidence))
            if all(c['busy'] for c in self.cards):return

    def schedule_after_receives(self):
        """R007 common optional execution order; original scheduler unchanged."""
        if not self.cfg.get('dispatch_before_source',False):return
        if not self.queued or all(c['busy'] for c in self.cards):return
        wall=time.monotonic_ns();cpu=time.process_time_ns()
        self.schedule()
        self.measure('post_receive_schedule',wall,cpu)

    def run(self):
        import torch
        from tools.validate_prefix_pipeline_gpu import prepare_inputs
        from live_batch_policies import OursBatchPolicy,EdgeServingBatchPolicy
        setup_start=time.monotonic_ns();cpu_setup=time.process_time_ns()
        self._preflight()
        torch.set_num_threads(1)
        _,_,bank=prepare_inputs(self.manifest_path,None)
        os.sched_setaffinity(0,{self.cfg['controller_cpu']})
        ctx=mp.get_context('spawn');self.epoch=ctx.Value('q',0)
        self.logs=ctx.Queue();self.inputs=ctx.Queue();self.results=ctx.Queue()
        self.response_input=ctx.Queue();self.delivered=ctx.Queue()
        self.commands=[ctx.Queue() for _ in range(2)]
        logger=ctx.Process(target=log_worker,args=(str(self.output),self.logs,self.cfg['logger_cpu'],self.epoch))
        workers=[ctx.Process(target=gpu_worker,args=(g,bank,self.commands[g],self.results,self.response_input,
                    self.epoch,self.cfg,str(self.profile_path),str(self.output))) for g in range(2)]
        responder=ctx.Process(target=response_worker,args=(self.response_input,self.delivered,
                    self.logs,self.epoch,self.cfg,str(self.output)))
        feeder=ctx.Process(target=source_worker,args=(self.manifest['source_trace']['path'],
                    self.inputs,self.epoch,self.cfg['source_cpu'],str(self.output)))
        self.processes=[logger,responder]+workers+[feeder]
        for process in self.processes:process.start()
        if self.arm in LITERATURE_ARMS:
            from literature_baselines.current_literature import create_policy
            self.policy=create_policy(self)
        else:self.policy=OursBatchPolicy(self) if self.arm=='ours' else EdgeServingBatchPolicy(self)
        ready=[];start=time.monotonic()
        try:
            while len(ready)<2:
                msg=self.results.get(timeout=120)
                if msg['type']=='ERROR':raise RuntimeError(msg['error'])
                if msg['type']!='READY':raise ValueError('Unexpected setup actor message')
                ready.append(msg)
            write_json(self.output/'actors.json',ready)
            setup_wall=(time.monotonic_ns()-setup_start)/NS
            setup_cpu=(time.process_time_ns()-cpu_setup)/NS
            self.epoch.value=time.monotonic_ns()+NS
            while self.now()<0:time.sleep(.001)
            cpu_run=time.process_time_ns();self.parent_cpu_baseline=cpu_snapshot();tick=0;last_progress=-1
            pending_source=deque()
            import gc
            defer_gc=self.cfg.get('defer_parent_gc',False)
            gc_enabled_before=configure_replay_gc(gc,defer_gc)
            gc_start=[];gc_metrics=dict(calls=0,total_wall_ns=0,max_wall_ns=0,over_1ms=0)
            gc_metrics.update(deferred=defer_gc,long_events=[],parent_peak_rss_kb_start=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            def gc_pause(phase,info):
                if phase=='start':gc_start.append(time.monotonic_ns())
                elif gc_start:
                    started_gc=gc_start.pop();finished_gc=time.monotonic_ns()
                    duration=finished_gc-started_gc
                    gc_metrics['calls']+=1;gc_metrics['total_wall_ns']+=duration
                    gc_metrics['max_wall_ns']=max(gc_metrics['max_wall_ns'],duration)
                    gc_metrics['over_1ms']+=duration>1000000
                    if duration>1000000:
                        gc_metrics['long_events'].append(dict(start_ns=started_gc-self.epoch.value,
                            end_ns=finished_gc-self.epoch.value,generation=info.get('generation')))
            gc.callbacks.append(gc_pause)
            while True:
                progressed=False
                # Response delivery is independent; consume acknowledgements
                # first. Bookkeeping may wait for BATCH_DONE, never the return.
                for inbox,handler,name in ((self.delivered,self.receive_return,'return_ack'),
                                            (self.results,self.receive_gpu,'gpu_receive')):
                    wall=time.monotonic_ns();cpu=time.process_time_ns()
                    for _ in range(32):
                        try:msg=inbox.get_nowait()
                        except queue.Empty:break
                        handler(msg);progressed=True
                        if time.monotonic_ns()-wall>=self.cfg['main_slice_ns']:break
                    self.measure(name,wall,cpu)
                self.schedule_after_receives()
                wall=time.monotonic_ns();cpu=time.process_time_ns()
                while time.monotonic_ns()-wall<self.cfg['main_slice_ns']:
                    if not pending_source:
                        try:typ,payload=self.inputs.get_nowait()
                        except queue.Empty:break
                        if typ=='ERROR':raise RuntimeError(payload)
                        if typ=='END':
                            self.source_done=True;self.source_metrics=payload
                        else:pending_source.extend(payload)
                        progressed=True
                    if pending_source:
                        raw,sent=pending_source.popleft();self.receive_source(raw,sent)
                self.measure('source_receive',wall,cpu)
                now=self.now()
                if now>=tick:
                    wall=time.monotonic_ns();before=time.process_time_ns()
                    while self.history and self.history[0][0]<now-2*NS:self.history.popleft()
                    while self.service_observations and self.service_observations[0][0]<now-2*NS:self.service_observations.popleft()
                    while self.prefix_mix and self.prefix_mix[0][0]<now-2*NS:self.prefix_mix.popleft()
                    self.policy.tick(now)
                    self.cpu['control_ns']+=time.process_time_ns()-before
                    tick=(now//100000000+1)*100000000
                    self.log('queues',dict(now_ns=now,Fast=len(self.public_jobs('Fast')),
                        Accurate=len(self.public_jobs('Accurate')),effective=[c['effective'] for c in self.cards]))
                    self.measure('control_tick',wall,before)
                wall=time.monotonic_ns();cpu=time.process_time_ns()
                if getattr(self.policy,'emergency_enabled',False):self.policy.maybe_emergency(self.now())
                self.schedule();self.measure('schedule',wall,cpu)
                seconds=int(now/NS)
                if seconds//10!=last_progress:
                    last_progress=seconds//10
                    self.log('progress',dict(now_s=now/NS,arrived=len(self.tasks),
                        ended=len(self.ended),queued=len(self.queued),batches=self.batch_seq))
                if (self.source_done and not pending_source and not self.queued and not self.batches
                        and len(self.ended)==len(self.tasks)):
                    if self.return_notifications or self.actual_returns!=self.expected_returns:
                        raise ValueError('Independent response reconciliation failed')
                    break
                if time.monotonic()-start>self.cfg['wall_cap_s']:raise TimeoutError('Arm wall cap')
                for process in self.processes:
                    if process.exitcode not in (None,0):raise RuntimeError('Runtime child failed')
                if not progressed:time.sleep(.0002)
            gc.callbacks.remove(gc_pause)
            self.gc_metrics=gc_metrics
            self.cpu['parent_steady_ns']=time.process_time_ns()-cpu_run
            self.virtual_end=self.now()
            self.policy.close()
            for command in self.commands:command.put(None)
            stopped=0
            while stopped<2:
                msg=self.results.get(timeout=30)
                if msg['type']=='ERROR':raise RuntimeError(msg['error'])
                self.receive_gpu(msg);stopped+=msg['type']=='STOPPED'
            for process in workers+[feeder]:process.join(timeout=30)
            self.response_input.put(None)
            self.response_input.close();self.response_input.join_thread()
            response_stop=self.delivered.get(timeout=30)
            if response_stop['type']!='RESPONSE_STOPPED':raise ValueError('Undrained response')
            responder.join(timeout=30)
            self.logs.put(None);logger.join(timeout=60)
            # Include the controller's asynchronous JSON/command queue writers.
            self.logs.close();self.logs.join_thread()
            for command in self.commands:command.close();command.join_thread()
            if any(p.exitcode!=0 for p in self.processes):raise RuntimeError('Unclean child shutdown')
            if self.source_digest.hexdigest()!=self.manifest['source_trace']['sha256']:
                raise ValueError('Received source SHA differs from frozen stream')
            if self.source_metrics['sha256']!=self.source_digest.hexdigest():raise ValueError('Source delivery mismatch')
            if len(self.tasks)!=self.manifest['arrivals']:raise ValueError('Original task count mismatch')
            if defer_gc:
                # All roots are terminal and child queues drained. Keep this
                # collection's full CPU inside the six-process final accounting.
                gc_wall=time.monotonic_ns();gc_cpu=time.process_time_ns()
                collected=gc.collect()
                self.cpu['deferred_gc_ns']=time.process_time_ns()-gc_cpu
                self.gc_metrics['final_collection']=dict(collected=collected,
                    wall_ns=time.monotonic_ns()-gc_wall,cpu_ns=self.cpu['deferred_gc_ns'])
            restore_replay_gc(gc,gc_enabled_before)
            self.gc_metrics['parent_peak_rss_kb_end']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return self.execution_summary(setup_wall,setup_cpu)
        except Exception:
            write_json(self.output/'failure.json',dict(error=traceback.format_exc(),elapsed_s=time.monotonic()-start))
            raise
        finally:
            if 'gc_pause' in locals() and gc_pause in gc.callbacks:gc.callbacks.remove(gc_pause)
            if 'gc_enabled_before' in locals():restore_replay_gc(gc,gc_enabled_before)
            for process in self.processes:
                if process.is_alive():process.terminate();process.join(timeout=5)

    def execution_summary(self,setup_wall,setup_cpu):
        # No labels, TCA, quantiles or paired statistics in the execution process.
        save_process_cpu(self.output,'controller',self.parent_cpu_baseline)
        process_names=('controller','source','gpu0','gpu1','logger','response')
        if self.arm in PLANNER_ARMS:process_names+=('literature_planner',)
        process_cpu={name:json.loads((self.output/(name+'_process_cpu.json')).read_text())
                     for name in process_names}
        steady=sum(row['replay_and_drain_s'] for row in process_cpu.values())
        parent=resource.getrusage(resource.RUSAGE_SELF)
        children=resource.getrusage(resource.RUSAGE_CHILDREN)
        lifetime=parent.ru_utime+parent.ru_stime+children.ru_utime+children.ru_stime
        metrics=dict(schema='real_gpu_batch_execution_v3',arm=self.arm,
            terminal_tracking_mode=('dense_preallocated' if self.cfg.get('dense_terminal_tracking',False) else 'original_set'),
            arrivals=len(self.tasks),batches=self.batch_seq,
            terminal_reasons=dict(self.counts),elapsed_steady_s=self.virtual_end/NS,
            gpu_timeline_service_s=[v/NS for v in self.gpu_busy],
            cpu_segments_s={k:v/NS for k,v in self.cpu.items()},
            cpu_process_accounting=process_cpu,
            section_timing=self.section_metrics,gc_pauses=self.gc_metrics,
            independent_return_count=len(self.actual_returns),
            response_cpu_s=process_cpu['response']['replay_and_drain_s'],
            cpu_all_processes_steady_s=steady,cpu_all_processes_replay_and_drain_s=steady,
            cpu_all_processes_lifetime_s=lifetime,
            source_cpu_s=process_cpu['source']['replay_and_drain_s'],
            gpu_actor_cpu_s=[process_cpu['gpu'+str(g)]['replay_and_drain_s'] for g in range(2)],
            logger_cpu_s=process_cpu['logger']['replay_and_drain_s'],
            setup_parent_wall_s=setup_wall,setup_parent_cpu_s=setup_cpu,
            source_observation_cpu_s=(self.policy.governor.source_event_cpu_ns/NS if self.arm=='ours' else 0.),
            algorithm_counters=(dict(self.policy.controller.counters) if self.arm=='ours' else
                                dict(choose_action_calls=self.policy.calls)),
            source_trace_sha256=self.source_digest.hexdigest(),source_frames=self.source_frames,
            all_tasks_unique_terminal=True,online_processes_stopped=True,
            cpu_overhead_in_latency=True,local_is_frozen_trace=True,gpu_predictions_are_real=True,
            cpu_accounting_scope='All recorded execution process user+system totals including every background thread and IPC/log drain; Jellyfish and Proteus include their separate seventh CPU planner. Nested timing segments are diagnostics, never added twice. Lifetime includes setup and waited helper subprocesses; excludes separate statistics.',
            software_delay_mode='actual_wall_clock',
            statistical_scope='Outcome-conditioned image pools. Local/upload are frozen trace; every edge CNN prediction is real eager GPU execution.')
        write_json(self.output/'execution.json',metrics)
        return metrics
