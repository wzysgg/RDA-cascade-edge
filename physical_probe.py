"""Actual frozen CNN execution on exactly two CUDA devices; no service replay."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import sys,json,time,csv,hashlib,subprocess,multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
from PIL import Image
PHASE=Path('/home/wangzengyu/edge/clra_phaseA')
ASSETS=Path('/home/wangzengyu/edge_paper/assets')
TASK17=Path('/home/wangzengyu/edge_paper/experiments/task17_four_quality_isolated_rebuild_20261002')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(PHASE.parent))
from clra_phaseA.models.model_factory import create_model
from clra_phaseA.models.source_space_protocol import apply_source_space_intervention,build_post_intervention_preprocess
from clra_phaseA.models.dataset import load_rows,CLASSES
from clra_phaseA.router.generate_router_dataset import image_quality_features
CHECKPOINTS={'Local':ASSETS/'models/local.pt','Fast':ASSETS/'models/fast.pt','Accurate':ASSETS/'models/accurate.pt'}
MODELS={'Local':'mobilenet_v3_small','Fast':'resnet18','Accurate':'convnext_tiny'}
UUIDS=['GPU-d1b82bfe-73db-7d85-16fa-2e38208c0064','GPU-7e400790-5792-8095-35cd-42d68b8eeffd']
# T010: a two-entry physical UUID mask selects its own disjoint pair.  The mask
# must be set before the first CUDA call; importing torch alone does not create
# a context, so an env-derived UUID list keeps N9/T009 behaviour unchanged.
_mask=[u.strip() for u in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if u.strip()]
if len(_mask) in (1,2,4) and all(u.startswith('GPU-') for u in _mask):UUIDS=_mask

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def configure(cpu=None):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if cpu is not None:os.sched_setaffinity(0,{cpu})
    torch.backends.cudnn.benchmark=False
def actor_cpu(name,default):
    """T012: two concurrent groups must sit on mutually disjoint spare cores.

    ``name`` may be a single environment variable or an ordered tuple of
    aliases; the first one that is present wins.  The tuple form lets T012
    export its own variable names while still honouring an inherited T011
    export, so the T011 defaults are never silently lost.
    """
    for key in (name if isinstance(name,(list,tuple)) else (name,)):
        raw=os.environ.get(key)
        if raw is None or raw=='':continue
        try:return int(raw)
        except (TypeError,ValueError):return int(default)
    return int(default)
def model(role,device):
    m=create_model(MODELS[role],num_classes=6,pretrained=False)
    saved=torch.load(CHECKPOINTS[role],map_location='cpu',weights_only=True)
    m.load_state_dict(saved['state_dict'],strict=True)
    return m.to(device).eval()
def fast_prefix(m,x):return m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x)))))
def fast_suffix(m,z):return m.fc(torch.flatten(m.avgpool(m.layer4(m.layer3(m.layer2(z)))),1))
def local_suffix(m,z,mid2=False):
    z=m.features[9 if mid2 else 5:](z)
    return m.classifier(torch.flatten(m.avgpool(z),1))

class Heads:
    """Actual CPU calls on live stage vectors; never an identity lookup."""
    roles={'MID1':['Local','Fast','Accurate'],'MID2':['Local','Fast','Accurate'],'EARLY':['Fast','Accurate','Rescue']}
    def __init__(self):
        self.models={};self.files={};self.fit_sources=set()
        for p in sorted((TASK17/'models').glob('*.json')):
            d=json.loads(p.read_text());m=d['metadata']
            if m['chain_id'].endswith('_final') and m['inner_group'] is None:
                self.models[m['stage'],m['target'],m['kind']]=d;self.files[p.name]=sha(p)
                self.fit_sources.update(m.get('fit_source_ids',[]))
        assert len(self.models)==18
    def one(self,d,x):
        if d['model_kind']=='constant_model':return float(d['constant_probability'])
        xx=np.r_[1.,(np.asarray(x)-np.asarray(d['mean']))/np.asarray(d['std'])]
        return float(1/(1+np.exp(-np.clip(xx@np.asarray(d['beta']),-35,35))))
    def predict(self,stage,vector,quality=None):
        x=np.asarray(vector,dtype=float)
        if stage!='EARLY':x=np.r_[x,quality]
        assert x.shape==({'MID1':43,'MID2':51,'EARLY':64}[stage],)
        result={}
        for role in self.roles[stage]:
            p=np.clip(self.one(self.models[stage,role,'base'],x),1e-12,1-1e-12)
            result[role]=self.one(self.models[stage,role,'calibrator'],[np.log(p/(1-p))])
        return result

def inputs(split='DEVELOPMENT',n=12):
    rows=load_rows(PHASE/'data/splits/clra_split_manifest.csv',split)[:n]
    prep=build_post_intervention_preprocess('resnet18');out=[]
    for condition in ['clean','illumination_0.4','blur_3.0']:
        kind,level=('clean',0.) if condition=='clean' else (condition.rsplit('_',1)[0],float(condition.rsplit('_',1)[1]))
        for r in rows:
            with Image.open(ASSETS/'datasets/NEU-DET'/r['original_path']) as f:
                im=apply_source_space_intervention(f.convert('RGB'),kind,level)
                out.append({'source':r['source_image_id'],'condition':condition,'tensor':prep(im).unsqueeze(0),
                      'quality':image_quality_features(im),'truth':CLASSES.index(r['class_label'])})
    return out

def gpu_probe(gpu,q):
    try:
        configure(gpu);assert torch.cuda.device_count()==2;torch.cuda.set_device(gpu)
        device=torch.device('cuda',gpu);fast=model('Fast',device);acc=model('Accurate',device)
        xs=[r['tensor'].to(device) for r in inputs()]
        hs=Heads();times={};equivalence=[];count={};vectors=[]
        with torch.inference_mode():
            zs=[fast_prefix(fast,x) for x in xs]
            for i in range(40):fast(xs[i%len(xs)]);acc(xs[i%len(xs)])
            torch.cuda.synchronize(device)
            for x,z in zip(xs,zs):
                a=fast(x);b=fast_suffix(fast,z)
                equivalence.append({'max_logit_error':float((a-b).abs().max()),'argmax_equal':int(a.argmax())==int(b.argmax())})
                vectors.append(torch.nn.functional.adaptive_avg_pool2d(z,1).flatten(1).cpu().numpy()[0])
            for kind in ['FAST_PREFIX','FAST_SUFFIX','FAST_FULL','ACCURATE']:
                vals=[];wall=[]
                for i in range(120):
                    x=xs[i%len(xs)];z=zs[i%len(xs)]
                    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
                    before=time.perf_counter_ns();s.record()
                    if kind=='FAST_PREFIX':value=fast_prefix(fast,x)
                    elif kind=='FAST_SUFFIX':value=fast_suffix(fast,z)
                    elif kind=='FAST_FULL':value=fast(x)
                    else:value=acc(x)
                    e.record();e.synchronize();after=time.perf_counter_ns()
                    assert value.is_cuda
                    vals.append(float(s.elapsed_time(e)));wall.append((after-before)/1e6)
                times[kind]={'cuda_mean_ms':float(np.mean(vals)),'cuda_p95_ms':float(np.percentile(vals,95)),
                             'wall_mean_ms':float(np.mean(wall)),'wall_p95_ms':float(np.percentile(wall,95)),
                             'samples':len(vals),'cuda_samples_ms':vals,'wall_samples_ms':wall}
                count[kind]=120
        r={'logical_gpu':gpu,'physical_uuid':UUIDS[gpu],'pid':os.getpid(),'gpu_name':torch.cuda.get_device_name(gpu),
           'actual_cuda_calls':count,'times':times,'split_equivalence':equivalence,
           'allocated_memory_bytes':torch.cuda.memory_allocated(gpu),'reserved_memory_bytes':torch.cuda.memory_reserved(gpu),
           'cuda_device_count':torch.cuda.device_count(),'no_cpu_fallback':True,'service_sample_replay':False,
           'max_fast_logit_error':max(x['max_logit_error'] for x in equivalence),
           'all_fast_argmax_equal':all(x['argmax_equal'] for x in equivalence)}
        q.put({'ok':True,'result':r})
    except Exception as e:
        import traceback
        q.put({'ok':False,'error':repr(e),'traceback':traceback.format_exc()})

def cpu_probe():
    from quality_need_v24 import quality_order, mid2_candidate
    configure(22);m=model('Local','cpu');hs=Heads();rows=inputs(n=12)
    values={k:[] for k in ['LOCAL_PREFIX','LOCAL_MID2','LOCAL_SUFFIX5','LOCAL_SUFFIX9','LOCAL_FULL','MID1_HEAD','MID2_HEAD','EARLY_HEAD']}
    eq=[];demand=[]
    with torch.inference_mode():
        for r in rows[:12]:m(r['tensor'])
        for r in rows:
            x=r['tensor'];start=time.perf_counter_ns();z=m.features[:5](x);end=time.perf_counter_ns();values['LOCAL_PREFIX'].append((end-start)/1e6)
            v=torch.nn.functional.adaptive_avg_pool2d(z,1).flatten().numpy();start=time.perf_counter_ns();p1=hs.predict('MID1',v,r['quality']);values['MID1_HEAD'].append((time.perf_counter_ns()-start)/1e6)
            start=time.perf_counter_ns();z2=m.features[5:9](z);values['LOCAL_MID2'].append((time.perf_counter_ns()-start)/1e6)
            v2=torch.nn.functional.adaptive_avg_pool2d(z2,1).flatten().numpy();start=time.perf_counter_ns();p2=hs.predict('MID2',v2,r['quality']);values['MID2_HEAD'].append((time.perf_counter_ns()-start)/1e6)
            start=time.perf_counter_ns();a=local_suffix(m,z);values['LOCAL_SUFFIX5'].append((time.perf_counter_ns()-start)/1e6)
            start=time.perf_counter_ns();b=local_suffix(m,z2,True);values['LOCAL_SUFFIX9'].append((time.perf_counter_ns()-start)/1e6)
            start=time.perf_counter_ns();full=m(x);values['LOCAL_FULL'].append((time.perf_counter_ns()-start)/1e6)
            eq.append({'max_logit_error':float(max((full-a).abs().max(),(full-b).abs().max())),
                       'argmax_equal':int(full.argmax())==int(a.argmax())==int(b.argmax())})
            trigger=mid2_candidate(p1)
            p=p2 if trigger else p1
            role=next(iter(quality_order(p, 'A')), 'Abstain')
            demand.append({'condition':r['condition'],'MID1':p1,'MID2':p2,'trigger':trigger,'quality_desired':role})
        for _ in range(100):
            start=time.perf_counter_ns();hs.predict('EARLY',np.zeros(64));values['EARLY_HEAD'].append((time.perf_counter_ns()-start)/1e6)
    # Verify our optimized live numeric interface against the serialized head author interface.
    sys.path.insert(0,str(TASK17/'code'))
    from public_auxiliary_predict import PublicAuxiliaryPredictor
    old=PublicAuxiliaryPredictor(TASK17/'models');errors=[]
    for stage,dim in [('MID1',40),('MID2',48),('EARLY',64)]:
        v=np.arange(dim)/dim;quality=[.4,.2,.01] if stage!='EARLY' else None
        got=hs.predict(stage,v,quality)
        reference=old.predict({'task_id':0,'feature_available_time_ms':0.,'stage':stage,'vector':v.tolist(),
             'quality3':dict(zip(['brightness','contrast','sharpness_blur_score'],quality)) if quality else None,
             'event_evidence':{'test':True},'vector_source_version':'test_numeric_equivalence'})
        errors.extend(abs(got[k]-reference['offline_reference_scores'][k]['calibrated']) for k in got)
    return {'times':{k:{'mean_ms':float(np.mean(v)),'p95_ms':float(np.percentile(v,95)),'samples_ms':v} for k,v in values.items()},
      'local_split_max_logit_error':max(x['max_logit_error'] for x in eq),'all_local_argmax_equal':all(x['argmax_equal'] for x in eq),
      'heads_numeric_max_error':max(errors),'head_kind':'Task17 frozen BinaryLogistic+calibrator, auxiliary; not RF or original OOF routing',
      'head_files_sha256':hs.files,'live_feature_demand_development_only':demand,
      'route_rule_status':'new explicit executable conservative bands(.5,+/-.1); historical per-fold A3 trigger is not a deployed global rule'}

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=','.join(UUIDS):raise RuntimeError('Explicit two physical UUIDs required')
    HERE.joinpath('evidence').mkdir(exist_ok=True)
    before=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv'],text=True)
    write(HERE/'evidence/provenance.json',{'host':subprocess.check_output(['hostname'],text=True).strip(),'logical_cpu_count':os.cpu_count(),
      'physical_gpu_uuids':UUIDS,'cuda_mask':os.environ['CUDA_VISIBLE_DEVICES'],'torch':torch.__version__,
      'CNN_checkpoints':{k:{'path':str(p.resolve()),'sha256':sha(p)} for k,p in CHECKPOINTS.items()},
      'source_sha256':{str(p):sha(p) for p in [PHASE/'models/model_factory.py',PHASE/'models/source_space_protocol.py',Path(__file__)]},
      'gpu_state_before':before,'deadline_ms':25,'batch_size':1,'precision':'FP32','source_intervention_before_spatial_preprocess':True})
    ctx=mp.get_context('spawn');q=ctx.Queue();ps=[ctx.Process(target=gpu_probe,args=(g,q)) for g in [0,1]]
    for p in ps:p.start()
    results=[q.get(timeout=300) for p in ps]
    for p in ps:p.join(timeout=30)
    write(HERE/'evidence/gpu_probe.json',results)
    if not all(r['ok'] for r in results):raise RuntimeError(results)
    cpu=cpu_probe();write(HERE/'evidence/cpu_probe.json',cpu)
    passed=all(r['result']['all_fast_argmax_equal'] and r['result']['max_fast_logit_error']<=1e-5 for r in results)
    passed &= cpu['all_local_argmax_equal'] and cpu['local_split_max_logit_error']<=1e-5 and cpu['heads_numeric_max_error']<=1e-12
    write(HERE/'evidence/probe_acceptance.json',{'pass':bool(passed),'actual_two_gpu_execution':True,
      'original_paper_full_equivalence':False,'reason':'new physical deployment and explicit auxiliary heads; full routing/resource protocol not yet accepted'})
    print(json.dumps({'pass':bool(passed),'gpus':[{k:r['result'][k] for k in ['physical_uuid','actual_cuda_calls','max_fast_logit_error']} for r in results],
                    'CPU':{k:v['p95_ms'] for k,v in cpu['times'].items()}},ensure_ascii=True),flush=True)
    if not passed:raise RuntimeError('Numeric equivalence failed')
if __name__=='__main__':main()
