"""Run one real-GPU batch arm, or a supervised sequential two-arm comparison.

No simulation entry is imported. Setup/warmup precede the common wall-clock
epoch. Original Local trace times/deadlines remain unchanged in every arm.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    def terminate(signum,frame):
        raise InterruptedError('Live arm interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,terminate)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['ours','edge_serving','comparison','supervised_ours',
                                  'wma_node','jellyfish_edge','edgeboost_edge','clockwork_edge','symphony_edge','proteus_edge','slackfit_edge',
                                  'pard_edge','ramsis_edge'],required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--config',default=str(ROOT/'config/live_gpu_batch.json'))
    p.add_argument('--allow-idle-resident-peers',action='store_true')
    args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(cfg['gpu_uuids'])
    os.environ['EDGE_GPU_CARD_COUNT']='2'
    os.environ['EDGE_DUAL_CLOCK_MODE']='OFF'
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
    from live_batch_runtime import LiveBatchRuntime,write_json,sha
    output=Path(args.output)
    if args.arm in ('comparison','supervised_ours'):
        output.mkdir(parents=True,exist_ok=False)
        status=dict(state='RUNNING',supervisor_pid=os.getpid(),started=time.time(),runs=[],
                    manifest_sha256=sha(args.manifest),config_sha256=sha(args.config),
                    automatic_rerun=False)
        write_json(output/'status.json',status)
        before_cpu=resource.getrusage(resource.RUSAGE_CHILDREN)
        for arm in (('ours','edge_serving') if args.arm=='comparison' else ('ours',)):
            command=[sys.executable,'-u','-B',str(Path(__file__).resolve()),'--arm',arm,
                     '--manifest',args.manifest,'--output',str(output/arm),'--config',args.config]
            if args.allow_idle_resident_peers:command+=['--allow-idle-resident-peers']
            started=time.time()
            with (output/(arm+'.log')).open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                run=dict(arm=arm,pid=child.pid,state='RUNNING',command=command)
                status['runs'].append(run);write_json(output/'status.json',status)
                try:code=child.wait(timeout=cfg['wall_cap_s']+240)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    code=124
                except BaseException:
                    child.terminate()
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    run.update(state='INTERRUPTED',exit_code=child.returncode)
                    status['state']='INTERRUPTED';write_json(output/'status.json',status)
                    raise
            after_cpu=resource.getrusage(resource.RUSAGE_CHILDREN)
            run.update(state='COMPLETE' if code==0 else 'FAILED',exit_code=code,
                wall_s=time.time()-started,
                os_process_tree_cpu_s=(after_cpu.ru_utime+after_cpu.ru_stime-
                                       before_cpu.ru_utime-before_cpu.ru_stime))
            before_cpu=after_cpu
            if code:
                status['state']='FAILED';write_json(output/'status.json',status)
                raise SystemExit(code)
            # A separate sibling process performs statistics after the complete
            # execution tree exits. OS CPU totals above contain no analysis.
            run['state']='STATISTICS';write_json(output/'status.json',status)
            subprocess.run([sys.executable,'-u','-B',str(ROOT/'tools/analyze_live_batch.py'),
                '--output',str(output/arm),'--cpu',str(cfg['statistics_cpu'])],check=True)
            statistics_cpu=resource.getrusage(resource.RUSAGE_CHILDREN)
            run['offline_statistics_os_cpu_s']=(statistics_cpu.ru_utime+statistics_cpu.ru_stime-
                                                after_cpu.ru_utime-after_cpu.ru_stime)
            before_cpu=statistics_cpu;run['state']='COMPLETE'
            run['metrics']=json.loads((output/arm/'metrics.json').read_text())
            write_json(output/'status.json',status)
        if args.arm=='supervised_ours':
            status.update(state='COMPLETE',ended=time.time())
            write_json(output/'status.json',status)
            print(json.dumps(status['runs'][0]['metrics']['totals']))
            return
        a,b=[run['metrics'] for run in status['runs']]
        if a['source_trace_sha256']!=b['source_trace_sha256'] or a['arrivals']!=b['arrivals']:
            raise ValueError('Same-input comparison failed')
        identities=[json.loads((output/arm/'identity.json').read_text()) for arm in ('ours','edge_serving')]
        for key in ('config_sha256','manifest_sha256','frozen_dependencies',
                    'calibration_sha256','git_commit','source_sha256'):
            if identities[0][key]!=identities[1][key]:raise ValueError('Arm identity changed: '+key)
        subprocess.run([sys.executable,'-u','-B',str(ROOT/'tools/analyze_live_batch.py'),
            '--output',str(output),'--cpu',str(cfg['statistics_cpu']),'--pair-tasks',str(a['arrivals'])],check=True)
        status.update(state='COMPLETE',ended=time.time(),same_frozen_source=True)
        write_json(output/'status.json',status)
        print(json.dumps({r['arm']:{k:r['metrics'][k] for k in ('arrivals','tca','totals','cpu_all_processes_steady_s')}
                          for r in status['runs']}))
        return
    public=json.loads((ROOT/'config/project.json').read_text())['public_lock']
    with open(public,'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime=LiveBatchRuntime(args.manifest,output,args.arm,args.config,args.allow_idle_resident_peers)
        execution=runtime.run()
        print(json.dumps({k:execution[k] for k in ('arrivals','cpu_all_processes_replay_and_drain_s')}))



if __name__=='__main__':main()
