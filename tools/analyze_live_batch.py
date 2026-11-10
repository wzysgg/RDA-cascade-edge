"""Separate CPU-only statistics command, after real GPU actors have exited."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from live_batch_statistics import analyze,pair

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--cpu',required=True,type=int)
    p.add_argument('--pair-tasks',type=int)
    a=p.parse_args()
    result=pair(a.output,a.pair_tasks,a.cpu) if a.pair_tasks is not None else analyze(a.output,a.cpu)
    print(json.dumps(result if a.pair_tasks else {k:result[k] for k in
        ('arrivals','tca','totals','cpu_all_processes_replay_and_drain_s','offline_statistics_cpu_s')}))
