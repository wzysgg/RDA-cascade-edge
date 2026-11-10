"""Update execution budgets from already completed public stages only."""
import copy,time
from collections import defaultdict
import numpy as np
class PastTimingBudgets:
    def __init__(self,rt):
        self.rt=rt
        self.floor_probe=copy.deepcopy(rt.local_probe_cost_node)
        self.floor_mid2=copy.deepcopy(rt.local_mid2_cost_node)
        self.floor_suffix=copy.deepcopy(rt.local_suffix_cost_node)
        self.floor_ipc=copy.deepcopy(rt.local_ipc_budget)
        self.samples=[defaultdict(list) for _ in range(20)];self.last=[{} for _ in range(20)]
    def observe_local(self,row,received_ns):
        n=int(row['resource'].split(':')[1]);j=row['job'];kind=j['kind']
        key=('SUFFIX_'+str(bool(row.get('MID2',False)))) if kind=='LOCAL_SUFFIX' else kind
        rr=self.samples[n];rr[key].append(row['finish_ns']-row['start_ns'])
        rr['IPC'].append(max(0,row['start_ns']-j['dispatch_ns'])+max(0,received_ns-row['finish_ns']))
        for k in (key,'IPC'):
            if len(rr[k])>160:del rr[k][:-160]
            if len(rr[k])<30 or received_ns-self.last[n].get(k,-1e18)<250_000_000:continue
            self.last[n][k]=received_ns;p95=float(np.percentile(rr[k],95))
            if k=='LOCAL_PROBE':self.rt.local_probe_cost_node[n]=max(self.floor_probe[n],p95)
            elif k=='LOCAL_MID2':self.rt.local_mid2_cost_node[n]=max(self.floor_mid2[n],p95)
            elif k.startswith('SUFFIX_'):
                flag=k=='SUFFIX_True';self.rt.local_suffix_cost_node[n][flag]=max(self.floor_suffix[n][flag],p95)
            else:self.rt.local_ipc_budget[n]=max(self.floor_ipc[n],p95+200_000)
        self.rt.local_total_cost_node[n]=(self.rt.local_probe_cost_node[n]+self.rt.local_suffix_cost_node[n][False]+2*self.rt.local_ipc_budget[n])
    def snapshot(self):
        return {'local_probe_ns':self.rt.local_probe_cost_node,'local_suffix_ns':self.rt.local_suffix_cost_node,
                'local_ipc_ns':self.rt.local_ipc_budget,'observations_only':True,'hard_deadline_guarantee':False}
