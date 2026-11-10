"""Source-disjoint DEVELOPMENT auxiliary joint-correctness model."""
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
QUALITY_THRESHOLDS={'Local': .8, 'Fast': .6, 'Accurate': .5}

def quality_eligible(role, probability):
    if role == 'Accurate':
        return probability > QUALITY_THRESHOLDS[role]-1e-14
    return probability >= QUALITY_THRESHOLDS[role]

def mid2_candidate(p):
    return (abs(p['Local']-.5) <= .1 or
            (not quality_eligible('Local', p['Local']) and abs(p['Fast']-.5) <= .1))

def refresh_quality_policy(environment):
    """Cached predictions stay fixed; derived routing flags use current gates."""
    for row in environment['bank'].values():
        row['MID2_candidate'] = mid2_candidate(row['MID1']['probabilities'])
    environment['quality_policy'] = dict(thresholds=dict(QUALITY_THRESHOLDS),
        local_fast_comparison='>=', accurate_need_threshold=.5,
        mid2_rule='Local near .5 +/- .1, or Local ineligible and Fast near .5 +/- .1')
    return environment
def features(stage,vector,quality,p):
    roles=('Fast','Accurate') if stage=='EARLY' else ('Local','Fast','Accurate')
    logits=[math.log(max(1e-6,min(1-1e-6,p[r]))/(1-max(1e-6,min(1-1e-6,p[r])))) for r in roles]
    return np.r_[np.asarray(vector,dtype=float),np.asarray(quality if quality is not None else [],dtype=float),logits]
class NeedHeads:
    def __init__(self):
        self.path=ROOT/'evidence/QUALITY_NEED_DEVELOPMENT_MODEL.json'
        self.data=json.loads(self.path.read_text())
        assert self.data['training_split']=='DEVELOPMENT' and self.data['evaluation_sources_used']==0
    def predict(self,stage,vector,quality,p):
        d=self.data['models'][stage];x=features(stage,vector,quality,p)
        assert len(x)==len(d['mean'])
        score=float(np.r_[1.,(x-np.asarray(d['mean']))/np.asarray(d['std'])]@np.asarray(d['beta']))
        return float(1/(1+np.exp(-np.clip(score,-35,35))))
    def enabled(self,stage):return self.data['validation'][stage]['accepted']
def quality_order(p,path,need_probability=None,enabled=False,allowed_roles=None):
    roles=['Fast','Accurate'] if path=='B' else ['Local','Fast','Accurate']
    if allowed_roles is not None:
        roles=[r for r in roles if r in allowed_roles]
    eligible=[r for r in roles if quality_eligible(r, p[r])]
    if enabled and need_probability is not None and need_probability>=.5 and 'Accurate' in eligible:
        eligible=['Accurate']+[r for r in eligible if r!='Accurate']
    return eligible


def baseline_source_frame(frame, role, upload_ns, direct_fast=False):
    """Single-model baseline using only quality already published by Local.

    Unprobed uploads go directly to the sole model in the current real run.
    direct_fast=False retains historical all-F Prefix/Head/Suffix semantics.
    Never inspect hidden EARLY values here.
    """
    if role not in ('Fast', 'Accurate'):
        raise ValueError('Single-model baseline requires Fast or Accurate')
    outcome = frame['outcome']
    probabilities = frame.get('quality_probabilities')
    if outcome == 'PREFIX_READY':
        if role == 'Fast' and not direct_fast:
            return frame
        return dict(frame, outcome='GPU_READY', stage='GPU_CONT',
                    gpu_candidates=[role], gpu_candidate_kinds={role:
                        'FAST_FULL' if role == 'Fast' else 'ACCURATE'},
                    reason='single-model unprobed upload; execute '+role+' directly',
                    baseline_unprobed_upload=True)
    quality_reject = (outcome == 'COMMON_ABSTAIN' and probabilities and
                      frame.get('reason') in ('no quality eligible Local or Accurate role',
                                              'no quality eligible Local Fast or Accurate role'))
    if outcome != 'GPU_READY' and not quality_reject:
        return frame
    if not probabilities or not quality_eligible(role, probabilities[role]):
        return dict(frame, outcome='COMMON_ABSTAIN', stage='ABSTAIN', prediction=None,
                    reason='single-model baseline below '+role+' quality threshold')
    at = frame['source_publish_ns'] + (upload_ns if outcome == 'COMMON_ABSTAIN' else 0)
    return dict(frame, outcome='GPU_READY', stage='GPU_CONT', prediction=None,
                source_publish_ns=at, gpu_candidates=[role],
                gpu_candidate_kinds={role: 'FAST_FULL' if role == 'Fast' else 'ACCURATE'},
                reason='single-model baseline with published Local quality')


