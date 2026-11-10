"""Measured mixed Fast batch costs; no policy, labels or future routes."""
import json
from pathlib import Path


def batch_kind(jobs):
    kinds={j['kind'] for j in jobs}
    if not kinds:raise ValueError('Empty batch')
    if len(kinds)==1:return next(iter(kinds))
    if kinds=={'FAST_FULL','FAST_PREFIX'}:return 'FAST_MIXED'
    raise ValueError('Incompatible GPU batch')


def fast_composition(jobs):
    if not jobs or any(j['kind'] not in ('FAST_FULL','FAST_PREFIX') for j in jobs):
        raise ValueError('Expected a nonempty Fast batch')
    return len(jobs),sum(j['kind']=='FAST_PREFIX' for j in jobs)


class MixedFastProfile:
    def __init__(self,path):
        data=json.loads(Path(path).read_text())
        if data['schema']!='mixed_fast_eager_v1' or data['cuda_graph_used']:
            raise ValueError('Expected actual eager mixed-batch measurement')
        self.rows={(r['batch'],r['unknown'],r['unknown_fast']):r for r in data['summary']}
        expected={(b,u,m) for b in range(2,11) for u in range(1,b) for m in range(u+1)}
        if set(self.rows)!=expected:raise ValueError('Incomplete mixed batch calibration')

    def mean(self,jobs,clock='mean_ms'):
        b,u=fast_composition(jobs)
        # Unknown outcomes are not visible at dispatch. Use the same maximum
        # over possible survivors as the original selective Prefix contract.
        return max(self.rows[b,u,m]['cuda_ms'][clock] for m in range(u+1))*1e6

    def units(self,b,u,m):
        row=self.rows[b,u,m]
        prefix,probe,suffix=(row['parts_mean_ns'][k] for k in ('prefix','probe','suffix'))
        n=b-u+m
        common=prefix/b
        known=common+suffix/n
        unknown=common+probe/u+(m/u)*suffix/n
        return known,unknown
