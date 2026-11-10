"""Observation-only CPU/wall profiling. Never advances the simulation clock."""
from array import array
from collections import Counter
from functools import wraps
import json
import os
import time


class Segment:
    # Each invocation owns both starts. Nested segments cannot overwrite them.
    def __init__(self, profiler, name):self.profiler,self.name=profiler,name
    def __enter__(self):
        self.cpu=time.process_time_ns();self.wall=time.perf_counter_ns()
        return self
    def __exit__(self,*exc):
        wall=time.perf_counter_ns()-self.wall;cpu=time.process_time_ns()-self.cpu
        self.profiler.record(self.name,cpu,wall)


class NullSegment:
    def __enter__(self):return self
    def __exit__(self,*exc):return False


NULL=NullSegment()
class Profiler:
    def __init__(self,enabled):
        self.enabled=enabled;self.samples={};self.counts=Counter()
    def span(self,name):return Segment(self,name) if self.enabled else NULL
    def record(self,name,cpu,wall):
        data=self.samples.setdefault(name,(array('Q'),array('Q')))
        data[0].append(cpu);data[1].append(wall)
    def summary(self):
        import numpy as np
        def stat(values):
            a=np.asarray(values,dtype=np.uint64)
            return dict(n=len(a),total_ns=int(a.sum()),mean_ns=float(a.mean()),
                p95_ns=float(np.percentile(a,95)),p99_ns=float(np.percentile(a,99)),max_ns=int(a.max()))
        return dict(enabled=self.enabled,clocks=dict(cpu='time.process_time_ns',wall='time.perf_counter_ns'),
            virtual_time_charged=False,raw_observed_times=True,nested_segments_are_not_additive=True,
            counters=dict(self.counts),segments={k:dict(cpu=stat(v[0]),wall=stat(v[1])) for k,v in self.samples.items()})


def profiler(rt):
    p=getattr(rt,'dispatch_profiler',None)
    if p is None:
        p=Profiler(os.environ.get('EDGE_DISPATCH_PROFILE','0')=='1');rt.dispatch_profiler=p
    return p


def timed_schedule(fn):
    @wraps(fn)
    def call(self,*a,**kw):
        p=profiler(self)
        if not p.enabled:return fn(self,*a,**kw)
        with p.span('dispatch_whole'):return fn(self,*a,**kw)
    return call


def save_profile(rt):
    p=profiler(rt)
    (rt.folder/'DISPATCH_PROFILE.json').write_text(json.dumps(p.summary(),indent=2)+'\n')
