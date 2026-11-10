"""Explicit staged interventions; only selected nodes change; no labels."""
import random

def build_plan(spec,meta):
    if spec.get('schema_version') in ('T003_mixed_v1','T004_mixed_v1'):
        import json as _json
        import mixed_scenario as _mixed
        from physical_probe import HERE as _HERE
        _cal=_json.loads((_HERE/'evidence/FROZEN_DEVELOPMENT_CALIBRATION.json').read_text(encoding='utf8'))
        _rates=_mixed.base_node_rates(_cal,spec['traffic_rho'])
        _schedule=spec.get('rate_schedule') or _mixed.build_rate_schedule(spec)[0]
        return _mixed.build_arrivals(spec,meta,_rates,rate_schedule=_schedule)
    bycondition={c:[r['bank_index'] for r in meta if r['condition']==c and r['split']=='EVALUATION'] for c in sorted(set(r['condition'] for r in meta))}
    affected=set(spec['affected_nodes']);plan=[]
    for node in range(20):
        for phase,p in enumerate(spec['phases']):
            rate=spec['hot_rates'][str(node)] if node in affected and p['load_increase'] else spec['base_node_tps']
            condition=p['condition'] if node in affected else 'clean'
            rng=random.Random(spec['seed']*1000+node*10+phase)
            images=random.Random(spec['seed']*2000+node*10+phase);at=p['start_s']
            while True:
                at+=rng.expovariate(rate)
                if at>=p['end_s']:break
                plan.append({'scheduled_offset_ns':round(at*1e9),'origin_node':node,'bank_index':images.choice(bycondition[condition])})
    plan.sort(key=lambda r:(r['scheduled_offset_ns'],r['origin_node']))
    return [{'task_id':i,**r} for i,r in enumerate(plan)]
