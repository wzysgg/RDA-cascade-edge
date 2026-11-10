"""Each intervention is 120 seconds; warmup60, baseline10, final recovery60."""
T003_WINDOWS=[(0,60),(60,140),(140,150),(150,260),(260,370),(370,380),(380,390),(390,500)]

def validate_t003_mixed(spec):
    assert spec['scenario']=='T003_mixed_partial_blur',spec.get('scenario')
    assert spec['policy'] in ('v24_unified','fixed_1a1f'),spec.get('policy')
    assert spec.get('controller_version')=='v24'
    assert spec.get('quality_need_enabled') is True and spec.get('protect_accurate_deadline') is True
    assert int(spec['total_nodes'])==20 and list(spec['affected_nodes'])==[0,1,2,3,4,10,11,12,13,14],spec.get('affected_nodes')
    assert float(spec['duration_s'])==500 and float(spec['controller_start_s'])==60
    assert float(spec['change_s'])==140 and float(spec['recover_s'])==380
    assert float(spec['deadline_ms'])==50 and float(spec['forecast_horizon_ms'])==50
    assert float(spec['initial_fast'])==1 and spec['requests']==[]
    assert float(spec['blur_radius'])==.55
    assert int(spec['rate_bin_s'])==5 and int(spec.get('rate_pair_s',10))==10
    assert 0.<float(spec['background_blur_prob'])<1. and 0.<float(spec['degraded_blur_prob'])<1.
    assert float(spec['background_blur_prob'])==.1 and float(spec['degraded_blur_prob'])==.8
    assert float(spec['traffic_rho'])==.6,spec.get('traffic_rho')
    assert spec.get('input_plan_path') and spec.get('input_plan_sha256') and spec.get('expected_plan_sha256'),'frozen input plan must be declared'
    expected=[(0.,140.,'mixed_background',False),(140.,380.,'mixed_degraded',False),(380.,500.,'mixed_recover',False)]
    got=[(float(p['start_s']),float(p['end_s']),p['condition'],bool(p['load_increase'])) for p in spec['phases']]
    assert got==expected,got

T004_WINDOWS=[(0,10),(10,70),(70,190),(190,250)]
T004_EXPECTED_PLAN_SHA256='ebd575740a29eb9c3923c51eeb47a7b0654f4e29e7bd2fe049d6fe10d37a9eaa'
T004_EXPECTED_ARRIVALS=226323

def validate_t004_mixed(spec,controller_version='v25',policies=('v25_demand','fixed_1a1f')):
    assert spec['scenario']=='T004_mixed_partial_blur',spec.get('scenario')
    assert spec['policy'] in policies,spec.get('policy')
    assert spec.get('controller_version')==controller_version
    assert spec.get('quality_need_enabled') is True and spec.get('protect_accurate_deadline') is True
    assert int(spec['total_nodes'])==20 and list(spec['affected_nodes'])==[0,1,2,3,4,10,11,12,13,14],spec.get('affected_nodes')
    assert float(spec['duration_s'])==250 and float(spec['controller_start_s'])==10
    assert float(spec['change_s'])==70 and float(spec['recover_s'])==190
    assert float(spec['deadline_ms'])==50 and float(spec['forecast_horizon_ms'])==50
    assert float(spec['initial_fast'])==1 and spec['requests']==[]
    assert float(spec['blur_radius'])==.55
    assert int(spec['rate_bin_s'])==5 and int(spec.get('rate_pair_s',10))==10
    assert float(spec['background_blur_prob'])==.1 and float(spec['degraded_blur_prob'])==.8
    assert float(spec['traffic_rho'])==.6,spec.get('traffic_rho')
    assert spec.get('input_plan_path') and spec.get('input_plan_sha256') and spec.get('expected_plan_sha256'),'frozen input plan must be declared'
    if spec.get('expected_plan_sha256') is not None:
        assert spec['expected_plan_sha256']==T004_EXPECTED_PLAN_SHA256,spec.get('expected_plan_sha256')
    if spec.get('expected_arrivals') is not None:
        assert int(spec['expected_arrivals'])==T004_EXPECTED_ARRIVALS,spec.get('expected_arrivals')
    expected=[(0.,70.,'mixed_background',False),(70.,190.,'mixed_degraded',False),(190.,250.,'mixed_recover',False)]
    got=[(float(p['start_s']),float(p['end_s']),p['condition'],bool(p['load_increase'])) for p in spec['phases']]
    assert got==expected,got

T008_CONTROLLER_VERSION='v26'
T008_POLICIES=('v26_effective','fixed_1a1f')

def validate_t008_mixed(spec):
    """T008 reuses the frozen T004_mixed_v1 inputs/scenario metadata; only the
    controller identity moves to v26 (TASK section 6).  Everything else is the
    same frozen 250 s, 20-node, d50, protect1/need1 contract."""
    return validate_t004_mixed(spec,controller_version=T008_CONTROLLER_VERSION,policies=T008_POLICIES)

T009_CONTROLLER_VERSION='qf2'
T009_POLICIES=('fixed_1a1f','qf2_feedback')

def validate_t009_mixed(spec):
    """T009 reuses the frozen T004_mixed_v1 inputs/scenario metadata byte for
    byte; only the controller identity moves to qf2 (TASK section 6).  The
    frozen experiment fields are still checked item by item by
    validate_t004_mixed."""
    return validate_t004_mixed(spec,controller_version=T009_CONTROLLER_VERSION,policies=T009_POLICIES)

T010_CONTROLLER_VERSION='qf3'
T010_POLICIES=('fixed_1a1f','qf3_feedback')

def validate_t010_mixed(spec):
    """T010 reuses the frozen T004_mixed_v1 inputs/scenario metadata byte for
    byte; only the controller identity moves to qf3 (TASK section 3).  The
    frozen experiment fields are still checked item by item by
    validate_t004_mixed."""
    return validate_t004_mixed(spec,controller_version=T010_CONTROLLER_VERSION,policies=T010_POLICIES)

T011_CONTROLLER_VERSION='qf4'
T011_POLICIES=('fixed_1a1f','qf4_feedback')

def validate_t011_mixed(spec):
    """T011 reuses the frozen T004_mixed_v1 inputs/scenario metadata byte for
    byte; only the controller identity moves to qf4 (TASK section 1).  The
    frozen experiment fields are still checked item by item by
    validate_t004_mixed."""
    return validate_t004_mixed(spec,controller_version=T011_CONTROLLER_VERSION,policies=T011_POLICIES)

T012_CONTROLLER_VERSION='qf4r'
T012_POLICIES=('fixed_1a1f','qf4r_feedback')

def validate_t012_mixed(spec):
    """T012 reuses the frozen T004_mixed_v1 inputs/scenario metadata byte for
    byte; only the controller identity moves to qf4r (TASK section 1).  The
    frozen experiment fields are still checked item by item by
    validate_t004_mixed."""
    return validate_t004_mixed(spec,controller_version=T012_CONTROLLER_VERSION,policies=T012_POLICIES)

def windows_for_spec(spec):
    if spec.get('schema_version')=='T004_mixed_v1':
        return list(T004_WINDOWS)
    if spec.get('schema_version')=='T003_mixed_v1':
        return list(T003_WINDOWS)
    if spec.get('repair_smoke'):
        windows=[(0,15),(15,20),(20,25),(25,40),(40,45),(45,60)]
        if spec['duration_s']==80:windows += [(60,65),(65,80)]
        return windows
    if spec.get('integration_smoke'):
        return [(0,6),(6,8),(8,15),(15,22),(22,30)]
    windows=[(0,60),(60,70),(70,80),(80,130),(130,180),(180,190),(190,200),(200,250)]
    if spec['duration_s']==370:
        windows += [(250,300),(300,310),(310,320),(320,370)]
    return windows

def validate_spec(spec):
    if spec.get('schema_version')=='T004_mixed_v1':
        if spec.get('controller_version')==T012_CONTROLLER_VERSION:
            validate_t012_mixed(spec)
        elif spec.get('controller_version')==T011_CONTROLLER_VERSION:
            validate_t011_mixed(spec)
        elif spec.get('controller_version')==T010_CONTROLLER_VERSION:
            validate_t010_mixed(spec)
        elif spec.get('controller_version')==T009_CONTROLLER_VERSION:
            validate_t009_mixed(spec)
        elif spec.get('controller_version')==T008_CONTROLLER_VERSION:
            validate_t008_mixed(spec)
        else:
            validate_t004_mixed(spec)
        return
    if spec.get('schema_version')=='T003_mixed_v1':
        validate_t003_mixed(spec)
        return
    if spec.get('controller_version')=='v24':
        assert spec['policy'] in ('v24_unified','fixed_1a1f','v23_growth')
        if spec.get('repair_smoke'):
            assert spec['initial_fast']==1 and spec['requests']==[]
            assert spec['controller_start_s']==15 and spec['change_s']==20 and spec['recover_s']==40
            assert spec['scenario'] in ('partial_blur','partial_load','blur_then_load')
            expected=[(0,20,'clean',False),(20,40,'clean' if spec['scenario']=='partial_load' else 'blur_'+str(spec['blur_radius']),spec['scenario']=='partial_load')]
            expected += [(40,60,'clean',True),(60,80,'clean',False)] if spec['scenario']=='blur_then_load' else [(40,60,'clean',False)]
            assert spec['duration_s']==(80 if spec['scenario']=='blur_then_load' else 60)
            assert [(p['start_s'],p['end_s'],p['condition'],p['load_increase']) for p in spec['phases']]==expected
            assert len(set(spec['affected_nodes'])) in (4,10,16) and spec['blur_radius'] in (.55,.6)
            return
        equivalent=dict(spec,controller_version='v23',policy='v23_growth' if spec['policy'] in ('v24_unified','v23_growth') else 'fixed_1a1f')
        return validate_spec(equivalent)
    if spec.get('controller_version')=='v23':
        assert spec['policy'] in ('v23_growth','fixed_1a1f')
        assert spec['initial_fast']==1 and spec['requests']==[] and spec['blur_radius'] in (.55,.6)
        assert len(set(spec['affected_nodes'])) in (4,10,16) and all(0<=n<20 for n in spec['affected_nodes'])
        if spec.get('integration_smoke'):
            assert spec['duration_s']==30 and spec['controller_start_s']==6 and spec['change_s']==8 and spec['recover_s']==22
            expected=[(0,8,'clean',False),(8,22,'blur_'+str(spec['blur_radius']),False),(22,30,'clean',False)]
        else:
            assert spec['controller_start_s']==60 and spec['change_s']==70 and spec['recover_s']==190
            assert spec['scenario'] in ('partial_blur','partial_load','blur_then_load')
            assert spec['duration_s']==(370 if spec['scenario']=='blur_then_load' else 250)
            expected=[(0,70,'clean',False),(70,190,'clean' if spec['scenario']=='partial_load' else 'blur_'+str(spec['blur_radius']),spec['scenario']=='partial_load')]
            expected += [(190,310,'clean',True),(310,370,'clean',False)] if spec['scenario']=='blur_then_load' else [(190,250,'clean',False)]
        assert [(p['start_s'],p['end_s'],p['condition'],p['load_increase']) for p in spec['phases']]==expected
        return
    if spec.get('integration_smoke'):
        assert spec['duration_s']==18 and spec['controller_start_s']==3
        assert spec['change_s']==5 and spec['recover_s']==13
        assert spec['policy']=='v22_demand' and spec['blur_radius']==.55
        assert spec['affected_nodes']==[0,1,2,3,4,10,11,12,13,14]
        assert spec['initial_fast']==1 and spec['requests']==[]
        assert spec['phases']==[{'start_s':0,'end_s':5,'condition':'clean','load_increase':False},
                               {'start_s':5,'end_s':13,'condition':'blur_0.55','load_increase':False},
                               {'start_s':13,'end_s':18,'condition':'clean','load_increase':False}]
        return
    assert spec['scenario']=='partial_blur' and spec['blur_radius'] in (.55,.6)
    assert spec['duration_s']==250
    assert spec['controller_start_s']==60 and spec['change_s']==70 and spec['recover_s']==190
    assert len(set(spec['affected_nodes'])) in (4,10,16)
    assert all(0<=n<20 for n in spec['affected_nodes'])
    assert spec['initial_fast']==1 and spec['requests']==[]
    assert spec['policy'] in ('fixed_1a1f','v20_resource','v22_demand')
    assert spec['phases'][0]=={'start_s':0,'end_s':70,'condition':'clean','load_increase':False}
    previous=0
    for p in spec['phases']:
        assert p['start_s']==previous and p['end_s']>p['start_s'];previous=p['end_s']
    assert previous==spec['duration_s']
    expected={
        'partial_blur':[(70,190,'blur_'+str(spec['blur_radius']),False),(190,250,'clean',False)],
        'partial_load':[(70,190,'clean',True),(190,250,'clean',False)],
        'blur_then_load':[(70,190,'blur_3.0',False),(190,310,'clean',True),(310,370,'clean',False)]}
    assert [(p['start_s'],p['end_s'],p['condition'],p['load_increase']) for p in spec['phases'][1:]]==expected[spec['scenario']]
