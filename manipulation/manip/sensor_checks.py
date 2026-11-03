"""Deferred real Fetch command-FIFO and observation-corruption engineering gates.

This module is imported only by released runtime checks. Fixtures prescribe inputs
and expected timelines; the actual wrapper/installed native simulator produces all
states and observations. An independent complete-state clone receives the literal
expected action at its native env boundary, avoiding a second copy of the queue or
sensor implementation as the acceptance oracle. No training/evaluation scope grows.
"""
import hashlib
import numpy as np
from .envs import make_env
from .sensor_checks_spec import (COMMANDS,FIFO_COMMANDS,FIFO_EXPECTED,NOISE_OFFSETS_M,
    DROPOUT_DRAWS,DROPOUT_PROBABILITY,DROPOUT_HOLD,HOLD_BLOCKS,SENSOR_COMMANDS)

def _hash_array(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()

def _same_array(actual,expected,where):
    if not np.array_equal(actual,expected):
        raise AssertionError('real sensor/latency exact array mismatch: '+where)

def _visible_fields(visible,clean,expected_position,where):
    # 0.1 micrometre tolerance accommodates observation storage casts; it is far
    # below the millimetre fixture offsets and is not a sensor-error estimate.
    position=np.asarray(expected_position)
    for label,value in [('object',visible['observation'][3:6]),('achieved_goal',visible['achieved_goal']),
                        ('relative',visible['observation'][6:9]+visible['observation'][:3])]:
        np.testing.assert_allclose(value,position,rtol=0,atol=1e-7,err_msg=where+': '+label)
    _same_array(visible['desired_goal'],clean['desired_goal'],where+': unchanged desired goal')
    _same_array(visible['observation'][:3],clean['observation'][:3],where+': unchanged grip')
    _same_array(visible['observation'][9:],clean['observation'][9:],where+': unchanged non-pose fields')

def _native_pair(env,reference,command,expected_action):
    """Actual outer env.step versus independent native env.step at the same state."""
    reference.restore(env.snapshot())
    visible,reward,terminated,truncated,info=env.step(np.asarray(command,dtype=np.float32))
    clean,native_reward,native_terminated,native_truncated,native_info=reference.env.step(
        np.asarray(expected_action,dtype=np.float32).copy())
    _same_array(env.integration_state(),reference.integration_state(),'complete integration state after expected native command')
    for key in ('observation','achieved_goal','desired_goal'):
        _same_array(env.clean_observation[key],clean[key],'independent native clean '+key)
    if (reward,terminated,truncated)!=(native_reward,native_terminated,native_truncated):
        raise AssertionError('corruption/latency altered native reward or timeout semantics')
    if bool(info['is_success'])!=bool(native_info['is_success']):
        raise AssertionError('visible corruption changed native success truth')
    expected=np.asarray(expected_action,dtype=np.float32)
    _same_array(env.last_applied_action,expected,'reported applied action versus literal expected FIFO command')
    _same_array(np.asarray(info['applied_action'],dtype=np.float32),expected,'step info applied action')
    return visible,clean,info

def command_fifo_checks(config):
    """Known full delay1/2 timelines, drain and reset clearing, using real physics."""
    config={**config,'training_randomization':None}
    rows=[]
    for delay in (1,2):
        env=make_env(config,{'latency':delay})
        reference=make_env(config,{'latency':delay})
        try:
            env.reset(seed=310+delay); reference.reset(seed=310+delay)
            # Populate real pending commands, then reset before they drain.
            for command in COMMANDS[:3]: env.step(np.asarray(command,dtype=np.float32))
            if not any(np.any(x) for x in env.queue):
                raise AssertionError('reset fixture did not leave an identifiable pending command')
            env.reset(seed=320+delay)
            if len(env.queue)!=delay or any(np.any(x) for x in env.queue) or np.any(env.last_applied_action):
                raise AssertionError('reset failed to clear the command FIFO/applied action')
            samples=[]
            for index,(command,expected) in enumerate(zip(FIFO_COMMANDS,FIFO_EXPECTED[delay])):
                visible,clean,_=_native_pair(env,reference,command,expected)
                _visible_fields(visible,clean,clean['observation'][3:6],f'delay{delay}/step{index+1}')
                samples.append({'control_step':index+1,'requested':list(command),'expected_applied':list(expected),
                    'actual_applied':env.last_applied_action.tolist(),'integration_state_sha256':_hash_array(env.integration_state())})
            rows.append({'delay_control_steps':delay,'actual_control_dt_seconds':float(env.contract['control_dt']),
                'declared_command_delay_seconds':delay*float(env.contract['control_dt']),
                'paired_controls':len(samples),'reset_cleared_pending_commands':True,'samples':samples,
                'acceptance':'actual outer env.step matches independent native clone on complete integration state/clean obs/reward/termination',
                'gripper_scope':'FetchPush blocks gripper; fixtures exercise XYZ commands, not gripper actuation'})
        finally:
            reference.close(); env.close()
    return rows

def _install_known_schedule(env):
    # Isolated engineering injection after a genuine reset, not an experiment
    # setting. Observation0 is checked before injection. Step1 is a fresh sample,
    # so every later hold anchor comes from actual native state + a literal offset.
    env.noise[:]=np.asarray(NOISE_OFFSETS_M,dtype=env.noise.dtype)
    env.dropout_draws[:]=np.asarray(DROPOUT_DRAWS,dtype=env.dropout_draws.dtype)
    env.drop_remaining=0

def observation_schedule_checks(config):
    """All51 visible observations, fixed noise/hold semantics and fresh reset state."""
    config={**config,'training_randomization':None}
    perturb={'noise_sigma':.005,'dropout_probability':DROPOUT_PROBABILITY,'dropout_hold':DROPOUT_HOLD}
    env=make_env(config,perturb); reference=make_env(config,perturb)
    try:
        initial,_=env.reset(seed=421); reference.reset(seed=421)
        _visible_fields(initial,env.clean_observation,env.clean_observation['observation'][3:6]+env.noise[0],'natural reset observation0')
        _install_known_schedule(env)
        clean_positions={}; samples=[]
        for step,command in enumerate(SENSOR_COMMANDS,1):
            visible,clean,_=_native_pair(env,reference,command,command)
            clean_positions[step]=clean['observation'][3:6].copy()
            # Declarative expected blocks, not the production counter/queue algorithm.
            source=next((anchor for first,last,anchor in HOLD_BLOCKS if first<=step<=last),step)
            expected=clean_positions[source]+np.asarray(NOISE_OFFSETS_M[source])
            _visible_fields(visible,clean,expected,f'known noise/dropout step{step}')
            samples.append({'control_step':step,'expected_visible_source_step':source,
                'clean_object_position_m':clean_positions[step].tolist(),'expected_visible_position_m':expected.tolist(),
                'actual_visible_position_m':visible['observation'][3:6].tolist(),
                'integration_state_sha256':_hash_array(env.integration_state())})
        # Reset during an active hold, not merely after the episode drains.
        env.reset(seed=422); _install_known_schedule(env)
        env.step(np.asarray(SENSOR_COMMANDS[0],dtype=np.float32)); env.step(np.asarray(SENSOR_COMMANDS[1],dtype=np.float32))
        if env.drop_remaining!=2: raise AssertionError('known step2 trigger did not enter the intended3-output hold')
        stale_pose=env.held_position.copy(); injected_noise=env.noise.copy(); injected_draws=env.dropout_draws.copy()
        reset_visible,_=env.reset(seed=431)
        expected_reset=env.clean_observation['observation'][3:6]+env.noise[0]
        if np.allclose(stale_pose,expected_reset,rtol=0,atol=1e-7):
            raise AssertionError('predeclared reset fixture has no distinguishable old/new pose')
        _visible_fields(reset_visible,env.clean_observation,expected_reset,'reset during active hold')
        np.testing.assert_allclose(env.held_position,expected_reset,rtol=0,atol=1e-7)
        if np.array_equal(injected_noise,env.noise) or np.array_equal(injected_draws,env.dropout_draws):
            raise AssertionError('reset retained the injected prior noise/dropout schedule')
        natural_noise=env.noise.copy(); natural_draws=env.dropout_draws.copy(); reset_position=reset_visible['observation'][3:6].copy()
        replay,_=env.reset(seed=431)
        _same_array(env.noise,natural_noise,'same-seed natural noise schedule replay')
        _same_array(env.dropout_draws,natural_draws,'same-seed natural dropout schedule replay')
        _same_array(replay['observation'][3:6],reset_position,'same-seed reset pose replay')
        return {'visible_observations_checked':51,'known_schedule_control_steps':50,'dropout_hold_outputs':DROPOUT_HOLD,
            'hold_blocks':[list(x) for x in HOLD_BLOCKS],'trigger_during_active_hold_does_not_extend':True,
            'reset_during_active_hold_clears_stale_pose_and_schedule':True,'same_seed_natural_schedule_replay':True,
            'reset_noise_sha256':_hash_array(natural_noise),'reset_dropout_draws_sha256':_hash_array(natural_draws),'samples':samples,
            'fixture_scope':'literal metre offsets/dropout draws verify application and timing through real env.step; no Gaussian-distribution/calibration estimate',
            'native_truth':'every step matches independent complete-state native clone; clean reward/success/timeout are not corrupted'}
    finally:
        reference.close(); env.close()
