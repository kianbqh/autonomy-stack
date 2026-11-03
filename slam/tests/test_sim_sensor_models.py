"""Deferred checks of the real existing 2-D simulator, not KITTI sensor calibration.

Requires NumPy and imports the repository's actual fusion_core.sim. NOT run
during private source preparation. No old simulator/result source is changed.
"""
import math
import numpy as np
from fusion_core.sim import RobotSim,SensorConfig

def test_zero_noise_synchronous_schedule_and_measurement_units():
    cfg=SensorConfig(gyro_noise=0,gyro_bias0=0,gyro_bias_rw=0,accel_noise=0,
                     odom_noise=0,odom_scale_err=0,gps_noise=0,gps_outlier_prob=0)
    sim=RobotSim(cfg,seed=3);counts={"imu":0,"odom":0,"gps":0}
    for tick in range(1,201):
        out=sim.step(.2,.1)
        assert abs(out["t"]-tick*.01)<1e-10
        assert out["imu"]==(.1,.2)
        assert ("odom" in out)==(tick%2==0)
        assert ("gps" in out)==(tick%100==0)
        for key in counts:counts[key]+=int(key in out)
        if "odom" in out:assert out["odom"]==sim.state[3]
        if "gps" in out:np.testing.assert_array_equal(out["gps"],sim.state[:2])
    assert counts=={"imu":200,"odom":100,"gps":2}

def test_fixed_seed_replays_actual_sensor_draws():
    a,b=RobotSim(seed=12),RobotSim(seed=12)
    for _ in range(300):
        x,y=a.step(.1,.05),b.step(.1,.05)
        assert x.keys()==y.keys()
        for key in x:np.testing.assert_array_equal(x[key],y[key])
        np.testing.assert_array_equal(a.state,b.state)
        assert a.bias==b.bias and a.odom_scale==b.odom_scale

def test_actual_noise_draw_scales_and_bias_random_walk():
    cfg=SensorConfig(gps_outlier_prob=0);sim=RobotSim(cfg,seed=41)
    gyro=[];accel=[];walk=[];odom=[]
    for _ in range(4096):
        before=sim.bias;out=sim.step(0,0)
        gyro.append(out["imu"][0]-sim.bias);accel.append(out["imu"][1]);walk.append(sim.bias-before)
        if "odom" in out:odom.append(out["odom"]-sim.state[3]*sim.odom_scale)
    for samples,sigma in ((gyro,.02),(accel,.15),(walk,.0005*math.sqrt(.01)),(odom,.05)):
        x=np.asarray(samples)
        assert abs(x.mean())<6*sigma/math.sqrt(len(x))
        assert .9*sigma<x.std(ddof=1)<1.1*sigma
    assert .97<=sim.odom_scale<=1.03

def test_gps_sigma_and_forced_multipath_radius():
    # gps_every=1 deliberately accelerates a unit fixture; it is not the
    # default 1Hz schedule, whose exact cadence is checked above.
    cfg=SensorConfig(gps_every=1,gps_outlier_prob=0)
    sim=RobotSim(cfg,seed=4);errors=[]
    for _ in range(4096):
        out=sim.step(0,0);errors.append(out["gps"]-sim.state[:2])
    x=np.asarray(errors)
    assert np.max(np.abs(x.mean(0)))<6*1.5/math.sqrt(len(x))
    assert np.all((x.std(0,ddof=1)>.9*1.5)&(x.std(0,ddof=1)<1.1*1.5))
    forced=RobotSim(SensorConfig(gps_every=1,gps_noise=0,gps_outlier_prob=1),seed=8)
    for _ in range(32):
        out=forced.step(0,0)
        assert abs(np.linalg.norm(out["gps"]-forced.state[:2])-8)<1e-10
