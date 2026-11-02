"""Synthetic official-source precision regressions; no real KITTI result."""
import math
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
from eval_traj import distance_pairs,kitti_error_components,kitti_segment_pairs,kitti_segments,rpe


def line_poses(positions):
    poses=np.repeat(np.eye(4)[None],len(positions),axis=0)
    poses[:,0,3]=positions
    return poses


def f32(value):
    return struct.unpack("<f",struct.pack("<f",value))[0]


def test_synthetic_float32_endpoint_skips_double_just_over_100m():
    truth=line_poses([0,50,100.0000001,150])
    assert distance_pairs(truth,100,stride=10)==[(0,2)]
    # Official dx/dist float casts turn the middle cumulative value into 100.
    assert kitti_segment_pairs(truth,[100],10)==[(0,3,100.0)]


def test_synthetic_full_1101_frames_all_lengths_stride_and_start_order():
    truth=line_poses(np.arange(1101,dtype=float))
    lengths=list(range(100,801,100))
    pairs=kitti_segment_pairs(truth,lengths,10)
    expected=[(i,i+length+1,float(length)) for i in range(0,1101,10)
              for length in lengths if i+length+1<1101]
    assert pairs==expected
    assert len(pairs)==520
    assert {length:sum(row[2]==length for row in pairs) for length in lengths}=={
        100:100,200:90,300:80,400:70,500:60,600:50,700:40,800:30}
    assert pairs[:8]==[(0,length+1,float(length)) for length in lengths]


def test_synthetic_overall_is_serial_float32_mean_of_all_segments():
    truth=line_poses(np.arange(32,dtype=float)*10)
    pred=line_poses(np.arange(32,dtype=float)*11)
    cfg={"segment_lengths_m":list(range(100,801,100)),"segment_start_stride_frames":10}
    result=kitti_segments(pred,truth,cfg)
    assert [(r["i"],r["j"],r["length_m"]) for r in result["samples"]]==[
        (0,11,100),(0,21,200),(0,31,300),(10,21,100),(10,31,200),(20,31,100)]
    expected_ratios=[f32(11/100),f32(21/200),f32(31/300),f32(11/100),f32(21/200),f32(11/100)]
    assert [row["translation_ratio"] for row in result["samples"]]==expected_ratios
    total=f32(0)
    for value in expected_ratios:total=f32(total+value)
    expected=f32(total/f32(len(expected_ratios)))
    assert result["translation_ratio_mean"]==expected
    assert result["translation_percent_mean"]==expected*100
    by_length_equal_weight=math.fsum(result["by_length"][str(length)]["translation_ratio_mean"] for length in (100,200,300))/3
    assert not math.isclose(expected,by_length_equal_weight,abs_tol=1e-6,rel_tol=1e-6)
    assert result["by_length"]["400"]["count"]==0
    assert result["by_length"]["400"]["translation_ratio_mean"] is None


def test_synthetic_scalar_rotation_cast_does_not_change_own_rpe():
    truth=line_poses([0,101]);pred=truth.copy()
    angle=1e-4;c,s=np.cos(angle),np.sin(angle)
    pred[1,:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
    # The float trace rounds to 3; the separate float64 RPE keeps the rotation.
    _,official_scalar=kitti_error_components(pred[1])
    assert official_scalar==0
    assert rpe(pred,truth,[(0,1)])["rotation_rmse_deg"]>0
