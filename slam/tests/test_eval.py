"""Numerical tests require NumPy; not executed during pure-stdlib preparation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
import numpy as np
from eval_traj import align_no_scale,distance_pairs,metrics,normalize,rpe
from project_lidar import project,depth_edges

def trajectory(n=30):
    p=np.repeat(np.eye(4)[None],n,axis=0)
    t=np.arange(n,dtype=float);p[:,0,3]=t*10;p[:,1,3]=np.sin(t/3)*8;p[:,2,3]=np.cos(t/4)
    return p

CFG={"segment_lengths_m":[100,200,300,400,500,600,700,800],"segment_start_stride_frames":10,"rpe_distance_m":100}

def test_se3_offset_rotation_and_nonzero_gt_origin():
    truth=trajectory();a=.4;R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=[100,-20,3]
    pred=T@truth
    # Synthetic identical motion with different nonzero global origins.
    m=metrics(pred,truth,CFG)
    assert m["ate"]["rmse_m"]<1e-10
    aligned,_=align_no_scale(pred,truth)
    assert np.max(np.abs(aligned-truth))<1e-9

def test_synthetic_metrics_invariant_to_independent_origins_with_drift():
    truth=trajectory();pred=truth.copy();t=np.arange(len(truth),dtype=float)
    # Deliberately nonzero synthetic translation and yaw drift; no real data.
    pred[:,:3,3]+=np.column_stack((.03*t,.1*np.sin(t/5),-.02*t))
    for i in range(len(pred)):
        a=.004*i;c,s=np.cos(a),np.sin(a)
        pred[i,:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
    def origin(yaw,translation):
        c,s=np.cos(yaw),np.sin(yaw);T=np.eye(4)
        T[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];T[:3,3]=translation
        return T
    reference=metrics(pred,truth,CFG)
    shifted=metrics(origin(-.35,[-40,12,8])@pred,origin(.7,[100,-20,3])@truth,CFG)
    np.testing.assert_allclose(shifted["ate"]["per_frame_m"],reference["ate"]["per_frame_m"],atol=1e-6,rtol=1e-6)
    for family in ("rpe_1frame","rpe_100m","kitti_segments"):
        left,right=shifted[family],reference[family]
        assert left["count"]==right["count"]
        assert [(r["i"],r["j"]) for r in left["samples"]]==[(r["i"],r["j"]) for r in right["samples"]]
        keys=("translation_percent","rotation_rad_per_m","rotation_deg_per_m","rotation_deg_per_100m") if family=="kitti_segments" else ("translation_m","rotation_rad")
        for key in keys:
            np.testing.assert_allclose([r[key] for r in left["samples"]],[r[key] for r in right["samples"]],atol=1e-6,rtol=1e-6)
        mean_keys=("translation_percent_mean","rotation_deg_per_100m_mean") if family=="kitti_segments" else ("translation_rmse_m","rotation_rmse_deg")
        for key in mean_keys:
            np.testing.assert_allclose(left[key],right[key],atol=1e-6,rtol=1e-6)

def test_scale_is_not_corrected():
    truth=trajectory();pred=truth.copy();pred[:,:3,3]*=1.2
    assert metrics(pred,truth,CFG)["ate"]["rmse_m"]>1

def test_short_sequence_has_null_segments_not_zero():
    m=metrics(trajectory(3),trajectory(3),CFG)
    assert m["kitti_segments"]["count"]==0
    assert m["kitti_segments"]["translation_percent_mean"] is None
    assert m["rpe_100m"]["translation_rmse_m"] is None

def test_strict_distance_pair_selection_and_mm_degree_units():
    truth=np.repeat(np.eye(4)[None],5,axis=0);truth[:,0,3]=[0,50,100,150,201]
    assert distance_pairs(truth,100)==[(0,3),(1,4),(2,4)]
    pred=truth.copy();pred[1,0,3]+=.001
    r=rpe(pred,truth,[(0,1)])
    assert abs(r["translation_rmse_m"]-.001)<1e-12
    assert r["rotation_rmse_deg"]==0

def test_zbuffer_depth_bounds_and_occlusion():
    P=np.array([[10,0,5,0],[0,10,5,0],[0,0,1,0]],float)
    points=np.array([[0,0,2],[0,0,3],[0,0,-1],[100,0,2]],float)
    depth,stats=project(points,P,np.eye(4),10,10,1,80)
    assert depth[5,5]==2 and stats["zbuffer_pixels"]==1 and stats["in_image_points"]==2
    assert stats["occluded_or_same_pixel_points"]==1
    assert stats["in_image_fraction_of_input"]==.5

def test_sparse_holes_do_not_become_depth_edges():
    depth=np.full((5,5),np.inf);depth[1,1]=2
    assert not depth_edges(depth,1,2).any()
    depth[1,2]=5
    assert depth_edges(depth,1,2).sum()==2
