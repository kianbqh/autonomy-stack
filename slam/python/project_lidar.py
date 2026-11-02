"""KITTI odometry projection: P2 @ Tr @ p_lidar, with nearest-depth z buffer."""
from __future__ import annotations
import numpy as np

def load_calibration(path):
    values={}
    for line in path.read_text().splitlines():
        key,raw=line.split(":",1)
        if key in values:raise ValueError("duplicate calibration key")
        values[key]=np.asarray([float(x) for x in raw.split()])
    if "Tr" not in values or "P2" not in values:raise ValueError("missing odometry Tr/P2")
    Tr=np.eye(4);Tr[:3,:]=values["Tr"].reshape(3,4);P2=values["P2"].reshape(3,4)
    if not np.isfinite(Tr).all() or not np.isfinite(P2).all() or not np.allclose(Tr[:3,:3].T@Tr[:3,:3],np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(Tr[:3,:3]),1,atol=1e-5):raise ValueError("invalid calibration SE3")
    return P2,Tr

def load_scan(path):
    if path.stat().st_size==0 or path.stat().st_size%16:raise ValueError("scan length not Nx4 float32")
    cloud=np.fromfile(path,dtype="<f4").reshape(-1,4)
    if not np.isfinite(cloud).all():raise ValueError("nonfinite scan")
    return cloud[:,:3].astype(np.float64)

def project(points,P2,Tr,height,width,minimum,maximum):
    if len(points)==0:raise ValueError("empty projection cloud")
    homogeneous=np.c_[points,np.ones(len(points))]
    camera=(Tr@homogeneous.T).T
    image=(P2@camera.T).T
    eligible=(camera[:,2]>=minimum)&(camera[:,2]<=maximum)&(image[:,2]>0)
    indices=np.flatnonzero(eligible)
    pixels=image[indices,:2]/image[indices,2,None]
    inside=(pixels[:,0]>=0)&(pixels[:,0]<width)&(pixels[:,1]>=0)&(pixels[:,1]<height)
    indices=indices[inside];pixels=pixels[inside]
    # Floor projects every accepted continuous coordinate to a valid pixel, with exact border semantics.
    u=np.floor(pixels[:,0]).astype(int);v=np.floor(pixels[:,1]).astype(int)
    z=camera[indices,2]
    depth=np.full((height,width),np.inf,dtype=np.float64)
    np.minimum.at(depth,(v,u),z)
    visible=np.isfinite(depth)
    return depth,{"input_points":len(points),"positive_depth_range_points":int(eligible.sum()),
                  "in_image_points":len(indices),"in_image_fraction_of_input":len(indices)/len(points),
                  "zbuffer_pixels":int(visible.sum()),"projection_coverage":float(visible.mean()),
                  "occluded_or_same_pixel_points":len(indices)-int(visible.sum()),
                  "projection":"P2*Tr*p_L; rectified camera0; no R0_rect", "zbuffer":"nearest rectified camera0 depth, floor pixel"}

def depth_edges(depth,threshold,radius):
    finite=np.isfinite(depth);edges=np.zeros(depth.shape,dtype=bool)
    for d in range(1,radius+1):
        for axis in (0,1):
            a=[slice(None),slice(None)];b=a.copy();a[axis]=slice(None,-d);b[axis]=slice(d,None)
            a,b=tuple(a),tuple(b)
            valid=finite[a]&finite[b]
            delta=np.zeros(valid.shape);np.subtract(depth[a],depth[b],out=delta,where=valid)
            hit=valid&(np.abs(delta)>=threshold)
            edges[a]|=hit;edges[b]|=hit
    return edges

def overlay(image,depth,maximum):
    import cv2
    out=image.copy();valid=np.isfinite(depth)
    colors=cv2.applyColorMap(np.clip(depth[valid]/maximum*255,0,255).astype(np.uint8).reshape(-1,1),cv2.COLORMAP_TURBO).reshape(-1,3)
    out[valid]=(0.25*out[valid]+0.75*colors).astype(np.uint8)
    return out
