/** Candidate integration component; native-browser/typecheck acceptance is not run. */
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { camera0ToThree, frameAtTime, parseSlamReplay, type Pose12, type SlamReplay } from './replay';

type EdgeMode='accepted'|'rejected'|'temporal'|'none';
function poseMatrix(p: Pose12): THREE.Matrix4 {
  return new THREE.Matrix4().set(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],0,0,0,1);
}
function path(poses: Pose12[],color: number): THREE.Line {
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(poses.flatMap(camera0ToThree),3));
  return new THREE.Line(geometry,new THREE.LineBasicMaterial({color}));
}
export function SlamScene({run,index,edgeMode}:{run:SlamReplay;index:number;edgeMode:EdgeMode}) {
  const host=useRef<HTMLDivElement>(null),current=useRef(index),mode=useRef(edgeMode);
  current.current=index;mode.current=edgeMode;
  const [error,setError]=useState<string|null>(null);
  useEffect(()=>{
    if(!host.current || !run.frames.length)return;
    const el=host.current;let renderer:THREE.WebGLRenderer;
    try{renderer=new THREE.WebGLRenderer({antialias:true});}catch(e){setError(String(e));return;}
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));el.appendChild(renderer.domElement);
    const scene=new THREE.Scene();scene.background=new THREE.Color(0x10151d);
    const positions=run.frames.flatMap(f=>[f.odom,...(f.optimized?[f.optimized]:[]),...(f.truth?[f.truth]:[])]).map(camera0ToThree);
    const bounds=new THREE.Box3();positions.forEach(p=>bounds.expandByPoint(new THREE.Vector3(...p)));
    const center=bounds.getCenter(new THREE.Vector3()),size=bounds.getSize(new THREE.Vector3());
    const span=Math.max(size.x,size.y,size.z,10),camera=new THREE.PerspectiveCamera(55,1,.1,span*20);
    const fitRadius=Math.max(bounds.getBoundingSphere(new THREE.Sphere()).radius,5);
    camera.position.copy(center).add(new THREE.Vector3(span*.6,span*.7,span*.6));
    const controls=new OrbitControls(camera,renderer.domElement);controls.target.copy(center);controls.update();
    scene.add(path(run.frames.map(f=>f.odom),0x60a5fa));
    const optimized=run.frames.flatMap(f=>f.optimized?[f.optimized]:[]),truth=run.frames.flatMap(f=>f.truth?[f.truth]:[]);
    if(optimized.length)scene.add(path(optimized,0xe879f9));if(truth.length)scene.add(path(truth,0xf8fafc));
    const transform=new THREE.Matrix4().makeScale(1,-1,-1);
    const markers=[new THREE.AxesHelper(2),new THREE.AxesHelper(2),new THREE.AxesHelper(2)];
    markers.forEach(m=>{m.matrixAutoUpdate=false;scene.add(m);});
    const edges=new THREE.Group();scene.add(edges);
    let lastEdgeFrame=-2,lastMode:EdgeMode='none';
    const rebuildEdges=(i:number)=>{
      for(const item of [...edges.children]){edges.remove(item);const line=item as THREE.Line;line.geometry.dispose();(line.material as THREE.Material).dispose();}
      const segments:number[]=[];
      for(const l of run.loops){
        if(!l.endpoints_available || l.query_frame>i)continue;
        const show=mode.current==='accepted'?l.accepted:mode.current==='temporal'?l.reason==='temporal_exclusion':mode.current==='rejected'?!l.accepted && l.reason!=='temporal_exclusion':false;
        if(!show)continue;
        segments.push(...[run.frames[l.candidate_frame].odom,run.frames[l.query_frame].odom].flatMap(camera0ToThree));
      }
      if(segments.length){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(segments,3));edges.add(new THREE.LineSegments(g,new THREE.LineBasicMaterial({color:mode.current==='accepted'?0x4ade80:mode.current==='temporal'?0x64748b:0xf59e0b,transparent:true,opacity:.5})));}
      lastEdgeFrame=i;lastMode=mode.current;
    };
    const failure=new THREE.Mesh(new THREE.SphereGeometry(Math.max(span*.004,.2),12,8),new THREE.MeshBasicMaterial({color:0xef4444}));
    failure.visible=run.status==='failed';failure.position.set(...camera0ToThree(run.frames[run.frames.length-1].odom));scene.add(failure);
    // The red marker is LAST ACCEPTED location, never the unknown failed-frame pose.
    let raf=0;
    const render=()=>{
      raf=requestAnimationFrame(render);const i=Math.max(0,Math.min(current.current,run.frames.length-1)),f=run.frames[i];
      const tracks=[f.odom,f.optimized,f.truth];tracks.forEach((p,k)=>{markers[k].visible=!!p;if(p)markers[k].matrix.multiplyMatrices(transform,poseMatrix(p));});
      if(i!==lastEdgeFrame || mode.current!==lastMode)rebuildEdges(i);
      controls.update();renderer.render(scene,camera);
    };
    const resize=()=>{
      const w=el.clientWidth,h=el.clientHeight;renderer.setSize(w,h);
      camera.aspect=Math.max(w,1)/Math.max(h,1);
      // Fit the sphere enclosing every 3D track to the limiting viewport angle.
      const halfV=THREE.MathUtils.degToRad(camera.fov/2),halfH=Math.atan(Math.tan(halfV)*camera.aspect);
      const distance=1.12*fitRadius/Math.sin(Math.min(halfV,halfH));
      const direction=camera.position.clone().sub(controls.target).normalize();
      camera.position.copy(center).addScaledVector(direction,distance);controls.target.copy(center);
      camera.far=Math.max(span*20,distance+2*fitRadius);camera.updateProjectionMatrix();controls.update();
    };
    resize();window.addEventListener('resize',resize);raf=requestAnimationFrame(render);
    return()=>{cancelAnimationFrame(raf);window.removeEventListener('resize',resize);controls.dispose();scene.traverse((o:any)=>{o.geometry?.dispose();if(Array.isArray(o.material))o.material.forEach((m:THREE.Material)=>m.dispose());else o.material?.dispose();});renderer.dispose();el.removeChild(renderer.domElement);};
  },[run]);
  return <>{error && <div role="alert">3D replay unavailable: {error}</div>}<div ref={host} style={{height:480,minHeight:320,width:'100%'}} data-testid="slam-viewport"/></>;
}

/** Root can mount this as a separate replay mode without modifying legacy Run. */
export default function KittiReplay({runUrl}:{runUrl:string}) {
  const [run,setRun]=useState<SlamReplay|null>(null),[error,setError]=useState<string|null>(null);
  const [time,setTime]=useState(0),[playing,setPlaying]=useState(false),[rate,setRate]=useState(4),[edgeMode,setEdgeMode]=useState<EdgeMode>('accepted');
  useEffect(()=>{const abort=new AbortController();setRun(null);setError(null);setPlaying(false);
    fetch(runUrl,{signal:abort.signal}).then(r=>{if(!r.ok)throw new Error(`Replay HTTP ${r.status}`);return r.json();}).then(raw=>{const r=parseSlamReplay(raw);setRun(r);setTime(r.frames[0]?.timestamp_s??0);}).catch(e=>{if(e.name!=='AbortError')setError(String(e));});
    return()=>abort.abort();},[runUrl]);
  useEffect(()=>{if(!playing || !run?.frames.length)return;let raf=0,last=performance.now();const end=run.frames[run.frames.length-1].timestamp_s;
    const tick=(now:number)=>{const delta=(now-last)/1000*rate;last=now;setTime(t=>{if(t+delta>=end){setPlaying(false);return end;}return t+delta;});raf=requestAnimationFrame(tick);};
    raf=requestAnimationFrame(tick);return()=>cancelAnimationFrame(raf);},[playing,rate,run]);
  const index=useMemo(()=>run?frameAtTime(run.frames,time):-1,[run,time]);
  if(error)return <div role="alert">Failed to load recorded SLAM replay: {error}</div>;
  if(!run)return <div>Loading recorded SLAM replay…</div>;
  const frame=run.frames[index],last=run.frames[run.frames.length-1],first=run.frames[0];
  const visibleLoops=run.loops.filter(l=>l.query_frame<=(frame?.frame_id??-1) && l.reason!=='temporal_exclusion').slice(-12);
  return <section aria-label="Recorded KITTI SLAM replay">
    <h2>{run.dataset} · sequence {run.sequence}</h2>
    <p data-testid="slam-status">{run.status==='complete'?'Completed recorded run':'Failed attempt; accepted prefix only'} · {run.accepted_frames}/{run.expected_frames} original frames · optimization {run.optimized_available?'available':'unavailable'}</p>
    {run.failure && <p role="status">Failure: {String(run.failure.message??'see raw journal')}. Failed frame: {String(run.failure.frame??'unknown')}. Red marker = last accepted location; the failed-frame pose is unknown.</p>}
    {frame?<>
      <SlamScene run={run} index={index} edgeMode={edgeMode}/>
      <div><button onClick={()=>setPlaying(p=>!p)}>{playing?'Pause':'Play'}</button>
        <input aria-label="SLAM time" type="range" min={first.timestamp_s} max={last.timestamp_s} step="any" value={time} onChange={e=>{setPlaying(false);setTime(Number(e.target.value));}}/>
        <span data-testid="slam-frame" data-frame-id={frame.frame_id}>Frame {frame.frame_id} · original timestamp {frame.timestamp_s.toFixed(6)} s</span>
        <label> playback <select value={rate} onChange={e=>setRate(Number(e.target.value))}>{[1,4,8].map(n=><option key={n} value={n}>{n}×</option>)}</select></label>
        <label> loops <select aria-label="SLAM loop layer" value={edgeMode} onChange={e=>setEdgeMode(e.target.value as EdgeMode)}>{['accepted','rejected','temporal','none'].map(s=><option key={s} value={s}>{s}</option>)}</select></label>
      </div>
      <p>Blue = ICP odometry · purple = optimized · white = ground truth. Loop green = geometrically accepted; amber = rejected. All tracks use the first camera0 frame, with full 3D poses and no scale alignment. No point cloud is included.</p>
      <p>Registration: {frame.registration?.registration_ms??'unavailable'} ms · residual {frame.registration?.rmse_m??'unavailable'} m · inliers {frame.registration?.inliers??'unavailable'}</p>
      <table data-testid="slam-loop-decisions"><thead><tr><th>Query → candidate</th><th>Decision</th><th>Reason</th></tr></thead><tbody>{visibleLoops.map(l=><tr key={l.proposal_id} data-proposal-id={l.proposal_id}><td>{l.query_frame} → {l.candidate_frame}</td><td>{l.accepted?'geometry accepted':'rejected'}</td><td>{l.reason}</td></tr>)}</tbody></table>
    </>:<p>No frame was accepted. The failure journal is retained; no trajectory has been synthesized.</p>}
    <small>{run.attribution}</small>
  </section>;
}
