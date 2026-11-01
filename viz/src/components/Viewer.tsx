import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { Run } from '../lib/types';
import { covEllipse, ellipsePoints, enuToThree } from '../lib/geometry';
import { indexAtTime, poseAt, rmseUpTo } from '../lib/timeline';
import { boundsOf, generateCloud } from '../lib/pointcloud';
import { FrameStats } from '../lib/fps';

export interface Layers { truth: boolean; ekf: boolean; dr: boolean; gps: boolean; ellipses: boolean; cloud: boolean }
interface Props { run: Run; points: number; layers: Layers; time: number; bench?: { seconds: number } | null; capture?: boolean; onStats?: (s: HudStats) => void }
export interface HudStats { fps: number; drawCalls: number; triangles: number; points: number; rmse: number }

function line(xy: number[][], color: number) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(xy.flatMap(([x, y]) => enuToThree(x, y, 0.25)), 3));
  return new THREE.Line(g, new THREE.LineBasicMaterial({ color }));
}

export default function Viewer({ run, points, layers, time, bench, capture = false, onStats }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const timeRef = useRef(time);
  timeRef.current = time;
  const groups = useRef<Record<keyof Layers, THREE.Object3D> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const el = host.current!;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
    } catch (e) { setError(String(e)); return; }
    renderer.setPixelRatio(1);
    renderer.setSize(el.clientWidth, el.clientHeight);
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0e1116);
    const s = run.samples;
    const b = boundsOf([...s.map((x) => x.truth), ...(run.world ? [...run.world.obstacles, run.world.goal] : [])], 10);
    const cx = (b.minX + b.maxX) / 2, cy = (b.minY + b.maxY) / 2, span = Math.max(b.maxX - b.minX, b.maxY - b.minY);
    const camera = new THREE.PerspectiveCamera(55, el.clientWidth / el.clientHeight, 0.1, span * 10);
    camera.position.set(...enuToThree(cx - span * 0.4, cy - span * 0.7, span * 0.6));
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(...enuToThree(cx, cy, 0));
    controls.update();

    const grid = new THREE.GridHelper(Math.ceil(span / 10) * 10, Math.ceil(span / 10), 0x334155, 0x1f2937);
    grid.position.set(...enuToThree(cx, cy, 0));
    scene.add(grid);
    if (run.world) {
      for (const [x, y, radius] of run.world.obstacles) {
        const obstacle = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 0.3, 32), new THREE.MeshBasicMaterial({ color: 0xb45309 }));
        obstacle.position.set(...enuToThree(x, y, 0.15));
        scene.add(obstacle);
      }
      const goal = new THREE.Mesh(new THREE.RingGeometry(0.5, 0.65, 32), new THREE.MeshBasicMaterial({ color: 0x4ade80, side: THREE.DoubleSide }));
      goal.geometry.rotateX(-Math.PI / 2);
      goal.position.set(...enuToThree(...run.world.goal, 0.05));
      scene.add(goal);
    }

    const truth = line(s.map((x) => [x.truth[0], x.truth[1]]), 0xffffff);
    const ekf = line(s.map((x) => [x.ekf[0], x.ekf[1]]), 0x3b82f6);
    const dr = line(s.map((x) => [x.dr[0], x.dr[1]]), 0x9ca3af);
    const gps = new THREE.Group();
    for (const accepted of [true, false]) {
      const fixes = run.gps.filter((g) => g.accepted === accepted);
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(fixes.flatMap((f) => enuToThree(f.xy[0], f.xy[1], 0.3)), 3));
      gps.add(new THREE.Points(g, new THREE.PointsMaterial({ color: accepted ? 0xf59e0b : 0xef4444, size: accepted ? 0.6 : 1.0 })));
    }
    // 2-sigma covariance ellipses, merged into a single LineSegments draw call.
    const segs: number[] = [];
    const ellipseStride = Math.max(1, Math.round(Number(run.meta.sample_hz ?? 10)));
    for (let i = 0; i < s.length; i += ellipseStride) {
      const p = ellipsePoints(s[i].ekf[0], s[i].ekf[1], covEllipse(...s[i].cov, 2), 24);
      for (let k = 0; k + 3 < p.length; k += 2) segs.push(...enuToThree(p[k], p[k + 1], 0.2), ...enuToThree(p[k + 2], p[k + 3], 0.2));
    }
    const eg = new THREE.BufferGeometry();
    eg.setAttribute('position', new THREE.Float32BufferAttribute(segs, 3));
    const ellipses = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.6 }));

    // point cloud: one BufferGeometry, one draw call; ENU -> three swizzle done once on upload
    const tBuild = performance.now();
    const cloudData = generateCloud(points, b, 7);
    const p3 = new Float32Array(points * 3);
    for (let i = 0; i < points; i++) {
      p3[i * 3] = cloudData.positions[i * 3]; p3[i * 3 + 1] = cloudData.positions[i * 3 + 2]; p3[i * 3 + 2] = -cloudData.positions[i * 3 + 1];
    }
    const cg = new THREE.BufferGeometry();
    cg.setAttribute('position', new THREE.BufferAttribute(p3, 3));
    cg.setAttribute('color', new THREE.BufferAttribute(cloudData.colors, 3));
    const cloud = new THREE.Points(cg, new THREE.PointsMaterial({ size: 0.12, vertexColors: true, sizeAttenuation: true }));
    const cloudGenerateMs = performance.now() - tBuild;
    const px = new Uint8Array(4);

    const robot = new THREE.Mesh(new THREE.ConeGeometry(0.6, 1.8, 12), new THREE.MeshBasicMaterial({ color: 0x22c55e }));
    robot.geometry.rotateZ(-Math.PI / 2); // point along +x
    const ghost = new THREE.Mesh(robot.geometry, new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true }));
    scene.add(truth, ekf, dr, gps, ellipses, cloud, robot, ghost);
    groups.current = { truth, ekf, dr, gps, ellipses, cloud };

    const stats = new FrameStats();
    let raf = 0, lastHud = 0, benchStart = 0, done = false, frozen = false, captureRequested = false;
    const benchCfg = bench;
    const freeze = () => {
      if (frozen) return;
      frozen = true;
      cancelAnimationFrame(raf);
      renderer.getContext().finish();
      renderer.domElement.dataset.renderState = 'frozen';
    };
    const requestCapture = () => { captureRequested = true; };
    if (capture) window.addEventListener('viewer:capture', requestCapture);
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop);
      const t = timeRef.current;
      const pe = poseAt(s, t, 'ekf'), pt = poseAt(s, t, 'truth');
      const frame = s[indexAtTime(s, t)];
      (robot.material as THREE.MeshBasicMaterial).color.set(frame.outcome === 'collision' ? 0xef4444 : 0x22c55e);
      robot.position.set(...enuToThree(pe[0], pe[1], 0.6)); robot.rotation.y = pe[2];
      ghost.position.set(...enuToThree(pt[0], pt[1], 0.6)); ghost.rotation.y = pt[2];
      if (benchCfg) { // orbit the camera so every frame does real work
        const a = now / 4000;
        camera.position.set(...enuToThree(cx + Math.cos(a) * span * 0.8, cy + Math.sin(a) * span * 0.8, span * 0.5));
        camera.lookAt(...enuToThree(cx, cy, 0));
      } else controls.update();
      renderer.render(scene, camera);
      if (benchCfg) { const gl = renderer.getContext(); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px); } // sync: frame time includes GPU work
      stats.tick(now);
      if (benchCfg && !done) {
        if (!benchStart && stats.count > 30) { benchStart = now; stats.reset(); }
        if (benchStart && now - benchStart > benchCfg.seconds * 1000) {
          done = true;
          const gl = renderer.getContext();
          const dbg = gl.getExtension('WEBGL_debug_renderer_info');
          freeze();
          (window as any).__bench = {
            ...stats.summary(), frameIntervalsMs: stats.intervals(), points, cloudGenerateMs, drawCalls: renderer.info.render.calls,
            gpu: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
            vendor: dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
            webglVersion: gl.getParameter(gl.VERSION), shadingLanguageVersion: gl.getParameter(gl.SHADING_LANGUAGE_VERSION),
            canvas: [renderer.domElement.width, renderer.domElement.height],
            captureFrozen: true,
            episode: run.meta.id ?? 'fusion-drive', outcome: run.meta.outcome ?? null,
          };
        }
      }
      if (captureRequested) freeze();
      if ((captureRequested || (benchCfg && done) || now - lastHud > 500) && onStats) {
        lastHud = now;
        const sm = stats.summary();
        onStats({ fps: sm.fps, drawCalls: renderer.info.render.calls, triangles: renderer.info.render.triangles, points,
                  rmse: rmseUpTo(s, indexAtTime(s, t)) });
        if (!benchCfg) stats.reset();
      }
    };
    raf = requestAnimationFrame(loop);
    const onResize = () => {
      camera.aspect = el.clientWidth / el.clientHeight; camera.updateProjectionMatrix();
      renderer.setSize(el.clientWidth, el.clientHeight);
    };
    window.addEventListener('resize', onResize);
    return () => {
      cancelAnimationFrame(raf); window.removeEventListener('resize', onResize); window.removeEventListener('viewer:capture', requestCapture); controls.dispose();
      scene.traverse((o: any) => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
      renderer.dispose(); el.removeChild(renderer.domElement); groups.current = null;
    };
  }, [run, points, bench, capture]);

  useEffect(() => {
    const g = groups.current;
    if (g) (Object.keys(g) as (keyof Layers)[]).forEach((k) => { g[k].visible = layers[k]; });
  }, [layers, run, points]);

  if (error) return <div className="error">WebGL unavailable: {error}</div>;
  return <div ref={host} className="viewport" data-testid="viewport" />;
}
