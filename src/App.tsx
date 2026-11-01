import { useEffect, useMemo, useState } from 'react';
import Viewer, { type HudStats, type Layers } from './components/Viewer';
import { parseRun } from './lib/parse';
import type { Run } from './lib/types';

const params = new URLSearchParams(window.location.search);
const BENCH = params.get('bench') ? { seconds: Number(params.get('seconds') ?? 5) } : null;
const POINT_OPTIONS = [10_000, 100_000, 1_000_000, 2_000_000];

export default function App() {
  const [run, setRun] = useState<Run | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [points, setPoints] = useState(Number(params.get('points') ?? 100_000));
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(!BENCH);
  const [speed, setSpeed] = useState(4);
  const [hud, setHud] = useState<HudStats | null>(null);
  const [layers, setLayers] = useState<Layers>({ truth: true, ekf: true, dr: false, gps: true, ellipses: true, cloud: true });

  useEffect(() => {
    fetch(params.get('run') ?? './data/run.json').then((r) => r.json()).then((j) => setRun(parseRun(j))).catch((e) => setErr(String(e)));
  }, []);
  const tMax = run ? run.samples[run.samples.length - 1].t : 0;
  useEffect(() => {
    if (!playing || !run) return;
    let raf = 0, prev = performance.now();
    const step = (now: number) => {
      setTime((t) => { const n = t + ((now - prev) / 1000) * speed; return n > tMax ? 0 : n; });
      prev = now; raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, run, tMax]);
  const bench = useMemo(() => BENCH, []);

  if (err) return <div className="error">Failed to load run: {err}</div>;
  if (!run) return <div className="loading">Loading…</div>;
  return (
    <div className="app">
      <Viewer run={run} points={points} layers={layers} time={time} bench={bench} onStats={setHud} />
      <div className="panel">
        <h1>Sensor-fusion run viewer</h1>
        <div className="row">
          <button onClick={() => setPlaying((p) => !p)}>{playing ? 'Pause' : 'Play'}</button>
          <input aria-label="time" type="range" min={0} max={tMax} step={0.1} value={time}
                 onChange={(e) => { setPlaying(false); setTime(Number(e.target.value)); }} />
          <span>{time.toFixed(1)} / {tMax.toFixed(0)} s</span>
        </div>
        <div className="row">
          <label>speed <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            {[1, 2, 4, 8].map((v) => <option key={v} value={v}>{v}×</option>)}</select></label>
          <label>points <select value={points} onChange={(e) => setPoints(Number(e.target.value))}>
            {POINT_OPTIONS.map((v) => <option key={v} value={v}>{v.toLocaleString()}</option>)}</select></label>
        </div>
        <div className="row layers">
          {(Object.keys(layers) as (keyof Layers)[]).map((k) => (
            <label key={k}><input type="checkbox" checked={layers[k]} onChange={() => setLayers({ ...layers, [k]: !layers[k] })} />{k}</label>
          ))}
        </div>
        {hud && <div className="hud">{hud.fps.toFixed(0)} FPS · {hud.points.toLocaleString()} pts · {hud.drawCalls} draw calls ·
          EKF RMSE so far {hud.rmse.toFixed(2)} m</div>}
        <div className="legend">white = ground truth · blue = EKF · gray = dead reckoning · amber/red = GPS accepted/rejected · cyan = 2σ ellipses</div>
      </div>
    </div>
  );
}
