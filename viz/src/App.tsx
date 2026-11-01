import { useEffect, useMemo, useState } from 'react';
import Viewer, { type HudStats, type Layers } from './components/Viewer';
import { parseRun } from './lib/parse';
import { indexAtTime } from './lib/timeline';
import type { ReplayChoice, Run } from './lib/types';

const params = new URLSearchParams(window.location.search);
const BENCH = params.get('bench') ? { seconds: Number(params.get('seconds') ?? 3) } : null;
const POINT_OPTIONS = [10_000, 100_000, 1_000_000, 2_000_000];
const FAULT_LABELS: Record<string, string> = { gps_dropout: 'GPS dropout', gps_bias: 'GPS bias', gps_spike_regime: 'GPS spikes', odom_dropout: 'Odometry dropout' };

export default function App() {
  const [choices, setChoices] = useState<ReplayChoice[]>([]);
  const [episode, setEpisode] = useState(params.get('episode') ?? 'normal');
  const [run, setRun] = useState<Run | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [points, setPoints] = useState(Number(params.get('points') ?? 100_000));
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(!BENCH);
  const [speed, setSpeed] = useState(4);
  const [hud, setHud] = useState<HudStats | null>(null);
  const [layers, setLayers] = useState<Layers>({ truth: true, ekf: true, dr: false, gps: true, ellipses: true, cloud: true });

  useEffect(() => {
    fetch('./data/replays/index.json').then((response) => {
      if (!response.ok) throw new Error(`Replay manifest: HTTP ${response.status}`);
      return response.json();
    }).then((entries: ReplayChoice[]) => {
      if (!Array.isArray(entries) || !entries.some((entry) => entry.id === 'normal') || !entries.some((entry) => entry.id === 'fault-collision')) throw new Error('Safety replay manifest is incomplete');
      setChoices([...entries, { id: 'fusion-drive', title: 'Fusion drive · 120 seconds', path: './data/run.json' }]);
    }).catch((error) => setErr(String(error)));
  }, []);
  useEffect(() => {
    if (!choices.length) return;
    const choice = choices.find((entry) => entry.id === episode);
    if (!choice) { setErr('Unknown replay'); return; }
    const abort = new AbortController();
    setErr(null);
    fetch(choice.path, { signal: abort.signal }).then((response) => {
      if (!response.ok) throw new Error(`Replay: HTTP ${response.status}`);
      return response.json();
    }).then((raw) => {
      const loaded = parseRun(raw);
      const end = loaded.samples[loaded.samples.length - 1].t;
      const requested = params.get('time') === 'end' ? end : Number(params.get('time') ?? 0);
      setTime(Math.min(end, Math.max(0, Number.isFinite(requested) ? requested : 0)));
      setRun(loaded);
      setHud(null);
    }).catch((error) => { if (error.name !== 'AbortError') setErr(String(error)); });
    return () => abort.abort();
  }, [episode, choices]);
  const tMax = run ? run.samples[run.samples.length - 1].t : 0;
  useEffect(() => {
    if (!playing || !run) return;
    let raf = 0, prev = performance.now();
    const step = (now: number) => {
      setTime((t) => { const next = t + ((now - prev) / 1000) * speed; return next > tMax ? 0 : next; });
      prev = now; raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, run, tMax]);
  const bench = useMemo(() => BENCH, []);
  if (err) return <div className="error">Failed to load replay: {err}</div>;
  if (!run) return <div className="loading">Loading safety replay…</div>;
  const frame = run.samples[indexAtTime(run.samples, time)];
  const faults = Object.entries(FAULT_LABELS).filter(([key]) => frame.faults?.[key]).map(([, label]) => label);
  const outcome = String(run.meta.outcome ?? 'recorded drive');
  const rawLabel = run.world ? 'held GPS' : 'dead reckoning';
  return (
    <div className="app">
      <Viewer run={run} points={points} layers={layers} time={time} bench={bench} capture={params.get('capture') === '1'} onStats={setHud} />
      <div className="panel">
        <h1>Autonomy replay</h1>
        <div className="row"><label>episode <select aria-label="episode" value={episode} onChange={(event) => { setPlaying(false); setEpisode(event.target.value); }}>
          {choices.map((choice) => <option key={choice.id} value={choice.id}>{choice.title}</option>)}
        </select></label></div>
        <div className={`status ${outcome}`} data-testid="episode-state" data-episode={String(run.meta.id ?? 'fusion-drive')}>
          <strong>Recorded outcome: {outcome}</strong><span>Frame: {frame.outcome ?? 'replay'}</span>
          <span data-testid="fault-state">Faults now: {faults.length ? faults.join(', ') : 'none'}</span>
          {frame.clearance !== undefined && <span>True clearance: {frame.clearance.toFixed(3)} m</span>}
        </div>
        <div className="row">
          <button onClick={() => setPlaying((value) => !value)}>{playing ? 'Pause' : 'Play'}</button>
          <input aria-label="time" type="range" min={0} max={tMax} step={0.05} value={time}
            onKeyDown={(event) => {
              if (event.key === 'End') {
                event.preventDefault(); setPlaying(false); setTime(tMax);
              }
            }}
            onChange={(event) => {
              setPlaying(false);
              const requested = Number(event.target.value);
              // Collisions can end between controller ticks; select their
              // actual terminal sample when scrubbing close to the end.
              setTime(requested >= tMax - 0.025 ? tMax : requested);
            }} />
          <span>{time.toFixed(2)} / {tMax.toFixed(2)} s</span>
        </div>
        <div className="row">
          <label>speed <select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
            {[1, 2, 4, 8].map((value) => <option key={value} value={value}>{value}×</option>)}</select></label>
          <label>points <select aria-label="points" value={points} onChange={(event) => setPoints(Number(event.target.value))}>
            {POINT_OPTIONS.map((value) => <option key={value} value={value}>{value.toLocaleString()}</option>)}</select></label>
        </div>
        <div className="row layers">
          {(Object.keys(layers) as (keyof Layers)[]).map((key) => (
            <label key={key}><input type="checkbox" checked={layers[key]} onChange={() => setLayers({ ...layers, [key]: !layers[key] })} />{key === 'dr' ? rawLabel : key}</label>
          ))}
        </div>
        {hud && <div className="hud">{hud.fps.toFixed(0)} FPS · {hud.points.toLocaleString()} pts · {hud.drawCalls} draw calls · estimate RMSE {hud.rmse.toFixed(2)} m</div>}
        <div className="legend">white = truth · blue = estimate · amber/red = GPS accepted/rejected · cyan = 2σ ellipses · brown = known obstacles · green ring = goal</div>
        <div className="legend">Synthetic 2-D world · procedural point cloud · offline perception is separate</div>
      </div>
    </div>
  );
}
