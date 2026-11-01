# webgl-3d-viewer

I wanted a way to inspect the trajectories and uncertainty from my sensor-fusion simulator together. This React, TypeScript and Three.js viewer replays a recorded drive with ground truth, EKF and dead-reckoning tracks, accepted and rejected GPS fixes, and 2σ position-covariance ellipses. Playback controls, a time scrubber, layer switches and an orbit camera help inspect individual parts of the run.

A deterministic procedural point cloud provides a separate rendering workload, from 10,000 to two million points. It uses one `BufferGeometry` and one point draw call; the covariance ellipses are merged into one line-segment geometry.

## Run

Use Node 24:

```bash
npm ci
npm run dev
```

Open the Vite URL, usually `http://localhost:5173`. Drag to orbit, scroll to zoom, pause and scrub the timeline, or change the layer and point-count controls. The HUD shows frame rate, draw calls and EKF position RMSE through the current time.

```bash
npm run typecheck
npm test
npm run build
npm run preview
```

A generated run is included in `public/data/run.json`. To export another one from [ros2-sensor-fusion-sim](https://github.com/kianbqh/ros2-sensor-fusion-sim), use its Python dependencies and point to that checkout:

```bash
python tools/export_run.py --fusion-repo /path/to/ros2-sensor-fusion-sim --seed 3 --duration 120
```

## Data and geometry

The included file was freshly exported from simulator commit `d47c6e01a08c8d25d2a07a395d1dd88b49b05d1b`, using seed 3 and a 120-second drive. It contains **1,200 pose samples at 10 Hz**, **120 GPS fixes**, and **5 rejected fixes**, in 174,518 bytes. [`results/data_provenance.json`](results/data_provenance.json) records the export time, source-file hashes and run-file SHA-256.

Trajectory, GPS and covariance values come from the simulated IMU, wheel odometry and GPS fusion run. The point cloud is generated procedurally with seed 7 to exercise point rendering independently of the simulation.

The parser checks increasing sample times, finite vectors and positive-semidefinite 2×2 position covariance. Ellipse axes and rotation come from the covariance eigendecomposition. ENU coordinates map into Three.js as `(east, up, -north)`. Pose interpolation follows the shortest yaw rotation, and time lookup uses binary search.

## Local checks

On 31 October 2026, **14/14 Vitest tests passed**, strict TypeScript checking passed, and the production build completed. The JavaScript bundle is **785.60 kB**, or **209.53 kB gzipped**. Checks used Node 24.22.2 and npm 11.19.1 in a Linux container backed by the local WSL2 disk. Details are in [`results/local_checks.json`](results/local_checks.json).

Tests cover covariance ellipse geometry (including the quadratic-form identity at each vertex), ENU conversion, timeline lookup and yaw interpolation, RMSE, run validation, deterministic cloud generation and frame statistics.

## Rendering checks

The [CI workflow](.github/workflows/ci.yml) runs headless Chromium with **SwiftShader software WebGL**. It checks Play/Pause, dragging the time slider, point-count changes and a layer toggle, then measures the original three workloads: **10,000**, **100,000** and **1,000,000** points at a **1280×720** viewport.

Each group uses 31 warmup frame intervals followed by a nominal three-second measurement window. The camera orbits throughout, and a one-pixel synchronous `readPixels` call runs after every render. Reported times are animation-frame intervals including that synchronization. `cloudGenerateMs` measures CPU cloud and geometry preparation before the first render. `pageLoadToCanvasMs` measures navigation to canvas detection.

The script validates the requested point count, a positive frame count, finite FPS and frame-time percentiles, draw calls, canvas size, and an actual renderer string containing `SwiftShader`. The `ci-swiftshader` artifact contains `bench.json`, browser and runner details, interaction results, and fresh screenshots. These measurements describe that CI software renderer.

To run the same headless checks in another environment:

```bash
npm run build
npx playwright-core install --with-deps chromium
npm run bench
```

The benchmark uses Playwright's managed Chromium by default; `CHROME_PATH` can select another Chromium executable. Its preview server has a bounded startup wait, and browser/server cleanup runs on success and failure.
Screenshot capture completes a final frame, synchronizes WebGL and stops the render loop before acknowledging the capture request.

## Files

| File | Purpose |
|---|---|
| `tools/export_run.py` | Export simulated trajectories, GPS decisions and covariance |
| `src/lib/parse.ts` | Validate the replay data contract |
| `src/lib/geometry.ts` | Covariance ellipses and coordinate conversion |
| `src/lib/timeline.ts` | Time lookup, pose interpolation and running RMSE |
| `src/lib/pointcloud.ts` | Procedural point positions and height colors |
| `src/components/Viewer.tsx` | Scene creation, rendering, cleanup and frame collection |
| `scripts/bench.mjs` | Headless interaction checks, rendering measurements and screenshots |
