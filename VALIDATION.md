# Validation

## Local execution

The run file was regenerated on 31 October 2026 from the current `ros2-sensor-fusion-sim` checkout, seed 3 and 120 seconds. Its source revision and SHA-256 hashes are in `results/data_provenance.json`. It has 1,200 samples, 120 GPS fixes and 5 rejected fixes.

Using Node 24.22.2/npm 11.19.1 in the existing `node:24-alpine` image with a private G-backed dependency/cache mount:

```bash
npm ci
npm run typecheck
npm test -- --reporter=default --reporter=junit --outputFile=results/vitest-junit.xml
npm run build
node --check scripts/bench.mjs
```

All 14 tests passed, none skipped; strict TypeScript checking and the production build passed. The initial revision is recorded in `results/local_checks.json`. The capture fix also passed Node syntax and TypeScript checks. Local execution did not launch a browser or collect rendering measurements.

## CI rendering

The workflow installs Playwright's matching Chromium, runs the same typecheck/tests/build, then executes `npm run bench`. The headless browser uses SwiftShader and records the actual renderer and browser version. It performs the minimal playback/scrubber/layer/point-count interactions and the original three rendering windows at 10k/100k/1M points.

Results must have positive frame counts, finite FPS/percentiles, the requested point count, positive draw calls, a 1280×720 canvas and a renderer string containing SwiftShader. CI produces full JSON, interaction assertions and fresh screenshots in the uploaded artifact.

[CI run 41269265850](https://github.com/kianbqh/webgl-3d-viewer/actions/runs/41269265850) passed on code revision `9796ebda1732633b5223eb704ccfbda7607dac7f`. It repeated all 14 tests/typechecking/build and produced the checked-in rendering JSON and screenshots. Runner: AMD EPYC 9V45, four exposed CPUs, Node 24.22.1, Chromium 156.0.8078.4, ANGLE/Vulkan SwiftShader Device (Subzero).

The three groups measured 154.6426, 52.9864 and 7.4357 FPS at 10k/100k/1M points, with 463/158/22 frame intervals and nine draw calls each. Frame p50/p95 values were 6.4/9.7, 19.0/32.6 and 134.6/137.7 ms. The JavaScript bundle is 785,925 bytes; Node's gzip output is 207,851 bytes. These values use the CI report's compression method.

Interaction checks passed Play/Pause, timeline dragging (61.3 s), point-count switching to 100k, and cloud-layer switching (9→8→9 draw calls). Exactly one canvas remained, no page errors occurred, and screenshots were taken after an explicit final-frame synchronization and render-loop freeze. The interactive HUD reports unsynchronized callback frequency; the benchmark uses a synchronous readback each frame.

Only documentation and the verified result files were added after the tested code revision; rendering was not repeated for that report commit.
