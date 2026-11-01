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

All 14 tests passed, none skipped; strict TypeScript checking and the production build passed. The JavaScript bundle is 785.60 kB, gzip 209.53 kB. The benchmark script passed its syntax check. Local execution did not launch a browser or collect rendering measurements.

## CI rendering

The workflow installs Playwright's matching Chromium, runs the same typecheck/tests/build, then executes `npm run bench`. The headless browser uses SwiftShader and records the actual renderer and browser version. It performs the minimal playback/scrubber/layer/point-count interactions and the original three rendering windows at 10k/100k/1M points.

Results must have positive frame counts, finite FPS/percentiles, the requested point count, positive draw calls, a 1280×720 canvas and a renderer string containing SwiftShader. CI produces full JSON, interaction assertions and fresh screenshots in the uploaded artifact.
