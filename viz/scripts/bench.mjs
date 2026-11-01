// Identical synchronized-frame protocol in native NVIDIA and explicitly software CI modes.
import assert from 'node:assert/strict';
import { spawn, execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { gzipSync } from 'node:zlib';
import { chromium } from 'playwright-core';

const hardware = process.env.BENCH_MODE === 'native-gpu';
const checksOnly = process.env.BENCH_CHECKS_ONLY === '1';
assert(hardware || !process.env.BENCH_MODE || process.env.BENCH_MODE === 'ci-software', 'Unknown BENCH_MODE');
const POINTS = (process.env.BENCH_POINTS ?? '10000,100000,1000000').split(',').map(Number);
const SECONDS = Number(process.env.BENCH_SECONDS ?? 3);
const PORT = Number(process.env.BENCH_PORT ?? 4176);
const output = path.resolve(process.env.BENCH_OUTPUT ?? (hardware ? 'results/native-gpu' : 'results/ci-software'));
const base = `http://127.0.0.1:${PORT}`;
assert(POINTS.length > 0 && POINTS.every((n) => Number.isInteger(n) && n > 0 && n <= 2_000_000));
assert(Number.isFinite(SECONDS) && SECONDS > 0 && SECONDS <= 60);
if (hardware) {
  assert.equal(process.platform, 'win32', 'Native GPU report requires Windows');
  for (const key of ['CHROME_PATH', 'BENCH_PROFILE', 'BENCH_CACHE']) {
    assert(process.env[key] && /^G:[\\/]/i.test(path.resolve(process.env[key])), `${key} must use G:`);
  }
}
mkdirSync(output, { recursive: true });
const manifest = JSON.parse(readFileSync('public/data/replays/index.json', 'utf8'));
const episode = manifest.find((entry) => entry.id === 'normal');
assert(episode, 'Actual normal safety export is required');
const episodeBytes = readFileSync(path.join('public', episode.path.replace(/^\.\//, '')));
const sourceEpisode = JSON.parse(episodeBytes);
const failureEpisode = JSON.parse(readFileSync('public/data/replays/fault-collision.json', 'utf8'));
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const report = {
  mode: hardware ? 'native Windows Chromium, NVIDIA hardware WebGL via ANGLE D3D11' : 'headless Chromium, SwiftShader (CPU software WebGL)',
  executionOrigin: process.env.GITHUB_ACTIONS === 'true' ? 'GitHub Actions' : 'local',
  revision: process.env.GITHUB_SHA ?? execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
  sourceSha256: Object.fromEntries(['scripts/bench.mjs', 'src/App.tsx', 'src/components/Viewer.tsx', 'src/lib/fps.ts'].map((file) => [file, hash(readFileSync(file))])),
  workflowRun: process.env.GITHUB_RUN_ID ?? null,
  startedUtc: new Date().toISOString(), browserVersion: null, nodeVersion: process.version,
  platform: `${os.platform()} ${os.release()} ${os.arch()}`, cpu: os.cpus()[0]?.model, logicalCpus: os.cpus().length,
  episode: { id: episode.id, sha256: hash(episodeBytes), dataKind: sourceEpisode.meta.data_kind, scenario: sourceEpisode.meta.scenario,
    seed: sourceEpisode.meta.seed, outcome: sourceEpisode.meta.outcome, world: sourceEpisode.world },
  bundles: readdirSync('dist/assets').filter((name) => name.endsWith('.js')).map((name) => {
    const bytes = readFileSync(`dist/assets/${name}`);
    return { name, bytes: bytes.length, gzipBytes: gzipSync(bytes).length, sha256: hash(bytes) };
  }),
  protocol: { secondsPerGroup: SECONDS, warmupFrameIntervals: 31, viewport: [1280, 720], pixelRatio: 1,
    cloudSeed: 7, camera: 'orbit', completion: 'one-pixel synchronous readPixels after each render',
    intervalClock: 'requestAnimationFrame timestamp', percentiles: 'linear interpolation at p*(N-1)',
    fps: '1000 / mean(frameIntervalsMs)', hud: 'unsynchronized submission frequency, not the benchmark' },
  completed: false, results: [],
  checksOnly,
};
if (hardware) {
  report.nvidiaSmi = execFileSync(process.env.NVIDIA_SMI ?? 'nvidia-smi', [
    '--query-gpu=name,driver_version,memory.total,pci.bus_id', '--format=csv,noheader'], { encoding: 'utf8' }).trim();
  assert.match(report.nvidiaSmi, /RTX\s*3080/i);
}
const save = () => writeFileSync(path.join(output, 'bench.json'), JSON.stringify(report, null, 2) + '\n');
const server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', 'preview', '--host', '127.0.0.1', '--port', String(PORT), '--strictPort'], { stdio: ['ignore', 'pipe', 'pipe'] });
let serverOutput = '', startupError, browser, context;
const isRunning = () => server.exitCode === null && server.signalCode === null;
server.on('error', (error) => { startupError = error; });
for (const stream of [server.stdout, server.stderr]) stream.on('data', (data) => { serverOutput = (serverOutput + data).slice(-8000); });
async function waitForServer() {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    if (startupError) throw startupError;
    if (!isRunning()) throw new Error(`Preview exited: ${serverOutput}`);
    try { if ((await fetch(base, { signal: AbortSignal.timeout(1000) })).ok) return; } catch { /* bounded retry */ }
    await delay(100);
  }
  throw new Error(`Preview startup timed out: ${serverOutput}`);
}
async function ready(page, id) {
  await page.waitForSelector(`canvas`);
  await page.locator(`[data-testid="episode-state"][data-episode="${id}"]`).waitFor();
}
async function frozenScreenshot(page, name) {
  await page.evaluate(() => window.dispatchEvent(new Event('viewer:capture')));
  await page.waitForSelector('canvas[data-render-state="frozen"]');
  await page.screenshot({ path: path.join(output, name), timeout: 30_000 });
}
function assertRenderer(renderer) {
  if (hardware) {
    assert.match(renderer, /NVIDIA.*(?:RTX\s*3080|GeForce)/i);
    assert.doesNotMatch(renderer, /SwiftShader|llvmpipe|WARP|software|Microsoft Basic/i);
    assert.match(renderer, /D3D11|Direct3D11/i);
  } else assert.match(renderer, /SwiftShader/i);
}
async function checkInteractions() {
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(String(error)));
  await page.goto(`${base}/?episode=normal&points=10000&capture=1`);
  await ready(page, 'normal');
  const renderer = await page.evaluate(() => {
    const gl = document.querySelector('canvas').getContext('webgl2');
    const debug = gl.getExtension('WEBGL_debug_renderer_info');
    return debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  });
  assertRenderer(renderer);
  report.verifiedRenderer = renderer;
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  const slider = page.getByRole('slider', { name: 'time' });
  const maxTime = Number(await slider.getAttribute('max'));
  await slider.press('Home');
  const box = await slider.boundingBox();
  assert(box && box.width > 0 && maxTime > 0);
  await page.mouse.move(box.x + 8, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height / 2, { steps: 5 });
  await page.mouse.up();
  const scrubbedTime = Number(await slider.inputValue());
  assert(scrubbedTime > maxTime * 0.25 && scrubbedTime < maxTime * 0.75);
  await page.getByRole('button', { name: 'Play', exact: true }).click();
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  await page.getByLabel('points', { exact: true }).selectOption('100000');
  await page.waitForFunction(() => document.querySelector('.hud')?.textContent?.includes('100,000 pts'));
  assert.equal(await page.locator('canvas').count(), 1);
  const drawCalls = () => page.locator('.hud').innerText().then((text) => Number(text.match(/(\d+) draw calls/)?.[1]));
  const before = await drawCalls();
  assert(Number.isFinite(before) && before > 0);
  await page.getByLabel('cloud', { exact: true }).uncheck();
  await page.waitForFunction((n) => Number(document.querySelector('.hud')?.textContent?.match(/(\d+) draw calls/)?.[1]) === n, before - 1);
  const hidden = await drawCalls();
  await page.getByLabel('cloud', { exact: true }).check();
  await page.waitForFunction((n) => Number(document.querySelector('.hud')?.textContent?.match(/(\d+) draw calls/)?.[1]) === n, before);
  assert.match(await page.getByText('held GPS', { exact: true }).innerText(), /held GPS/);
  await frozenScreenshot(page, 'ui-normal.png');
  // Switching the actual loader to the exported collision case is part of correctness.
  await page.getByLabel('episode', { exact: true }).selectOption('fault-collision');
  await ready(page, 'fault-collision');
  await slider.press('End');
  await page.waitForFunction(() => document.querySelector('[data-testid="episode-state"]')?.textContent?.includes('Frame: collision'));
  const failureState = await page.getByTestId('episode-state').innerText();
  assert.match(failureState, /Recorded outcome: collision/);
  assert.match(await page.getByTestId('fault-state').innerText(), /GPS bias/);
  await frozenScreenshot(page, 'ui-failure.png');
  const expectedFailureRmse = failureEpisode.meta.metrics.loc_rmse.toFixed(2);
  await page.waitForFunction((rmse) => document.querySelector('.hud')?.textContent?.includes(`estimate RMSE ${rmse} m`), expectedFailureRmse);
  const failureHud = await page.locator('.hud').innerText();
  assert.equal(await page.locator('.error').count(), 0);
  assert.deepEqual(errors, []);
  writeFileSync(path.join(output, 'ui-smoke.json'), JSON.stringify({
    browserVersion: browser.version(), mode: report.mode, executionOrigin: report.executionOrigin,
    playPause: true, scrubbedTime, normalDurationS: maxTime, pointCountAfterChange: 100000,
    canvasCount: 1, drawCallsCloudVisible: before, drawCallsCloudHidden: hidden,
    replaySelection: ['normal', 'fault-collision'], failureState, failureHud, expectedFailureRmse,
    terminalRmseMatchesExport: true, captureFrozen: true, pageErrors: errors,
  }, null, 2) + '\n');
  await page.close();
}
try {
  await waitForServer();
  console.log(`BROWSER_START ${new Date().toISOString()}`);
  const args = hardware ? ['--use-gl=angle', '--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist', '--disable-software-rasterizer',
    '--disable-gpu-vsync', '--disable-frame-rate-limit', `--disk-cache-dir=${path.resolve(process.env.BENCH_CACHE)}`] :
    ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--disable-gpu-vsync', '--disable-frame-rate-limit'];
  report.browserArguments = args;
  const options = { ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}), headless: !hardware, args };
  if (hardware) {
    context = await chromium.launchPersistentContext(path.resolve(process.env.BENCH_PROFILE), { ...options, viewport: { width: 1280, height: 720 } });
    browser = context.browser();
  } else {
    browser = await chromium.launch(options);
    context = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  }
  report.browserVersion = browser.version();
  const cdp = await browser.newBrowserCDPSession();
  const system = await cdp.send('SystemInfo.getInfo');
  report.chromiumGpu = system.gpu;
  if (hardware) assert(system.gpu.devices.some((device) => device.vendorId === 0x10de), 'NVIDIA physical GPU missing in Chromium SystemInfo');
  await cdp.detach();
  await checkInteractions();
  console.log(`INTERACTIONS_COMPLETE ${new Date().toISOString()}`);
  for (const n of checksOnly ? [] : POINTS) {
    const groupStartedUtc = new Date().toISOString();
    console.log(`GROUP_START ${n} ${groupStartedUtc}`);
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', (error) => errors.push(String(error)));
    const t0 = Date.now();
    await page.goto(`${base}/?episode=normal&bench=1&points=${n}&seconds=${SECONDS}`);
    await ready(page, 'normal');
    const pageLoadToCanvasMs = Date.now() - t0;
    await page.waitForFunction(() => window.__bench, null, { timeout: (SECONDS + 300) * 1000 });
    const result = await page.evaluate(() => window.__bench);
    assert.equal(result.points, n);
    assert.equal(result.episode, 'normal');
    assert(Number.isInteger(result.frames) && result.frames > 0);
    assert.equal(result.frameIntervalsMs.length, result.frames);
    assert(result.frameIntervalsMs.every((x) => Number.isFinite(x) && x > 0));
    const mean = result.frameIntervalsMs.reduce((a, b) => a + b, 0) / result.frames;
    assert(Math.abs(result.fps - 1000 / mean) < 1e-8);
    for (const field of ['fps', 'frameMsP50', 'frameMsP95', 'cloudGenerateMs']) assert(Number.isFinite(result[field]) && result[field] > 0, field);
    assert(result.frameMsP95 >= result.frameMsP50);
    assert(Number.isInteger(result.drawCalls) && result.drawCalls > 0);
    assertRenderer(result.gpu);
    assert.deepEqual(result.canvas, [1280, 720]);
    assert.equal(result.captureFrozen, true);
    await page.waitForSelector('canvas[data-render-state="frozen"]');
    assert.deepEqual(errors, []);
    if (n === 100000) await page.screenshot({ path: path.join(output, 'screenshot_100k.png') });
    report.results.push({ ...result, pageLoadToCanvasMs, groupStartedUtc, groupFinishedUtc: new Date().toISOString() });
    save();
    console.log(`${n.toLocaleString()} points: ${result.fps.toFixed(2)} FPS; p50/p95 ${result.frameMsP50.toFixed(2)}/${result.frameMsP95.toFixed(2)} ms; ${result.drawCalls} draw calls; ${result.gpu}`);
    await page.close();
  }
  report.completed = true;
} finally {
  try { if (context) await context.close(); if (browser?.isConnected()) await browser.close(); }
  finally {
    if (server.pid && isRunning()) {
      server.kill('SIGTERM');
      const deadline = Date.now() + 5000;
      while (isRunning() && Date.now() < deadline) await delay(50);
      if (isRunning()) server.kill('SIGKILL');
    }
    report.finishedUtc = new Date().toISOString(); save();
    writeFileSync(path.join(output, 'preview.log'), serverOutput);
  }
}
