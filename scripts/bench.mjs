// Three fixed rendering windows plus a small interaction check, all in headless SwiftShader.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import { setTimeout as delay } from 'node:timers/promises';
import { gzipSync } from 'node:zlib';
import { chromium } from 'playwright-core';

const POINTS = (process.env.BENCH_POINTS ?? '10000,100000,1000000').split(',').map(Number);
const SECONDS = Number(process.env.BENCH_SECONDS ?? 3);
assert(POINTS.length > 0 && POINTS.every((n) => Number.isInteger(n) && n > 0 && n <= 2_000_000));
assert(Number.isFinite(SECONDS) && SECONDS > 0 && SECONDS <= 60);
mkdirSync('results', { recursive: true });

const report = {
  mode: 'headless Chromium, SwiftShader (CPU software WebGL)',
  executionOrigin: process.env.GITHUB_ACTIONS === 'true' ? 'GitHub Actions' : 'local',
  revision: process.env.GITHUB_SHA ?? null,
  workflowRun: process.env.GITHUB_RUN_ID ?? null,
  startedUtc: new Date().toISOString(),
  browserVersion: null,
  nodeVersion: process.version,
  platform: `${os.platform()} ${os.release()} ${os.arch()}`,
  cpu: os.cpus()[0]?.model,
  logicalCpus: os.cpus().length,
  runDataSha256: createHash('sha256').update(readFileSync('public/data/run.json')).digest('hex'),
  bundles: readdirSync('dist/assets').filter((name) => name.endsWith('.js')).map((name) => {
    const bytes = readFileSync(`dist/assets/${name}`);
    return { name, bytes: bytes.length, gzipBytes: gzipSync(bytes).length, sha256: createHash('sha256').update(bytes).digest('hex') };
  }),
  seconds: SECONDS,
  warmupFrameIntervals: 31,
  viewport: [1280, 720],
  completed: false,
  results: [],
};
const save = () => writeFileSync('results/bench.json', JSON.stringify(report, null, 2) + '\n');
const server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', 'preview', '--host', '127.0.0.1', '--port', '4173', '--strictPort'], { stdio: ['ignore', 'pipe', 'pipe'] });
let serverOutput = '', startupError, browser;
const isRunning = () => server.exitCode === null && server.signalCode === null;
server.on('error', (error) => { startupError = error; });
for (const stream of [server.stdout, server.stderr]) stream.on('data', (data) => { serverOutput = (serverOutput + data).slice(-8000); });

async function waitForServer() {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    if (startupError) throw startupError;
    if (!isRunning()) throw new Error(`Preview exited: ${serverOutput}`);
    if (!serverOutput.includes('4173')) { await delay(100); continue; }
    try {
      const response = await fetch('http://127.0.0.1:4173/', { signal: AbortSignal.timeout(1000) });
      if (response.ok) return;
    } catch { /* retry until the bounded deadline */ }
    await delay(100);
  }
  throw new Error(`Preview startup timed out: ${serverOutput}`);
}

async function checkInteractions() {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(String(error)));
  await page.goto('http://127.0.0.1:4173/?points=10000&capture=1');
  await page.waitForSelector('canvas');
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  await page.getByRole('button', { name: 'Play', exact: true }).waitFor();
  const slider = page.getByRole('slider', { name: 'time' });
  await slider.press('Home');
  const box = await slider.boundingBox();
  assert(box && box.width > 0);
  await page.mouse.move(box.x + 8, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height / 2, { steps: 5 });
  await page.mouse.up();
  const scrubbedTime = Number(await slider.inputValue());
  assert(scrubbedTime > 20 && scrubbedTime < 100);
  await page.getByRole('button', { name: 'Play', exact: true }).click();
  await page.getByRole('button', { name: 'Pause', exact: true }).click();
  await page.getByLabel('points').selectOption('100000');
  await page.waitForFunction(() => document.querySelector('.hud')?.textContent?.includes('100,000 pts'));
  assert.equal(await page.locator('canvas').count(), 1);
  const drawCalls = () => page.locator('.hud').innerText().then((text) => Number(text.match(/(\d+) draw calls/)?.[1]));
  const before = await drawCalls();
  assert(Number.isFinite(before) && before > 0);
  await page.getByLabel('cloud', { exact: true }).uncheck();
  await page.waitForFunction((expected) => Number(document.querySelector('.hud')?.textContent?.match(/(\d+) draw calls/)?.[1]) === expected, before - 1);
  const hidden = await drawCalls();
  await page.getByLabel('cloud', { exact: true }).check();
  await page.waitForFunction((expected) => Number(document.querySelector('.hud')?.textContent?.match(/(\d+) draw calls/)?.[1]) === expected, before);
  assert.equal(await page.locator('.error').count(), 0);
  assert.deepEqual(pageErrors, []);
  await page.evaluate(() => window.dispatchEvent(new Event('viewer:capture')));
  await page.waitForSelector('canvas[data-render-state="frozen"]');
  await page.screenshot({ path: 'results/ui-smoke.png' });
  writeFileSync('results/ui-smoke.json', JSON.stringify({
    browserVersion: browser.version(), executionOrigin: report.executionOrigin,
    playPause: true, scrubbedTime, pointCountAfterChange: 100000,
    canvasCount: 1, drawCallsCloudVisible: before, drawCallsCloudHidden: hidden,
    restoredDrawCalls: await drawCalls(), captureFrozen: true, pageErrors,
  }, null, 2) + '\n');
  await page.close();
}

try {
  await waitForServer();
  browser = await chromium.launch({
    ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}),
    headless: true,
    args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--disable-gpu-vsync', '--disable-frame-rate-limit'],
  });
  report.browserVersion = browser.version();
  await checkInteractions();
  for (const n of POINTS) {
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    const pageErrors = [];
    page.on('pageerror', (error) => pageErrors.push(String(error)));
    const t0 = Date.now();
    await page.goto(`http://127.0.0.1:4173/?bench=1&points=${n}&seconds=${SECONDS}`);
    await page.waitForSelector('canvas');
    const pageLoadToCanvasMs = Date.now() - t0;
    await page.waitForFunction(() => window.__bench, null, { timeout: (SECONDS + 300) * 1000 });
    const result = await page.evaluate(() => window.__bench);
    assert.equal(result.points, n);
    assert(Number.isInteger(result.frames) && result.frames > 0);
    for (const field of ['fps', 'frameMsP50', 'frameMsP95', 'cloudGenerateMs']) assert(Number.isFinite(result[field]) && result[field] > 0, field);
    assert(result.frameMsP95 >= result.frameMsP50);
    assert(Number.isInteger(result.drawCalls) && result.drawCalls > 0);
    assert.match(result.gpu, /SwiftShader/i);
    assert.deepEqual(result.canvas, [1280, 720]);
    assert.equal(result.captureFrozen, true);
    await page.waitForSelector('canvas[data-render-state="frozen"]');
    assert.deepEqual(pageErrors, []);
    if (n === 100000) await page.screenshot({ path: 'results/screenshot_100k.png' });
    report.results.push({ ...result, pageLoadToCanvasMs });
    save();
    console.log(`${n.toLocaleString()} points: ${result.fps.toFixed(1)} FPS; p50/p95 ${result.frameMsP50.toFixed(1)}/${result.frameMsP95.toFixed(1)} ms; ${result.drawCalls} draw calls; ${result.gpu}`);
    await page.close();
  }
  report.completed = true;
} finally {
  try { if (browser) await browser.close(); }
  finally {
    if (server.pid && isRunning()) {
      server.kill('SIGTERM');
      const deadline = Date.now() + 5000;
      while (isRunning() && Date.now() < deadline) await delay(50);
      if (isRunning()) server.kill('SIGKILL');
    }
    report.finishedUtc = new Date().toISOString();
    save();
    writeFileSync('results/preview.log', serverOutput);
  }
}
