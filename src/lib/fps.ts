/** Frame-time statistics over a window of frames. */
export class FrameStats {
  private times: number[] = [];
  private last: number | null = null;
  tick(now: number) {
    if (this.last !== null) this.times.push(now - this.last);
    this.last = now;
  }
  reset() { this.times = []; this.last = null; }
  get count() { return this.times.length; }
  summary() {
    const s = [...this.times].sort((a, b) => a - b);
    const q = (p: number) => (s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))] : NaN);
    const mean = s.reduce((a, b) => a + b, 0) / Math.max(s.length, 1);
    return { frames: s.length, fps: 1000 / mean, frameMsP50: q(0.5), frameMsP95: q(0.95) };
  }
}
