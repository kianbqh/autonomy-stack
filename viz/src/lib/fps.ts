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
  intervals() { return [...this.times]; }
  summary() {
    const s = [...this.times].sort((a, b) => a - b);
    const q = (p: number) => {
      if (!s.length) return 0;
      const index = p * (s.length - 1), lower = Math.floor(index), upper = Math.ceil(index);
      return s[lower] + (s[upper] - s[lower]) * (index - lower);
    };
    const mean = s.reduce((a, b) => a + b, 0) / Math.max(s.length, 1);
    return { frames: s.length, fps: mean > 0 ? 1000 / mean : 0, frameMsP50: q(0.5), frameMsP95: q(0.95) };
  }
}
