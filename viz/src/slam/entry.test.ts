// Authored source candidate only. Vitest/TypeScript/browser have not run here.
import { describe, expect, it } from 'vitest';
import { resolveViewerEntry } from './entry';

describe('viewer page entry', () => {
  it('preserves all existing safety, fault and benchmark entry URLs', () => {
    for (const query of ['', '?episode=normal', '?episode=fault-collision&time=end', '?bench=1&seconds=1&points=10000', '?view=safety&episode=fault-collision', '?replay=unused.json']) {
      expect(resolveViewerEntry(`http://localhost:5176/${query}`)).toEqual({ kind: 'safety' });
    }
  });
  it('resolves an explicit real same-origin JSON path under a base directory', () => {
    expect(resolveViewerEntry('http://localhost:5176/viewer/?view=slam&replay=.%2Fdata%2Fslam%2Frun%2Freplay.json')).toEqual({
      kind: 'slam', runUrl: 'http://localhost:5176/viewer/data/slam/run/replay.json', safetyUrl: '/viewer/',
    });
  });
  it('preserves the requested legacy episode on the return link', () => {
    const entry = resolveViewerEntry('http://localhost:5176/?view=slam&replay=/data/replay.json&episode=fault-collision&time=end');
    expect(entry.kind).toBe('slam');
    if (entry.kind === 'slam') expect(entry.safetyUrl).toBe('/?episode=fault-collision&time=end');
  });
  it('reports missing, ambiguous, unsupported or off-origin input without fallback data', () => {
    for (const query of [
      '?view=slam', '?view=slam&replay=', '?view=slam&replay=a.json&replay=b.json',
      '?view=slam&view=safety&replay=a.json', '?view=other',
      '?view=slam&replay=https%3A%2F%2Felsewhere.test%2Freplay.json',
      '?view=slam&replay=file%3A%2F%2FG%3A%2Fdata%2Freplay.json',
      '?view=slam&replay=data%3Aapplication%2Fjson%2C%7B%7D',
      '?view=slam&replay=/data/raw.bin', '?view=slam&replay=/data/replay.json%23fragment',
    ]) expect(resolveViewerEntry(`http://localhost:5176/${query}`).kind).toBe('error');
  });
});
