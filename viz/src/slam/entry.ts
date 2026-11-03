/** Explicit page-entry selection; no fetch, fallback dataset or hook lives here. */
export type ViewerEntry =
  | { kind: 'safety' }
  | { kind: 'slam'; runUrl: string; safetyUrl: string }
  | { kind: 'error'; message: string; safetyUrl: string };

export function resolveViewerEntry(href: string): ViewerEntry {
  const current = new URL(href);
  const modes = current.searchParams.getAll('view');
  // Existing episode/bench/time/points parameters keep their original meaning.
  if (modes.length === 0 || (modes.length === 1 && modes[0] === 'safety')) return { kind: 'safety' };

  const safety = new URL(current.href);
  safety.searchParams.delete('view');
  safety.searchParams.delete('replay');
  const safetyUrl = safety.pathname + safety.search + safety.hash;
  const invalid = (message: string): ViewerEntry => ({ kind: 'error', message, safetyUrl });
  if (modes.length !== 1 || modes[0] !== 'slam') return invalid('This replay view is not available.');

  const files = current.searchParams.getAll('replay');
  if (files.length !== 1 || !files[0].trim()) return invalid('Choose one exported replay JSON file.');
  let replay: URL;
  try { replay = new URL(files[0], current); }
  catch { return invalid('The replay path is invalid.'); }
  if (!['http:', 'https:'].includes(replay.protocol) || replay.origin !== current.origin || replay.username || replay.password) {
    return invalid('Choose a replay JSON served by this viewer.');
  }
  if (!replay.pathname.toLowerCase().endsWith('.json') || replay.hash) return invalid('Choose an exported replay JSON file.');
  return { kind: 'slam', runUrl: replay.href, safetyUrl };
}
