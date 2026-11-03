import KittiReplay from './KittiReplay';
import type { ViewerEntry } from './entry';

/** Separate component identity keeps the legacy App and its hooks untouched. */
export default function SlamEntry({ entry }: { entry: Exclude<ViewerEntry, { kind: 'safety' }> }) {
  return (
    <main className="slam-shell" aria-label="Recorded SLAM replay">
      <header className="slam-shell__nav">
        <h1>Offline SLAM replay</h1>
        <a href={entry.safetyUrl}>Safety replay</a>
      </header>
      {entry.kind === 'error'
        ? <p className="slam-shell__message" role="alert">{entry.message}</p>
        : <KittiReplay runUrl={entry.runUrl} />}
    </main>
  );
}
