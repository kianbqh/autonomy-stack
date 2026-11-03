import { createRoot } from 'react-dom/client';
import App from './App';
import { resolveViewerEntry } from './slam/entry';
import SlamEntry from './slam/SlamEntry';
import './style.css';

const entry = resolveViewerEntry(window.location.href);
if (entry.kind !== 'safety') document.title = 'Offline SLAM replay';
createRoot(document.getElementById('root')!).render(
  entry.kind === 'safety' ? <App /> : <SlamEntry entry={entry} />,
);
