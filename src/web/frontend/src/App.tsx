import { useEffect, useMemo, useState } from 'react';

import { SystemDrawer } from './components/SystemDrawer';
import { ApiDocsPage } from './pages/ApiDocsPage';
import { DocumentsPage } from './pages/DocumentsPage';
import { KnowledgeGraphPage } from './pages/KnowledgeGraphPage';
import { RetrievalPage } from './pages/RetrievalPage';

type TabKey = 'documents' | 'knowledge-graph' | 'retrieval' | 'api';

const TABS: Array<{ key: TabKey; label: string }> = [
  { key: 'documents', label: '文档' },
  { key: 'knowledge-graph', label: '知识图谱' },
  { key: 'retrieval', label: '检索' },
  { key: 'api', label: 'API' },
];

function resolveInitialTab(): TabKey {
  const params = new URLSearchParams(window.location.search);
  const candidate = params.get('tab') as TabKey | null;
  if (candidate && TABS.some((tab) => tab.key === candidate)) return candidate;
  return 'documents';
}

export default function App() {
  const [activeTab, setActiveTab] = useState<TabKey>(resolveInitialTab);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const appVersion = useMemo(() => import.meta.env.VITE_APP_VERSION ?? 'v0.2.0', []);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    params.set('tab', activeTab);
    const next = `${window.location.pathname}?${params.toString()}${window.location.hash}`;
    window.history.replaceState(null, '', next);
  }, [activeTab]);

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b border-app-border bg-white/90 backdrop-blur">
        <div className="mx-auto flex w-[min(1480px,96vw)] flex-wrap items-center justify-between gap-3 px-2 py-2">
          <div className="flex min-w-0 items-center gap-2">
            <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-50 text-emerald-600">LR</span>
            <div className="min-w-0">
              <h1 className="m-0 truncate font-display text-xl font-semibold">LightRAG | My Graph KB</h1>
            </div>
          </div>

          <nav className="flex flex-wrap items-center gap-1 rounded-xl bg-slate-100 p-1" aria-label="main tabs">
            {TABS.map((tab) => (
              <button
                key={tab.key}
                type="button"
                className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${
                  activeTab === tab.key ? 'bg-emerald-500 text-white shadow-sm' : 'text-slate-600 hover:bg-white'
                }`}
                onClick={() => setActiveTab(tab.key)}
              >
                {tab.label}
              </button>
            ))}
            <span className="rounded-md bg-amber-100 px-2 py-1 text-xs font-semibold text-amber-700">无需登陆</span>
          </nav>

          <div className="flex items-center gap-2">
            <span className="text-sm text-app-muted">{appVersion}</span>
            <button type="button" className="soft-button" onClick={() => setDrawerOpen(true)}>
              日志/系统
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto w-[min(1480px,96vw)] px-2 py-4">
        {activeTab === 'documents' ? <DocumentsPage /> : null}
        {activeTab === 'knowledge-graph' ? <KnowledgeGraphPage /> : null}
        {activeTab === 'retrieval' ? <RetrievalPage /> : null}
        {activeTab === 'api' ? <ApiDocsPage /> : null}
      </main>

      <SystemDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} />
    </div>
  );
}
