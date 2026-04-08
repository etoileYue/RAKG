import { useState } from 'react';

import { KGBuildPage } from './pages/KGBuildPage';
import { LogsPage } from './pages/LogsPage';
import { QAPage } from './pages/QAPage';
import { SystemPage } from './pages/SystemPage';

type TabKey = 'kg' | 'qa' | 'logs' | 'system';

const TABS: Array<{ key: TabKey; label: string }> = [
  { key: 'kg', label: 'KG 构建' },
  { key: 'qa', label: 'QA 检索' },
  { key: 'logs', label: '日志中心' },
  { key: 'system', label: '系统状态' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<TabKey>('kg');

  return (
    <div className="app-shell">
      <header className="top-header">
        <div>
          <h1>RAKG Workbench</h1>
          <p>V1: KG / QA / 日志 / 系统状态</p>
        </div>
      </header>

      <nav className="tab-nav" aria-label="main navigation">
        {TABS.map((item) => (
          <button
            key={item.key}
            type="button"
            className={item.key === activeTab ? 'tab-button active' : 'tab-button'}
            onClick={() => setActiveTab(item.key)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <main className="content-wrap">
        {activeTab === 'kg' ? <KGBuildPage /> : null}
        {activeTab === 'qa' ? <QAPage /> : null}
        {activeTab === 'logs' ? <LogsPage /> : null}
        {activeTab === 'system' ? <SystemPage /> : null}
      </main>
    </div>
  );
}
