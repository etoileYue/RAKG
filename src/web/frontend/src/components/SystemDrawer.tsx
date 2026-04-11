import { useEffect, useMemo, useState } from 'react';

import { getHealth, getLogs, listTasks } from '../api';
import type { HealthResponse, LogsResponse, TaskSummary } from '../types';

type DrawerTab = 'logs' | 'system';

interface SystemDrawerProps {
  open: boolean;
  onClose: () => void;
}

export function SystemDrawer({ open, onClose }: SystemDrawerProps) {
  const [tab, setTab] = useState<DrawerTab>('logs');

  const [taskId, setTaskId] = useState('');
  const [keyword, setKeyword] = useState('');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);

  const [logs, setLogs] = useState<LogsResponse | null>(null);
  const [logsLoading, setLogsLoading] = useState(false);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [systemLoading, setSystemLoading] = useState(false);

  const [error, setError] = useState('');

  const maxPage = useMemo(() => {
    if (!logs) return 1;
    return Math.max(1, Math.ceil(logs.total / logs.page_size));
  }, [logs]);

  const loadLogs = async () => {
    setLogsLoading(true);
    setError('');
    try {
      const data = await getLogs({
        page,
        page_size: pageSize,
        task_id: taskId || undefined,
        keyword: keyword || undefined,
      });
      setLogs(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLogsLoading(false);
    }
  };

  const loadSystem = async () => {
    setSystemLoading(true);
    setError('');
    try {
      const [healthData, taskData] = await Promise.all([
        getHealth(),
        listTasks({ page: 1, page_size: 8 }),
      ]);
      setHealth(healthData);
      setTasks(taskData.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSystemLoading(false);
    }
  };

  useEffect(() => {
    if (!open) return;
    if (tab === 'logs') {
      void loadLogs();
      return;
    }
    void loadSystem();
  }, [open, tab, page, pageSize]);

  useEffect(() => {
    if (!open || tab !== 'system') return;
    const timer = window.setInterval(() => {
      void loadSystem();
    }, 5000);
    return () => window.clearInterval(timer);
  }, [open, tab]);

  if (!open) return null;

  return (
    <>
      <button
        type="button"
        className="fixed inset-0 z-40 bg-slate-900/30"
        aria-label="关闭侧边抽屉"
        onClick={onClose}
      />
      <aside className="fixed right-0 top-0 z-50 h-full w-full max-w-[540px] overflow-hidden border-l border-app-border bg-white shadow-2xl">
        <header className="flex items-center justify-between border-b border-app-border px-4 py-3">
          <div>
            <h2 className="m-0 font-display text-base font-semibold">运行面板</h2>
            <p className="m-0 text-xs text-app-muted">日志中心 / 系统状态</p>
          </div>
          <button type="button" onClick={onClose} className="soft-button px-2 py-1 text-xs">
            关闭
          </button>
        </header>

        <div className="flex gap-2 border-b border-app-border px-4 py-2">
          <button
            type="button"
            className={tab === 'logs' ? 'status-chip active' : 'status-chip'}
            onClick={() => setTab('logs')}
          >
            日志
          </button>
          <button
            type="button"
            className={tab === 'system' ? 'status-chip active' : 'status-chip'}
            onClick={() => setTab('system')}
          >
            系统
          </button>
        </div>

        <div className="custom-scrollbar h-[calc(100%-112px)] overflow-y-auto p-4">
          {error ? <p className="mb-3 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-600">{error}</p> : null}

          {tab === 'logs' ? (
            <section className="space-y-3">
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                <input
                  className="field"
                  placeholder="task_id"
                  value={taskId}
                  onChange={(event) => setTaskId(event.target.value)}
                />
                <input
                  className="field"
                  placeholder="keyword"
                  value={keyword}
                  onChange={(event) => setKeyword(event.target.value)}
                />
              </div>

              <div className="flex items-center gap-2">
                <label className="text-xs text-app-muted">
                  page_size
                  <input
                    className="field mt-1 w-28"
                    type="number"
                    min={10}
                    max={200}
                    value={pageSize}
                    onChange={(event) => setPageSize(Number(event.target.value) || 20)}
                  />
                </label>
                <button
                  type="button"
                  className="soft-button mt-4"
                  onClick={() => {
                    setPage(1);
                    void loadLogs();
                  }}
                  disabled={logsLoading}
                >
                  查询
                </button>
              </div>

              <div className="space-y-2">
                {!logs || logs.items.length === 0 ? (
                  <div className="panel px-3 py-4 text-sm text-app-muted">暂无日志</div>
                ) : (
                  logs.items.map((item) => (
                    <article key={item.id} className="panel animate-fadeInUp p-3">
                      <header className="mb-1 flex items-center justify-between text-xs">
                        <strong className="text-slate-700">{item.level}</strong>
                        <span className="text-app-muted">{item.created_at}</span>
                      </header>
                      <p className="mb-1 text-xs text-app-muted">
                        source={item.source} task_id={item.task_id || '-'}
                      </p>
                      <p className="m-0 whitespace-pre-wrap text-sm text-slate-800">{item.message}</p>
                    </article>
                  ))
                )}
              </div>

              <div className="flex items-center justify-between">
                <button
                  type="button"
                  className="soft-button"
                  onClick={() => setPage((prev) => Math.max(1, prev - 1))}
                  disabled={page <= 1 || logsLoading}
                >
                  上一页
                </button>
                <span className="text-xs text-app-muted">
                  page {page}/{maxPage}
                </span>
                <button
                  type="button"
                  className="soft-button"
                  onClick={() => setPage((prev) => prev + 1)}
                  disabled={page >= maxPage || logsLoading}
                >
                  下一页
                </button>
              </div>
            </section>
          ) : (
            <section className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="m-0 text-sm font-semibold">健康状态</h3>
                <button type="button" className="soft-button" onClick={() => void loadSystem()} disabled={systemLoading}>
                  刷新
                </button>
              </div>

              {!health ? (
                <div className="panel px-3 py-4 text-sm text-app-muted">加载中...</div>
              ) : (
                <div className="grid grid-cols-2 gap-2">
                  <div className="panel p-3">
                    <p className="m-0 text-xs text-app-muted">Healthy</p>
                    <p className="m-0 text-sm font-semibold">{String(health.healthy)}</p>
                  </div>
                  <div className="panel p-3">
                    <p className="m-0 text-xs text-app-muted">Queued</p>
                    <p className="m-0 text-sm font-semibold">{health.queue.queued}</p>
                  </div>
                  <div className="panel p-3">
                    <p className="m-0 text-xs text-app-muted">Running</p>
                    <p className="m-0 text-sm font-semibold">{health.queue.running}</p>
                  </div>
                  <div className="panel p-3">
                    <p className="m-0 text-xs text-app-muted">Running Task</p>
                    <p className="m-0 truncate text-sm font-semibold">{health.queue.running_task_id || '-'}</p>
                  </div>
                </div>
              )}

              <details className="panel p-3">
                <summary className="cursor-pointer text-sm font-semibold">模型配置摘要</summary>
                <pre className="json-panel mt-2">{JSON.stringify(health?.model_config_summary ?? {}, null, 2)}</pre>
              </details>

              <div className="space-y-2">
                <h3 className="m-0 text-sm font-semibold">最近任务</h3>
                {tasks.length === 0 ? (
                  <div className="panel px-3 py-4 text-sm text-app-muted">暂无任务</div>
                ) : (
                  tasks.map((task) => (
                    <article key={task.task_id} className="panel p-3">
                      <header className="mb-1 flex items-center justify-between">
                        <strong className="text-sm">{task.task_id}</strong>
                        <span className="text-xs text-app-muted">{task.status}</span>
                      </header>
                      <p className="m-0 text-xs text-app-muted">{task.message}</p>
                    </article>
                  ))
                )}
              </div>
            </section>
          )}
        </div>
      </aside>
    </>
  );
}
