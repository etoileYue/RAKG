import { useEffect, useState } from 'react';

import { getHealth, listTasks } from '../api';
import type { HealthResponse, TaskSummary } from '../types';

export function SystemPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [latestTasks, setLatestTasks] = useState<TaskSummary[]>([]);
  const [error, setError] = useState('');

  const load = async () => {
    setError('');
    try {
      const [healthData, taskData] = await Promise.all([
        getHealth(),
        listTasks({ page: 1, page_size: 10 }),
      ]);
      setHealth(healthData);
      setLatestTasks(taskData.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  useEffect(() => {
    load();
    const timer = window.setInterval(load, 5000);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <div className="page-grid">
      <section className="panel">
        <h2>系统状态</h2>
        <div className="button-row">
          <button type="button" onClick={load}>
            刷新
          </button>
        </div>

        {error ? <p className="error-text">{error}</p> : null}

        {!health ? (
          <div className="empty-card">加载中...</div>
        ) : (
          <div className="status-grid">
            <div>
              <span>Healthy</span>
              <strong>{String(health.healthy)}</strong>
            </div>
            <div>
              <span>Queued</span>
              <strong>{health.queue.queued}</strong>
            </div>
            <div>
              <span>Running</span>
              <strong>{health.queue.running}</strong>
            </div>
            <div>
              <span>Running Task</span>
              <strong>{health.queue.running_task_id || '-'}</strong>
            </div>
          </div>
        )}
      </section>

      <section className="panel">
        <h2>模型配置摘要</h2>
        <pre className="json-view">{JSON.stringify(health?.model_config_summary ?? {}, null, 2)}</pre>
      </section>

      <section className="panel full-row">
        <h2>最近任务</h2>
        {latestTasks.length === 0 ? (
          <div className="empty-card">暂无任务</div>
        ) : (
          <div className="task-list">
            {latestTasks.map((task) => (
              <article key={task.task_id} className="task-item">
                <header>
                  <strong>{task.task_id}</strong>
                  <span>{task.status}</span>
                </header>
                <p>{task.message}</p>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
