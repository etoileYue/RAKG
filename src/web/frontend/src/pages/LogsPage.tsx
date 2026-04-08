import { useEffect, useState } from 'react';

import { getLogs } from '../api';
import type { LogsResponse } from '../types';

export function LogsPage() {
  const [taskId, setTaskId] = useState('');
  const [keyword, setKeyword] = useState('');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(30);
  const [data, setData] = useState<LogsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const load = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await getLogs({
        page,
        page_size: pageSize,
        task_id: taskId || undefined,
        keyword: keyword || undefined,
      });
      setData(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [page, pageSize]);

  return (
    <div className="page-grid single-column">
      <section className="panel">
        <h2>日志中心</h2>
        <div className="inline-form-grid">
          <label>
            task_id
            <input value={taskId} onChange={(event) => setTaskId(event.target.value)} />
          </label>
          <label>
            keyword
            <input value={keyword} onChange={(event) => setKeyword(event.target.value)} />
          </label>
          <label>
            page_size
            <input type="number" min={10} max={200} value={pageSize} onChange={(event) => setPageSize(Number(event.target.value))} />
          </label>
        </div>

        <div className="button-row">
          <button
            type="button"
            onClick={() => {
              setPage(1);
              load();
            }}
            disabled={loading}
          >
            查询
          </button>
        </div>

        {error ? <p className="error-text">{error}</p> : null}

        <div className="log-list">
          {!data || data.items.length === 0 ? (
            <div className="empty-card">暂无日志</div>
          ) : (
            data.items.map((item) => (
              <article className="log-item" key={item.id}>
                <header>
                  <strong>{item.level}</strong>
                  <span>{item.created_at}</span>
                </header>
                <p className="log-meta">
                  source={item.source} task_id={item.task_id || '-'}
                </p>
                <p>{item.message}</p>
              </article>
            ))
          )}
        </div>

        <div className="button-row split">
          <button type="button" onClick={() => setPage((prev) => Math.max(1, prev - 1))} disabled={page <= 1 || loading}>
            上一页
          </button>
          <span>
            page {page} / {data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1}
          </span>
          <button
            type="button"
            onClick={() => setPage((prev) => prev + 1)}
            disabled={loading || !data || page >= Math.ceil(data.total / data.page_size)}
          >
            下一页
          </button>
        </div>
      </section>
    </div>
  );
}
