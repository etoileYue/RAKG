import { type FormEvent, useEffect, useState } from 'react';

import { getTask, listTasks, qaQuery } from '../api';
import type { QAResponse, TaskSummary } from '../types';

export function QAPage() {
  const [kgPath, setKgPath] = useState('');
  const [question, setQuestion] = useState('蝴蝶的生命周期包括哪四个主要阶段？');
  const [maxHop, setMaxHop] = useState(2);
  const [seedTopK, setSeedTopK] = useState(5);
  const [maxContextItems, setMaxContextItems] = useState(30);

  const [taskOptions, setTaskOptions] = useState<TaskSummary[]>([]);
  const [result, setResult] = useState<QAResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const loadTasks = async () => {
      try {
        const response = await listTasks({ page: 1, page_size: 50, task_type: 'kg_build', status: 'succeeded' });
        setTaskOptions(response.items);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    };
    loadTasks();
  }, []);

  const onLoadFromTask = async (taskId: string) => {
    if (!taskId) return;
    try {
      const detail = await getTask(taskId);
      const latest = String(detail.output_payload.latest_graph_path ?? '').trim();
      if (latest) setKgPath(latest);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError('');
    setLoading(true);
    setResult(null);

    try {
      const data = await qaQuery({
        kg_path: kgPath,
        question,
        max_hop: maxHop,
        seed_top_k: seedTopK,
        max_context_items: maxContextItems,
      });
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-grid single-column">
      <section className="panel">
        <h2>QA 检索</h2>
        <form onSubmit={onSubmit} className="form-grid">
          <label>
            KG 文件路径
            <input value={kgPath} onChange={(event) => setKgPath(event.target.value)} placeholder="/abs/path/to/kg.json" />
          </label>

          <label>
            已完成任务（可选）
            <select onChange={(event) => onLoadFromTask(event.target.value)} defaultValue="">
              <option value="">请选择任务</option>
              {taskOptions.map((task) => (
                <option key={task.task_id} value={task.task_id}>
                  {task.task_id}
                </option>
              ))}
            </select>
          </label>

          <label>
            问题
            <textarea value={question} onChange={(event) => setQuestion(event.target.value)} rows={4} />
          </label>

          <div className="inline-form-grid">
            <label>
              max_hop
              <input type="number" min={1} max={3} value={maxHop} onChange={(event) => setMaxHop(Number(event.target.value))} />
            </label>
            <label>
              seed_top_k
              <input
                type="number"
                min={1}
                max={20}
                value={seedTopK}
                onChange={(event) => setSeedTopK(Number(event.target.value))}
              />
            </label>
            <label>
              max_context_items
              <input
                type="number"
                min={1}
                max={100}
                value={maxContextItems}
                onChange={(event) => setMaxContextItems(Number(event.target.value))}
              />
            </label>
          </div>

          <div className="button-row">
            <button type="submit" disabled={loading}>
              {loading ? '检索中...' : '发起 QA'}
            </button>
          </div>
        </form>

        {error ? <p className="error-text">{error}</p> : null}
      </section>

      <section className="panel">
        <h2>回答结果</h2>
        {!result ? (
          <div className="empty-card">输入问题后展示结果</div>
        ) : (
          <>
            <pre className="answer-block">{result.formatted_answer}</pre>
            <details>
              <summary>检索上下文</summary>
              <pre className="json-view">{JSON.stringify(result.retrieved_context, null, 2)}</pre>
            </details>
            <details>
              <summary>中间信息</summary>
              <pre className="json-view">{JSON.stringify(result.intermediate, null, 2)}</pre>
            </details>
          </>
        )}
      </section>
    </div>
  );
}
