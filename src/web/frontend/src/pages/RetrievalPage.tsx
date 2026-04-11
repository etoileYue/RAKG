import { FormEvent, useEffect, useState } from 'react';

import { getTask, listTasks, qaQuery } from '../api';
import type { QAResponse, TaskSummary } from '../types';

export function RetrievalPage() {
  const [kgPath, setKgPath] = useState('');
  const [question, setQuestion] = useState('蝴蝶的生命周期包括哪四个主要阶段？');
  const [maxHop, setMaxHop] = useState(2);
  const [seedTopK, setSeedTopK] = useState(5);
  const [maxContextItems, setMaxContextItems] = useState(30);

  const [taskOptions, setTaskOptions] = useState<TaskSummary[]>([]);
  const [result, setResult] = useState<QAResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const loadTaskOptions = async () => {
    try {
      const response = await listTasks({ page: 1, page_size: 100, task_type: 'kg_build', status: 'succeeded' });
      setTaskOptions(response.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  useEffect(() => {
    void loadTaskOptions();
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
    <div className="space-y-3">
      <section className="panel p-4">
        <div className="mb-3">
          <h2 className="m-0 font-display text-lg font-semibold">检索</h2>
          <p className="m-0 mt-1 text-sm text-app-muted">基于指定知识图谱执行 QA 检索</p>
        </div>

        <form onSubmit={onSubmit} className="space-y-3">
          <label className="block text-xs text-app-muted">
            KG 文件路径
            <input className="field mt-1" value={kgPath} onChange={(event) => setKgPath(event.target.value)} placeholder="/abs/path/to/kg.json" />
          </label>

          <label className="block text-xs text-app-muted">
            从已完成任务回填（可选）
            <select className="field mt-1" defaultValue="" onChange={(event) => void onLoadFromTask(event.target.value)}>
              <option value="">请选择任务</option>
              {taskOptions.map((task) => (
                <option key={task.task_id} value={task.task_id}>
                  {task.task_id}
                </option>
              ))}
            </select>
          </label>

          <label className="block text-xs text-app-muted">
            问题
            <textarea className="field mt-1 min-h-28" value={question} onChange={(event) => setQuestion(event.target.value)} />
          </label>

          <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
            <label className="block text-xs text-app-muted">
              max_hop
              <input
                className="field mt-1"
                type="number"
                min={1}
                max={3}
                value={maxHop}
                onChange={(event) => setMaxHop(Number(event.target.value) || 2)}
              />
            </label>

            <label className="block text-xs text-app-muted">
              seed_top_k
              <input
                className="field mt-1"
                type="number"
                min={1}
                max={20}
                value={seedTopK}
                onChange={(event) => setSeedTopK(Number(event.target.value) || 5)}
              />
            </label>

            <label className="block text-xs text-app-muted">
              max_context_items
              <input
                className="field mt-1"
                type="number"
                min={1}
                max={100}
                value={maxContextItems}
                onChange={(event) => setMaxContextItems(Number(event.target.value) || 30)}
              />
            </label>
          </div>

          <button type="submit" className="brand-button" disabled={loading}>
            {loading ? '检索中...' : '发起检索'}
          </button>
        </form>

        {error ? <p className="mb-0 mt-3 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-600">{error}</p> : null}
      </section>

      <section className="panel p-4">
        <h3 className="m-0 text-sm font-semibold">回答结果</h3>
        {!result ? (
          <p className="mt-2 text-sm text-app-muted">输入问题后展示结果</p>
        ) : (
          <div className="mt-3 space-y-3">
            <article className="rounded-xl border border-app-border bg-slate-50 p-3 text-sm leading-6 text-slate-800">
              {result.formatted_answer}
            </article>

            <details>
              <summary className="cursor-pointer text-xs font-semibold text-app-muted">检索上下文</summary>
              <pre className="json-panel mt-2 max-h-96">{JSON.stringify(result.retrieved_context, null, 2)}</pre>
            </details>

            <details>
              <summary className="cursor-pointer text-xs font-semibold text-app-muted">中间信息</summary>
              <pre className="json-panel mt-2 max-h-96">{JSON.stringify(result.intermediate, null, 2)}</pre>
            </details>
          </div>
        )}
      </section>
    </div>
  );
}
