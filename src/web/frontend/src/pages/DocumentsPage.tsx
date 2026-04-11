import { FormEvent, useEffect, useMemo, useState } from 'react';

import { cancelTask, createKGBuildTask, getTask, listKGCandidates, listTasks } from '../api';
import type { KGCandidate, TaskDetail, TaskStatus, TaskSummary } from '../types';

const FILTERS: Array<{ key: 'all' | TaskStatus; label: string }> = [
  { key: 'all', label: '全部' },
  { key: 'succeeded', label: '已完成' },
  { key: 'running', label: '处理中' },
  { key: 'queued', label: '等待中' },
  { key: 'failed', label: '失败' },
  { key: 'canceled', label: '已取消' },
];

const TERMINAL = new Set<TaskStatus>(['succeeded', 'failed', 'canceled']);

export function DocumentsPage() {
  const [inputType, setInputType] = useState<'text' | 'json_path'>('text');
  const [topic, setTopic] = useState('web_topic');
  const [text, setText] = useState('');
  const [jsonPath, setJsonPath] = useState('data/raw/MINE_short10.json');
  const [outputDir, setOutputDir] = useState('');
  const [useExistingKg, setUseExistingKg] = useState(false);

  const [kgCandidates, setKgCandidates] = useState<KGCandidate[]>([]);
  const [selectedExistingKgPath, setSelectedExistingKgPath] = useState('');

  const [filter, setFilter] = useState<'all' | TaskStatus>('all');
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState('');
  const [selectedTask, setSelectedTask] = useState<TaskDetail | null>(null);

  const [submitting, setSubmitting] = useState(false);
  const [loadingTasks, setLoadingTasks] = useState(false);
  const [error, setError] = useState('');

  const progressText = useMemo(() => {
    if (!selectedTask) return '0%';
    return `${Math.round((selectedTask.progress ?? 0) * 100)}%`;
  }, [selectedTask]);

  const loadCandidates = async () => {
    try {
      const data = await listKGCandidates();
      setKgCandidates(data.items);
      setSelectedExistingKgPath((current) => {
        if (current && data.items.some((item) => item.path === current)) return current;
        return data.items[0]?.path ?? '';
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const loadTasks = async () => {
    setLoadingTasks(true);
    setError('');
    try {
      const data = await listTasks({
        page: 1,
        page_size: 80,
        task_type: 'kg_build',
        status: filter === 'all' ? undefined : filter,
      });
      setTasks(data.items);
      setSelectedTaskId((current) => {
        if (current && data.items.some((item) => item.task_id === current)) return current;
        return data.items[0]?.task_id ?? '';
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingTasks(false);
    }
  };

  const loadTaskDetail = async (taskId: string) => {
    if (!taskId) {
      setSelectedTask(null);
      return;
    }

    try {
      const detail = await getTask(taskId);
      setSelectedTask(detail);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  useEffect(() => {
    void loadCandidates();
  }, []);

  useEffect(() => {
    void loadTasks();
  }, [filter]);

  useEffect(() => {
    void loadTaskDetail(selectedTaskId);
  }, [selectedTaskId]);

  useEffect(() => {
    if (!selectedTaskId || !selectedTask || TERMINAL.has(selectedTask.status)) return;

    const timer = window.setInterval(() => {
      void loadTaskDetail(selectedTaskId);
      void loadTasks();
    }, 1800);
    return () => window.clearInterval(timer);
  }, [selectedTaskId, selectedTask?.status]);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setError('');

    try {
      const existingKg = useExistingKg ? selectedExistingKgPath.trim() : '';
      if (useExistingKg && !existingKg) {
        throw new Error('请先选择已有 KG，再启用 existing_kg。');
      }

      const payload =
        inputType === 'text'
          ? {
              input_type: 'text' as const,
              text,
              topic,
              output_dir: outputDir || undefined,
              existing_kg: existingKg || undefined,
            }
          : {
              input_type: 'json_path' as const,
              json_path: jsonPath,
              output_dir: outputDir || undefined,
              existing_kg: existingKg || undefined,
            };

      const created = await createKGBuildTask(payload);
      setSelectedTaskId(created.task_id);
      await loadTasks();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const onCancelSelected = async () => {
    if (!selectedTaskId) return;
    try {
      await cancelTask(selectedTaskId);
      await Promise.all([loadTaskDetail(selectedTaskId), loadTasks()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className="space-y-3">
      <section className="panel p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="m-0 font-display text-lg font-semibold">文档管理</h2>
            <p className="m-0 mt-1 text-sm text-app-muted">KG 构建任务提交与状态追踪</p>
          </div>
          <div className="flex items-center gap-2">
            <button type="button" className="soft-button" onClick={() => void loadTasks()} disabled={loadingTasks}>
              扫描/重试
            </button>
            <button type="button" className="soft-button" onClick={() => void loadCandidates()}>
              流水线
            </button>
          </div>
        </div>
      </section>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[420px_1fr]">
        <section className="panel p-4">
          <h3 className="m-0 text-sm font-semibold">提交构建任务</h3>
          <form className="mt-3 space-y-3" onSubmit={onSubmit}>
            <label className="block text-xs text-app-muted">
              输入类型
              <select className="field mt-1" value={inputType} onChange={(event) => setInputType(event.target.value as 'text' | 'json_path')}>
                <option value="text">文本输入</option>
                <option value="json_path">JSON 文件路径</option>
              </select>
            </label>

            {inputType === 'text' ? (
              <>
                <label className="block text-xs text-app-muted">
                  Topic
                  <input className="field mt-1" value={topic} onChange={(event) => setTopic(event.target.value)} />
                </label>
                <label className="block text-xs text-app-muted">
                  文本内容
                  <textarea
                    className="field mt-1 min-h-36"
                    value={text}
                    onChange={(event) => setText(event.target.value)}
                    placeholder="输入需要构建图谱的文本"
                  />
                </label>
              </>
            ) : (
              <label className="block text-xs text-app-muted">
                JSON 路径
                <input className="field mt-1" value={jsonPath} onChange={(event) => setJsonPath(event.target.value)} />
              </label>
            )}

            <label className="block text-xs text-app-muted">
              输出目录（可选）
              <input
                className="field mt-1"
                value={outputDir}
                onChange={(event) => setOutputDir(event.target.value)}
                placeholder="例如 data/web/custom_job"
              />
            </label>

            <label className="flex items-center gap-2 text-xs text-app-muted">
              <input type="checkbox" checked={useExistingKg} onChange={(event) => setUseExistingKg(event.target.checked)} />
              在已有知识图谱基础上构建（existing_kg）
            </label>

            <label className="block text-xs text-app-muted">
              已有 KG
              <select
                className="field mt-1"
                value={selectedExistingKgPath}
                onChange={(event) => setSelectedExistingKgPath(event.target.value)}
                disabled={kgCandidates.length === 0}
              >
                <option value="">请选择已有 KG</option>
                {kgCandidates.map((candidate) => (
                  <option key={`${candidate.source}:${candidate.path}`} value={candidate.path}>
                    {candidate.display_name}
                  </option>
                ))}
              </select>
            </label>

            <button type="submit" className="brand-button" disabled={submitting}>
              {submitting ? '提交中...' : '上传/构建'}
            </button>
          </form>
        </section>

        <section className="panel p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h3 className="m-0 text-sm font-semibold">已上传文档 / 任务列表</h3>
            <div className="flex flex-wrap gap-2">
              {FILTERS.map((item) => (
                <button
                  key={item.key}
                  type="button"
                  className={filter === item.key ? 'status-chip active' : 'status-chip'}
                  onClick={() => setFilter(item.key)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-3 grid grid-cols-1 gap-3 xl:grid-cols-[1fr_360px]">
            <div className="custom-scrollbar max-h-[500px] space-y-2 overflow-auto pr-1">
              {tasks.length === 0 ? (
                <div className="rounded-xl border border-dashed border-app-border p-6 text-center text-sm text-app-muted">无文档</div>
              ) : (
                tasks.map((task) => {
                  const active = task.task_id === selectedTaskId;
                  return (
                    <button
                      key={task.task_id}
                      type="button"
                      className={`w-full rounded-xl border p-3 text-left transition ${
                        active ? 'border-emerald-300 bg-emerald-50' : 'border-app-border bg-white hover:bg-slate-50'
                      }`}
                      onClick={() => setSelectedTaskId(task.task_id)}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <strong className="truncate text-sm">{task.task_id}</strong>
                        <span className="text-xs text-app-muted">{task.status}</span>
                      </div>
                      <p className="m-0 mt-1 truncate text-xs text-app-muted">{task.message}</p>
                      <div className="mt-2 h-1.5 rounded-full bg-slate-200">
                        <div
                          className="h-full rounded-full bg-emerald-500"
                          style={{ width: `${Math.round((task.progress ?? 0) * 100)}%` }}
                        />
                      </div>
                    </button>
                  );
                })
              )}
            </div>

            <div className="rounded-xl border border-app-border bg-slate-50 p-3">
              <div className="mb-2 flex items-center justify-between">
                <h4 className="m-0 text-sm font-semibold">任务详情</h4>
                <button
                  type="button"
                  className="soft-button"
                  onClick={onCancelSelected}
                  disabled={!selectedTask || TERMINAL.has(selectedTask.status)}
                >
                  取消任务
                </button>
              </div>

              {!selectedTask ? (
                <p className="text-sm text-app-muted">选择左侧任务查看详情</p>
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="rounded-lg border border-app-border bg-white p-2">
                      <p className="m-0 text-app-muted">状态</p>
                      <p className="m-0 mt-1 font-semibold">{selectedTask.status}</p>
                    </div>
                    <div className="rounded-lg border border-app-border bg-white p-2">
                      <p className="m-0 text-app-muted">进度</p>
                      <p className="m-0 mt-1 font-semibold">{progressText}</p>
                    </div>
                  </div>

                  <p className="mb-0 mt-2 text-xs text-app-muted">{selectedTask.message}</p>

                  <details className="mt-3">
                    <summary className="cursor-pointer text-xs font-semibold text-app-muted">输出 payload</summary>
                    <pre className="json-panel mt-2 max-h-72">{JSON.stringify(selectedTask.output_payload, null, 2)}</pre>
                  </details>

                  <details className="mt-2">
                    <summary className="cursor-pointer text-xs font-semibold text-app-muted">输入 payload</summary>
                    <pre className="json-panel mt-2 max-h-72">{JSON.stringify(selectedTask.input_payload, null, 2)}</pre>
                  </details>
                </>
              )}
            </div>
          </div>

          {error ? <p className="mb-0 mt-3 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-600">{error}</p> : null}
        </section>
      </div>
    </div>
  );
}
