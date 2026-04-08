import { type FormEvent, useEffect, useMemo, useState } from 'react';

import { cancelTask, createKGBuildTask, getTask, readGraphArtifact } from '../api';
import { GraphView } from '../components/GraphView';
import type { GraphData, TaskDetail } from '../types';

const TERMINAL = new Set(['succeeded', 'failed', 'canceled']);

export function KGBuildPage() {
  const [inputType, setInputType] = useState<'text' | 'json_path'>('text');
  const [topic, setTopic] = useState('web_topic');
  const [text, setText] = useState('');
  const [jsonPath, setJsonPath] = useState('data/raw/MINE_short10.json');
  const [outputDir, setOutputDir] = useState('');

  const [taskId, setTaskId] = useState('');
  const [task, setTask] = useState<TaskDetail | null>(null);
  const [graphData, setGraphData] = useState<GraphData | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!taskId) return;

    let stopped = false;
    let timer: number | undefined;

    const poll = async () => {
      try {
        const detail = await getTask(taskId);
        if (stopped) return;
        setTask(detail);

        if (detail.status === 'succeeded') {
          const latestPath = String(detail.output_payload.latest_graph_path ?? '').trim();
          if (latestPath) {
            const graph = await readGraphArtifact(latestPath);
            if (!stopped) setGraphData(graph.data);
          }
        }

        if (!TERMINAL.has(detail.status)) {
          timer = window.setTimeout(poll, 1800);
        }
      } catch (err) {
        if (!stopped) {
          setError(err instanceof Error ? err.message : String(err));
          timer = window.setTimeout(poll, 2400);
        }
      }
    };

    // 轮询任务状态，前端无需保持长连接。
    poll();
    return () => {
      stopped = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [taskId]);

  const progressText = useMemo(() => {
    if (!task) return '0%';
    return `${Math.round((task.progress ?? 0) * 100)}%`;
  }, [task]);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError('');
    setLoading(true);
    setTask(null);
    setGraphData(null);

    try {
      const payload =
        inputType === 'text'
          ? {
              input_type: 'text' as const,
              text,
              topic,
              output_dir: outputDir || undefined,
            }
          : {
              input_type: 'json_path' as const,
              json_path: jsonPath,
              output_dir: outputDir || undefined,
            };

      const created = await createKGBuildTask(payload);
      setTaskId(created.task_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const onCancel = async () => {
    if (!taskId) return;
    try {
      await cancelTask(taskId);
      const detail = await getTask(taskId);
      setTask(detail);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const loadGraphFromPath = async () => {
    if (!task) return;
    const latestPath = String(task.output_payload.latest_graph_path ?? '').trim();
    if (!latestPath) return;
    try {
      const graph = await readGraphArtifact(latestPath);
      setGraphData(graph.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className="page-grid">
      <section className="panel">
        <h2>KG 构建</h2>
        <form onSubmit={onSubmit} className="form-grid">
          <label>
            输入类型
            <select value={inputType} onChange={(event) => setInputType(event.target.value as 'text' | 'json_path')}>
              <option value="text">文本输入</option>
              <option value="json_path">JSON 文件路径</option>
            </select>
          </label>

          {inputType === 'text' ? (
            <>
              <label>
                Topic
                <input value={topic} onChange={(event) => setTopic(event.target.value)} placeholder="topic name" />
              </label>
              <label>
                文本内容
                <textarea
                  value={text}
                  onChange={(event) => setText(event.target.value)}
                  rows={8}
                  placeholder="输入需要构建图谱的文本"
                />
              </label>
            </>
          ) : (
            <label>
              JSON 路径
              <input value={jsonPath} onChange={(event) => setJsonPath(event.target.value)} />
            </label>
          )}

          <label>
            输出目录（可选）
            <input
              value={outputDir}
              onChange={(event) => setOutputDir(event.target.value)}
              placeholder="例如 data/web/custom_job"
            />
          </label>

          <div className="button-row">
            <button type="submit" disabled={loading}>
              {loading ? '提交中...' : '提交任务'}
            </button>
            <button type="button" onClick={onCancel} disabled={!task || TERMINAL.has(task.status)}>
              取消任务
            </button>
            <button type="button" onClick={loadGraphFromPath} disabled={!task}>
              重新加载图谱
            </button>
          </div>
        </form>

        {error ? <p className="error-text">{error}</p> : null}
      </section>

      <section className="panel">
        <h2>任务状态</h2>
        {!task ? (
          <div className="empty-card">提交任务后可查看进度与结果</div>
        ) : (
          <>
            <div className="status-grid">
              <div>
                <span>Task ID</span>
                <strong>{task.task_id}</strong>
              </div>
              <div>
                <span>状态</span>
                <strong>{task.status}</strong>
              </div>
              <div>
                <span>进度</span>
                <strong>{progressText}</strong>
              </div>
              <div>
                <span>消息</span>
                <strong>{task.message}</strong>
              </div>
            </div>

            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${Math.round((task.progress ?? 0) * 100)}%` }} />
            </div>

            <pre className="json-view">{JSON.stringify(task.output_payload, null, 2)}</pre>
          </>
        )}
      </section>

      <section className="panel full-row">
        <h2>图谱预览</h2>
        <GraphView data={graphData} />
        <details>
          <summary>原始图谱 JSON</summary>
          <pre className="json-view">{JSON.stringify(graphData, null, 2)}</pre>
        </details>
      </section>
    </div>
  );
}
