import { FormEvent, useEffect, useState } from 'react';

import { listKGCandidates, qaQuery } from '../api';
import type { KGCandidate, QAResponse } from '../types';

function resolveDefaultGraphPath(items: KGCandidate[]): string {
  if (!items.length) return '';
  const recentTask = items.find((item) => item.source === 'task');
  return recentTask?.path ?? items[0].path;
}

export function RetrievalPage() {
  const [kgCandidates, setKgCandidates] = useState<KGCandidate[]>([]);
  const [selectedKgPath, setSelectedKgPath] = useState('');
  const [question, setQuestion] = useState('蝴蝶的生命周期包括哪四个主要阶段？');
  const [maxHop, setMaxHop] = useState(2);
  const [seedTopK, setSeedTopK] = useState(5);
  const [maxContextItems, setMaxContextItems] = useState(30);

  const [result, setResult] = useState<QAResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const loadKGCandidates = async () => {
    try {
      const response = await listKGCandidates();
      setKgCandidates(response.items);
      setSelectedKgPath((current) => {
        if (current && response.items.some((item) => item.path === current)) return current;
        return resolveDefaultGraphPath(response.items);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  useEffect(() => {
    void loadKGCandidates();
  }, []);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError('');
    setLoading(true);
    setResult(null);

    try {
      if (!selectedKgPath) {
        throw new Error('请先选择一个可用图谱。');
      }
      const data = await qaQuery({
        kg_path: selectedKgPath,
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
        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <h2 className="m-0 font-display text-lg font-semibold">检索</h2>
            <p className="m-0 mt-1 text-sm text-app-muted">基于已有知识图谱执行 QA 检索</p>
          </div>
          <button type="button" className="soft-button" onClick={() => void loadKGCandidates()}>
            刷新图谱源
          </button>
        </div>

        <form onSubmit={onSubmit} className="space-y-3">
          <label className="block text-xs text-app-muted">
            可用图谱
            <select className="field mt-1" value={selectedKgPath} onChange={(event) => setSelectedKgPath(event.target.value)}>
              <option value="">请选择图谱</option>
              {kgCandidates.map((candidate) => (
                <option key={`${candidate.source}:${candidate.path}`} value={candidate.path}>
                  {candidate.display_name}
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

          <button type="submit" className="brand-button" disabled={loading || !selectedKgPath}>
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
            <article className="rounded-xl border border-app-border bg-slate-50 p-3 text-sm leading-6 text-slate-800">{result.formatted_answer}</article>

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
