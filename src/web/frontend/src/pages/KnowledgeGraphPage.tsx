import { useEffect, useState } from 'react';

import { listKGCandidates, readGraphArtifact } from '../api';
import { SigmaGraphPanel } from '../components/SigmaGraphPanel';
import type { GraphData, KGCandidate } from '../types';

export function KnowledgeGraphPage() {
  const [candidates, setCandidates] = useState<KGCandidate[]>([]);
  const [selectedPath, setSelectedPath] = useState('');
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [resolvedPath, setResolvedPath] = useState('');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const loadCandidates = async () => {
    setError('');
    try {
      const data = await listKGCandidates();
      setCandidates(data.items);
      setSelectedPath((current) => {
        if (current && data.items.some((item) => item.path === current)) return current;
        return data.items[0]?.path ?? '';
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const loadGraph = async (path: string) => {
    if (!path) return;
    setLoading(true);
    setError('');
    try {
      const data = await readGraphArtifact(path);
      setGraphData(data.data);
      setResolvedPath(data.path);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadCandidates();
  }, []);

  useEffect(() => {
    if (!selectedPath) return;
    void loadGraph(selectedPath);
  }, [selectedPath]);

  return (
    <div className="space-y-3">
      <section className="panel p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="m-0 font-display text-lg font-semibold">知识图谱</h2>
            <p className="m-0 mt-1 text-sm text-app-muted">知识图谱预览</p>
          </div>
          <div className="flex items-center gap-2">
            <button type="button" className="soft-button" onClick={() => void loadCandidates()}>
              刷新图谱源
            </button>
            <button type="button" className="soft-button" onClick={() => void loadGraph(selectedPath)} disabled={!selectedPath || loading}>
              {loading ? '加载中...' : '重载图'}
            </button>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-[1fr_240px]">
          <select className="field" value={selectedPath} onChange={(event) => setSelectedPath(event.target.value)}>
            <option value="">请选择图谱</option>
            {candidates.map((candidate) => (
              <option key={`${candidate.source}:${candidate.path}`} value={candidate.path}>
                {candidate.display_name}
              </option>
            ))}
          </select>
          <div className="rounded-xl border border-app-border bg-slate-50 px-3 py-2 text-xs text-app-muted">
            共 {candidates.length} 个可用图谱
          </div>
        </div>
      </section>

      {error ? <p className="rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-600">{error}</p> : null}

      <SigmaGraphPanel data={graphData} />

      <details className="panel p-3">
        <summary className="cursor-pointer text-sm font-semibold">原始图谱 JSON</summary>
        <pre className="json-panel mt-2 max-h-96">{JSON.stringify(graphData, null, 2)}</pre>
      </details>
    </div>
  );
}
