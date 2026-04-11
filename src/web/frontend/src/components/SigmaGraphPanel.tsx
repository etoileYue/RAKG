import { useEffect, useMemo, useRef, useState } from 'react';
import { MultiGraph } from 'graphology';
import type Sigma from 'sigma';

import { getNodeColor, normalizeGraph } from '../lib/graph';
import type { GraphData, NormalizedGraphNode } from '../types';

interface SigmaGraphPanelProps {
  data: GraphData | null;
}

export function SigmaGraphPanel({ data }: SigmaGraphPanelProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sigmaRef = useRef<Sigma | null>(null);
  const graphRef = useRef<MultiGraph | null>(null);
  const nodeMapRef = useRef<Map<string, NormalizedGraphNode>>(new Map());

  const { nodes, edges } = useMemo(() => normalizeGraph(data), [data]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [searchText, setSearchText] = useState('');
  const [runtimeError, setRuntimeError] = useState('');

  const selectedNode = useMemo(() => {
    if (!selectedNodeId) return null;
    return nodeMapRef.current.get(selectedNodeId) ?? null;
  }, [selectedNodeId, nodes]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    setRuntimeError('');

    if (sigmaRef.current) {
      sigmaRef.current.kill();
      sigmaRef.current = null;
    }

    const graph = new MultiGraph();
    const nodeMap = new Map<string, NormalizedGraphNode>();
    nodeMapRef.current = nodeMap;

    if (nodes.length === 0) {
      graphRef.current = graph;
      setSelectedNodeId(null);
      return;
    }

    let canceled = false;
    let mountedSigma: Sigma | null = null;

    const mountSigma = async () => {
      try {
        const sigmaModule = await import('sigma');
        if (canceled) return;
        const SigmaCtor = sigmaModule.default;

        const radius = 14;
        nodes.forEach((node, index) => {
          const angle = (2 * Math.PI * index) / nodes.length;
          graph.addNode(node.id, {
            x: Math.cos(angle) * radius,
            y: Math.sin(angle) * radius,
            size: 6,
            label: node.label,
            color: getNodeColor(node.entityType),
          });
          nodeMap.set(node.id, node);
        });

        edges.forEach((edge) => {
          if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) return;
          if (graph.hasEdge(edge.id)) return;
          graph.addEdgeWithKey(edge.id, edge.source, edge.target, {
            size: 1,
            label: edge.relation,
            color: '#9ca3af',
          });
        });

        graph.forEachNode((nodeKey) => {
          const degree = graph.degree(nodeKey);
          graph.setNodeAttribute(nodeKey, 'size', Math.max(6, Math.min(14, 5 + degree * 0.9)));
        });

        mountedSigma = new SigmaCtor(graph, container, {
          renderEdgeLabels: false,
          labelDensity: 0.07,
          labelGridCellSize: 64,
          labelRenderedSizeThreshold: 8,
          defaultNodeType: 'circle',
          defaultEdgeType: 'line',
        });

        mountedSigma.on('clickNode', ({ node }) => {
          setSelectedNodeId(node);
          const attrs = graph.getNodeAttributes(node);
          const camera = mountedSigma?.getCamera();
          if (!camera) return;
          camera.animate({ x: attrs.x as number, y: attrs.y as number, ratio: 0.35 }, { duration: 300 });
        });

        mountedSigma.on('clickStage', () => setSelectedNodeId(null));

        graphRef.current = graph;
        sigmaRef.current = mountedSigma;
      } catch (err) {
        setRuntimeError(err instanceof Error ? err.message : String(err));
        graphRef.current = null;
        sigmaRef.current = null;
      }
    };

    void mountSigma();

    return () => {
      canceled = true;
      if (mountedSigma) {
        mountedSigma.kill();
      }
      sigmaRef.current = null;
    };
  }, [nodes, edges]);

  useEffect(() => {
    const sigma = sigmaRef.current;
    const graph = graphRef.current;
    if (!sigma || !graph) return;

    try {
      const keyword = searchText.trim().toLowerCase();
      const neighborSet = new Set<string>();

      if (selectedNodeId && graph.hasNode(selectedNodeId)) {
        graph.forEachNeighbor(selectedNodeId, (neighbor) => {
          neighborSet.add(neighbor);
        });
      }

      sigma.setSetting('nodeReducer', (node, attrs) => {
        const rawLabel = String(attrs.label ?? '').toLowerCase();
        const isMatch = !keyword || rawLabel.includes(keyword);
        const isSelected = !!selectedNodeId && node === selectedNodeId;
        const isNeighbor = !!selectedNodeId && neighborSet.has(node);
        const isFocusedArea = !selectedNodeId || isSelected || isNeighbor;

        if (!isMatch) {
          return { ...attrs, color: '#e2e8f0', label: '' };
        }

        if (!isFocusedArea) {
          return { ...attrs, color: '#d1d5db', label: '' };
        }

        if (isSelected) {
          return {
            ...attrs,
            color: '#059669',
            size: Math.max(Number(attrs.size ?? 8), 10),
            label: String(attrs.label ?? ''),
            zIndex: 1,
          };
        }

        return attrs;
      });

      sigma.setSetting('edgeReducer', (edge, attrs) => {
        if (!selectedNodeId || !graph.hasNode(selectedNodeId)) return attrs;
        const source = graph.source(edge);
        const target = graph.target(edge);
        if (source === selectedNodeId || target === selectedNodeId) {
          return { ...attrs, color: '#6b7280', size: 1.8 };
        }
        return { ...attrs, hidden: true };
      });

      sigma.refresh();
    } catch (err) {
      setRuntimeError(err instanceof Error ? err.message : String(err));
    }
  }, [selectedNodeId, searchText]);

  const zoom = (factor: number) => {
    const sigma = sigmaRef.current;
    if (!sigma) return;
    const camera = sigma.getCamera();
    const state = camera.getState();
    camera.animate({ ratio: Math.max(0.05, Math.min(2.2, state.ratio * factor)) }, { duration: 180 });
  };

  const resetView = () => {
    const sigma = sigmaRef.current;
    if (!sigma) return;
    setSelectedNodeId(null);
    sigma.getCamera().animate({ x: 0, y: 0, ratio: 1, angle: 0 }, { duration: 250 });
  };

  if (nodes.length === 0) {
    return <div className="panel flex h-[520px] items-center justify-center text-sm text-app-muted">无图数据(请先构建或加载图谱)</div>;
  }

  if (runtimeError) {
    return (
      <div className="panel flex h-[520px] flex-col items-center justify-center gap-2 px-4 text-center">
        <p className="m-0 text-sm font-semibold text-rose-600">图谱渲染失败</p>
        <p className="m-0 text-xs text-app-muted">{runtimeError}</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-3 xl:grid-cols-[1fr_320px]">
      <section className="panel p-3">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <input
              className="field w-72"
              value={searchText}
              onChange={(event) => setSearchText(event.target.value)}
              placeholder="页面内搜索节点..."
            />
            <button type="button" className="soft-button" onClick={resetView}>
              重置
            </button>
          </div>

          <div className="flex items-center gap-2">
            <button type="button" className="soft-button" onClick={() => zoom(0.82)}>
              放大
            </button>
            <button type="button" className="soft-button" onClick={() => zoom(1.22)}>
              缩小
            </button>
          </div>
        </div>

        <div ref={containerRef} className="h-[460px] overflow-hidden rounded-xl border border-app-border bg-slate-50" />
      </section>

      <aside className="panel p-3">
        <h3 className="m-0 text-sm font-semibold">节点属性</h3>
        {!selectedNode ? (
          <p className="mt-3 text-sm text-app-muted">点击图中节点查看详情</p>
        ) : (
          <div className="mt-3 space-y-2">
            <div>
              <p className="m-0 text-xs text-app-muted">名称</p>
              <p className="m-0 text-sm font-semibold">{selectedNode.label}</p>
            </div>
            <div>
              <p className="m-0 text-xs text-app-muted">类型</p>
              <p className="m-0 text-sm">{selectedNode.entityType}</p>
            </div>
            <div>
              <p className="m-0 text-xs text-app-muted">描述</p>
              <p className="m-0 whitespace-pre-wrap text-sm">{selectedNode.description || '暂无'}</p>
            </div>

            <details>
              <summary className="cursor-pointer text-xs font-semibold text-app-muted">原始节点 JSON</summary>
              <pre className="json-panel mt-2 max-h-72">{JSON.stringify(selectedNode.raw, null, 2)}</pre>
            </details>
          </div>
        )}
      </aside>
    </div>
  );
}
