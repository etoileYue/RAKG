import { useMemo, useState } from 'react';

import type { GraphData } from '../types';

type GraphNode = {
  name: string;
  type: string;
  description: string;
  raw: Record<string, unknown>;
};

type GraphEdge = {
  source: string;
  target: string;
  relation: string;
  description: string;
};

function normalizeGraph(data: GraphData | null | undefined): {
  nodes: GraphNode[];
  edges: GraphEdge[];
} {
  if (!data) return { nodes: [], edges: [] };

  const entities = Array.isArray(data.entities) ? data.entities : [];
  const relations = Array.isArray(data.relations) ? data.relations : [];

  const nodeMap = new Map<string, GraphNode>();

  for (const item of entities) {
    if (!item || typeof item !== 'object') continue;
    const obj = item as Record<string, unknown>;
    const name = String(obj.name ?? '').trim();
    if (!name) continue;
    nodeMap.set(name, {
      name,
      type: String(obj.type ?? 'Unknown'),
      description: String(obj.description ?? ''),
      raw: obj,
    });
  }

  const edges: GraphEdge[] = [];
  for (const rel of relations) {
    if (Array.isArray(rel) && rel.length >= 3) {
      const [source, relation, target, description = ''] = rel;
      const sourceName = String(source);
      const targetName = String(target);
      if (!nodeMap.has(sourceName)) {
        nodeMap.set(sourceName, {
          name: sourceName,
          type: 'Unknown',
          description: '',
          raw: { name: sourceName, type: 'Unknown' },
        });
      }
      if (!nodeMap.has(targetName)) {
        nodeMap.set(targetName, {
          name: targetName,
          type: 'Unknown',
          description: '',
          raw: { name: targetName, type: 'Unknown' },
        });
      }
      edges.push({
        source: sourceName,
        target: targetName,
        relation: String(relation),
        description: String(description ?? ''),
      });
      continue;
    }

    if (rel && typeof rel === 'object') {
      const obj = rel as Record<string, unknown>;
      const sourceName = String(obj.source ?? '').trim();
      const targetName = String(obj.target ?? '').trim();
      const relationName = String(obj.relation ?? '').trim();
      if (!sourceName || !targetName || !relationName) continue;

      if (!nodeMap.has(sourceName)) {
        nodeMap.set(sourceName, {
          name: sourceName,
          type: 'Unknown',
          description: '',
          raw: { name: sourceName, type: 'Unknown' },
        });
      }
      if (!nodeMap.has(targetName)) {
        nodeMap.set(targetName, {
          name: targetName,
          type: 'Unknown',
          description: '',
          raw: { name: targetName, type: 'Unknown' },
        });
      }

      edges.push({
        source: sourceName,
        target: targetName,
        relation: relationName,
        description: String(obj.description ?? obj.rel_description ?? ''),
      });
    }
  }

  return {
    nodes: Array.from(nodeMap.values()),
    edges,
  };
}

export function GraphView({ data }: { data: GraphData | null }) {
  const { nodes, edges } = useMemo(() => normalizeGraph(data), [data]);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

  if (nodes.length === 0) {
    return <div className="empty-card">暂无图谱数据</div>;
  }

  const width = 900;
  const height = 560;
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.max(140, Math.min(width, height) * 0.35);

  // V1 使用固定圆环布局，避免引入复杂布局库。
  const positions = new Map<string, { x: number; y: number }>();
  nodes.forEach((node, idx) => {
    const angle = (2 * Math.PI * idx) / nodes.length;
    positions.set(node.name, {
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius,
    });
  });

  return (
    <div className="graph-layout">
      <svg viewBox={`0 0 ${width} ${height}`} className="graph-canvas" role="img" aria-label="knowledge graph">
        <defs>
          <marker
            id="arrowhead"
            markerWidth="8"
            markerHeight="6"
            refX="7"
            refY="3"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <path d="M0,0 L8,3 L0,6 z" fill="#5b6876" />
          </marker>
        </defs>

        {edges.map((edge, idx) => {
          const source = positions.get(edge.source);
          const target = positions.get(edge.target);
          if (!source || !target) return null;
          const midX = (source.x + target.x) / 2;
          const midY = (source.y + target.y) / 2;
          return (
            <g key={`${edge.source}-${edge.target}-${edge.relation}-${idx}`}>
              <line
                x1={source.x}
                y1={source.y}
                x2={target.x}
                y2={target.y}
                stroke="#6d7c8d"
                strokeWidth="1.2"
                markerEnd="url(#arrowhead)"
              >
                <title>{`${edge.source} -[${edge.relation}]-> ${edge.target}\n${edge.description}`}</title>
              </line>
              <text x={midX} y={midY - 4} textAnchor="middle" className="edge-label">
                {edge.relation}
              </text>
            </g>
          );
        })}

        {nodes.map((node) => {
          const pos = positions.get(node.name);
          if (!pos) return null;
          const active = selectedNode?.name === node.name;
          return (
            <g key={node.name} transform={`translate(${pos.x}, ${pos.y})`} className="node-group">
              <circle
                r={active ? 31 : 27}
                fill={active ? '#0f766e' : '#124e66'}
                stroke={active ? '#5eead4' : '#88c5d7'}
                strokeWidth={active ? 3 : 2}
                onClick={() => setSelectedNode(node)}
              >
                <title>{`${node.name} (${node.type})`}</title>
              </circle>
              <text y={5} textAnchor="middle" className="node-label">
                {node.name.length > 10 ? `${node.name.slice(0, 10)}...` : node.name}
              </text>
            </g>
          );
        })}
      </svg>

      <div className="node-panel">
        {selectedNode ? (
          <>
            <h4>{selectedNode.name}</h4>
            <p>类型: {selectedNode.type}</p>
            <p>描述: {selectedNode.description || '暂无'}</p>
            <pre>{JSON.stringify(selectedNode.raw, null, 2)}</pre>
          </>
        ) : (
          <p>点击节点查看属性</p>
        )}
      </div>
    </div>
  );
}
