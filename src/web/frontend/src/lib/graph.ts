import type { GraphData, NormalizedGraphEdge, NormalizedGraphNode } from '../types';

const ENTITY_COLORS: Record<string, string> = {
  Person: '#0ea5e9',
  Organization: '#f59e0b',
  Event: '#10b981',
  Location: '#8b5cf6',
  Unknown: '#64748b',
};

export function getNodeColor(entityType: string): string {
  return ENTITY_COLORS[entityType] ?? '#14b8a6';
}

export function normalizeGraph(data: GraphData | null | undefined): {
  nodes: NormalizedGraphNode[];
  edges: NormalizedGraphEdge[];
} {
  if (!data) return { nodes: [], edges: [] };

  const entities = Array.isArray(data.entities) ? data.entities : [];
  const relations = Array.isArray(data.relations) ? data.relations : [];

  const nodeMap = new Map<string, NormalizedGraphNode>();

  for (const item of entities) {
    if (!item || typeof item !== 'object') continue;
    const obj = item as Record<string, unknown>;
    const name = String(obj.name ?? '').trim();
    if (!name) continue;
    const entityType = String(obj.type ?? obj.entity_type ?? 'Unknown');
    nodeMap.set(name, {
      id: name,
      label: name,
      entityType,
      description: String(obj.description ?? ''),
      raw: obj,
    });
  }

  const edges: NormalizedGraphEdge[] = [];

  for (const rel of relations) {
    if (Array.isArray(rel) && rel.length >= 3) {
      const [source, relation, target, description = ''] = rel;
      const sourceName = String(source).trim();
      const targetName = String(target).trim();
      const relationName = String(relation).trim();
      if (!sourceName || !targetName || !relationName) continue;
      addUnknownNode(nodeMap, sourceName);
      addUnknownNode(nodeMap, targetName);
      edges.push({
        id: `${sourceName}-${relationName}-${targetName}-${edges.length}`,
        source: sourceName,
        target: targetName,
        relation: relationName,
        description: String(description ?? ''),
        raw: { source: sourceName, target: targetName, relation: relationName, description },
      });
      continue;
    }

    if (rel && typeof rel === 'object') {
      const obj = rel as Record<string, unknown>;
      const sourceName = String(obj.source ?? '').trim();
      const targetName = String(obj.target ?? '').trim();
      const relationName = String(obj.relation ?? obj.type ?? '').trim();
      if (!sourceName || !targetName || !relationName) continue;

      addUnknownNode(nodeMap, sourceName);
      addUnknownNode(nodeMap, targetName);

      edges.push({
        id: `${sourceName}-${relationName}-${targetName}-${edges.length}`,
        source: sourceName,
        target: targetName,
        relation: relationName,
        description: String(obj.description ?? obj.rel_description ?? ''),
        raw: obj,
      });
    }
  }

  return { nodes: Array.from(nodeMap.values()), edges };
}

function addUnknownNode(nodeMap: Map<string, NormalizedGraphNode>, id: string): void {
  if (nodeMap.has(id)) return;
  nodeMap.set(id, {
    id,
    label: id,
    entityType: 'Unknown',
    description: '',
    raw: { name: id, type: 'Unknown' },
  });
}
