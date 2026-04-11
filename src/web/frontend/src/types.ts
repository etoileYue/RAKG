export type TaskStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'canceled';

export interface TaskSummary {
  task_id: string;
  task_type: string;
  status: TaskStatus;
  progress: number;
  message: string;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
}

export interface TaskDetail extends TaskSummary {
  input_payload: Record<string, unknown>;
  output_payload: Record<string, unknown>;
  error_message?: string | null;
  cancel_requested: boolean;
}

export interface TaskListResponse {
  total: number;
  page: number;
  page_size: number;
  items: TaskSummary[];
}

export interface LogsResponse {
  total: number;
  page: number;
  page_size: number;
  items: Array<{
    id: number;
    task_id?: string | null;
    level: string;
    source: string;
    message: string;
    created_at: string;
  }>;
}

export interface QAResponse {
  question: string;
  formatted_answer: string;
  answer: string;
  retrieved_context: Record<string, unknown>;
  evidence_sources: Array<Record<string, string>>;
  graph_paths: string[];
  intermediate: Record<string, unknown>;
}

export interface HealthResponse {
  healthy: boolean;
  queue: {
    queued: number;
    running: number;
    running_task_id?: string | null;
  };
  model_config_summary: Record<string, unknown>;
}

export interface GraphData {
  entities?: Array<Record<string, unknown>>;
  relations?: Array<Record<string, unknown> | [string, string, string, string?]>;
}

export interface KGCandidate {
  path: string;
  source: 'seed' | 'task';
  display_name: string;
  task_id?: string | null;
}

export interface KGCandidateListResponse {
  total: number;
  items: KGCandidate[];
}

export interface NormalizedGraphNode {
  id: string;
  label: string;
  entityType: string;
  description: string;
  raw: Record<string, unknown>;
}

export interface NormalizedGraphEdge {
  id: string;
  source: string;
  target: string;
  relation: string;
  description: string;
  raw: Record<string, unknown>;
}
