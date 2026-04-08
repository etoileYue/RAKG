import type {
  GraphData,
  HealthResponse,
  LogsResponse,
  QAResponse,
  TaskDetail,
  TaskListResponse,
} from './types';

// Dev 默认走 Vite 同源代理，避免跨域与端口漂移导致的 fetch 失败。
const API_BASE = import.meta.env.VITE_API_BASE ?? '/api/v1';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `request failed: ${response.status}`);
  }

  return (await response.json()) as T;
}

export async function createKGBuildTask(payload: {
  input_type: 'text' | 'json_path';
  text?: string;
  topic?: string;
  json_path?: string;
  output_dir?: string;
}): Promise<{ task_id: string }> {
  return request('/tasks/kg-build', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function getTask(taskId: string): Promise<TaskDetail> {
  return request(`/tasks/${taskId}`);
}

export async function listTasks(params?: {
  page?: number;
  page_size?: number;
  task_type?: string;
  status?: string;
}): Promise<TaskListResponse> {
  const search = new URLSearchParams();
  if (params?.page) search.set('page', String(params.page));
  if (params?.page_size) search.set('page_size', String(params.page_size));
  if (params?.task_type) search.set('task_type', params.task_type);
  if (params?.status) search.set('status', params.status);
  const suffix = search.size ? `?${search.toString()}` : '';
  return request(`/tasks${suffix}`);
}

export async function cancelTask(taskId: string): Promise<{ task_id: string; accepted: boolean; status: string }> {
  return request(`/tasks/${taskId}/cancel`, { method: 'POST' });
}

export async function qaQuery(payload: {
  kg_path: string;
  question: string;
  max_hop: number;
  seed_top_k: number;
  max_context_items: number;
}): Promise<QAResponse> {
  return request('/qa/query', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function getLogs(params?: {
  page?: number;
  page_size?: number;
  task_id?: string;
  keyword?: string;
}): Promise<LogsResponse> {
  const search = new URLSearchParams();
  if (params?.page) search.set('page', String(params.page));
  if (params?.page_size) search.set('page_size', String(params.page_size));
  if (params?.task_id) search.set('task_id', params.task_id);
  if (params?.keyword) search.set('keyword', params.keyword);
  const suffix = search.size ? `?${search.toString()}` : '';
  return request(`/logs${suffix}`);
}

export async function getHealth(): Promise<HealthResponse> {
  return request('/system/health');
}

export async function readGraphArtifact(path: string): Promise<{ path: string; data: GraphData }> {
  const search = new URLSearchParams({ path });
  return request(`/artifacts/kg?${search.toString()}`);
}
