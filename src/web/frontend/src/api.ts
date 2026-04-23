import type {
  GraphData,
  HealthResponse,
  QADeleteConversationResponse,
  QAConversationSummary,
  QAConversationDetailResponse,
  QAConversationListResponse,
  KGCandidateListResponse,
  LogsResponse,
  QAResponse,
  QASendMessageResponse,
  TaskDetail,
  TaskListResponse,
} from './types';

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api/v1';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers ?? {});
  if (!(init?.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
  });

  if (!response.ok) {
    const rawText = await response.text();
    let message = rawText || `request failed: ${response.status}`;
    try {
      const parsed = JSON.parse(rawText) as { detail?: string };
      if (parsed.detail) message = parsed.detail;
    } catch {
      // keep raw text
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}

export function getApiDocsUrl(): string {
  return '/docs';
}

export async function createKGBuildTask(payload: {
  input_type: 'text' | 'json_path';
  text?: string;
  topic?: string;
  json_path?: string;
  output_dir?: string;
  existing_kg?: string;
  force_rebuild?: boolean;
}): Promise<{ task_id: string }> {
  return request('/tasks/kg-build', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function createKGBuildTaskUpload(payload: {
  input_type: 'json_text' | 'json_file';
  json_text?: string;
  json_file?: File;
  output_dir?: string;
  existing_kg?: string;
  force_rebuild?: boolean;
}): Promise<{ task_id: string }> {
  const form = new FormData();
  form.set('input_type', payload.input_type);
  if (payload.input_type === 'json_text') {
    form.set('json_text', payload.json_text ?? '');
  } else if (payload.json_file) {
    form.set('json_file', payload.json_file);
  }
  if (payload.output_dir) form.set('output_dir', payload.output_dir);
  if (payload.existing_kg) form.set('existing_kg', payload.existing_kg);
  if (payload.force_rebuild) form.set('force_rebuild', String(payload.force_rebuild));

  return request('/tasks/kg-build/upload', {
    method: 'POST',
    body: form,
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

export async function listQAConversations(): Promise<QAConversationListResponse> {
  return request('/qa/conversations');
}

export async function createQAConversation(payload: { kg_path: string; title?: string }): Promise<QAConversationSummary> {
  return request('/qa/conversations', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function getQAConversation(conversationId: string): Promise<QAConversationDetailResponse> {
  return request(`/qa/conversations/${conversationId}`);
}

export async function sendQAConversationMessage(
  conversationId: string,
  payload: {
    question: string;
    max_hop: number;
    seed_top_k: number;
    max_context_items: number;
  }
): Promise<QASendMessageResponse> {
  return request(`/qa/conversations/${conversationId}/messages`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function deleteQAConversation(conversationId: string): Promise<QADeleteConversationResponse> {
  return request(`/qa/conversations/${conversationId}`, {
    method: 'DELETE',
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

export async function listKGCandidates(): Promise<KGCandidateListResponse> {
  return request('/artifacts/kg/candidates');
}
