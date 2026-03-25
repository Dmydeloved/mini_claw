/**
 * API client for Mini-OpenClaw backend.
 */

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, '') ??
  '/api';

export interface ChatResponse {
  reply: string;
  session_id: string;
}

export interface Skill {
  name: string;
  description: string;
  trigger: string;
  enabled: boolean;
  location: string;
}

export interface SkillContent {
  name: string;
  content: string;
  path: string;
}

export interface FileContent {
  path: string;
  content: string;
}

export interface SessionInfo {
  session_id: string;
  title: string;
  updated_at: string;
}

export interface CreateSessionResponse {
  session_id: string;
}

export interface RawMessage {
  role: string;
  content: string;
}

export interface RawMessagesResponse {
  session_id: string;
  updated_at: string;
  message_count: number;
  messages: RawMessage[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers ?? {}),
      },
      ...init,
    });
  } catch {
    throw new Error('Connection error: 无法连接后端服务，请确认 backend 已启动。');
  }

  if (!response.ok) {
    let detail = '';

    try {
      const data = (await response.json()) as { detail?: string };
      detail = data.detail || '';
    } catch {
      detail = '';
    }

    throw new Error(detail || `API error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

export async function sendMessage(
  message: string,
  sessionId?: string
): Promise<ChatResponse> {
  return request<ChatResponse>('/chat', {
    method: 'POST',
    body: JSON.stringify({
      message,
      session_id: sessionId,
      stream: false,
    }),
  });
}

export async function getSkills(): Promise<Skill[]> {
  return request<Skill[]>('/skills');
}

export async function getSkillContent(skillName: string): Promise<SkillContent> {
  return request<SkillContent>(`/skills/${encodeURIComponent(skillName)}`);
}

export async function getFile(path: string): Promise<FileContent> {
  return request<FileContent>(`/files?path=${encodeURIComponent(path)}`);
}

export async function saveFile(path: string, content: string): Promise<void> {
  await request<{ status: string }>('/files', {
    method: 'POST',
    body: JSON.stringify({ path, content }),
  });
}

export async function getSessions(): Promise<SessionInfo[]> {
  return request<SessionInfo[]>('/sessions');
}

export async function createSession(): Promise<CreateSessionResponse> {
  return request<CreateSessionResponse>('/sessions', {
    method: 'POST',
  });
}

export async function getRawMessages(sessionId: string): Promise<RawMessagesResponse> {
  return request<RawMessagesResponse>(`/sessions/${encodeURIComponent(sessionId)}/raw-messages`);
}

export async function previewRawMessages(sessionId?: string): Promise<RawMessagesResponse> {
  const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
  return request<RawMessagesResponse>(`/raw-messages/preview${query}`);
}
