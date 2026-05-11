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

interface ChatStreamHandlers {
  onDelta?: (chunk: string, fullReply: string) => void;
  onSession?: (sessionId: string) => void;
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

export interface RuntimeConfig {
  workspace_dir: string;
  memory_dir: string;
  sessions_dir: string;
  memory_file: string;
  topic_memory_store_file: string;
  topic_memory_store_dir: string;
  raw_messages_dir: string;
}

export interface TopicMemoryOverview {
  version: number;
  updated_at: string;
  counts: {
    experiences: number;
    segments: number;
    qas: number;
    relations: number;
  };
  current_runtime_state?: {
    conversation_id?: string;
    current_experience_id?: string | null;
    current_segment_id?: string | null;
    latest_qa_id?: string | null;
    recent_segment_ids?: string[];
    updated_at?: string;
  } | null;
  latest_experience?: {
    topic?: string;
    goal?: string;
    summary_short?: string;
  } | null;
  latest_segment?: {
    topic?: string;
    intent?: string;
    status?: string;
    summary?: string;
  } | null;
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

function parseSseChunk(chunk: string): { event: string; data: string } | null {
  const lines = chunk.split(/\r?\n/);
  let event = 'message';
  const dataLines: string[] = [];

  for (const line of lines) {
    if (!line || line.startsWith(':')) continue;
    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trim();
      continue;
    }
    if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).trimStart());
    }
  }

  if (!dataLines.length) {
    return null;
  }

  return { event, data: dataLines.join('\n') };
}

export async function streamMessage(
  message: string,
  sessionId?: string,
  handlers: ChatStreamHandlers = {}
): Promise<ChatResponse> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        stream: true,
      }),
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

  if (!response.body) {
    throw new Error('Streaming is not supported in the current browser.');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let reply = '';
  let resolvedSessionId = sessionId ?? '';

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });

    const chunks = buffer.split(/\r?\n\r?\n/);
    buffer = chunks.pop() ?? '';

    for (const chunk of chunks) {
      const parsed = parseSseChunk(chunk);
      if (!parsed) continue;

      let payload: { content?: string; reply?: string; session_id?: string; detail?: string } = {};
      try {
        payload = JSON.parse(parsed.data) as typeof payload;
      } catch {
        payload = {};
      }

      if (parsed.event === 'session' && payload.session_id) {
        resolvedSessionId = payload.session_id;
        handlers.onSession?.(payload.session_id);
      }

      if (parsed.event === 'delta' && payload.content) {
        reply += payload.content;
        handlers.onDelta?.(payload.content, reply);
      }

      if (parsed.event === 'error') {
        throw new Error(payload.detail || 'Stream error');
      }

      if (parsed.event === 'done') {
        return {
          reply: payload.reply ?? reply,
          session_id: payload.session_id ?? resolvedSessionId,
        };
      }
    }

    if (done) {
      break;
    }
  }

  return {
    reply,
    session_id: resolvedSessionId,
  };
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

export async function getRuntimeConfig(): Promise<RuntimeConfig> {
  return request<RuntimeConfig>('/runtime-config');
}

export async function getRawMessages(sessionId: string): Promise<RawMessagesResponse> {
  return request<RawMessagesResponse>(`/sessions/${encodeURIComponent(sessionId)}/raw-messages`);
}

export async function previewRawMessages(sessionId?: string): Promise<RawMessagesResponse> {
  const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
  return request<RawMessagesResponse>(`/raw-messages/preview${query}`);
}

export async function getTopicMemoryOverview(sessionId?: string): Promise<TopicMemoryOverview> {
  const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
  return request<TopicMemoryOverview>(`/topic-memory/overview${query}`);
}
