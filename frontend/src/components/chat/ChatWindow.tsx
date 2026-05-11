'use client';

import Editor from '@monaco-editor/react';
import { marked } from 'marked';
import { useEffect, useMemo, useRef, useState } from 'react';

import {
  createSession,
  getFile,
  getRawMessages,
  getRuntimeConfig,
  getSessions,
  getSkillContent,
  getSkills,
  getTopicMemoryOverview,
  previewRawMessages,
  saveFile,
  streamMessage,
  type RuntimeConfig,
  type SessionInfo,
  type Skill,
  type TopicMemoryOverview,
} from '@/lib/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface FilePanelItem {
  path: string;
  label: string;
  description: string;
}

type PanelTab = 'chat' | 'memory' | 'skills';
type ChatInspectorTab = 'session' | 'log' | 'raw';

const DEFAULT_RUNTIME_CONFIG: RuntimeConfig = {
  workspace_dir: 'workspace',
  memory_dir: 'memory',
  sessions_dir: 'sessions',
  memory_file: 'memory/MEMORY.md',
  topic_memory_store_file: 'memory/topic_memory_store.json',
  topic_memory_store_dir: 'memory/topic_memory_store',
  raw_messages_dir: 'sessions/_raw_messages',
};

function buildMemoryFiles(config: RuntimeConfig): FilePanelItem[] {
  return [
    {
      path: config.memory_file,
      label: 'MEMORY.md',
      description: 'Rendered topic memory snapshot',
    },
    {
      path: config.topic_memory_store_file,
      label: 'topic_memory_store.json',
      description: 'Topic memory overview index',
    },
    { path: 'workspace/SOUL.md', label: 'SOUL.md', description: 'System soul' },
    { path: 'workspace/IDENTITY.md', label: 'IDENTITY.md', description: 'Identity rules' },
    { path: 'workspace/USER.md', label: 'USER.md', description: 'User profile' },
    { path: 'workspace/AGENTS.md', label: 'AGENTS.md', description: 'Agent protocol' },
    {
      path: 'workspace/SKILLS_SNAPSHOT.md',
      label: 'SKILLS_SNAPSHOT.md',
      description: 'Skills snapshot',
    },
  ];
}

function formatTodayLogPath(memoryDir: string) {
  const now = new Date();
  const year = now.getFullYear();
  const month = `${now.getMonth() + 1}`.padStart(2, '0');
  const day = `${now.getDate()}`.padStart(2, '0');
  return `${memoryDir}/logs/${year}-${month}-${day}.md`;
}

function getSessionFilePath(sessionId: string, sessionsDir: string) {
  return `${sessionsDir}/${sessionId}.json`;
}

function getRawMessagesFilePath(sessionId: string, rawMessagesDir: string) {
  return `${rawMessagesDir}/${sessionId}.json`;
}

function inferEditorLanguage(path: string) {
  if (path.endsWith('.json')) return 'json';
  if (path.endsWith('.md')) return 'markdown';
  if (path.endsWith('.py')) return 'python';
  return 'plaintext';
}

function parseSessionMessages(content: string): Message[] {
  try {
    const history = JSON.parse(content) as Array<{ role?: string; content?: string }>;
    return history
      .filter(
        (message): message is Message =>
          (message.role === 'user' || message.role === 'assistant') &&
          typeof message.content === 'string'
      )
      .map((message) => ({
        role: message.role,
        content: message.content,
      }));
  } catch {
    return [];
  }
}

export default function ChatWindow() {
  const [runtimeConfig, setRuntimeConfig] = useState<RuntimeConfig>(DEFAULT_RUNTIME_CONFIG);
  const [activeTab, setActiveTab] = useState<PanelTab>('chat');
  const [chatInspectorTab, setChatInspectorTab] = useState<ChatInspectorTab>('session');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [chatInspectorLoading, setChatInspectorLoading] = useState(false);
  const [skills, setSkills] = useState<Skill[]>([]);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [topicMemoryOverview, setTopicMemoryOverview] = useState<TopicMemoryOverview | null>(null);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [editorPath, setEditorPath] = useState(DEFAULT_RUNTIME_CONFIG.memory_file);
  const [editorContent, setEditorContent] = useState('');
  const [editorLabel, setEditorLabel] = useState('Memory Editor');
  const [editorDirty, setEditorDirty] = useState(false);
  const [sessionRecordContent, setSessionRecordContent] = useState('[]');
  const [todayLogContent, setTodayLogContent] = useState('# Today Log\n\nNo log for today yet.');
  const [rawMessagesContent, setRawMessagesContent] = useState(
    '{\n  "message_count": 0,\n  "messages": []\n}'
  );
  const [statusText, setStatusText] = useState('Ready');
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const memoryFiles = useMemo(() => buildMemoryFiles(runtimeConfig), [runtimeConfig]);
  const todayLogPath = useMemo(
    () => formatTodayLogPath(runtimeConfig.memory_dir),
    [runtimeConfig.memory_dir]
  );

  const editorTitle = useMemo(() => {
    if (activeTab === 'memory') return 'Memory Editor';
    if (editorLabel) return editorLabel;
    const parts = editorPath.split('/');
    return parts[parts.length - 1] ?? editorPath;
  }, [activeTab, editorLabel, editorPath]);

  const chatInspectorPath =
    chatInspectorTab === 'session'
      ? sessionId
        ? getSessionFilePath(sessionId, runtimeConfig.sessions_dir)
        : `${runtimeConfig.sessions_dir}/session.json`
      : chatInspectorTab === 'raw'
        ? sessionId
          ? getRawMessagesFilePath(sessionId, runtimeConfig.raw_messages_dir)
          : 'preview'
        : todayLogPath;
  const chatInspectorTitle =
    chatInspectorTab === 'session'
      ? 'Session Record'
      : chatInspectorTab === 'raw'
        ? 'Raw Prompt'
        : 'Today Log';
  const chatInspectorContent =
    chatInspectorTab === 'session'
      ? sessionRecordContent
      : chatInspectorTab === 'raw'
        ? rawMessagesContent
        : todayLogContent;

  const layoutColumns = `${leftCollapsed ? '76px' : '320px'} minmax(0,1fr) ${
    rightCollapsed ? '76px' : '460px'
  }`;

  const isErrorMessage = (message: Message) =>
    message.role === 'assistant' &&
    (message.content.startsWith('Error:') || message.content.includes('Connection error'));

  useEffect(() => {
    void bootstrap();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  async function loadChatArtifacts(
    targetSessionId?: string,
    config: RuntimeConfig = runtimeConfig
  ) {
    const sessionPath = targetSessionId
      ? getSessionFilePath(targetSessionId, config.sessions_dir)
      : '';

    const sessionFilePromise = sessionPath
      ? getFile(sessionPath).catch(() => ({ path: sessionPath, content: '[]' }))
      : Promise.resolve({ path: '', content: '[]' });

    const rawMessagesFilePromise = (targetSessionId
      ? getRawMessages(targetSessionId)
      : previewRawMessages()
    )
      .then((payload) => ({
        path: targetSessionId
          ? getRawMessagesFilePath(targetSessionId, config.raw_messages_dir)
          : 'preview',
        content: JSON.stringify(payload, null, 2),
      }))
      .catch(() => ({
        path: targetSessionId
          ? getRawMessagesFilePath(targetSessionId, config.raw_messages_dir)
          : 'preview',
        content: '{\n  "message_count": 0,\n  "messages": []\n}',
      }));

    const logPath = formatTodayLogPath(config.memory_dir);
    const logFilePromise = getFile(logPath).catch(() => ({
      path: logPath,
      content: '# Today Log\n\nNo log for today yet.',
    }));
    const topicMemoryPromise = getTopicMemoryOverview(targetSessionId).catch(() => null);

    const [sessionFile, rawMessagesFile, logFile, topicMemoryOverview] = await Promise.all([
      sessionFilePromise,
      rawMessagesFilePromise,
      logFilePromise,
      topicMemoryPromise,
    ]);
    return { sessionFile, rawMessagesFile, logFile, topicMemoryOverview };
  }

  async function refreshChatInspector(
    targetSessionId?: string,
    config: RuntimeConfig = runtimeConfig
  ) {
    try {
      setChatInspectorLoading(true);
      const { sessionFile, rawMessagesFile, logFile, topicMemoryOverview } =
        await loadChatArtifacts(targetSessionId, config);
      setSessionRecordContent(sessionFile.content);
      setRawMessagesContent(rawMessagesFile.content);
      setTodayLogContent(logFile.content);
      setTopicMemoryOverview(topicMemoryOverview);
    } catch (error) {
      console.error(error);
      setStatusText('Failed to refresh chat context');
    } finally {
      setChatInspectorLoading(false);
    }
  }

  async function bootstrap() {
    try {
      const [configResult, skillsResult, sessionsResult] = await Promise.allSettled([
        getRuntimeConfig(),
        getSkills(),
        getSessions(),
      ]);
      const resolvedConfig =
        configResult.status === 'fulfilled' ? configResult.value : DEFAULT_RUNTIME_CONFIG;
      const memoryResult = await getFile(resolvedConfig.memory_file).catch(() => ({
        path: resolvedConfig.memory_file,
        content: '',
      }));

      const skillsData = skillsResult.status === 'fulfilled' ? skillsResult.value : [];
      const sessionsData = sessionsResult.status === 'fulfilled' ? sessionsResult.value : [];
      const memoryFile = memoryResult;

      setRuntimeConfig(resolvedConfig);
      setSkills(skillsData);
      setSessions(sessionsData);
      setEditorPath(memoryFile.path);
      setEditorLabel('Memory Editor');
      setEditorContent(memoryFile.content);
      setEditorDirty(false);
      setStatusText('Workspace loaded');

      if (sessionsData.length > 0) {
        await openSession(sessionsData[0].session_id, true, resolvedConfig);
      } else {
        await refreshChatInspector(undefined, resolvedConfig);
      }
    } catch (error) {
      console.error(error);
      setStatusText('Failed to load workspace');
    }
  }

  async function openFile(path: string) {
    try {
      const file = await getFile(path);
      const selectedFile = memoryFiles.find((item) => item.path === path);
      setEditorPath(file.path);
      setEditorLabel(selectedFile?.label ?? 'Memory Editor');
      setEditorContent(file.content);
      setEditorDirty(false);
      setActiveTab('memory');
      setRightCollapsed(false);
      setStatusText(`Opened ${file.path}`);
    } catch (error) {
      console.error(error);
      setStatusText(`Failed to open ${path}`);
    }
  }

  async function openSkill(skillName: string) {
    try {
      const skill = await getSkillContent(skillName);
      setEditorPath(skill.path);
      setEditorLabel(skill.name);
      setEditorContent(skill.content);
      setEditorDirty(false);
      setActiveTab('skills');
      setRightCollapsed(false);
      setStatusText(`Opened skill ${skillName}`);
    } catch (error) {
      console.error(error);
      setStatusText(`Failed to open skill ${skillName}`);
    }
  }

  async function openSession(
    nextSessionId: string,
    silent = false,
    config: RuntimeConfig = runtimeConfig
  ) {
    try {
      const { sessionFile, rawMessagesFile, logFile, topicMemoryOverview } =
        await loadChatArtifacts(nextSessionId, config);

      setSessionId(nextSessionId);
      setMessages(parseSessionMessages(sessionFile.content));
      setSessionRecordContent(sessionFile.content);
      setRawMessagesContent(rawMessagesFile.content);
      setTodayLogContent(logFile.content);
      setTopicMemoryOverview(topicMemoryOverview);
      setActiveTab('chat');
      setChatInspectorTab('session');
      setRightCollapsed(false);

      if (!silent) {
        setStatusText(`Opened session ${nextSessionId}`);
      }
    } catch (error) {
      console.error(error);
      setStatusText(`Failed to open session ${nextSessionId}`);
    }
  }

  async function startNewChat() {
    try {
      setStatusText('Creating a fresh session');
      setInput('');
      setMessages([]);
      setActiveTab('chat');
      setChatInspectorTab('session');
      setRightCollapsed(false);

      const { session_id: nextSessionId } = await createSession();
      const [sessionsData, artifacts] = await Promise.all([
        getSessions(),
        loadChatArtifacts(nextSessionId),
      ]);

      setSessionId(nextSessionId);
      setSessions(sessionsData);
      setSessionRecordContent(artifacts.sessionFile.content);
      setRawMessagesContent(artifacts.rawMessagesFile.content);
      setTodayLogContent(artifacts.logFile.content);
      setTopicMemoryOverview(artifacts.topicMemoryOverview);
      setMessages([]);
      setStatusText(`New session ${nextSessionId}`);
    } catch (error) {
      console.error(error);
      setStatusText('Failed to create a new session');
    }
  }

  async function handleSave() {
    try {
      setSaving(true);
      await saveFile(editorPath, editorContent);
      setEditorDirty(false);
      setStatusText(`Saved ${editorPath}`);

      if (editorPath.endsWith('/SKILL.md')) {
        const [skillsData, sessionsData] = await Promise.all([getSkills(), getSessions()]);
        setSkills(skillsData);
        setSessions(sessionsData);
      }
    } catch (error) {
      console.error(error);
      setStatusText(`Failed to save ${editorPath}`);
    } finally {
      setSaving(false);
    }
  }

  function upsertAssistantMessage(content: string) {
    setMessages((prev) => {
      const next = [...prev];
      const lastMessage = next[next.length - 1];

      if (lastMessage?.role === 'assistant') {
        next[next.length - 1] = { ...lastMessage, content };
        return next;
      }

      return [...next, { role: 'assistant', content }];
    });
  }

  async function handleSend() {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: userMessage },
      { role: 'assistant', content: '' },
    ]);
    setLoading(true);
    setStatusText('Agent is streaming');
    setActiveTab('chat');
    setChatInspectorTab('session');
    setRightCollapsed(false);

    try {
      const response = await streamMessage(userMessage, sessionId, {
        onSession: (nextSessionId) => {
          setSessionId(nextSessionId);
        },
        onDelta: (_chunk, fullReply) => {
          upsertAssistantMessage(fullReply);
        },
      });
      const [sessionsData, artifacts] = await Promise.all([
        getSessions(),
        loadChatArtifacts(response.session_id),
      ]);

      setSessionId(response.session_id);
      setSessions(sessionsData);
      setSessionRecordContent(artifacts.sessionFile.content);
      setRawMessagesContent(artifacts.rawMessagesFile.content);
      setTodayLogContent(artifacts.logFile.content);
      setTopicMemoryOverview(artifacts.topicMemoryOverview);
      setMessages(parseSessionMessages(artifacts.sessionFile.content));
      setStatusText(`Updated session ${response.session_id}`);
    } catch (error) {
      console.error(error);
      const fallbackMessage =
        error instanceof Error
          ? error.message
          : 'Connection error: backend is unavailable.';
      upsertAssistantMessage(fallbackMessage);
      setStatusText('Chat request failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="h-screen overflow-hidden bg-app">
      <div className="mx-auto flex h-full max-w-[1880px] flex-col px-4 py-4">
        <header className="glass-panel mb-4 flex shrink-0 items-center justify-between px-5 py-4">
          <div>
            <p className="text-xs uppercase tracking-[0.28em] text-slate-500">IDE Workspace</p>
            <h1 className="text-xl font-semibold text-slate-900">mini OpenClaw</h1>
          </div>
          <a
            href="#"
            className="rounded-full border border-blue-200 bg-white/80 px-4 py-2 text-sm text-blue-700 transition hover:border-blue-400 hover:text-blue-900"
          >
            Workspace
          </a>
        </header>

        <div className="min-h-0 flex-1">
          <div className="grid h-full min-h-0 gap-4" style={{ gridTemplateColumns: layoutColumns }}>
            <aside className="glass-panel flex min-h-0 flex-col overflow-hidden">
              {leftCollapsed ? (
                <div className="flex h-full flex-col items-center justify-between py-4">
                  <button
                    onClick={() => setLeftCollapsed(false)}
                    className="panel-toggle-button"
                    aria-label="Expand left panel"
                  >
                    &gt;
                  </button>
                  <div className="flex flex-col items-center gap-3">
                    <button
                      onClick={() => {
                        setActiveTab('chat');
                        setLeftCollapsed(false);
                      }}
                      className="collapsed-rail-button"
                    >
                      C
                    </button>
                    <button
                      onClick={() => {
                        setActiveTab('memory');
                        setLeftCollapsed(false);
                      }}
                      className="collapsed-rail-button"
                    >
                      M
                    </button>
                    <button
                      onClick={() => {
                        setActiveTab('skills');
                        setLeftCollapsed(false);
                      }}
                      className="collapsed-rail-button"
                    >
                      S
                    </button>
                  </div>
                  <div className="vertical-caption">Left</div>
                </div>
              ) : (
                <>
                  <div className="border-b border-white/70 px-4 py-4">
                    <div className="mb-3 flex items-center justify-between gap-3">
                      <p className="panel-label">Sidebar</p>
                      <button
                        onClick={() => setLeftCollapsed(true)}
                        className="panel-toggle-button"
                        aria-label="Collapse left panel"
                      >
                        &lt;
                      </button>
                    </div>
                    <div className="flex rounded-2xl bg-slate-100 p-1">
                      {(['chat', 'memory', 'skills'] as PanelTab[]).map((tab) => (
                        <button
                          key={tab}
                          onClick={() => setActiveTab(tab)}
                          className={`flex-1 rounded-xl px-3 py-2 text-sm capitalize transition ${
                            activeTab === tab
                              ? 'bg-white text-slate-900 shadow-sm'
                              : 'text-slate-500 hover:text-slate-800'
                          }`}
                        >
                          {tab}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
                    {activeTab === 'chat' && (
                      <section className="space-y-5">
                        <div>
                          <div className="mb-3 flex items-center justify-between gap-3">
                            <p className="panel-label">Sessions</p>
                            <div className="flex items-center gap-2">
                              <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs text-slate-500">
                                {sessions.length}
                              </span>
                              <button
                                onClick={() => void startNewChat()}
                                className="rounded-full border border-blue-200 bg-white px-3 py-1 text-xs text-blue-700 transition hover:border-blue-300 hover:text-blue-900"
                              >
                                New Chat
                              </button>
                            </div>
                          </div>
                          <div className="space-y-2">
                            {sessions.length === 0 && (
                              <p className="text-sm text-slate-500">No saved sessions yet.</p>
                            )}
                            {sessions.map((session) => (
                              <button
                                key={session.session_id}
                                onClick={() => void openSession(session.session_id)}
                                className={`w-full rounded-2xl border px-3 py-3 text-left transition ${
                                  sessionId === session.session_id
                                    ? 'border-blue-400 bg-blue-50'
                                    : 'border-white/70 bg-white/70 hover:border-blue-200'
                                }`}
                              >
                                <p className="line-clamp-2 text-sm font-medium text-slate-900">
                                  {session.title}
                                </p>
                                <p className="mt-1 text-xs text-slate-500">
                                  {new Date(session.updated_at).toLocaleString()}
                                </p>
                              </button>
                            ))}
                          </div>
                        </div>
                      </section>
                    )}

                    {activeTab === 'memory' && (
                      <section>
                        {topicMemoryOverview && (
                          <div className="mb-4 rounded-2xl border border-blue-100 bg-blue-50/70 px-3 py-3">
                            <p className="text-sm font-medium text-slate-900">Topic Memory</p>
                            <p className="mt-2 text-xs text-slate-600">
                              Experience {topicMemoryOverview.counts.experiences} · Segment{' '}
                              {topicMemoryOverview.counts.segments} · QA {topicMemoryOverview.counts.qas} ·
                              Relation {topicMemoryOverview.counts.relations}
                            </p>
                            <p className="mt-2 text-xs text-slate-500">
                              Current Segment:{' '}
                              {topicMemoryOverview.current_runtime_state?.current_segment_id ?? 'none'}
                            </p>
                          </div>
                        )}
                        <p className="panel-label">Memory Files</p>
                        <div className="mt-3 space-y-2">
                          {memoryFiles.map((item) => (
                            <button
                              key={item.path}
                              onClick={() => void openFile(item.path)}
                              className={`w-full rounded-2xl border px-3 py-3 text-left transition ${
                                editorPath === item.path
                                  ? 'border-blue-400 bg-blue-50'
                                  : 'border-white/70 bg-white/70 hover:border-blue-200'
                              }`}
                            >
                              <p className="text-sm font-medium text-slate-900">{item.label}</p>
                              <p className="mt-1 text-xs text-slate-500">{item.description}</p>
                            </button>
                          ))}
                        </div>
                      </section>
                    )}

                    {activeTab === 'skills' && (
                      <section>
                        <p className="panel-label">Skills</p>
                        <div className="mt-3 space-y-2">
                          {skills.length === 0 && (
                            <p className="text-sm text-slate-500">No skills found in backend/skills.</p>
                          )}
                          {skills.map((skill) => (
                            <button
                              key={skill.name}
                              onClick={() => void openSkill(skill.name)}
                              className="w-full rounded-2xl border border-white/70 bg-white/75 px-3 py-3 text-left transition hover:border-blue-200"
                            >
                              <p className="text-sm font-medium text-slate-900">{skill.name}</p>
                              <p className="mt-2 text-[11px] uppercase tracking-[0.2em] text-blue-700">
                                {skill.location}
                              </p>
                            </button>
                          ))}
                        </div>
                      </section>
                    )}
                  </div>
                </>
              )}
            </aside>

            <main className="glass-panel flex min-h-0 min-w-0 flex-col overflow-hidden">
              <div className="border-b border-white/70 px-5 py-4">
                <div className="flex items-center justify-between gap-4">
                  <p className="panel-label">Stage</p>
                  <button
                    onClick={() => {
                      setActiveTab('chat');
                      setChatInspectorTab('raw');
                      setRightCollapsed(false);
                      void refreshChatInspector(sessionId);
                    }}
                    className="rounded-full border border-slate-200 bg-white/80 px-3 py-1.5 text-xs text-slate-600 transition hover:border-blue-300 hover:text-blue-700"
                  >
                    Raw Prompt
                  </button>
                </div>
                <div className="mt-3 flex items-center justify-between gap-4">
                  <div className="text-sm text-slate-500">
                    {sessionId ? `Session ${sessionId}` : 'New chat session'}
                  </div>
                  <div className="rounded-full bg-blue-50 px-3 py-1 text-xs text-blue-700">
                    {statusText}
                  </div>
                </div>
              </div>

              <div className="flex-1 overflow-y-auto px-5 py-5">
                {messages.length === 0 ? (
                  <div className="mx-auto mt-12 max-w-xl rounded-[28px] border border-dashed border-blue-200 bg-white/70 px-8 py-10 text-center">
                    <p className="text-sm uppercase tracking-[0.28em] text-blue-700">Local First</p>
                    <h3 className="mt-3 text-2xl font-semibold text-slate-900">
                      Start a fresh chat or review the workspace memory files.
                    </h3>
                    <p className="mt-3 text-sm leading-7 text-slate-600">
                      The right panel shows the session record, raw prompt, and today&apos;s log
                      during chat, and becomes an editable file panel in the memory workspace.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-5">
                    {messages.map((message, index) => (
                      <div
                        key={`${message.role}-${index}`}
                        className={`message-row ${
                          message.role === 'user' ? 'message-row-user' : 'message-row-assistant'
                        }`}
                      >
                        <div
                          className={`message-avatar ${
                            message.role === 'user'
                              ? 'message-avatar-user'
                              : isErrorMessage(message)
                                ? 'message-avatar-error'
                                : 'message-avatar-assistant'
                          }`}
                        >
                          {message.role === 'user' ? 'U' : isErrorMessage(message) ? '!' : 'AI'}
                        </div>
                        <article
                          className={`message-card ${
                            message.role === 'user'
                              ? 'message-user-card'
                              : isErrorMessage(message)
                                ? 'message-error-card'
                                : 'message-assistant-card'
                          }`}
                        >
                          <div className="mb-3 flex items-center justify-between gap-3">
                            <p className="message-role">
                              {message.role === 'user'
                                ? 'User'
                                : isErrorMessage(message)
                                  ? 'System Notice'
                                  : 'Assistant'}
                            </p>
                            {isErrorMessage(message) && (
                              <span className="message-error-badge">Connection Issue</span>
                            )}
                          </div>
                          {message.role === 'assistant' ? (
                            <div
                              className="markdown-content"
                              dangerouslySetInnerHTML={{
                                __html: marked.parse(message.content) as string,
                              }}
                            />
                          ) : (
                            <p className="whitespace-pre-wrap text-sm leading-7">{message.content}</p>
                          )}
                        </article>
                      </div>
                    ))}
                  </div>
                )}

                {loading && (
                  <div className="mt-4 inline-flex items-center gap-2 rounded-full bg-white/80 px-4 py-2 text-sm text-slate-500">
                    <span className="h-2 w-2 animate-pulse rounded-full bg-blue-500" />
                    Agent is thinking
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              <div className="border-t border-white/70 px-5 py-4">
                <div className="chat-composer rounded-[30px] border border-blue-100 bg-white/90 p-3 shadow-[0_20px_60px_rgba(15,23,42,0.08)]">
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        void handleSend();
                      }
                    }}
                    placeholder="Type a message. Press Enter to send, Shift + Enter for a new line."
                    className="h-28 w-full resize-none border-0 bg-transparent px-3 py-3 text-sm leading-7 text-slate-800 outline-none placeholder:text-slate-400"
                    disabled={loading}
                  />
                  <div className="flex items-center justify-between px-3 pt-2">
                    <div>
                      <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Session</p>
                      <p className="text-xs text-slate-500">{sessionId ?? 'new session'}</p>
                    </div>
                    <button
                      onClick={() => void handleSend()}
                      disabled={loading || !input.trim()}
                      className="rounded-full bg-gradient-to-r from-blue-700 to-blue-500 px-5 py-2.5 text-sm font-medium text-white transition hover:from-blue-800 hover:to-blue-600 disabled:cursor-not-allowed disabled:from-slate-300 disabled:to-slate-300"
                    >
                      Send
                    </button>
                  </div>
                </div>
              </div>
            </main>

            <section className="glass-panel flex min-h-0 flex-col overflow-hidden">
              {rightCollapsed ? (
                <div className="flex h-full flex-col items-center justify-between py-4">
                  <button
                    onClick={() => setRightCollapsed(false)}
                    className="panel-toggle-button"
                    aria-label="Expand right panel"
                  >
                    &lt;
                  </button>
                  <div className="vertical-caption">{activeTab === 'chat' ? 'Context' : 'Editor'}</div>
                  <button
                    onClick={() => setRightCollapsed(false)}
                    className="collapsed-rail-button"
                  >
                    E
                  </button>
                </div>
              ) : activeTab === 'chat' ? (
                <>
                  <div className="border-b border-white/70 px-5 py-4">
                    <div className="mb-2 flex items-center justify-between gap-3">
                      <p className="panel-label">Chat Context</p>
                      <button
                        onClick={() => setRightCollapsed(true)}
                        className="panel-toggle-button"
                        aria-label="Collapse right panel"
                      >
                        &gt;
                      </button>
                    </div>

                    <div className="mb-3 flex rounded-2xl bg-slate-100 p-1">
                      <button
                        onClick={() => setChatInspectorTab('session')}
                        className={`flex-1 rounded-xl px-3 py-2 text-sm transition ${
                          chatInspectorTab === 'session'
                            ? 'bg-white text-slate-900 shadow-sm'
                            : 'text-slate-500 hover:text-slate-800'
                        }`}
                      >
                        Session
                      </button>
                      <button
                        onClick={() => setChatInspectorTab('log')}
                        className={`flex-1 rounded-xl px-3 py-2 text-sm transition ${
                          chatInspectorTab === 'log'
                            ? 'bg-white text-slate-900 shadow-sm'
                            : 'text-slate-500 hover:text-slate-800'
                        }`}
                      >
                        Today Log
                      </button>
                      <button
                        onClick={() => setChatInspectorTab('raw')}
                        className={`flex-1 rounded-xl px-3 py-2 text-sm transition ${
                          chatInspectorTab === 'raw'
                            ? 'bg-white text-slate-900 shadow-sm'
                            : 'text-slate-500 hover:text-slate-800'
                        }`}
                      >
                        Raw Prompt
                      </button>
                    </div>

                    <div className="flex items-center justify-between gap-3">
                      <div className="min-w-0">
                        <h2 className="truncate text-lg font-semibold text-slate-900">
                          {chatInspectorTitle}
                        </h2>
                        <p className="truncate text-sm text-slate-500">{chatInspectorPath}</p>
                      </div>

                      <button
                        onClick={() => void refreshChatInspector(sessionId)}
                        className="rounded-full border border-blue-300 px-4 py-2 text-sm text-blue-700 transition hover:border-blue-500 hover:text-blue-900"
                      >
                        {chatInspectorLoading ? 'Refreshing...' : 'Refresh'}
                      </button>
                    </div>
                  </div>

                  <div className="min-h-0 flex-1">
                    <Editor
                      height="100%"
                      path={chatInspectorPath}
                      language={inferEditorLanguage(chatInspectorPath)}
                      theme="vs"
                      value={chatInspectorContent}
                      options={{
                        readOnly: true,
                        minimap: { enabled: false },
                        fontSize: 13,
                        lineNumbers: 'on',
                        wordWrap: 'on',
                        roundedSelection: false,
                        scrollBeyondLastLine: false,
                        automaticLayout: true,
                      }}
                    />
                  </div>
                </>
              ) : (
                <>
                  <div className="border-b border-white/70 px-5 py-4">
                    <div className="mb-2 flex items-center justify-between gap-3">
                      <p className="panel-label">Editor</p>
                      <button
                        onClick={() => setRightCollapsed(true)}
                        className="panel-toggle-button"
                        aria-label="Collapse right panel"
                      >
                        &gt;
                      </button>
                    </div>

                    <div className="flex items-center justify-between gap-3">
                      {activeTab === 'memory' ? (
                        <div className="min-w-0">
                          <h2 className="truncate text-lg font-semibold text-slate-900">
                            {editorTitle}
                          </h2>
                          <p className="truncate text-sm text-slate-500">
                            Edit workspace memory files from the left panel.
                          </p>
                        </div>
                      ) : (
                        <div className="min-w-0">
                          <h2 className="truncate text-lg font-semibold text-slate-900">
                            {editorTitle}
                          </h2>
                          <p className="truncate text-sm text-slate-500">{editorPath}</p>
                        </div>
                      )}

                      <button
                        onClick={() => void handleSave()}
                        disabled={saving || !editorDirty}
                        className="rounded-full border border-blue-300 px-4 py-2 text-sm text-blue-700 transition hover:border-blue-500 hover:text-blue-900 disabled:cursor-not-allowed disabled:border-slate-200 disabled:text-slate-300"
                      >
                        {saving ? 'Saving...' : 'Save'}
                      </button>
                    </div>
                  </div>

                  <div className="min-h-0 flex-1">
                    <Editor
                      height="100%"
                      path={editorPath}
                      language={inferEditorLanguage(editorPath)}
                      theme="vs"
                      value={editorContent}
                      onChange={(value) => {
                        setEditorContent(value ?? '');
                        setEditorDirty(true);
                      }}
                      options={{
                        minimap: { enabled: false },
                        fontSize: 13,
                        lineNumbers: 'on',
                        wordWrap: 'on',
                        roundedSelection: false,
                        scrollBeyondLastLine: false,
                        automaticLayout: true,
                      }}
                    />
                  </div>
                </>
              )}
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}
