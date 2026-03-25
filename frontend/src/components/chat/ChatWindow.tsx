'use client';

import Editor from '@monaco-editor/react';
import { marked } from 'marked';
import { useEffect, useMemo, useRef, useState } from 'react';

import {
  createSession,
  getFile,
  getRawMessages,
  getSessions,
  getSkillContent,
  getSkills,
  previewRawMessages,
  saveFile,
  sendMessage,
  type RawMessagesResponse,
  type SessionInfo,
  type Skill,
} from '@/lib/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

type PanelTab = 'chat' | 'memory' | 'skills';
type InspectorMode = 'editor' | 'raw';

const MEMORY_FILE = 'memory/MEMORY.md';
const WORKSPACE_FILES = [
  'workspace/SKILLS_SNAPSHOT.md',
  'workspace/SOUL.md',
  'workspace/IDENTITY.md',
  'workspace/USER.md',
  'workspace/AGENTS.md',
];

export default function ChatWindow() {
  const [activeTab, setActiveTab] = useState<PanelTab>('chat');
  const [inspectorMode, setInspectorMode] = useState<InspectorMode>('editor');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [rawLoading, setRawLoading] = useState(false);
  const [skills, setSkills] = useState<Skill[]>([]);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [editorPath, setEditorPath] = useState(MEMORY_FILE);
  const [editorContent, setEditorContent] = useState('');
  const [editorDirty, setEditorDirty] = useState(false);
  const [statusText, setStatusText] = useState('Ready');
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [rawMessages, setRawMessages] = useState<RawMessagesResponse | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const editorTitle = useMemo(() => {
    const parts = editorPath.split('/');
    return parts[parts.length - 1] ?? editorPath;
  }, [editorPath]);

  const layoutColumns = `${leftCollapsed ? '76px' : '320px'} minmax(0,1fr) ${
    rightCollapsed ? '76px' : '460px'
  }`;

  const isErrorMessage = (message: Message) =>
    message.role === 'assistant' &&
    (message.content.startsWith('Error:') || message.content.includes('模型调用失败'));

  useEffect(() => {
    void bootstrap();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  async function bootstrap() {
    try {
      const [skillsData, sessionsData, memoryFile, rawPreview] = await Promise.all([
        getSkills(),
        getSessions(),
        getFile(MEMORY_FILE),
        previewRawMessages(),
      ]);

      setSkills(skillsData);
      setSessions(sessionsData);
      setEditorContent(memoryFile.content);
      setRawMessages(rawPreview);
      setStatusText('Workspace loaded');

      if (sessionsData.length > 0) {
        await openSession(sessionsData[0].session_id, true);
      }
    } catch (error) {
      console.error(error);
      setStatusText('Failed to load workspace');
    }
  }

  async function refreshRawMessages(targetSessionId?: string) {
    try {
      setRawLoading(true);
      const payload = targetSessionId
        ? await getRawMessages(targetSessionId)
        : await previewRawMessages();
      setRawMessages(payload);
    } catch (error) {
      console.error(error);
      const fallback = targetSessionId
        ? await previewRawMessages(targetSessionId)
        : await previewRawMessages();
      setRawMessages(fallback);
    } finally {
      setRawLoading(false);
    }
  }

  async function openFile(path: string) {
    try {
      const file = await getFile(path);
      setEditorPath(file.path);
      setEditorContent(file.content);
      setEditorDirty(false);
      setInspectorMode('editor');
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
      setEditorContent(skill.content);
      setEditorDirty(false);
      setInspectorMode('editor');
      setActiveTab('skills');
      setStatusText(`Opened skill ${skillName}`);
    } catch (error) {
      console.error(error);
      setStatusText(`Failed to open skill ${skillName}`);
    }
  }

  async function openSession(nextSessionId: string, silent = false) {
    try {
      const [file, payload] = await Promise.all([
        getFile(`sessions/${nextSessionId}.json`),
        getRawMessages(nextSessionId),
      ]);

      const history = JSON.parse(file.content) as Array<{
        role: string;
        content: string;
      }>;

      setSessionId(nextSessionId);
      setRawMessages(payload);
      setMessages(
        history
          .filter(
            (message): message is Message =>
              (message.role === 'user' || message.role === 'assistant') &&
              typeof message.content === 'string'
          )
          .map((message) => ({
            role: message.role,
            content: message.content,
          }))
      );

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
      setInspectorMode('raw');
      setRightCollapsed(false);

      const { session_id: nextSessionId } = await createSession();
      const [sessionsData, payload] = await Promise.all([
        getSessions(),
        getRawMessages(nextSessionId),
      ]);

      setSessionId(nextSessionId);
      setSessions(sessionsData);
      setRawMessages(payload);
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

      if (editorPath === MEMORY_FILE || editorPath.endsWith('/SKILL.md')) {
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

  async function handleSend() {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);
    setStatusText('Agent is thinking');

    try {
      const response = await sendMessage(userMessage, sessionId);
      setSessionId(response.session_id);
      setMessages((prev) => [...prev, { role: 'assistant', content: response.reply }]);

      const [sessionsData, payload] = await Promise.all([
        getSessions(),
        getRawMessages(response.session_id),
      ]);

      setSessions(sessionsData);
      setRawMessages(payload);
      setInspectorMode('raw');
      setRightCollapsed(false);
      setStatusText(`Updated session ${response.session_id}`);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content:
            error instanceof Error
              ? error.message
              : '连接失败：无法连接后端服务，请检查 backend 是否已启动。',
        },
      ]);
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
            赋范空间
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
                        <p className="panel-label">Memory Files</p>
                        <div className="mt-3 space-y-2">
                          <button
                            onClick={() => openFile(MEMORY_FILE)}
                            className="sidebar-item w-full text-left"
                          >
                            Core Memory
                          </button>
                          {WORKSPACE_FILES.map((path) => (
                            <button
                              key={path}
                              onClick={() => openFile(path)}
                              className="sidebar-item w-full text-left"
                            >
                              {path.replace('workspace/', '')}
                            </button>
                          ))}
                        </div>
                      </section>
                    )}

                    {activeTab === 'skills' && (
                      <section>
                        <p className="panel-label">Installed Skills</p>
                        <div className="mt-3 space-y-2">
                          {skills.map((skill) => (
                            <button
                              key={skill.name}
                              onClick={() => openSkill(skill.name)}
                              className="w-full rounded-2xl border border-white/70 bg-white/75 px-3 py-3 text-left transition hover:border-blue-200"
                            >
                              <p className="text-sm font-medium text-slate-900">{skill.name}</p>
                              <p className="mt-1 text-xs text-slate-500">{skill.description}</p>
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
                    onClick={async () => {
                      setInspectorMode('raw');
                      setRightCollapsed(false);
                      await refreshRawMessages(sessionId);
                    }}
                    className="rounded-full border border-slate-200 bg-white/80 px-3 py-1.5 text-xs text-slate-600 transition hover:border-blue-300 hover:text-blue-700"
                  >
                    Show Raw Messages
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
                      Start a fresh chat or inspect the current system prompt.
                    </h3>
                    <p className="mt-3 text-sm leading-7 text-slate-600">
                      当前右侧 `Raw Messages` 会在没有用户输入时也显示 system 内容，方便直接检查 prompt 拼装。
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
                    placeholder="输入消息，按 Enter 发送，Shift + Enter 换行"
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
                  <div className="vertical-caption">Inspector</div>
                  <button
                    onClick={() => setRightCollapsed(false)}
                    className="collapsed-rail-button"
                  >
                    E
                  </button>
                </div>
              ) : (
                <>
                  <div className="border-b border-white/70 px-5 py-4">
                    <div className="mb-2 flex items-center justify-between gap-3">
                      <p className="panel-label">Inspector</p>
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
                        onClick={() => setInspectorMode('editor')}
                        className={`flex-1 rounded-xl px-3 py-2 text-sm transition ${
                          inspectorMode === 'editor'
                            ? 'bg-white text-slate-900 shadow-sm'
                            : 'text-slate-500 hover:text-slate-800'
                        }`}
                      >
                        Editor
                      </button>
                      <button
                        onClick={async () => {
                          setInspectorMode('raw');
                          await refreshRawMessages(sessionId);
                        }}
                        className={`flex-1 rounded-xl px-3 py-2 text-sm transition ${
                          inspectorMode === 'raw'
                            ? 'bg-white text-slate-900 shadow-sm'
                            : 'text-slate-500 hover:text-slate-800'
                        }`}
                      >
                        Raw Messages
                      </button>
                    </div>

                    <div className="flex items-center justify-between gap-3">
                      <div className="min-w-0">
                        <h2 className="truncate text-lg font-semibold text-slate-900">
                          {inspectorMode === 'editor' ? editorTitle : 'Raw Messages'}
                        </h2>
                        <p className="truncate text-sm text-slate-500">
                          {inspectorMode === 'editor'
                            ? editorPath
                            : rawMessages?.updated_at
                              ? `${rawMessages.message_count} messages • ${new Date(
                                  rawMessages.updated_at
                                ).toLocaleString()}`
                              : 'Current prompt payload preview'}
                        </p>
                      </div>

                      {inspectorMode === 'editor' ? (
                        <button
                          onClick={() => void handleSave()}
                          disabled={saving || !editorDirty}
                          className="rounded-full border border-blue-300 px-4 py-2 text-sm text-blue-700 transition hover:border-blue-500 hover:text-blue-900 disabled:cursor-not-allowed disabled:border-slate-200 disabled:text-slate-300"
                        >
                          {saving ? 'Saving...' : 'Save'}
                        </button>
                      ) : (
                        <button
                          onClick={() => void refreshRawMessages(sessionId)}
                          className="rounded-full border border-blue-300 px-4 py-2 text-sm text-blue-700 transition hover:border-blue-500 hover:text-blue-900"
                        >
                          {rawLoading ? 'Refreshing...' : 'Refresh'}
                        </button>
                      )}
                    </div>
                  </div>

                  {inspectorMode === 'editor' ? (
                    <>
                      <div className="grid gap-2 border-b border-white/70 px-5 py-3 sm:grid-cols-2">
                        <button
                          onClick={() => openFile(MEMORY_FILE)}
                          className="sidebar-item text-left"
                        >
                          Open MEMORY.md
                        </button>
                        <button
                          onClick={() => openFile('workspace/SKILLS_SNAPSHOT.md')}
                          className="sidebar-item text-left"
                        >
                          Open Snapshot
                        </button>
                      </div>

                      <div className="min-h-0 flex-1">
                        <Editor
                          height="100%"
                          defaultLanguage="markdown"
                          language="markdown"
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
                  ) : (
                    <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
                      {rawMessages?.messages.length ? (
                        <div className="space-y-4">
                          {rawMessages.messages.map((message, index) => (
                            <section
                              key={`${message.role}-${index}`}
                              className={`raw-message-card raw-role-${message.role}`}
                            >
                              <div className="mb-3 flex items-center justify-between gap-3">
                                <p className="raw-message-role">{message.role}</p>
                                <span className="raw-message-chip">
                                  {message.role === 'system'
                                    ? 'System Prompt'
                                    : message.role === 'user'
                                      ? 'User Input'
                                      : message.role === 'assistant'
                                        ? 'Assistant History'
                                        : 'Message'}
                                </span>
                              </div>
                              <pre className="raw-message-content">{message.content}</pre>
                            </section>
                          ))}
                        </div>
                      ) : (
                        <div className="rounded-3xl border border-dashed border-slate-200 bg-white/70 px-6 py-8 text-sm text-slate-500">
                          当前还没有模型消息记录。
                        </div>
                      )}
                    </div>
                  )}
                </>
              )}
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}
