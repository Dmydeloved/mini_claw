"""
Memory System for Mini-OpenClaw
Handles workspace files, MEMORY.md, daily logs, and session management.
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """Represents a single message in conversation"""

    role: str  # 'user', 'assistant', or 'tool'
    content: str
    timestamp: str
    tool_calls: Optional[List[Dict]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class MemoryManager:
    """
    Manages three-layer memory system:
    1. MEMORY.md - Rendered snapshot of topic memory
    2. Daily logs - Chronological conversation records
    3. Session JSON - Active session history
    """

    def __init__(self, memory_dir: str, sessions_dir: str, workspace_dir: str):
        self.memory_dir = os.path.abspath(memory_dir)
        self.sessions_dir = os.path.abspath(sessions_dir)
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.logs_dir = os.path.join(self.memory_dir, "logs")
        self.raw_messages_dir = os.path.join(self.sessions_dir, "_raw_messages")

        # Ensure directories exist
        os.makedirs(self.memory_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.sessions_dir, exist_ok=True)
        os.makedirs(self.raw_messages_dir, exist_ok=True)
        os.makedirs(self.workspace_dir, exist_ok=True)

        # Initialize MEMORY.md if not exists
        self.memory_file = os.path.join(self.memory_dir, "MEMORY.md")
        if not os.path.exists(self.memory_file):
            self._initialize_memory()

        self._ensure_workspace_files()

    def _initialize_memory(self):
        """Create initial MEMORY.md file"""
        initial_content = """# Topic Memory Snapshot

此文件由系统根据三层主题记忆自动生成，请勿手工编辑。

## Runtime State

- conversation_id: none
- current_experience_id: none
- current_segment_id: none
- latest_qa_id: none
- recent_segment_ids: none
- updated_at: none

## Counts

- experiences: 0
- segments: 0
- qas: 0
- relations: 0
"""
        with open(self.memory_file, "w", encoding="utf-8") as f:
            f.write(initial_content)

    def _ensure_workspace_files(self):
        """Create default workspace prompt files required by the PRD."""
        defaults = {
            "IDENTITY.md": """# Identity

你是 Mini-OpenClaw，一个透明、可控、本地优先的 AI Agent。

- 你会优先依赖本地文件、技能说明和工具结果，而不是凭空猜测。
- 你必须明确区分“已知信息”“文件内容”“工具结果”“你的推断”。
- 你应当保持简洁、可靠、可执行。
""",
            "USER.md": """# User Profile

- Name: 未记录
- Preferences: 未记录
- Working Style: 偏向直接执行并给出清晰结果
""",
            "AGENTS.md": """# 操作指南

## 技能调用协议 (SKILL PROTOCOL)
Prompt 中只会注入技能摘要，也就是每个 skill 的名称和描述，不会直接注入 skill 的完整正文。
当你要使用某个技能时，必须严格遵守以下步骤：

1. 你的第一步行动永远是先使用 `read_file` 读取 `workspace/SKILLS_SNAPSHOT.md`，找到目标 skill 的真实文件位置。
2. 然后再使用 `read_file` 读取对应的 `SKILL.md` Markdown 文件。
3. 仔细阅读文件中的内容、步骤和示例。
4. 根据文件中的指示，结合你内置的 Core Tools (`terminal`, `python_repl`, `fetch_url`) 来执行具体任务。

禁止直接猜测技能的参数或用法，必须先读取文件。

## 记忆协议

1. 主题长期记忆采用 `Experience -> Segment -> QA` 三层结构，并由系统自动写入 `memory/topic_memory_store/`。
2. `memory/MEMORY.md` 是主题记忆快照渲染文件，不是手工维护的事实库。
3. 会话历史保存在 `sessions/*.json`。
4. 如需引用项目规则或用户画像，请优先参考工作区文件。
5. 如果你不确定某个事实是否准确，应明确说明不确定，而不是把猜测写进记忆。
""",
        }

        for filename, content in defaults.items():
            path = os.path.join(self.workspace_dir, filename)
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)

    def get_memory_content(self) -> str:
        """Read MEMORY.md content"""
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading memory: {e}")
            return ""

    def update_memory(self, new_content: str):
        """Update MEMORY.md with new content"""
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception as e:
            print(f"Error updating memory: {e}")

    def append_to_memory(self, section: str, content: str):
        """Append content to a specific section in MEMORY.md"""
        current_memory = self.get_memory_content()

        # Simple append to end
        updated_memory = current_memory + f"\n\n## {section}\n\n{content}"

        self.update_memory(updated_memory)

    def _truncate_text(self, text: str, max_chars: int = 1200) -> str:
        """Trim oversized log snippets to keep memory files readable."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...[truncated]"

    def _summarize_message_for_log(self, msg: Message) -> str:
        """Summarize verbose message content before writing it to the daily log."""
        content = msg.content or ""

        if msg.role != "tool":
            return self._truncate_text(content, max_chars=1500)

        tool_name = msg.name or (msg.tool_calls[0].get("name") if msg.tool_calls else "tool")
        tool_args = msg.tool_calls[0].get("args", {}) if msg.tool_calls else {}

        if tool_name == "read_file":
            file_path = str(tool_args.get("file_path", ""))
            return f"Read file `{file_path}`.\n{self._truncate_text(content, max_chars=1000)}"

        if tool_name == "fetch_url":
            url = str(tool_args.get("url", ""))
            return f"Fetched `{url}`.\n{self._truncate_text(content, max_chars=1000)}"

        if tool_name == "write_file":
            file_path = str(tool_args.get("file_path", ""))
            return f"Wrote file `{file_path}`.\n{self._truncate_text(content, max_chars=600)}"

        return self._truncate_text(content, max_chars=1000)

    def save_to_daily_log(self, conversation: List[Message]):
        """Save conversation to today's daily log"""
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self.logs_dir, f"{today}.md")

        # Format conversation as markdown
        log_content = f"\n\n---\n\n## Conversation at {datetime.now().strftime('%H:%M:%S')}\n\n"

        for msg in conversation:
            role_label = {
                "user": "User",
                "assistant": "Assistant",
                "tool": "Tool",
            }.get(msg.role, msg.role.title())
            if msg.role == "tool" and msg.name:
                role_label = f"Tool `{msg.name}`"
            log_content += f"**{role_label}**: {self._summarize_message_for_log(msg)}\n\n"

            if msg.tool_calls:
                log_content += "**Tool Calls**:\n"
                for tool_call in msg.tool_calls:
                    log_content += f"- {tool_call.get('name', 'unknown')}: {tool_call.get('args', {})}\n"
                log_content += "\n"

        # Append to log file
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_content)
        except Exception as e:
            print(f"Error writing to daily log: {e}")

    def load_session(self, session_id: str) -> List[Message]:
        """Load session history from JSON"""
        session_file = os.path.join(self.sessions_dir, f"{session_id}.json")

        if not os.path.exists(session_file):
            return []

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [Message(**msg) for msg in data]
        except Exception as e:
            print(f"Error loading session: {e}")
            return []

    def save_session(self, session_id: str, messages: List[Message]):
        """Save session history to JSON"""
        session_file = os.path.join(self.sessions_dir, f"{session_id}.json")

        try:
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump([asdict(msg) for msg in messages], f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving session: {e}")

    def create_new_session(self) -> str:
        """Create a new session ID"""
        base_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = base_session_id
        counter = 1

        while os.path.exists(os.path.join(self.sessions_dir, f"{session_id}.json")):
            session_id = f"{base_session_id}_{counter:02d}"
            counter += 1

        return session_id

    def save_raw_messages(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        prompt_preview: Optional[Dict[str, Any]] = None,
        prompt_layers: Optional[Dict[str, Any]] = None,
    ):
        """Persist the exact prompt payload sent to the chat model."""
        raw_messages_file = os.path.join(self.raw_messages_dir, f"{session_id}.json")
        prompt_layers = prompt_layers or {}

        payload = {
            "session_id": session_id,
            "updated_at": datetime.now().isoformat(),
            "message_count": len(messages),
            "messages": messages,
            "system": prompt_layers.get("system", {}),
            "skills": prompt_layers.get("skills", {}),
            "tools": prompt_layers.get("tools", {}),
            "topic_memory": prompt_layers.get("topic_memory", {}),
            "session": prompt_layers.get("session", {}),
            "user": prompt_layers.get("user", {}),
            "prompt_preview": prompt_preview or {},
        }

        try:
            with open(raw_messages_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving raw messages: {e}")

    def load_raw_messages(self, session_id: str) -> Dict:
        """Load the latest raw prompt payload for a session."""
        raw_messages_file = os.path.join(self.raw_messages_dir, f"{session_id}.json")

        if not os.path.exists(raw_messages_file):
            return {
                "session_id": session_id,
                "updated_at": "",
                "message_count": 0,
                "messages": [],
                "system": {},
                "skills": {},
                "tools": {},
                "topic_memory": {},
                "session": {},
                "user": {},
            }

        try:
            with open(raw_messages_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading raw messages: {e}")
            return {
                "session_id": session_id,
                "updated_at": "",
                "message_count": 0,
                "messages": [],
                "system": {},
                "skills": {},
                "tools": {},
                "topic_memory": {},
                "session": {},
                "user": {},
            }

    def list_sessions(self) -> List[Dict[str, str]]:
        """Return saved sessions sorted by modification time descending."""
        sessions: List[Dict[str, str]] = []

        for filename in os.listdir(self.sessions_dir):
            if not filename.endswith(".json"):
                continue

            session_id = filename[:-5]
            path = os.path.join(self.sessions_dir, filename)
            updated_at = datetime.fromtimestamp(os.path.getmtime(path)).isoformat()

            title = session_id
            history = self.load_session(session_id)
            for msg in history:
                if msg.role == "user" and msg.content.strip():
                    title = msg.content.strip()[:50]
                    break

            sessions.append(
                {
                    "session_id": session_id,
                    "title": title,
                    "updated_at": updated_at,
                }
            )

        sessions.sort(key=lambda item: item["updated_at"], reverse=True)
        return sessions

    def get_recent_logs(self, days: int = 7) -> str:
        """Get recent daily logs for context"""
        logs_content = []

        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            log_file = os.path.join(self.logs_dir, date.strftime("%Y-%m-%d") + ".md")
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    logs_content.append(f"# {date.strftime('%Y-%m-%d')}\n\n{f.read()}")

        return "\n\n".join(logs_content) if logs_content else "No recent logs."

    def get_recent_activity_summary(
        self,
        max_sessions: int = 3,
        max_messages_per_session: int = 4,
        max_chars: int = 280,
        exclude_session_id: Optional[str] = None,
    ) -> str:
        """Build a lightweight summary of recent user/assistant activity."""
        sessions = [
            session
            for session in self.list_sessions()
            if not exclude_session_id or session["session_id"] != exclude_session_id
        ][:max_sessions]
        if not sessions:
            return "No recent session activity."

        lines: List[str] = []

        for session in sessions:
            history = self.load_session(session["session_id"])
            relevant_messages = [msg for msg in history if msg.role in {"user", "assistant"}]
            if not relevant_messages:
                continue

            lines.append(f"Session `{session['session_id']}`:")
            for msg in relevant_messages[-max_messages_per_session:]:
                label = "User" if msg.role == "user" else "Assistant"
                snippet = " ".join((msg.content or "").split())
                lines.append(f"- {label}: {self._truncate_text(snippet, max_chars=max_chars)}")

        return "\n".join(lines) if lines else "No recent session activity."

    def get_workspace_file(self, filename: str) -> str:
        """Read a workspace prompt file."""
        path = os.path.join(self.workspace_dir, filename)
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
