"""
Agent Graph for Mini-OpenClaw
Uses LangChain's create_agent API with LangGraph runtime
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from tools import (
    create_terminal_tool,
    create_python_repl_tool,
    create_fetch_url_tool,
    create_read_file_tool,
    create_rag_search_tool,
)
from graph.skills import SkillManager
from graph.memory import MemoryManager, Message


class MiniOpenClawAgent:
    """
    Main Agent class for Mini-OpenClaw
    Integrates tools, skills, and memory systems
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model_name: str = "gpt-4",
        default_headers: Optional[Dict[str, str]] = None,
        use_responses_api: bool = False,
        root_dir: Optional[str] = None,
        skills_dir: Optional[str] = None,
        memory_dir: Optional[str] = None,
        sessions_dir: Optional[str] = None,
        knowledge_dir: Optional[str] = None,
        storage_dir: Optional[str] = None,
    ):
        # Set default directories
        if root_dir is None:
            root_dir = os.getcwd()

        self.root_dir = root_dir
        self.skills_dir = skills_dir or os.path.join(root_dir, "skills")
        self.memory_dir = memory_dir or os.path.join(root_dir, "memory")
        self.sessions_dir = sessions_dir or os.path.join(root_dir, "sessions")
        self.workspace_dir = os.path.join(root_dir, "workspace")
        self.knowledge_dir = knowledge_dir or os.path.join(root_dir, "knowledge")
        self.storage_dir = storage_dir or os.path.join(root_dir, "storage")

        # Initialize LLM
        llm_kwargs = {
            "api_key": api_key,
            "model": model_name,
            "use_responses_api": use_responses_api,
        }
        if base_url:
            llm_kwargs["base_url"] = base_url
        if default_headers:
            llm_kwargs["default_headers"] = default_headers

        self.llm = ChatOpenAI(**llm_kwargs)

        # Initialize tools
        self.tools = [
            create_terminal_tool(root_dir=self.root_dir),
            create_python_repl_tool(),
            create_fetch_url_tool(),
            create_read_file_tool(root_dir=self.root_dir),
            create_rag_search_tool(
                knowledge_dir=self.knowledge_dir, storage_dir=self.storage_dir
            ),
        ]

        # Initialize managers
        self.skill_manager = SkillManager(skills_dir=self.skills_dir, root_dir=self.root_dir)
        self.memory_manager = MemoryManager(
            memory_dir=self.memory_dir,
            sessions_dir=self.sessions_dir,
            workspace_dir=self.workspace_dir,
        )
        self.skill_manager.write_snapshot(self.workspace_dir)

        # Create agent
        self.agent = self._create_agent()

    def _truncate_component(self, content: str, max_chars: int = 20000) -> str:
        """Trim oversized prompt sections while keeping the source explicit."""
        if len(content) <= max_chars:
            return content
        return content[:max_chars] + "\n...[truncated]"

    def _read_workspace_file(self, filename: str, fallback: str = "") -> str:
        """Read a workspace file with fallback content."""
        path = os.path.join(self.workspace_dir, filename)
        if not os.path.exists(path):
            return fallback
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_system_prompt(self) -> str:
        """Build the PRD-aligned system prompt in the required order."""
        snapshot = self.skill_manager.write_snapshot(self.workspace_dir)

        sections = [
            ("SKILLS_SNAPSHOT.md", snapshot),
            (
                "SOUL.md",
                self._read_workspace_file(
                    "SOUL.md", "You are Mini-OpenClaw, a transparent local-first AI assistant."
                ),
            ),
            ("IDENTITY.md", self.memory_manager.get_workspace_file("IDENTITY.md")),
            ("USER.md", self.memory_manager.get_workspace_file("USER.md")),
            ("AGENTS.md", self.memory_manager.get_workspace_file("AGENTS.md")),
            ("MEMORY.md", self.memory_manager.get_memory_content()),
        ]

        rendered_sections = []
        for filename, content in sections:
            rendered_sections.append(f"# {filename}\n\n{self._truncate_component(content)}")

        rendered_sections.append(
            "# CORE_TOOLS\n\n"
            "- `terminal`: execute safe shell commands inside the project root.\n"
            "- `python_repl`: run Python snippets for calculation and parsing.\n"
            "- `fetch_url`: fetch URLs and return cleaned text.\n"
            "- `read_file`: read project files, especially skill instructions.\n"
            "- `search_knowledge_base`: query the local knowledge base.\n"
        )

        return "\n\n".join(rendered_sections)

    def _create_agent(self):
        """Create agent using LangChain's create_agent API"""
        return create_agent(model=self.llm, tools=self.tools)

    def _format_agent_error(self, error: Exception) -> str:
        """Convert provider and network exceptions into readable Chinese messages."""
        message = str(error).strip() or error.__class__.__name__
        lower_message = message.lower()

        if "connection error" in lower_message:
            return (
                "模型调用失败：无法连接到模型服务。"
                "请检查 `OPENAI_BASE_URL`、网络连接、代理设置以及服务端是否可访问。"
                "如果你使用的是 OpenAI 兼容渠道，请确认该渠道支持 `chat.completions` 接口。"
            )

        if "403" in lower_message and "model" in lower_message:
            return (
                "模型调用失败：当前 `MODEL_NAME` 在所配置的接口渠道中不可用。"
                "请检查 `.env` 中的 `MODEL_NAME` 和 `OPENAI_BASE_URL` 是否匹配。"
                f"\n\n原始错误：{message}"
            )

        if "401" in lower_message or "authentication" in lower_message:
            return (
                "模型调用失败：API Key 无效或未授权。"
                "请检查 `.env` 中的 `OPENAI_API_KEY`。"
                f"\n\n原始错误：{message}"
            )

        return f"模型调用失败：{message}"

    def _extract_reply_and_tool_messages(self, response) -> Tuple[str, List[Message]]:
        """Normalize LangChain responses into a user-facing reply plus tool history."""
        tool_messages: List[Message] = []

        if isinstance(response, dict):
            messages = response.get("messages", [])
            for message in messages:
                message_type = getattr(message, "type", "")
                content = getattr(message, "content", "")
                if message_type == "tool":
                    tool_messages.append(
                        Message(
                            role="tool",
                            content=str(content),
                            timestamp=datetime.now().isoformat(),
                        )
                    )

            for message in reversed(messages):
                if getattr(message, "type", "") == "ai" and getattr(message, "content", ""):
                    return str(message.content), tool_messages

            if isinstance(response.get("output"), str):
                return response["output"], tool_messages

        return "I apologize, but I couldn't generate a response.", tool_messages

    def _build_raw_messages(
        self,
        system_prompt: str,
        history: List[Message],
        message: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build the exact message list sent to the model."""
        raw_messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

        for item in history:
            if item.role in {"user", "assistant"}:
                raw_messages.append({"role": item.role, "content": item.content})

        if message:
            raw_messages.append({"role": "user", "content": message})
        return raw_messages

    def chat(self, message: str, session_id: Optional[str] = None) -> Dict:
        """
        Process a chat message

        Args:
            message: User message
            session_id: Session ID for conversation continuity

        Returns:
            Dict with reply and session_id
        """
        # Create new session if needed
        if session_id is None:
            session_id = self.memory_manager.create_new_session()

        # Load session history
        history = self.memory_manager.load_session(session_id)

        # Build system prompt
        system_prompt = self._build_system_prompt()
        raw_messages = self._build_raw_messages(system_prompt, history, message)
        self.memory_manager.save_raw_messages(session_id, raw_messages)

        # Convert history to LangChain messages
        chat_history = []
        for msg in history:
            if msg.role == "user":
                chat_history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                chat_history.append(AIMessage(content=msg.content))

        # Invoke agent
        tool_messages: List[Message] = []
        try:
            response = self.agent.invoke(
                {
                    "messages": [("system", system_prompt), *chat_history, ("human", message)]
                }
            )

            reply, tool_messages = self._extract_reply_and_tool_messages(response)

        except Exception as e:
            reply = self._format_agent_error(e)
            print(f"Agent error: {e}")

        # Save to session
        timestamp = datetime.now().isoformat()

        history.append(Message(role="user", content=message, timestamp=timestamp))
        history.extend(tool_messages)
        history.append(Message(role="assistant", content=reply, timestamp=timestamp))

        self.memory_manager.save_session(session_id, history)

        # Save to daily log
        self.memory_manager.save_to_daily_log(
            [
                Message(role="user", content=message, timestamp=timestamp),
                Message(role="assistant", content=reply, timestamp=timestamp),
            ]
        )

        return {"reply": reply, "session_id": session_id}

    def create_session(self) -> Dict[str, str]:
        """Create an empty session so the UI can inspect a fresh prompt state."""
        session_id = self.memory_manager.create_new_session()
        self.memory_manager.save_session(session_id, [])

        system_prompt = self._build_system_prompt()
        raw_messages = self._build_raw_messages(system_prompt, [])
        self.memory_manager.save_raw_messages(session_id, raw_messages)

        return {"session_id": session_id}

    def get_skills(self) -> List[Dict]:
        """Get all skills information"""
        return self.skill_manager.get_all_skills_info()

    def update_skill(self, name: str, content: str) -> bool:
        """Update a skill"""
        return self.skill_manager.update_skill(name, content)

    def get_memory(self) -> str:
        """Get memory content"""
        return self.memory_manager.get_memory_content()

    def update_memory(self, content: str):
        """Update memory content"""
        self.memory_manager.update_memory(content)

    def list_sessions(self) -> List[Dict[str, str]]:
        """List saved sessions."""
        return self.memory_manager.list_sessions()

    def get_raw_messages(self, session_id: str) -> Dict:
        """Get the latest raw prompt payload for a session."""
        payload = self.memory_manager.load_raw_messages(session_id)
        if payload.get("messages"):
            return payload

        history = self.memory_manager.load_session(session_id)
        system_prompt = self._build_system_prompt()
        preview_messages = self._build_raw_messages(system_prompt, history)
        return {
            "session_id": session_id,
            "updated_at": datetime.now().isoformat(),
            "message_count": len(preview_messages),
            "messages": preview_messages,
        }

    def preview_raw_messages(self, session_id: Optional[str] = None) -> Dict:
        """Preview the current prompt payload even before the next user message is sent."""
        history = self.memory_manager.load_session(session_id) if session_id else []
        system_prompt = self._build_system_prompt()
        preview_messages = self._build_raw_messages(system_prompt, history)

        return {
            "session_id": session_id or "",
            "updated_at": datetime.now().isoformat(),
            "message_count": len(preview_messages),
            "messages": preview_messages,
        }
