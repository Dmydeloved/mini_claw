"""
Agent Graph for Mini-OpenClaw
Uses an explicit autonomous tool loop while preserving the existing
memory, skills, session, and snapshot architecture.
"""

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from tools import (
    create_fetch_url_tool,
    create_python_repl_tool,
    create_rag_search_tool,
    create_read_file_tool,
    create_terminal_tool,
    create_write_file_tool,
)
from graph.context_selector import TopicContextSelector, TopicSessionContext
from graph.experience_miner import ExperienceMiner
from graph.memory import MemoryManager, Message
from graph.skills import SkillManager
from graph.topic_memory_manager import TopicMemoryManager, TopicPromptContext


@dataclass
class PromptLayers:
    """Explicit prompt-engineering layers used to build one model call."""

    system: Dict[str, str]
    skills: Dict[str, Any]
    tools: Dict[str, Any]
    topic_memory: Dict[str, Any]
    session: Dict[str, Any]
    user: Dict[str, Any]


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
        workspace_dir: Optional[str] = None,
        skills_dir: Optional[str] = None,
        memory_dir: Optional[str] = None,
        sessions_dir: Optional[str] = None,
        knowledge_dir: Optional[str] = None,
        storage_dir: Optional[str] = None,
        max_tool_iterations: int = 12,
    ):
        # Set default directories
        if root_dir is None:
            root_dir = os.getcwd()

        self.root_dir = os.path.abspath(root_dir)
        self.workspace_dir = os.path.abspath(
            workspace_dir or os.path.join(self.root_dir, "workspace")
        )
        self.skills_dir = os.path.abspath(
            skills_dir or os.path.join(self.root_dir, "skills")
        )
        self.memory_dir = os.path.abspath(
            memory_dir or os.path.join(self.workspace_dir, "memory")
        )
        self.sessions_dir = os.path.abspath(
            sessions_dir or os.path.join(self.workspace_dir, "sessions")
        )
        self.knowledge_dir = os.path.abspath(
            knowledge_dir or os.path.join(self.root_dir, "knowledge")
        )
        self.storage_dir = os.path.abspath(
            storage_dir or os.path.join(self.root_dir, "storage")
        )
        self.max_tool_iterations = max(1, max_tool_iterations)

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
            create_write_file_tool(root_dir=self.root_dir),
            create_rag_search_tool(
                knowledge_dir=self.knowledge_dir, storage_dir=self.storage_dir
            ),
        ]
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        # Initialize managers
        self.skill_manager = SkillManager(skills_dir=self.skills_dir, root_dir=self.root_dir)
        self.memory_manager = MemoryManager(
            memory_dir=self.memory_dir,
            sessions_dir=self.sessions_dir,
            workspace_dir=self.workspace_dir,
        )
        self.topic_memory_manager = TopicMemoryManager(
            memory_dir=self.memory_dir,
            sessions_dir=self.sessions_dir,
        )
        self.experience_miner = ExperienceMiner(memory_dir=self.memory_dir)
        self.context_selector = TopicContextSelector()
        self.skill_manager.write_snapshot(self.workspace_dir)

        # Create model interface with tool binding
        self.agent = self._create_agent()

    def _stringify_content(self, content: Any) -> str:
        """Normalize mixed message content into a stable string."""
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue

                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type in {"text", "output_text"} and item.get("text"):
                        parts.append(str(item["text"]))
                    elif item_type == "tool_result" and item.get("content"):
                        parts.append(self._stringify_content(item["content"]))
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                    continue

                parts.append(str(item))

            return "\n".join(part for part in parts if part)

        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False, indent=2)

        return str(content)

    def _truncate_component(self, content: Any, max_chars: int = 20000) -> str:
        """Trim oversized prompt sections while keeping the source explicit."""
        normalized = self._stringify_content(content)
        if len(normalized) <= max_chars:
            return normalized
        return normalized[:max_chars] + "\n...[truncated]"

    def _read_workspace_file(self, filename: str, fallback: str = "") -> str:
        """Read a workspace file with fallback content."""
        path = os.path.join(self.workspace_dir, filename)
        if not os.path.exists(path):
            return fallback
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _strip_markdown_title(self, content: str) -> str:
        """Remove the first markdown title to avoid repeated headings in composite sections."""
        normalized = (content or "").strip()
        return re.sub(r"^# .*\n+", "", normalized, count=1).strip()

    def _build_tool_definitions(self) -> str:
        """Build a compact prompt-visible tool policy without duplicating bind schema."""
        lines = [
            "Tools are already bound natively to the model.",
            "Use actual tool calls instead of describing pseudo-calls in plain text.",
            "Do not invent tools or tool arguments.",
            "Prefer tool results over unsupported assumptions.",
            "Use `read_file` to inspect skill files before relying on a skill.",
        ]

        return "\n".join(lines).strip()

    def _build_runtime_context(
        self,
        session_id: Optional[str],
        user_message: Optional[str],
        loop_messages: Optional[List[Message]] = None,
        iteration: int = 0,
        topic_prompt_context: Optional[TopicPromptContext] = None,
        session_context: Optional[TopicSessionContext] = None,
    ) -> str:
        """Build volatile runtime context included before each model call."""
        loop_messages = loop_messages or []
        lines = [
            f"- session_id: {session_id or '(new session)'}",
            f"- loop_iteration: {iteration + 1}/{self.max_tool_iterations}",
        ]

        matched_skill = self.skill_manager.match_trigger(user_message or "")
        if matched_skill:
            lines.append(
                "- matched_skill_hint: "
                f"{matched_skill.name} ({matched_skill.relative_path})"
            )

        tool_result_count = sum(1 for msg in loop_messages if msg.role == "tool")
        if tool_result_count:
            lines.append(f"- tool_results_collected_in_current_turn: {tool_result_count}")

        if topic_prompt_context:
            lines.append(
                "- prompt_new_session_detected: "
                f"{'true' if topic_prompt_context.is_new_session else 'false'}"
            )
            if topic_prompt_context.transition_notes:
                lines.append(
                    "- prompt_flow_actions: "
                    f"{', '.join(topic_prompt_context.transition_notes)}"
                )

        if session_context:
            lines.append(f"- session_context_strategy: {session_context.strategy}")
            lines.append(
                "- session_original_messages: "
                f"{session_context.original_message_count}"
            )
            lines.append(
                "- session_selected_messages: "
                f"{session_context.selected_message_count}"
            )
            lines.append(
                "- session_prompt_messages: "
                f"{session_context.prompt_message_count}"
            )
            lines.append(
                "- session_selection_applied: "
                f"{'true' if session_context.selection_applied else 'false'}"
            )
            lines.append(
                "- session_compression_applied: "
                f"{'true' if session_context.compression_applied else 'false'}"
            )

        embedded_skill_draft = self._inspect_embedded_skill_draft(user_message)
        if embedded_skill_draft:
            lines.append("- embedded_skill_draft_detected: true")
            lines.append(f"- embedded_skill_name: {embedded_skill_draft['name']}")
            lines.append(
                f"- embedded_skill_file_path_hint: {embedded_skill_draft['file_path']}"
            )
            if embedded_skill_draft.get("validation_error"):
                lines.append(
                    "- embedded_skill_draft_status: "
                    f"incomplete ({embedded_skill_draft['validation_error']})"
                )
            else:
                lines.append("- embedded_skill_draft_status: valid frontmatter detected")

            if self._has_skill_write_in_turn(loop_messages):
                lines.append("- embedded_skill_persistence_status: already written in this turn")

        return "\n".join(lines)

    def _build_memory_context(self, session_id: Optional[str] = None) -> str:
        """Build the prompt-visible memory context injected into the system prompt."""
        memory_content = self.memory_manager.get_memory_content()
        recent_activity = self.memory_manager.get_recent_activity_summary(
            exclude_session_id=session_id
        )
        return (
            "## Persistent User Memory\n\n"
            f"{memory_content}\n\n"
            "## Other Recent Sessions\n\n"
            f"{recent_activity}"
        )

    def _build_role_context(self) -> str:
        """Build a concise role section from persona-related workspace files."""
        soul = self._strip_markdown_title(
            self._read_workspace_file(
                "SOUL.md",
                "You are Mini-OpenClaw, a transparent local-first AI assistant.",
            )
        )
        identity = self._strip_markdown_title(
            self.memory_manager.get_workspace_file("IDENTITY.md")
        )
        user_profile = self._strip_markdown_title(
            self.memory_manager.get_workspace_file("USER.md")
        )

        parts = ["## Soul", soul]
        if identity:
            parts.extend(["", "## Identity", identity])
        if user_profile:
            parts.extend(["", "## User Profile", user_profile])
        return "\n".join(part for part in parts if part is not None).strip()

    def _build_execution_rules(self) -> str:
        """Build consolidated execution rules to replace overlapping prompt sections."""
        agents_rules = self._strip_markdown_title(
            self.memory_manager.get_workspace_file("AGENTS.md")
        )
        loop_rules = (
            "## Autonomous Loop\n"
            "1. Before every answer, rely on the prompt you were given in this call.\n"
            "2. If you need external information or file contents, emit tool calls instead of stopping early.\n"
            "3. After tool results are returned, continue the task automatically from the updated context.\n"
            "4. Only provide the final natural-language answer when no further tool call is necessary.\n"
            "5. Base conclusions on files and tool results, and distinguish facts from inference.\n"
            "6. When working with skills, first read `workspace/SKILLS_SNAPSHOT.md`, then read the target `SKILL.md`.\n"
            "7. When asked to create or update a skill, write a valid `skills/<skill_name>/SKILL.md`."
        )
        parts = ["## Project Rules", agents_rules, "", loop_rules]
        return "\n".join(part for part in parts if part is not None).strip()

    def _build_prompt_layers(
        self,
        session_id: Optional[str],
        topic_memory_context: Optional[str],
        runtime_context: Optional[str],
        session_context: Optional[TopicSessionContext],
        user_message: Optional[str],
    ) -> PromptLayers:
        """Build explicit prompt layers so the engineering structure is easy to inspect and evolve."""
        self.skill_manager.reload_skills(verbose=False)
        self.skill_manager.write_snapshot(self.workspace_dir)

        # The first four layers are rendered into the actual system message.
        system_layer = {
            "role_context": self._build_role_context(),
            "execution_rules": self._build_execution_rules(),
            "memory_context": self._build_memory_context(session_id=session_id),
        }
        skills_layer = {
            "summary": self.skill_manager.build_skills_prompt(),
        }
        tools_layer = {
            "policy": self._build_tool_definitions(),
            "tool_bind_active": True,
        }
        topic_memory_layer = {
            "content": topic_memory_context or "No topic memory context available.",
        }

        # Session stays outside the system block because it is the most dynamic layer.
        session_layer = {
            "runtime": runtime_context or "",
            "strategy": session_context.strategy if session_context else "none",
            "selection_applied": session_context.selection_applied if session_context else False,
            "compression_applied": session_context.compression_applied if session_context else False,
            "selected_history": [
                self._serialize_session_message(msg)
                for msg in (session_context.prompt_history if session_context else [])
            ],
        }
        user_layer = {
            "current_message": user_message or "",
        }

        return PromptLayers(
            system=system_layer,
            skills=skills_layer,
            tools=tools_layer,
            topic_memory=topic_memory_layer,
            session=session_layer,
            user=user_layer,
        )

    def _serialize_session_message(self, msg: Message) -> Dict[str, Any]:
        """Convert one persisted message into a compact prompt-layer preview payload."""
        payload: Dict[str, Any] = {
            "role": msg.role,
            "content": self._truncate_component(msg.content, max_chars=600),
        }
        if msg.name:
            payload["name"] = msg.name
        if msg.tool_calls:
            payload["tool_calls"] = msg.tool_calls
        return payload

    def _build_system_prompt(
        self,
        prompt_layers: PromptLayers,
    ) -> str:
        """Render the system-facing layers into the single system message sent to the model."""
        sections = [
            (
                "SYSTEM.md",
                "\n\n".join(
                    [
                        "## Role Context",
                        prompt_layers.system.get("role_context", ""),
                        "",
                        "## Execution Rules",
                        prompt_layers.system.get("execution_rules", ""),
                        "",
                        "## Memory Context",
                        prompt_layers.system.get("memory_context", ""),
                    ]
                ).strip(),
            ),
            ("SKILLS.md", str(prompt_layers.skills.get("summary", ""))),
            ("TOOLS.md", str(prompt_layers.tools.get("policy", ""))),
        ]

        topic_context = str(prompt_layers.topic_memory.get("content", "")).strip()
        if topic_context:
            sections.append(("TOPIC_MEMORY.md", topic_context))

        runtime_context = str(prompt_layers.session.get("runtime", "")).strip()
        if runtime_context:
            sections.append(("RUNTIME.md", runtime_context))

        rendered_sections = []
        for filename, content in sections:
            rendered_sections.append(f"# {filename}\n\n{self._truncate_component(content)}")

        return "\n\n".join(rendered_sections)

    def _create_agent(self):
        """Create the tool-enabled chat model used by the autonomous loop."""
        return self.llm.bind_tools(self.tools)

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
        """Normalize model responses into a user-facing reply plus tool history."""
        tool_messages: List[Message] = []

        if isinstance(response, AIMessage):
            return self._stringify_content(response.content), tool_messages

        if isinstance(response, dict):
            messages = response.get("messages", [])
            for message in messages:
                message_type = getattr(message, "type", "")
                content = getattr(message, "content", "")
                if message_type == "tool":
                    tool_name = getattr(message, "name", None)
                    tool_call_id = getattr(message, "tool_call_id", None)
                    tool_calls = None
                    if tool_name or tool_call_id:
                        tool_calls = [
                            {
                                "id": tool_call_id,
                                "name": tool_name,
                            }
                        ]
                    tool_messages.append(
                        Message(
                            role="tool",
                            content=self._stringify_content(content),
                            timestamp=datetime.now().isoformat(),
                            tool_calls=tool_calls,
                            name=tool_name,
                            tool_call_id=tool_call_id,
                        )
                    )

            for message in reversed(messages):
                if getattr(message, "type", "") == "ai":
                    return self._stringify_content(getattr(message, "content", "")), tool_messages

            if isinstance(response.get("output"), str):
                return response["output"], tool_messages

        if hasattr(response, "content"):
            return self._stringify_content(getattr(response, "content")), tool_messages

        return "I apologize, but I couldn't generate a response.", tool_messages

    def _normalize_tool_calls(self, tool_calls: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Normalize provider tool-call payloads into a stable internal format."""
        normalized: List[Dict[str, Any]] = []

        for index, tool_call in enumerate(tool_calls or []):
            if not isinstance(tool_call, dict):
                continue

            args = tool_call.get("args", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass

            normalized.append(
                {
                    "id": tool_call.get("id")
                    or f"tool_call_{datetime.now().timestamp()}_{index}",
                    "name": tool_call.get("name", "unknown_tool"),
                    "args": args,
                    "type": tool_call.get("type", "tool_call"),
                }
            )

        return normalized

    def _extract_embedded_skill_markdown(self, user_message: Optional[str]) -> Optional[str]:
        """Extract an embedded SKILL.md draft from a user message when present."""
        if not user_message:
            return None

        stripped = user_message.strip()
        if stripped.startswith("---"):
            return stripped

        match = re.search(r"(---\s*\n.*?\n---\s*\n.*)", user_message, re.DOTALL)
        if match:
            return match.group(1).strip()

        return None

    def _inspect_embedded_skill_draft(self, user_message: Optional[str]) -> Optional[Dict[str, Any]]:
        """Inspect an embedded SKILL.md draft without inferring the user's intent."""
        embedded_markdown = self._extract_embedded_skill_markdown(user_message)
        if not embedded_markdown:
            return None

        inspection = self.skill_manager.inspect_skill_content(embedded_markdown)
        if not inspection:
            return None

        metadata = inspection.get("metadata", {})
        skill_name = str(metadata.get("name", "")).strip()
        if not skill_name:
            return None

        return {
            "name": skill_name,
            "file_path": self.skill_manager.build_skill_relative_path(skill_name),
            "validation_error": inspection.get("validation_error"),
            "is_valid": inspection.get("is_valid", False),
            "content": embedded_markdown,
        }

    def _has_skill_write_in_turn(self, loop_messages: List[Message]) -> bool:
        """Check whether the current turn already wrote a skill file."""
        for msg in loop_messages:
            if msg.role != "tool" or msg.name != "write_file":
                continue
            for tool_call in msg.tool_calls or []:
                file_path = str(tool_call.get("args", {}).get("file_path", ""))
                if self._is_skill_file_path(file_path):
                    return True
        return False

    def _is_skill_file_path(self, file_path: str) -> bool:
        """Check whether a file path points to a skill definition file."""
        normalized = file_path.replace("\\", "/").lstrip("./")
        return normalized.startswith("skills/") and normalized.endswith("/SKILL.md")

    def _postprocess_write_file_result(self, tool_args: Any, content: str) -> str:
        """Reload skills after write_file and explain whether the written skill was loaded."""
        if not isinstance(tool_args, dict):
            return content

        file_path = str(tool_args.get("file_path", ""))
        if not self._is_skill_file_path(file_path):
            return content

        if content.startswith("Error:"):
            return content

        self.skill_manager.reload_skills(verbose=False)
        self.skill_manager.write_snapshot(self.workspace_dir)

        inspection = self.skill_manager.inspect_skill_content(str(tool_args.get("text", "")))
        if not inspection:
            return (
                content
                + "\nWarning: The file was saved, but the content could not be parsed as a skill."
            )

        skill_name = str(inspection.get("metadata", {}).get("name", "")).strip()
        if not skill_name:
            return content + "\nWarning: The file was saved, but no skill name could be identified."

        loaded_skill = self.skill_manager.get_skill_by_name(skill_name)
        if loaded_skill:
            return (
                content
                + f"\nSkill `{skill_name}` is now loaded and available at `{loaded_skill.relative_path}`."
            )

        validation_error = inspection.get("validation_error") or (
            "the saved file did not satisfy the skill loader requirements"
        )
        return (
            content
            + "\nWarning: The file was saved, but the skill was not loaded because "
            + validation_error
            + "."
        )

    def _extract_tool_calls(self, response) -> List[Dict[str, Any]]:
        """Extract normalized tool calls from a model response."""
        if isinstance(response, AIMessage):
            return self._normalize_tool_calls(getattr(response, "tool_calls", None))

        if isinstance(response, dict):
            messages = response.get("messages", [])
            for message in reversed(messages):
                if getattr(message, "type", "") == "ai":
                    return self._normalize_tool_calls(getattr(message, "tool_calls", None))

        return []

    def _format_tool_calls_summary(self, tool_calls: Optional[List[Dict[str, Any]]]) -> str:
        """Render tool calls as readable text for raw prompt snapshots."""
        normalized = self._normalize_tool_calls(tool_calls)
        if not normalized:
            return ""

        lines = ["[Tool Calls]"]
        for tool_call in normalized:
            args_text = self._stringify_content(tool_call.get("args", {}))
            lines.append(
                f"- {tool_call['name']} (id={tool_call['id']}): {args_text}"
            )

        return "\n".join(lines)

    def _format_tool_result_for_prompt(self, tool_name: Optional[str], content: str) -> str:
        """Format a tool result when serialized back into prompt history."""
        if tool_name:
            return f"[Tool `{tool_name}` Result]\n{content}"
        return f"[Tool Result]\n{content}"

    def _append_tool_summary_if_missing(self, content: str, summary: str) -> str:
        """Avoid duplicating tool-call summaries when content already contains them."""
        normalized_content = (content or "").strip()
        normalized_summary = (summary or "").strip()
        if not normalized_summary:
            return normalized_content
        if not normalized_content:
            return normalized_summary
        if normalized_summary in normalized_content:
            return normalized_content
        if normalized_content.startswith("[Tool Calls]"):
            return normalized_content
        return f"{normalized_content}\n\n{normalized_summary}".strip()

    def _summarize_tool_content_for_history(self, msg: Message) -> str:
        """Compress verbose tool outputs before replaying them into model history."""
        tool_name = msg.name or (msg.tool_calls[0].get("name") if msg.tool_calls else None) or "tool"
        tool_args = msg.tool_calls[0].get("args", {}) if msg.tool_calls else {}
        raw_content = msg.content or ""

        if tool_name == "read_file":
            file_path = str(tool_args.get("file_path", ""))
            if self._is_skill_file_path(file_path):
                inspection = self.skill_manager.inspect_skill_content(raw_content)
                if inspection:
                    metadata = inspection.get("metadata", {})
                    validation_error = inspection.get("validation_error")
                    status = "valid" if inspection.get("is_valid") else f"invalid ({validation_error})"
                    return (
                        f"Read skill file `{file_path}`.\n"
                        f"- name: {metadata.get('name', '')}\n"
                        f"- description: {metadata.get('description', '')}\n"
                        f"- status: {status}"
                    )
            return f"Read file `{file_path}`.\n{self._truncate_component(raw_content, max_chars=1200)}"

        if tool_name == "fetch_url":
            url = str(tool_args.get("url", ""))
            return f"Fetched `{url}`.\n{self._truncate_component(raw_content, max_chars=1600)}"

        if tool_name == "write_file":
            file_path = str(tool_args.get("file_path", ""))
            return f"Wrote `{file_path}`.\n{self._truncate_component(raw_content, max_chars=600)}"

        if tool_name == "terminal":
            return self._truncate_component(raw_content, max_chars=1200)

        return self._truncate_component(raw_content, max_chars=1000)

    def _history_to_langchain_messages(self, history: List[Message]) -> List[BaseMessage]:
        """Convert persisted message history into LangChain messages."""
        chat_history: List[BaseMessage] = []
        known_tool_call_ids = set()

        for index, msg in enumerate(history):
            if msg.role == "user":
                chat_history.append(
                    HumanMessage(content=self._truncate_component(msg.content, max_chars=16000))
                )
                continue

            if msg.role == "assistant":
                normalized_tool_calls = self._normalize_tool_calls(msg.tool_calls)
                if normalized_tool_calls:
                    concrete_tool_calls: List[Dict[str, Any]] = []
                    for tool_index, tool_call in enumerate(normalized_tool_calls):
                        tool_call_id = tool_call.get("id") or f"stored_tool_call_{index}_{tool_index}"
                        concrete_tool_calls.append(
                            {
                                "id": tool_call_id,
                                "name": tool_call.get("name", "unknown_tool"),
                                "args": tool_call.get("args", {}),
                                "type": tool_call.get("type", "tool_call"),
                            }
                        )
                        known_tool_call_ids.add(tool_call_id)

                    assistant_content = self._truncate_component(msg.content, max_chars=16000)
                    if not assistant_content:
                        assistant_content = self._format_tool_calls_summary(concrete_tool_calls)

                    chat_history.append(
                        AIMessage(content=assistant_content, tool_calls=concrete_tool_calls)
                    )
                else:
                    chat_history.append(
                        AIMessage(content=self._truncate_component(msg.content, max_chars=16000))
                    )
                continue

            if msg.role == "tool":
                tool_call_id = msg.tool_call_id
                if not tool_call_id and msg.tool_calls:
                    tool_call_id = msg.tool_calls[0].get("id")

                tool_name = msg.name
                if not tool_name and msg.tool_calls:
                    tool_name = msg.tool_calls[0].get("name")

                content = self._summarize_tool_content_for_history(msg)
                if tool_call_id and tool_call_id in known_tool_call_ids:
                    chat_history.append(
                        ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)
                    )
                else:
                    chat_history.append(
                        AIMessage(content=self._format_tool_result_for_prompt(tool_name, content))
                    )
                continue

            chat_history.append(AIMessage(content=self._truncate_component(msg.content, max_chars=16000)))

        return chat_history

    def _serialize_prompt_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Serialize the exact message payload sent to the model."""
        raw_messages: List[Dict[str, Any]] = []

        for message in messages:
            role = getattr(message, "type", "assistant")
            if role == "ai":
                role = "assistant"
            elif role == "human":
                role = "user"

            content = self._stringify_content(getattr(message, "content", ""))
            payload: Dict[str, Any] = {"role": role, "content": content}

            if role == "assistant":
                tool_calls = self._extract_tool_calls(message)
                if tool_calls:
                    payload["tool_calls"] = tool_calls
                    summary = self._format_tool_calls_summary(tool_calls)
                    payload["content"] = self._append_tool_summary_if_missing(content, summary)

            if role == "tool":
                tool_name = getattr(message, "name", None)
                tool_call_id = getattr(message, "tool_call_id", None)
                if tool_name:
                    payload["name"] = tool_name
                    payload["content"] = self._format_tool_result_for_prompt(tool_name, content)
                if tool_call_id:
                    payload["tool_call_id"] = tool_call_id

            raw_messages.append(payload)

        return raw_messages

    def _parse_system_sections(self, system_prompt: str) -> List[Dict[str, str]]:
        """Parse the rendered system prompt into named sections for UI inspection."""
        sections: List[Dict[str, str]] = []
        matches = list(
            re.finditer(r"^# ([A-Za-z0-9_./-]+\.md)\n\n", system_prompt, re.MULTILINE)
        )

        for index, match in enumerate(matches):
            name = match.group(1).strip()
            content_start = match.end()
            content_end = matches[index + 1].start() if index + 1 < len(matches) else len(system_prompt)
            content = system_prompt[content_start:content_end].strip()
            sections.append({"name": name, "content": content})

        return sections

    def _build_prompt_preview(
        self,
        raw_messages: List[Dict[str, Any]],
        prompt_layers: Optional[PromptLayers] = None,
    ) -> Dict[str, Any]:
        """Build a structured preview of the prompt for easier inspection."""
        system_message = next(
            (message for message in raw_messages if message.get("role") == "system"),
            None,
        )
        system_sections = self._parse_system_sections(system_message.get("content", "")) if system_message else []
        section_map = {section["name"]: section["content"] for section in system_sections}
        conversation_messages = [message for message in raw_messages if message.get("role") != "system"]

        current_user_message = ""
        for message in reversed(conversation_messages):
            if message.get("role") == "user" and str(message.get("content", "")).strip():
                current_user_message = str(message.get("content", ""))
                break

        current_turn_start = None
        for index in range(len(conversation_messages) - 1, -1, -1):
            message = conversation_messages[index]
            if message.get("role") == "user" and str(message.get("content", "")).strip():
                current_turn_start = index
                break

        current_turn_messages = (
            conversation_messages[current_turn_start:]
            if current_turn_start is not None
            else []
        )
        session_history_messages = (
            conversation_messages[:current_turn_start]
            if current_turn_start is not None
            else conversation_messages
        )

        structured_prompt = (
            {
                "system": prompt_layers.system,
                "skills": prompt_layers.skills,
                "tools": prompt_layers.tools,
                "topic_memory": prompt_layers.topic_memory,
                "session": {
                    **prompt_layers.session,
                    "current_turn": current_turn_messages,
                },
                "user": prompt_layers.user,
            }
            if prompt_layers
            else {
                "system": {
                    "content": section_map.get("SYSTEM.md", ""),
                },
                "skills": {
                    "summary": section_map.get("SKILLS.md", ""),
                },
                "tools": {
                    "policy": section_map.get("TOOLS.md", ""),
                    "tool_bind_active": True,
                },
                "topic_memory": {
                    "content": section_map.get("TOPIC_MEMORY.md", ""),
                },
                "session": {
                    "runtime": section_map.get("RUNTIME.md", ""),
                    "selected_history": session_history_messages,
                    "current_turn": current_turn_messages,
                },
                "user": {
                    "current_message": current_user_message,
                },
            }
        )

        return {
            "system_sections": system_sections,
            "persona_sections": [
                section
                for section in system_sections
                if section["name"] in {"SYSTEM.md"}
            ],
            "skills_summary": section_map.get("SKILLS.md", ""),
            "tools_and_functions": section_map.get("TOOLS.md", ""),
            "retrieved_memory": section_map.get("SYSTEM.md", ""),
            "topic_memory": section_map.get("TOPIC_MEMORY.md", ""),
            "current_context": section_map.get("RUNTIME.md", ""),
            "autonomous_agent_loop": section_map.get("SYSTEM.md", ""),
            "current_user_message": current_user_message,
            "conversation_messages": conversation_messages,
            "structured_prompt": structured_prompt,
        }

    def _build_raw_prompt_payload(
        self,
        session_id: str,
        raw_messages: List[Dict[str, Any]],
        prompt_layers: Optional[PromptLayers] = None,
        updated_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the stored/raw API payload with explicit top-level prompt layers."""
        prompt_preview = self._build_prompt_preview(raw_messages, prompt_layers=prompt_layers)
        structured_prompt = prompt_preview.get("structured_prompt", {})

        return {
            "session_id": session_id,
            "updated_at": updated_at or datetime.now().isoformat(),
            "message_count": len(raw_messages),
            "messages": raw_messages,
            "system": structured_prompt.get("system", {}),
            "skills": structured_prompt.get("skills", {}),
            "tools": structured_prompt.get("tools", {}),
            "topic_memory": structured_prompt.get("topic_memory", {}),
            "session": structured_prompt.get("session", {}),
            "user": structured_prompt.get("user", {}),
            "prompt_preview": prompt_preview,
        }

    def _hydrate_raw_prompt_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Backfill the new top-level prompt layers for old raw snapshots."""
        hydrated = dict(payload or {})
        messages = hydrated.get("messages") or []
        prompt_preview = hydrated.get("prompt_preview") or {}

        if not prompt_preview and messages:
            prompt_preview = self._build_prompt_preview(messages)
            hydrated["prompt_preview"] = prompt_preview

        structured_prompt = prompt_preview.get("structured_prompt", {}) if isinstance(prompt_preview, dict) else {}
        for key in ("system", "skills", "tools", "topic_memory", "session", "user"):
            hydrated[key] = hydrated.get(key) or structured_prompt.get(key, {})

        hydrated.setdefault("session_id", "")
        hydrated.setdefault("updated_at", "")
        hydrated["message_count"] = len(messages)
        hydrated["messages"] = messages
        return hydrated

    def _build_prompt_messages(
        self,
        system_prompt: str,
        history: List[Message],
        message: Optional[str] = None,
    ) -> List[BaseMessage]:
        """Build the exact LangChain message list sent to the model."""
        combined_history = list(history)
        if message is not None:
            combined_history.append(
                Message(
                    role="user",
                    content=message,
                    timestamp=datetime.now().isoformat(),
                )
            )

        return [SystemMessage(content=system_prompt), *self._history_to_langchain_messages(combined_history)]

    def _build_raw_messages(
        self,
        system_prompt: str,
        history: List[Message],
        message: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build the exact serialized payload sent to the model."""
        prompt_messages = self._build_prompt_messages(system_prompt, history, message)
        return self._serialize_prompt_messages(prompt_messages)

    def _invoke_model(self, prompt_messages: List[BaseMessage]) -> AIMessage:
        """Invoke the tool-enabled model and normalize the response."""
        response = self.agent.invoke(prompt_messages)

        if isinstance(response, AIMessage):
            return response

        if isinstance(response, dict):
            messages = response.get("messages", [])
            for message in reversed(messages):
                if isinstance(message, AIMessage):
                    return message

            output = response.get("output")
            if isinstance(output, AIMessage):
                return output
            if isinstance(output, str):
                return AIMessage(content=output)

        if hasattr(response, "content"):
            return AIMessage(
                content=self._stringify_content(getattr(response, "content", "")),
                tool_calls=self._normalize_tool_calls(getattr(response, "tool_calls", None)),
            )

        return AIMessage(content=self._stringify_content(response))

    def _prepare_tool_input(self, tool, args: Any) -> Any:
        """Coerce model tool arguments into the shape expected by the tool."""
        schema = getattr(tool, "args", {}) or {}
        schema_keys = list(schema.keys())

        if not isinstance(args, dict):
            if len(schema_keys) == 1:
                return {schema_keys[0]: args}
            return args

        if len(schema_keys) != 1:
            return args

        expected_key = schema_keys[0]
        if expected_key in args:
            return args

        for alias in ("input", "query", "command", "commands", "url", "file_path", "path"):
            if alias in args:
                return {expected_key: args[alias]}

        if len(args) == 1:
            return {expected_key: next(iter(args.values()))}

        return args

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Message:
        """Execute one tool call and wrap the result as a history message."""
        tool_name = tool_call.get("name", "unknown_tool")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id") or f"tool_call_{datetime.now().timestamp()}"
        timestamp = datetime.now().isoformat()

        tool = self.tools_by_name.get(tool_name)
        if tool is None:
            return Message(
                role="tool",
                content=f"Error: Tool `{tool_name}` is not available.",
                timestamp=timestamp,
                tool_calls=[
                    {
                        "id": tool_call_id,
                        "name": tool_name,
                        "args": tool_args,
                        "status": "missing_tool",
                    }
                ],
                name=tool_name,
                tool_call_id=tool_call_id,
            )

        try:
            prepared_input = self._prepare_tool_input(tool, tool_args)
            result = tool.invoke(prepared_input)
            content = self._stringify_content(result)
            if tool_name == "write_file":
                content = self._postprocess_write_file_result(prepared_input, content)
            status = "success"
        except Exception as exc:
            content = f"Error executing tool `{tool_name}`: {exc}"
            status = "error"

        return Message(
            role="tool",
            content=content,
            timestamp=timestamp,
            tool_calls=[
                {
                    "id": tool_call_id,
                    "name": tool_name,
                    "args": tool_args,
                    "status": status,
                }
            ],
            name=tool_name,
            tool_call_id=tool_call_id,
        )

    def _run_autonomous_loop(
        self,
        session_id: str,
        history: List[Message],
        user_message: str,
    ) -> Tuple[str, List[Message]]:
        """Run the explicit OpenClaw-style autonomous tool loop."""
        timestamp = datetime.now().isoformat()
        turn_messages: List[Message] = [
            Message(role="user", content=user_message, timestamp=timestamp)
        ]
        final_reply = ""
        topic_prompt_context = self.topic_memory_manager.build_prompt_context(
            session_id=session_id,
            user_message=user_message,
            history=history,
            prepare_state=True,
        )
        session_context = self.context_selector.build_session_context(
            history=history,
            analysis=topic_prompt_context.analysis,
        )

        for iteration in range(self.max_tool_iterations):
            runtime_context = self._build_runtime_context(
                session_id=session_id,
                user_message=user_message,
                loop_messages=turn_messages,
                iteration=iteration,
                topic_prompt_context=topic_prompt_context,
                session_context=session_context,
            )
            prompt_layers = self._build_prompt_layers(
                session_id=session_id,
                topic_memory_context=topic_prompt_context.rendered,
                runtime_context=runtime_context,
                session_context=session_context,
                user_message=user_message,
            )
            system_prompt = self._build_system_prompt(prompt_layers=prompt_layers)
            combined_history = [*session_context.prompt_history, *turn_messages]
            prompt_messages = self._build_prompt_messages(system_prompt, combined_history)
            raw_messages = self._serialize_prompt_messages(prompt_messages)
            raw_prompt_payload = self._build_raw_prompt_payload(
                session_id=session_id,
                raw_messages=raw_messages,
                prompt_layers=prompt_layers,
            )
            self.memory_manager.save_raw_messages(
                session_id,
                raw_messages,
                prompt_preview=raw_prompt_payload["prompt_preview"],
                prompt_layers={
                    "system": raw_prompt_payload["system"],
                    "skills": raw_prompt_payload["skills"],
                    "tools": raw_prompt_payload["tools"],
                    "topic_memory": raw_prompt_payload["topic_memory"],
                    "session": raw_prompt_payload["session"],
                    "user": raw_prompt_payload["user"],
                },
            )

            response = self._invoke_model(prompt_messages)
            reply, _ = self._extract_reply_and_tool_messages(response)
            tool_calls = self._extract_tool_calls(response)

            assistant_message = Message(
                role="assistant",
                content=reply,
                timestamp=datetime.now().isoformat(),
                tool_calls=tool_calls or None,
            )
            turn_messages.append(assistant_message)

            if tool_calls:
                for tool_call in tool_calls:
                    turn_messages.append(self._execute_tool_call(tool_call))
                continue

            final_reply = reply or "任务已完成，但模型没有返回可见文本。"
            return final_reply, turn_messages

        final_reply = (
            "任务尚未完成，但已达到自动工具循环上限。"
            "请检查工具结果，或将任务拆分后重试。"
        )
        turn_messages.append(
            Message(
                role="assistant",
                content=final_reply,
                timestamp=datetime.now().isoformat(),
            )
        )
        return final_reply, turn_messages

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

        try:
            reply, turn_messages = self._run_autonomous_loop(
                session_id=session_id,
                history=history,
                user_message=message,
            )
        except Exception as e:
            reply = self._format_agent_error(e)
            print(f"Agent error: {e}")
            turn_messages = [
                Message(
                    role="user",
                    content=message,
                    timestamp=datetime.now().isoformat(),
                ),
                Message(
                    role="assistant",
                    content=reply,
                    timestamp=datetime.now().isoformat(),
                ),
            ]

        # Save to session
        updated_history = [*history, *turn_messages]
        self.memory_manager.save_session(session_id, updated_history)
        topic_update = self.topic_memory_manager.remember_turn(
            session_id=session_id,
            user_message=message,
            assistant_reply=reply,
            turn_messages=turn_messages,
        )
        self.experience_miner.record_turn(
            session_id=session_id,
            user_message=message,
            assistant_reply=reply,
            turn_messages=turn_messages,
            topic_update=topic_update,
        )

        # Save to daily log
        self.memory_manager.save_to_daily_log(turn_messages)

        return {"reply": reply, "session_id": session_id}

    def create_session(self) -> Dict[str, str]:
        """Create an empty session so the UI can inspect a fresh prompt state."""
        session_id = self.memory_manager.create_new_session()
        self.memory_manager.save_session(session_id, [])

        topic_prompt_context = self.topic_memory_manager.build_prompt_context(
            session_id=session_id,
            user_message=None,
            history=[],
        )
        session_context = self.context_selector.build_session_context(
            history=[],
            analysis=topic_prompt_context.analysis,
        )
        runtime_context = self._build_runtime_context(
            session_id=session_id,
            user_message=None,
            topic_prompt_context=topic_prompt_context,
            session_context=session_context,
        )
        prompt_layers = self._build_prompt_layers(
            session_id=session_id,
            topic_memory_context=topic_prompt_context.rendered,
            runtime_context=runtime_context,
            session_context=session_context,
            user_message=None,
        )
        system_prompt = self._build_system_prompt(prompt_layers=prompt_layers)
        raw_messages = self._build_raw_messages(system_prompt, [])
        raw_prompt_payload = self._build_raw_prompt_payload(
            session_id=session_id,
            raw_messages=raw_messages,
            prompt_layers=prompt_layers,
        )
        self.memory_manager.save_raw_messages(
            session_id,
            raw_messages,
            prompt_preview=raw_prompt_payload["prompt_preview"],
            prompt_layers={
                "system": raw_prompt_payload["system"],
                "skills": raw_prompt_payload["skills"],
                "tools": raw_prompt_payload["tools"],
                "topic_memory": raw_prompt_payload["topic_memory"],
                "session": raw_prompt_payload["session"],
                "user": raw_prompt_payload["user"],
            },
        )

        return {"session_id": session_id}

    def get_skills(self) -> List[Dict]:
        """Get all skills information"""
        return self.skill_manager.get_all_skills_info()

    def update_skill(self, name: str, content: str) -> bool:
        """Update a skill"""
        updated = self.skill_manager.update_skill(name, content)
        if updated:
            self.skill_manager.write_snapshot(self.workspace_dir)
        return updated

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
        payload = self._hydrate_raw_prompt_payload(
            self.memory_manager.load_raw_messages(session_id)
        )
        if payload.get("messages"):
            return payload

        history = self.memory_manager.load_session(session_id)
        topic_prompt_context = self.topic_memory_manager.build_prompt_context(
            session_id=session_id,
            user_message=None,
            history=history,
        )
        session_context = self.context_selector.build_session_context(
            history=history,
            analysis=topic_prompt_context.analysis,
        )
        runtime_context = self._build_runtime_context(
            session_id=session_id,
            user_message=None,
            topic_prompt_context=topic_prompt_context,
            session_context=session_context,
        )
        prompt_layers = self._build_prompt_layers(
            session_id=session_id,
            topic_memory_context=topic_prompt_context.rendered,
            runtime_context=runtime_context,
            session_context=session_context,
            user_message=None,
        )
        system_prompt = self._build_system_prompt(prompt_layers=prompt_layers)
        preview_messages = self._build_raw_messages(system_prompt, session_context.prompt_history)
        return self._build_raw_prompt_payload(
            session_id=session_id,
            raw_messages=preview_messages,
            prompt_layers=prompt_layers,
        )

    def preview_raw_messages(self, session_id: Optional[str] = None) -> Dict:
        """Preview the current prompt payload even before the next user message is sent."""
        history = self.memory_manager.load_session(session_id) if session_id else []
        topic_prompt_context = self.topic_memory_manager.build_prompt_context(
            session_id=session_id,
            user_message=None,
            history=history,
        )
        session_context = self.context_selector.build_session_context(
            history=history,
            analysis=topic_prompt_context.analysis,
        )
        runtime_context = self._build_runtime_context(
            session_id=session_id,
            user_message=None,
            topic_prompt_context=topic_prompt_context,
            session_context=session_context,
        )
        prompt_layers = self._build_prompt_layers(
            session_id=session_id,
            topic_memory_context=topic_prompt_context.rendered,
            runtime_context=runtime_context,
            session_context=session_context,
            user_message=None,
        )
        system_prompt = self._build_system_prompt(prompt_layers=prompt_layers)
        preview_messages = self._build_raw_messages(system_prompt, session_context.prompt_history)
        return self._build_raw_prompt_payload(
            session_id=session_id or "",
            raw_messages=preview_messages,
            prompt_layers=prompt_layers,
        )
