"""
Agent Graph for Mini-OpenClaw
Uses an explicit autonomous tool loop while preserving the existing
memory, skills, session, and snapshot architecture.
"""

import json
import os
import re
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
from graph.memory import MemoryManager, Message
from graph.skills import SkillManager


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
        max_tool_iterations: int = 12,
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

    def _build_tool_definitions(self) -> str:
        """Serialize all available tool/function definitions for the prompt."""
        blocks: List[str] = []

        for tool in self.tools:
            args_schema = getattr(tool, "args", {}) or {}
            try:
                schema_text = json.dumps(args_schema, ensure_ascii=False, indent=2)
            except TypeError:
                schema_text = self._stringify_content(args_schema)

            blocks.extend(
                [
                    f"## {tool.name}",
                    tool.description.strip(),
                    "",
                    "Arguments schema:",
                    "```json",
                    schema_text,
                    "```",
                    "",
                ]
            )

        return "\n".join(blocks).strip() or "No tools currently available."

    def _build_runtime_context(
        self,
        session_id: Optional[str],
        user_message: Optional[str],
        loop_messages: Optional[List[Message]] = None,
        iteration: int = 0,
    ) -> str:
        """Build volatile runtime context included before each model call."""
        loop_messages = loop_messages or []
        lines = [
            f"- session_id: {session_id or '(new session)'}",
            f"- loop_iteration: {iteration + 1}/{self.max_tool_iterations}",
        ]

        if user_message:
            lines.append(f"- current_user_query: {user_message}")

        matched_skill = self.skill_manager.match_trigger(user_message or "")
        if matched_skill:
            lines.append(
                "- matched_skill_hint: "
                f"{matched_skill.name} ({matched_skill.relative_path})"
            )

        tool_result_count = sum(1 for msg in loop_messages if msg.role == "tool")
        if tool_result_count:
            lines.append(f"- tool_results_collected_in_current_turn: {tool_result_count}")

        skill_request = self._extract_skill_creation_request(user_message)
        if skill_request:
            lines.append("- skill_creation_request_detected: true")
            lines.append(f"- requested_skill_name: {skill_request['name']}")
            lines.append(f"- suggested_skill_file_path: {skill_request['file_path']}")
            if skill_request.get("validation_error"):
                lines.append(
                    "- user_skill_draft_status: "
                    f"draft is incomplete ({skill_request['validation_error']})"
                )
            else:
                lines.append("- user_skill_draft_status: draft already includes valid skill frontmatter")

            if self._has_skill_write_in_turn(loop_messages):
                lines.append("- requested_skill_persistence_status: already written in this turn")
            else:
                lines.append("- requested_skill_persistence_status: pending")
                lines.append(
                    "- required_action: use `write_file` to persist a valid "
                    "`skills/<skill_name>/SKILL.md` file before finishing"
                )

        return "\n".join(lines)

    def _build_system_prompt(self, extra_context: Optional[str] = None) -> str:
        """Build the full OpenClaw-style prompt scaffold before every LLM call."""
        self.skill_manager.reload_skills(verbose=False)
        self.skill_manager.write_snapshot(self.workspace_dir)

        sections = [
            (
                "SOUL.md",
                self._read_workspace_file(
                    "SOUL.md", "You are Mini-OpenClaw, a transparent local-first AI assistant."
                ),
            ),
            ("IDENTITY.md", self.memory_manager.get_workspace_file("IDENTITY.md")),
            ("USER.md", self.memory_manager.get_workspace_file("USER.md")),
            ("AGENTS.md", self.memory_manager.get_workspace_file("AGENTS.md")),
            ("SKILLS_SUMMARY.md", self.skill_manager.build_skills_prompt()),
            ("TOOLS_AND_FUNCTIONS.md", self._build_tool_definitions()),
            ("MEMORY.md", self.memory_manager.get_memory_content()),
            ("RECENT_LOGS.md", self.memory_manager.get_recent_logs()),
        ]

        if extra_context:
            sections.append(("CURRENT_CONTEXT.md", extra_context))

        sections.append(
            (
                "AUTONOMOUS_AGENT_LOOP.md",
                (
                    "You are running in fully autonomous agent mode.\n\n"
                    "Rules:\n"
                    "1. Before every answer, rely on the prompt you were given in this call: "
                    "system prompt, all skills, all tools/functions, memory, history, context, and the user query.\n"
                    "2. If you need external information or file contents, emit tool calls instead of stopping early.\n"
                    "3. After tool results are returned, continue the task automatically from the updated context.\n"
                    "4. Only provide the final natural-language answer when no further tool call is necessary.\n"
                    "5. Follow the skill protocol strictly: the prompt only contains each skill's name and description. "
                    "If you need to use a skill, first read `workspace/SKILLS_SNAPSHOT.md` to locate it, then read the skill file with `read_file`.\n"
                    "6. Base conclusions on files and tool results, and distinguish facts from inference.\n"
                    "7. When the user asks to add, create, save, or update a skill, you must use `write_file` "
                    "to create or update `skills/<skill_name>/SKILL.md` instead of only describing the plan.\n"
                    "8. A valid `SKILL.md` file must contain YAML frontmatter with at least "
                    "`name` and `description`. `trigger` is optional and only serves as a routing hint. "
                    "Add `enabled: true` unless the user requests otherwise.\n"
                    "9. If the user supplied only a partial skill draft, first transform it into a valid `SKILL.md`, "
                    "then write the file."
                ),
            )
        )

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

    def _is_skill_creation_request(self, user_message: Optional[str]) -> bool:
        """Detect whether the user is asking to add or update a skill."""
        if not user_message:
            return False

        lowered = user_message.lower()
        patterns = (
            "添加这个技能",
            "添加技能",
            "新增技能",
            "创建技能",
            "保存技能",
            "帮我添加",
            "帮我创建",
            "add this skill",
            "add the skill",
            "create this skill",
            "create a skill",
            "save this skill",
            "update this skill",
        )
        return any(pattern in lowered for pattern in patterns) or "skill" in lowered and (
            "添加" in user_message or "新增" in user_message or "创建" in user_message or "add" in lowered
        )

    def _extract_skill_creation_request(self, user_message: Optional[str]) -> Optional[Dict[str, Any]]:
        """Extract a requested skill draft and its target path from the user message."""
        if not self._is_skill_creation_request(user_message):
            return None

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

                content = self._truncate_component(msg.content, max_chars=12000)
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
                    payload["content"] = (
                        f"{content}\n\n{summary}".strip() if content else summary
                    )

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
        requested_skill = self._extract_skill_creation_request(user_message)
        turn_messages: List[Message] = [
            Message(role="user", content=user_message, timestamp=timestamp)
        ]
        final_reply = ""

        for iteration in range(self.max_tool_iterations):
            system_prompt = self._build_system_prompt(
                extra_context=self._build_runtime_context(
                    session_id=session_id,
                    user_message=user_message,
                    loop_messages=turn_messages,
                    iteration=iteration,
                )
            )
            combined_history = [*history, *turn_messages]
            prompt_messages = self._build_prompt_messages(system_prompt, combined_history)
            raw_messages = self._serialize_prompt_messages(prompt_messages)
            self.memory_manager.save_raw_messages(session_id, raw_messages)

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

            if (
                iteration == 0
                and requested_skill
                and not self._has_skill_write_in_turn(turn_messages)
            ):
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

        # Save to daily log
        self.memory_manager.save_to_daily_log(turn_messages)

        return {"reply": reply, "session_id": session_id}

    def create_session(self) -> Dict[str, str]:
        """Create an empty session so the UI can inspect a fresh prompt state."""
        session_id = self.memory_manager.create_new_session()
        self.memory_manager.save_session(session_id, [])

        system_prompt = self._build_system_prompt(
            extra_context=self._build_runtime_context(session_id=session_id, user_message=None)
        )
        raw_messages = self._build_raw_messages(system_prompt, [])
        self.memory_manager.save_raw_messages(session_id, raw_messages)

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
        payload = self.memory_manager.load_raw_messages(session_id)
        if payload.get("messages"):
            return payload

        history = self.memory_manager.load_session(session_id)
        system_prompt = self._build_system_prompt(
            extra_context=self._build_runtime_context(session_id=session_id, user_message=None)
        )
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
        system_prompt = self._build_system_prompt(
            extra_context=self._build_runtime_context(session_id=session_id, user_message=None)
        )
        preview_messages = self._build_raw_messages(system_prompt, history)

        return {
            "session_id": session_id or "",
            "updated_at": datetime.now().isoformat(),
            "message_count": len(preview_messages),
            "messages": preview_messages,
        }
