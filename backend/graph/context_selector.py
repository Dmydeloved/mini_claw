"""
Topic-aware conversation history selection.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .memory import Message
from .topic_memory_manager import TopicIntent


@dataclass
class TopicSessionContext:
    strategy: str
    used_full_history: bool
    selected_history: List[Message]
    prompt_history: List[Message]
    compression_applied: bool
    compression_summary: str
    selection_applied: bool
    original_message_count: int
    selected_message_count: int
    prompt_message_count: int


class TopicContextSelector:
    """Select history messages that are most relevant to the active topic."""

    def select_history(
        self,
        history: Sequence[Message],
        analysis: Optional[TopicIntent],
        max_messages: int = 14,
        preserve_recent_messages: int = 6,
    ) -> List[Message]:
        if len(history) <= max_messages or analysis is None:
            return list(history[-max_messages:])

        history_list = list(history)
        recent = history_list[-preserve_recent_messages:]
        older = history_list[:-preserve_recent_messages]
        older_budget = max(0, max_messages - len(recent))

        ranked: List[Tuple[float, int, Message]] = []
        for index, msg in enumerate(older):
            score = self._score_message(msg, analysis)
            if score <= 0.12:
                continue
            ranked.append((score, index, msg))

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected_indices = {index for _, index, _ in ranked[:older_budget]}

        augmented_indices = set(selected_indices)
        for index in list(selected_indices):
            if older[index].role == "assistant" and index > 0 and older[index - 1].role == "user":
                augmented_indices.add(index - 1)
            if older[index].role == "tool" and index > 0:
                augmented_indices.add(index - 1)

        selected_older = [older[index] for index in sorted(augmented_indices)]
        combined = [*selected_older, *recent]
        return combined[-max_messages:]

    def build_session_context(
        self,
        history: Sequence[Message],
        analysis: Optional[TopicIntent],
        full_history_message_threshold: int = 10,
        full_history_char_threshold: int = 3200,
        max_selected_messages: int = 24,
        max_prompt_messages: int = 14,
        preserve_recent_messages: int = 6,
        compression_char_threshold: int = 6000,
    ) -> TopicSessionContext:
        """Build prompt-ready session context with full -> select -> compress flow."""
        history_list = list(history)
        # Small session context is cheaper and clearer if we keep it verbatim.
        if self._within_threshold(
            history_list,
            max_messages=full_history_message_threshold,
            max_chars=full_history_char_threshold,
        ):
            prompt_history = history_list
            return TopicSessionContext(
                strategy="full_history",
                used_full_history=True,
                selected_history=history_list,
                prompt_history=prompt_history,
                compression_applied=False,
                compression_summary="",
                selection_applied=False,
                original_message_count=len(history_list),
                selected_message_count=len(history_list),
                prompt_message_count=len(prompt_history),
            )

        selected_history = self.select_history(
            history=history,
            analysis=analysis,
            max_messages=max_selected_messages,
            preserve_recent_messages=max(4, preserve_recent_messages + 2),
        )

        # If topic-filtered history already fits the budget, skip compression.
        if not self._needs_compression(
            selected_history,
            max_messages=max_prompt_messages,
            max_chars=compression_char_threshold,
        ):
            prompt_history = selected_history[-max_prompt_messages:]
            return TopicSessionContext(
                strategy="selected_history",
                used_full_history=False,
                selected_history=selected_history,
                prompt_history=prompt_history,
                compression_applied=False,
                compression_summary="",
                selection_applied=True,
                original_message_count=len(history_list),
                selected_message_count=len(selected_history),
                prompt_message_count=len(prompt_history),
            )

        recent_history = selected_history[-preserve_recent_messages:]
        older_history = selected_history[:-preserve_recent_messages]
        # Only older topic-related context is compressed; the newest turns stay verbatim.
        compression_summary = self._compress_history(older_history, analysis)
        summary_message = Message(
            role="assistant",
            content=(
                "[Compressed Topic Context]\n"
                f"{compression_summary}"
            ),
            timestamp="",
        )
        prompt_history = [summary_message, *recent_history]
        prompt_history = prompt_history[-max_prompt_messages:]
        return TopicSessionContext(
            strategy="selected_then_compressed",
            used_full_history=False,
            selected_history=selected_history,
            prompt_history=prompt_history,
            compression_applied=True,
            compression_summary=compression_summary,
            selection_applied=True,
            original_message_count=len(history_list),
            selected_message_count=len(selected_history),
            prompt_message_count=len(prompt_history),
        )

    def _score_message(self, msg: Message, analysis: TopicIntent) -> float:
        content = (msg.content or "").lower()
        if not content.strip():
            return 0.0

        keyword_hits = sum(1 for keyword in analysis.keywords if keyword and keyword in content)
        intent_hits = sum(1 for intent in analysis.intents if intent and intent in content)
        topic_hits = sum(1 for token in analysis.topic.lower().split(" / ") if token and token in content)

        role_weight = {
            "user": 1.0,
            "assistant": 0.9,
            "tool": 0.55,
        }.get(msg.role, 0.75)

        return (
            min(1.0, keyword_hits * 0.18 + intent_hits * 0.12 + topic_hits * 0.2)
            * role_weight
        )

    def _needs_compression(
        self,
        history: Sequence[Message],
        max_messages: int,
        max_chars: int,
    ) -> bool:
        if len(history) > max_messages:
            return True
        total_chars = sum(len(msg.content or "") for msg in history)
        return total_chars > max_chars

    def _within_threshold(
        self,
        history: Sequence[Message],
        max_messages: int,
        max_chars: int,
    ) -> bool:
        if len(history) > max_messages:
            return False
        total_chars = sum(len(msg.content or "") for msg in history)
        return total_chars <= max_chars

    def _compress_history(
        self,
        history: Sequence[Message],
        analysis: Optional[TopicIntent],
        max_entries: int = 8,
    ) -> str:
        lines = []
        if analysis:
            lines.append(
                f"topic={analysis.topic}; intents={', '.join(analysis.intents)}; "
                f"keywords={', '.join(analysis.keywords[:6])}"
            )

        for msg in history[-max_entries:]:
            label = {
                "user": "User",
                "assistant": "Assistant",
                "tool": "Tool",
            }.get(msg.role, msg.role.title())
            snippet = " ".join((msg.content or "").split())[:180]
            if not snippet:
                continue
            if msg.role == "tool" and msg.name:
                label = f"Tool `{msg.name}`"
            lines.append(f"- {label}: {snippet}")

        return "\n".join(lines) if lines else "No earlier topic-related context to compress."
