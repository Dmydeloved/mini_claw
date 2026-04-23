"""
Experience mining and skill-candidate generation for Mini-OpenClaw.
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from .memory import Message
from .topic_memory_manager import TopicIntent, TopicTurnUpdate


@dataclass
class SkillCandidate:
    candidate_id: str
    name: str
    description: str
    source_topic: str
    domain: str
    intents: List[str] = field(default_factory=list)
    tool_sequence: List[str] = field(default_factory=list)
    occurrence_count: int = 0
    confidence: float = 0.0
    session_ids: List[str] = field(default_factory=list)
    file_path: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return slug or "candidate"


class ExperienceMiner:
    """Track repeated tool-use patterns and turn them into candidate skills."""

    def __init__(self, memory_dir: str):
        self.memory_dir = os.path.abspath(memory_dir)
        self.experience_dir = os.path.join(self.memory_dir, "experience")
        self.skill_candidates_dir = os.path.join(self.experience_dir, "skill_candidates")
        self.pattern_index_file = os.path.join(self.experience_dir, "pattern_index.json")

        os.makedirs(self.experience_dir, exist_ok=True)
        os.makedirs(self.skill_candidates_dir, exist_ok=True)

        if not os.path.exists(self.pattern_index_file):
            self._save_json(self.pattern_index_file, {"patterns": {}})

    def record_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_reply: str,
        turn_messages: Sequence[Message],
        topic_update: Optional[TopicTurnUpdate],
    ) -> Optional[SkillCandidate]:
        """Capture execution patterns from a finished turn and emit a skill candidate when justified."""
        analysis = topic_update.analysis if topic_update else None
        if not analysis:
            return None

        tool_sequence = [msg.name for msg in turn_messages if msg.role == "tool" and msg.name]
        if len(tool_sequence) < 2:
            return None

        signature = self._build_pattern_signature(analysis, tool_sequence)
        pattern_index = self._load_json(self.pattern_index_file, {"patterns": {}})
        patterns = pattern_index.setdefault("patterns", {})
        entry = patterns.get(signature, {})

        entry["domain"] = analysis.domain
        entry["topic"] = analysis.topic
        entry["intents"] = list(analysis.intents)
        entry["tool_sequence"] = list(tool_sequence)
        entry["count"] = int(entry.get("count", 0)) + 1
        entry["examples"] = self._append_limited(
            entry.get("examples", []),
            {
                "session_id": session_id,
                "user_message": user_message[:220],
                "assistant_reply": assistant_reply[:260],
                "updated_at": datetime.now().isoformat(),
            },
            limit=5,
        )
        entry["session_ids"] = self._append_unique(entry.get("session_ids", []), session_id, limit=12)
        entry["updated_at"] = datetime.now().isoformat()
        patterns[signature] = entry
        self._save_json(self.pattern_index_file, pattern_index)

        if not self._should_generate_candidate(entry, topic_update):
            return None

        return self._write_skill_candidate(signature, entry)

    def _build_pattern_signature(self, analysis: TopicIntent, tool_sequence: Sequence[str]) -> str:
        intents = ",".join(sorted(analysis.intents))
        tools = ">".join(tool_sequence)
        return f"{analysis.domain}|{intents}|{tools}"

    def _should_generate_candidate(
        self,
        entry: Dict,
        topic_update: Optional[TopicTurnUpdate],
    ) -> bool:
        count = int(entry.get("count", 0))
        tool_sequence = entry.get("tool_sequence", [])
        if count >= 2:
            return True
        if len(tool_sequence) >= 4:
            return True
        if topic_update and topic_update.topic_shift and len(tool_sequence) >= 3:
            return True
        return False

    def _write_skill_candidate(self, signature: str, entry: Dict) -> SkillCandidate:
        topic = str(entry.get("topic", "general workflow"))
        domain = str(entry.get("domain", "general_dialog"))
        tool_sequence = [str(item) for item in entry.get("tool_sequence", [])]
        intents = [str(item) for item in entry.get("intents", [])]
        occurrence_count = int(entry.get("count", 0))
        confidence = min(0.95, 0.35 + occurrence_count * 0.15 + len(tool_sequence) * 0.05)

        name = self._build_candidate_name(domain, topic, tool_sequence)
        candidate_id = _slugify(signature.replace("|", "-"))
        file_path = os.path.join(self.skill_candidates_dir, f"{candidate_id}.md")
        description = (
            f"Candidate reusable workflow for {topic} using "
            f"{' -> '.join(tool_sequence)}."
        )

        content = self._render_candidate_markdown(
            name=name,
            description=description,
            domain=domain,
            topic=topic,
            intents=intents,
            tool_sequence=tool_sequence,
            occurrence_count=occurrence_count,
            confidence=confidence,
            examples=entry.get("examples", []),
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return SkillCandidate(
            candidate_id=candidate_id,
            name=name,
            description=description,
            source_topic=topic,
            domain=domain,
            intents=intents,
            tool_sequence=tool_sequence,
            occurrence_count=occurrence_count,
            confidence=round(confidence, 2),
            session_ids=[str(item) for item in entry.get("session_ids", [])],
            file_path=file_path,
            updated_at=datetime.now().isoformat(),
        )

    def _build_candidate_name(
        self,
        domain: str,
        topic: str,
        tool_sequence: Sequence[str],
    ) -> str:
        condensed_topic = topic.replace(" / ", " ").strip()
        condensed_topic = re.sub(r"\s+", " ", condensed_topic)[:36]
        primary_tool = tool_sequence[0] if tool_sequence else "workflow"
        return f"{domain}:{condensed_topic}:{primary_tool}"

    def _render_candidate_markdown(
        self,
        name: str,
        description: str,
        domain: str,
        topic: str,
        intents: Sequence[str],
        tool_sequence: Sequence[str],
        occurrence_count: int,
        confidence: float,
        examples: Sequence[Dict],
    ) -> str:
        lines = [
            f"# Skill Candidate: {name}",
            "",
            "## Summary",
            f"- Description: {description}",
            f"- Domain: {domain}",
            f"- Topic: {topic}",
            f"- Intents: {', '.join(intents) or 'general'}",
            f"- Tool sequence: {' -> '.join(tool_sequence)}",
            f"- Occurrence count: {occurrence_count}",
            f"- Confidence: {confidence:.2f}",
            "",
            "## Why This Candidate Exists",
            "This candidate was generated because the agent repeated a similar tool-use pattern.",
            "",
            "## Evidence",
        ]

        if not examples:
            lines.append("- No examples recorded.")
        else:
            for example in examples[-3:]:
                lines.append(f"- Session: {example.get('session_id', '')}")
                lines.append(f"  user: {example.get('user_message', '')}")
                lines.append(f"  assistant: {example.get('assistant_reply', '')}")

        lines.extend(
            [
                "",
                "## Proposed Skill Prototype",
                "```md",
                "---",
                f"name: \"{name}\"",
                f"description: \"{description}\"",
                f"trigger: \"{self._build_trigger(topic, intents)}\"",
                "enabled: false",
                "---",
                "",
                "# Goal",
                f"Handle recurring workflow for {topic}.",
                "",
                "# Recommended Steps",
            ]
        )

        for index, tool_name in enumerate(tool_sequence, start=1):
            lines.append(f"{index}. Use `{tool_name}` when its prerequisite inputs are available.")

        lines.extend(
            [
                "",
                "# Validation Checklist",
                "1. Confirm the trigger matches a genuinely repeatable task.",
                "2. Confirm each tool call is necessary and ordered correctly.",
                "3. Confirm the workflow generalizes beyond one session.",
                "```",
                "",
            ]
        )
        return "\n".join(lines)

    def _build_trigger(self, topic: str, intents: Sequence[str]) -> str:
        topic_hint = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", " ", topic).strip().lower()
        intent_hint = "|".join(re.escape(intent) for intent in intents[:2] if intent)
        if intent_hint:
            return f"{topic_hint}|{intent_hint}"
        return topic_hint or "workflow"

    def _append_unique(self, items: Sequence[str], value: str, limit: int) -> List[str]:
        merged = list(items)
        if value not in merged:
            merged.append(value)
        return merged[-limit:]

    def _append_limited(self, items: Sequence[Dict], value: Dict, limit: int) -> List[Dict]:
        merged = list(items)
        merged.append(value)
        return merged[-limit:]

    def _load_json(self, path: str, default):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _save_json(self, path: str, payload):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
