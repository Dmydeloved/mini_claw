"""
Topic-oriented memory management for Mini-OpenClaw.

This module adds a lightweight topic layer on top of the existing
profile/session memory system so the agent can:
1. infer the current domain/topic/intent from a user message
2. retrieve cross-session topic memories
3. track topic slices inside each session
4. consolidate new turns into reusable topic cards
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set, Tuple

from graph.memory import Message


CHINESE_STOPWORDS = {
    "帮我",
    "一下",
    "这个",
    "那个",
    "现在",
    "需要",
    "想要",
    "我们",
    "你们",
    "请问",
    "一个",
    "一些",
    "已经",
    "还有",
    "以及",
    "因为",
    "所以",
    "但是",
    "如果",
    "然后",
    "继续",
    "进行",
    "如何",
    "怎么",
    "什么",
    "为什么",
    "是否",
    "可以",
    "需要",
    "对于",
    "关于",
    "相关",
    "方面",
    "问题",
    "内容",
    "基于",
    "系统",
    "会话",
    "主题",
    "上下文",
    "记忆",
    "经验",
}

ENGLISH_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "about",
    "what",
    "when",
    "where",
    "which",
    "while",
    "should",
    "would",
    "could",
    "need",
    "want",
    "help",
    "please",
    "session",
    "memory",
    "context",
    "topic",
    "agent",
}

DOMAIN_KEYWORDS = {
    "software_engineering": {
        "python",
        "java",
        "golang",
        "typescript",
        "javascript",
        "bug",
        "debug",
        "测试",
        "代码",
        "接口",
        "前端",
        "后端",
        "数据库",
        "部署",
        "编程",
        "脚本",
        "函数",
        "类",
    },
    "finance_analysis": {
        "股票",
        "基金",
        "银行",
        "证券",
        "macd",
        "kdj",
        "boll",
        "布林带",
        "均线",
        "技术面",
        "支撑位",
        "压力位",
        "财报",
        "估值",
        "行情",
        "600036",
    },
    "research_writing": {
        "论文",
        "研究",
        "综述",
        "摘要",
        "框架",
        "方案",
        "方法",
        "设计",
        "写作",
        "报告",
        "课题",
        "项目",
    },
    "product_planning": {
        "prd",
        "需求",
        "原型",
        "功能",
        "流程",
        "页面",
        "交互",
        "设计稿",
        "方案",
        "规划",
    },
}

INTENT_KEYWORDS = {
    "analysis": {"分析", "判断", "评估", "review", "analyze", "compare", "比较"},
    "implementation": {"实现", "开发", "编写", "修改", "修复", "build", "implement", "fix"},
    "planning": {"方案", "规划", "设计", "架构", "思路", "plan", "design", "architecture"},
    "explanation": {"解释", "说明", "讲解", "介绍", "why", "explain", "how"},
    "summarization": {"总结", "归纳", "提炼", "摘要", "summary", "summarize"},
    "retrieval": {"查找", "搜索", "检索", "load", "find", "search"},
}


@dataclass
class TopicIntent:
    domain: str
    topic: str
    intents: List[str]
    keywords: List[str]
    confidence: float

    @property
    def topic_id(self) -> str:
        return _slugify(f"{self.domain}-{self.topic}") or "general-topic"


@dataclass
class TopicCard:
    topic_id: str
    domain: str
    topic: str
    intents: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    summaries: List[str] = field(default_factory=list)
    tool_patterns: List[str] = field(default_factory=list)
    related_sessions: List[str] = field(default_factory=list)
    importance: float = 0.5
    reuse_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SessionTopicSlice:
    slice_id: str
    topic_id: str
    domain: str
    topic: str
    intents: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    summary: str = ""
    turn_count: int = 0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SessionTopicState:
    session_id: str
    current_topic: Optional[TopicIntent] = None
    slices: List[SessionTopicSlice] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TopicPromptContext:
    analysis: Optional[TopicIntent]
    topic_shift: bool
    retrieved_cards: List[TopicCard]
    related_slices: List[SessionTopicSlice]
    rendered: str


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", value.lower())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized[:96]


class TopicMemoryManager:
    """Manage topic extraction, retrieval, consolidation, and topic slices."""

    def __init__(self, memory_dir: str, sessions_dir: str):
        self.memory_dir = os.path.abspath(memory_dir)
        self.sessions_dir = os.path.abspath(sessions_dir)
        self.topics_dir = os.path.join(self.memory_dir, "topics")
        self.index_dir = os.path.join(self.memory_dir, "topic_index")
        self.index_file = os.path.join(self.index_dir, "inverted_index.json")
        self.session_topics_dir = os.path.join(self.sessions_dir, "_topic_state")

        os.makedirs(self.topics_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.session_topics_dir, exist_ok=True)

        if not os.path.exists(self.index_file):
            self._save_json(
                self.index_file,
                {"keywords": {}, "intents": {}, "domains": {}},
            )

    def analyze_message(
        self,
        message: Optional[str],
        previous: Optional[TopicIntent] = None,
    ) -> Optional[TopicIntent]:
        """Infer domain/topic/intent from the latest user message."""
        if not message or not str(message).strip():
            return previous

        normalized_message = str(message).lower()
        tokens = self._tokenize(str(message))
        matched_keywords = self._extract_known_keywords(normalized_message)
        keywords = self._merge_unique(matched_keywords, self._select_keywords(tokens), limit=8)
        domain = self._infer_domain(tokens, normalized_message, previous)
        intents = self._infer_intents(tokens, normalized_message)
        topic = self._infer_topic(keywords, message, previous)
        confidence = min(0.96, 0.35 + len(keywords) * 0.08 + len(intents) * 0.05)

        return TopicIntent(
            domain=domain,
            topic=topic,
            intents=intents,
            keywords=keywords,
            confidence=round(confidence, 2),
        )

    def build_prompt_context(
        self,
        session_id: Optional[str],
        user_message: Optional[str],
    ) -> TopicPromptContext:
        """Prepare topic-aware retrieval context for the next model call."""
        state = self.load_session_state(session_id) if session_id else None
        previous = state.current_topic if state else None
        analysis = self.analyze_message(user_message, previous=previous)
        topic_shift = self.is_topic_shift(previous, analysis)

        if not analysis:
            return TopicPromptContext(
                analysis=None,
                topic_shift=False,
                retrieved_cards=[],
                related_slices=[],
                rendered="No active topic inferred yet.",
            )

        cards = self.retrieve_topic_cards(analysis)
        related_slices = self.retrieve_related_slices(
            analysis,
            exclude_session_id=session_id,
        )
        rendered = self._render_prompt_context(analysis, topic_shift, cards, related_slices)

        return TopicPromptContext(
            analysis=analysis,
            topic_shift=topic_shift,
            retrieved_cards=cards,
            related_slices=related_slices,
            rendered=rendered,
        )

    def is_topic_shift(
        self,
        previous: Optional[TopicIntent],
        current: Optional[TopicIntent],
    ) -> bool:
        """Detect whether the conversation likely switched topic."""
        if current is None:
            return False
        if previous is None:
            return True

        similarity = self.topic_similarity(previous, current)
        if previous.domain != current.domain and similarity < 0.45:
            return True
        return similarity < 0.32

    def topic_similarity(self, left: TopicIntent, right: TopicIntent) -> float:
        """Compute a lightweight semantic similarity based on domain/topic/intent overlap."""
        left_keywords = set(left.keywords)
        right_keywords = set(right.keywords)
        keyword_overlap = self._jaccard(left_keywords, right_keywords)
        intent_overlap = self._jaccard(set(left.intents), set(right.intents))
        domain_score = 1.0 if left.domain == right.domain else 0.0
        topic_score = 1.0 if left.topic == right.topic else self._soft_topic_overlap(left.topic, right.topic)
        return round(
            keyword_overlap * 0.45
            + intent_overlap * 0.2
            + domain_score * 0.15
            + topic_score * 0.2,
            3,
        )

    def retrieve_topic_cards(
        self,
        analysis: TopicIntent,
        limit: int = 3,
    ) -> List[TopicCard]:
        """Retrieve cross-session topic cards relevant to the current topic."""
        candidates = self._candidate_topic_ids(analysis)
        cards = []
        for topic_id in candidates:
            card = self.load_topic_card(topic_id)
            if not card:
                continue
            score = self._score_card(card, analysis)
            if score <= 0.15:
                continue
            cards.append((score, card))

        if not cards:
            for filename in os.listdir(self.topics_dir):
                if not filename.endswith(".json"):
                    continue
                card = self.load_topic_card(filename[:-5])
                if not card:
                    continue
                score = self._score_card(card, analysis)
                if score > 0.18:
                    cards.append((score, card))

        cards.sort(key=lambda item: item[0], reverse=True)
        top_cards = [card for _, card in cards[:limit]]
        for card in top_cards:
            card.reuse_count += 1
            card.updated_at = datetime.now().isoformat()
            self.save_topic_card(card)
        return top_cards

    def retrieve_related_slices(
        self,
        analysis: TopicIntent,
        exclude_session_id: Optional[str] = None,
        limit: int = 2,
    ) -> List[SessionTopicSlice]:
        """Retrieve related topic slices from other sessions."""
        ranked: List[Tuple[float, SessionTopicSlice]] = []

        for filename in os.listdir(self.session_topics_dir):
            if not filename.endswith(".json"):
                continue

            session_id = filename[:-5]
            if exclude_session_id and session_id == exclude_session_id:
                continue

            state = self.load_session_state(session_id)
            for session_slice in state.slices:
                score = self._score_slice(session_slice, analysis)
                if score <= 0.18:
                    continue
                ranked.append((score, session_slice))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [session_slice for _, session_slice in ranked[:limit]]

    def remember_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_reply: str,
        turn_messages: Sequence[Message],
    ) -> Optional[TopicIntent]:
        """Persist the new turn into session topic slices and cross-session topic cards."""
        state = self.load_session_state(session_id)
        previous = state.current_topic
        analysis = self.analyze_message(user_message, previous=previous)
        if not analysis:
            return previous

        topic_shift = self.is_topic_shift(previous, analysis)
        turn_summary = self._summarize_turn(user_message, assistant_reply, turn_messages)

        if topic_shift or not state.slices:
            state.slices.append(
                SessionTopicSlice(
                    slice_id=f"{analysis.topic_id}-{len(state.slices) + 1}",
                    topic_id=analysis.topic_id,
                    domain=analysis.domain,
                    topic=analysis.topic,
                    intents=list(analysis.intents),
                    keywords=list(analysis.keywords),
                    summary=turn_summary,
                    turn_count=1,
                )
            )
        else:
            current_slice = state.slices[-1]
            current_slice.updated_at = datetime.now().isoformat()
            current_slice.turn_count += 1
            current_slice.intents = self._merge_unique(current_slice.intents, analysis.intents, limit=6)
            current_slice.keywords = self._merge_unique(current_slice.keywords, analysis.keywords, limit=10)
            current_slice.summary = self._merge_summaries(current_slice.summary, turn_summary)

        state.current_topic = analysis
        state.updated_at = datetime.now().isoformat()
        self.save_session_state(state)

        self._merge_turn_into_topic_card(session_id, analysis, turn_summary, turn_messages)
        return analysis

    def load_session_state(self, session_id: Optional[str]) -> SessionTopicState:
        """Load the topic state for a session."""
        if not session_id:
            return SessionTopicState(session_id="")

        path = os.path.join(self.session_topics_dir, f"{session_id}.json")
        if not os.path.exists(path):
            return SessionTopicState(session_id=session_id)

        payload = self._load_json(path, {})
        current_topic_payload = payload.get("current_topic")
        current_topic = TopicIntent(**current_topic_payload) if current_topic_payload else None
        slices = [SessionTopicSlice(**item) for item in payload.get("slices", [])]
        return SessionTopicState(
            session_id=payload.get("session_id", session_id),
            current_topic=current_topic,
            slices=slices,
            updated_at=payload.get("updated_at", datetime.now().isoformat()),
        )

    def save_session_state(self, state: SessionTopicState):
        """Persist per-session topic slices and current topic state."""
        if not state.session_id:
            return

        payload = {
            "session_id": state.session_id,
            "current_topic": asdict(state.current_topic) if state.current_topic else None,
            "slices": [asdict(item) for item in state.slices],
            "updated_at": state.updated_at,
        }
        path = os.path.join(self.session_topics_dir, f"{state.session_id}.json")
        self._save_json(path, payload)

    def load_topic_card(self, topic_id: str) -> Optional[TopicCard]:
        """Load a single topic card from disk."""
        path = os.path.join(self.topics_dir, f"{topic_id}.json")
        if not os.path.exists(path):
            return None
        payload = self._load_json(path, {})
        if not payload:
            return None
        return TopicCard(**payload)

    def save_topic_card(self, card: TopicCard):
        """Persist a topic card and refresh the inverted index."""
        path = os.path.join(self.topics_dir, f"{card.topic_id}.json")
        self._save_json(path, asdict(card))
        self._refresh_index_for_card(card)

    def get_current_topic_summary(self, session_id: Optional[str]) -> str:
        """Return a compact description of the current session topic."""
        state = self.load_session_state(session_id)
        if not state.current_topic:
            return "No active topic tracked for this session."

        current = state.current_topic
        return (
            f"domain={current.domain}; topic={current.topic}; "
            f"intents={', '.join(current.intents) or 'general'}; "
            f"keywords={', '.join(current.keywords[:6]) or 'n/a'}"
        )

    def _tokenize(self, text: str) -> List[str]:
        raw_tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_+-]{1,}|[0-9]{3,}", text)
        tokens: List[str] = []
        for token in raw_tokens:
            normalized = token.strip().lower()
            if not normalized:
                continue
            if normalized in ENGLISH_STOPWORDS or normalized in CHINESE_STOPWORDS:
                continue
            if len(normalized) == 1:
                continue
            tokens.append(normalized)
        return tokens

    def _select_keywords(self, tokens: Sequence[str], limit: int = 8) -> List[str]:
        seen: Set[str] = set()
        ranked = sorted(tokens, key=self._keyword_sort_key)
        keywords: List[str] = []
        for token in ranked:
            if token in seen:
                continue
            seen.add(token)
            keywords.append(token)
            if len(keywords) >= limit:
                break
        return keywords

    def _keyword_sort_key(self, token: str) -> Tuple[int, int, str]:
        is_numeric = 1 if token.isdigit() else 0
        is_indicator = 1 if token.upper() in {"MACD", "KDJ", "RSI", "BOLL"} else 0
        return (-is_indicator, -is_numeric, -len(token), token)

    def _infer_domain(
        self,
        tokens: Sequence[str],
        normalized_message: str,
        previous: Optional[TopicIntent],
    ) -> str:
        token_set = set(tokens)
        best_domain = "general_dialog"
        best_score = 0

        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = len(token_set & keywords) + sum(
                1 for keyword in keywords if keyword in normalized_message
            )
            if score > best_score:
                best_domain = domain
                best_score = score

        if best_score == 0 and previous:
            return previous.domain
        return best_domain

    def _infer_intents(self, tokens: Sequence[str], normalized_message: str) -> List[str]:
        token_set = set(tokens)
        intents: List[str] = []
        for intent, keywords in INTENT_KEYWORDS.items():
            if (token_set & keywords) or any(keyword in normalized_message for keyword in keywords):
                intents.append(intent)
        return intents or ["general"]

    def _extract_known_keywords(self, normalized_message: str) -> List[str]:
        keywords: List[str] = []
        for group in (DOMAIN_KEYWORDS, INTENT_KEYWORDS):
            for terms in group.values():
                for term in terms:
                    if term in normalized_message and term not in keywords:
                        keywords.append(term)
        return keywords

    def _infer_topic(
        self,
        keywords: Sequence[str],
        message: str,
        previous: Optional[TopicIntent],
    ) -> str:
        if keywords:
            return " / ".join(list(keywords)[:3])

        normalized = " ".join(message.strip().split())
        if normalized:
            return normalized[:48]

        if previous:
            return previous.topic

        return "general topic"

    def _candidate_topic_ids(self, analysis: TopicIntent) -> Set[str]:
        index = self._load_json(self.index_file, {"keywords": {}, "intents": {}, "domains": {}})
        candidates: Set[str] = set()

        for keyword in analysis.keywords:
            candidates.update(index.get("keywords", {}).get(keyword, []))
        for intent in analysis.intents:
            candidates.update(index.get("intents", {}).get(intent, []))
        candidates.update(index.get("domains", {}).get(analysis.domain, []))
        return candidates

    def _score_card(self, card: TopicCard, analysis: TopicIntent) -> float:
        keyword_score = self._jaccard(set(card.keywords), set(analysis.keywords))
        intent_score = self._jaccard(set(card.intents), set(analysis.intents))
        domain_score = 1.0 if card.domain == analysis.domain else 0.0
        topic_score = self._soft_topic_overlap(card.topic, analysis.topic)
        importance_score = min(1.0, card.importance)
        return round(
            keyword_score * 0.4
            + intent_score * 0.2
            + domain_score * 0.15
            + topic_score * 0.15
            + importance_score * 0.1,
            3,
        )

    def _score_slice(self, session_slice: SessionTopicSlice, analysis: TopicIntent) -> float:
        keyword_score = self._jaccard(set(session_slice.keywords), set(analysis.keywords))
        intent_score = self._jaccard(set(session_slice.intents), set(analysis.intents))
        domain_score = 1.0 if session_slice.domain == analysis.domain else 0.0
        topic_score = self._soft_topic_overlap(session_slice.topic, analysis.topic)
        return round(
            keyword_score * 0.45
            + intent_score * 0.15
            + domain_score * 0.15
            + topic_score * 0.25,
            3,
        )

    def _render_prompt_context(
        self,
        analysis: TopicIntent,
        topic_shift: bool,
        cards: Sequence[TopicCard],
        related_slices: Sequence[SessionTopicSlice],
    ) -> str:
        lines = [
            "## Current Topic Analysis",
            f"- domain: {analysis.domain}",
            f"- topic: {analysis.topic}",
            f"- intents: {', '.join(analysis.intents)}",
            f"- keywords: {', '.join(analysis.keywords)}",
            f"- topic_shift_detected: {'yes' if topic_shift else 'no'}",
            "",
            "## Retrieved Topic Memories",
        ]

        if not cards:
            lines.append("- No strongly related long-term topic memories found yet.")
        else:
            for card in cards:
                lines.append(f"- [{card.domain}] {card.topic}")
                if card.summaries:
                    lines.append(f"  summary: {card.summaries[-1]}")
                if card.tool_patterns:
                    lines.append(f"  tool patterns: {', '.join(card.tool_patterns[:4])}")

        lines.extend(["", "## Related Historical Topic Slices"])
        if not related_slices:
            lines.append("- No related topic slices from prior sessions.")
        else:
            for session_slice in related_slices:
                lines.append(
                    f"- [{session_slice.domain}] {session_slice.topic}: {session_slice.summary}"
                )

        return "\n".join(lines).strip()

    def _summarize_turn(
        self,
        user_message: str,
        assistant_reply: str,
        turn_messages: Sequence[Message],
    ) -> str:
        user_snippet = " ".join(user_message.split())[:180]
        assistant_snippet = " ".join((assistant_reply or "").split())[:220]
        tools = [msg.name for msg in turn_messages if msg.role == "tool" and msg.name]
        tool_summary = f"; tools={', '.join(self._merge_unique([], tools, limit=4))}" if tools else ""
        return f"user={user_snippet}; assistant={assistant_snippet}{tool_summary}"

    def _merge_turn_into_topic_card(
        self,
        session_id: str,
        analysis: TopicIntent,
        turn_summary: str,
        turn_messages: Sequence[Message],
    ):
        card = self.load_topic_card(analysis.topic_id)
        if not card:
            card = TopicCard(
                topic_id=analysis.topic_id,
                domain=analysis.domain,
                topic=analysis.topic,
            )

        card.intents = self._merge_unique(card.intents, analysis.intents, limit=8)
        card.keywords = self._merge_unique(card.keywords, analysis.keywords, limit=12)
        card.summaries = self._merge_unique(card.summaries, [turn_summary], limit=8)
        tool_names = [msg.name for msg in turn_messages if msg.role == "tool" and msg.name]
        card.tool_patterns = self._merge_unique(card.tool_patterns, tool_names, limit=8)
        card.related_sessions = self._merge_unique(card.related_sessions, [session_id], limit=12)
        card.importance = min(1.0, round(card.importance + 0.06, 2))
        card.updated_at = datetime.now().isoformat()
        self.save_topic_card(card)

    def _refresh_index_for_card(self, card: TopicCard):
        index = self._load_json(self.index_file, {"keywords": {}, "intents": {}, "domains": {}})
        index = self._remove_topic_from_index(index, card.topic_id)

        for keyword in card.keywords:
            index.setdefault("keywords", {}).setdefault(keyword, []).append(card.topic_id)
        for intent in card.intents:
            index.setdefault("intents", {}).setdefault(intent, []).append(card.topic_id)
        index.setdefault("domains", {}).setdefault(card.domain, []).append(card.topic_id)

        self._save_json(self.index_file, self._deduplicate_index(index))

    def _remove_topic_from_index(self, index: Dict, topic_id: str) -> Dict:
        for bucket in ("keywords", "intents", "domains"):
            for key, values in list(index.get(bucket, {}).items()):
                index[bucket][key] = [value for value in values if value != topic_id]
                if not index[bucket][key]:
                    index[bucket].pop(key, None)
        return index

    def _deduplicate_index(self, index: Dict) -> Dict:
        for bucket in ("keywords", "intents", "domains"):
            cleaned = {}
            for key, values in index.get(bucket, {}).items():
                cleaned[key] = list(dict.fromkeys(values))
            index[bucket] = cleaned
        return index

    def _merge_summaries(self, existing: str, new_summary: str) -> str:
        if not existing:
            return new_summary[:420]
        if new_summary in existing:
            return existing[:420]
        merged = f"{existing} | {new_summary}"
        return merged[:420]

    def _merge_unique(
        self,
        existing: Sequence[str],
        incoming: Sequence[str],
        limit: int,
    ) -> List[str]:
        merged = list(existing)
        for item in incoming:
            normalized = str(item).strip()
            if not normalized or normalized in merged:
                continue
            merged.append(normalized)
        return merged[-limit:]

    def _soft_topic_overlap(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        return self._jaccard(set(left.split(" / ")), set(right.split(" / ")))

    def _jaccard(self, left: Set[str], right: Set[str]) -> float:
        if not left or not right:
            return 0.0
        intersection = len(left & right)
        union = len(left | right)
        if union == 0:
            return 0.0
        return intersection / union

    def _load_json(self, path: str, default):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _save_json(self, path: str, payload):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
