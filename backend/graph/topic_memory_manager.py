"""
Topic-oriented memory management for Mini-OpenClaw.

The topic layer now centers on:
1. TopicSession: a topic thread inside one conversation session
2. TopicQA: one completed QA turn, including tool interactions

Prompt construction keeps the original session-history flow while adding
structured topic sessions and topic-aware retrieval.
"""

import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .memory import Message

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:  # pragma: no cover - optional dependency behavior
    OpenAIEmbeddings = None  # type: ignore


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

CHINESE_STOPWORDS = {
    "帮我",
    "一下",
    "这个",
    "那个",
    "现在",
    "需要",
    "想要",
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
    "信息",
    "情况",
    "详情",
    "介绍",
    "了解",
    "查询",
    "看看",
    "最近",
    "近期",
}

GENERIC_TOKEN_PREFIXES = (
    "我想了解下",
    "我想了解一下",
    "想了解下",
    "想了解一下",
    "了解下",
    "了解一下",
    "帮我查询下",
    "帮我查询一下",
    "帮我查下",
    "帮我看看",
    "请帮我",
    "请问",
    "我想",
    "想",
)

GENERIC_TOKEN_SUFFIXES = (
    "的情况",
    "的信息",
    "情况",
    "信息",
    "内容",
    "方面",
    "一下",
    "的",
)

GENERIC_RETRIEVAL_TERMS = {
    "了解",
    "查询",
    "看看",
    "query",
    "find",
    "search",
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
        "a股",
        "b股",
        "h股",
        "港股",
        "美股",
        "股市",
        "macd",
        "kdj",
        "boll",
        "rsi",
        "ema",
        "ma",
        "600036",
        "股票",
        "个股",
        "大盘",
        "板块",
        "指数",
        "上证",
        "深证",
        "沪深",
        "沪深300",
        "创业板",
        "科创板",
        "涨跌",
        "资金流向",
        "基金",
        "银行",
        "证券",
        "布林带",
        "均线",
        "技术面",
        "支撑位",
        "压力位",
        "财报",
        "估值",
        "行情",
        "成交量",
    },
    "research_writing": {
        "paper",
        "research",
        "summary",
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
        "prototype",
        "feature",
        "需求",
        "原型",
        "功能",
        "流程",
        "页面",
        "交互",
        "设计稿",
        "规划",
    },
}

INTENT_KEYWORDS = {
    "analysis": {
        "review",
        "analyze",
        "compare",
        "分析",
        "判断",
        "评估",
        "比较",
    },
    "implementation": {
        "build",
        "implement",
        "fix",
        "实现",
        "开发",
        "编写",
        "修改",
        "修复",
    },
    "planning": {
        "plan",
        "design",
        "architecture",
        "方案",
        "规划",
        "设计",
        "架构",
        "思路",
    },
    "explanation": {
        "why",
        "explain",
        "how",
        "解释",
        "说明",
        "讲解",
        "介绍",
    },
    "summarization": {
        "summary",
        "summarize",
        "总结",
        "归纳",
        "提炼",
        "摘要",
    },
    "retrieval": {
        "load",
        "find",
        "search",
        "query",
        "查找",
        "搜索",
        "检索",
        "查询",
        "了解",
    },
}

FOLLOW_UP_HINTS = {
    "继续",
    "刚才",
    "上一个",
    "上次",
    "这个",
    "那个",
    "它",
    "继续说",
    "继续做",
    "接着",
    "然后呢",
    "再展开",
}

WAITING_USER_HINTS = (
    "请告诉我",
    "请提供",
    "请确认",
    "请补充",
    "是否需要",
    "你希望",
    "您希望",
    "请问",
    "需要我继续",
    "?",
    "？",
)

BLOCKED_HINTS = (
    "无法",
    "失败",
    "错误",
    "没有权限",
    "超时",
    "连接",
    "blocked",
    "permission",
    "error",
    "failed",
    "timeout",
)


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
class TopicQA:
    qa_id: str
    domain: str
    topic: str
    intents: List[str] = field(default_factory=list)
    summary: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_names: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TopicSession:
    topic_session_id: str
    topic_id: str
    domain: str
    topic: str
    intents: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    status: str = "resolved"
    summary: str = ""
    qa_count: int = 0
    qas: List[TopicQA] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SessionTopicState:
    session_id: str
    version: int = 2
    current_topic: Optional[TopicIntent] = None
    current_topic_session_id: Optional[str] = None
    topic_sessions: List[TopicSession] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TopicPromptContext:
    analysis: Optional[TopicIntent]
    topic_shift: bool
    is_new_session: bool
    retrieved_topic_sessions: List[TopicSession]
    selected_qas: List[TopicQA]
    transition_notes: List[str]
    rendered: str


@dataclass
class TopicTurnUpdate:
    analysis: Optional[TopicIntent]
    topic_shift: bool
    topic_session_id: Optional[str]
    topic_session_status: Optional[str]


@dataclass
class RetrievalDocument:
    doc_id: str
    text: str
    tokens: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", value.lower())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized[:96]


class TopicMemoryManager:
    """Manage topic extraction, hybrid retrieval, and topic-session persistence."""

    def __init__(self, memory_dir: str, sessions_dir: str):
        self.memory_dir = os.path.abspath(memory_dir)
        self.sessions_dir = os.path.abspath(sessions_dir)
        self.session_topics_dir = os.path.join(self.sessions_dir, "_topic_state")

        os.makedirs(self.session_topics_dir, exist_ok=True)

        self._embedding_cache: Dict[str, List[float]] = {}
        self._embedder = self._create_embedder()

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
        topic = self._infer_topic(keywords, str(message), previous)
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
        history: Optional[Sequence[Message]] = None,
        prepare_state: bool = False,
    ) -> TopicPromptContext:
        """Prepare topic-aware retrieval context for the next model call."""
        state = self.load_session_state(session_id) if session_id else None
        history = list(history or [])
        previous = state.current_topic if state else None
        is_new_session = self._is_new_session(state, history)
        analysis = self.analyze_message(user_message, previous=previous)
        transition_notes: List[str] = []

        if not analysis:
            return TopicPromptContext(
                analysis=None,
                topic_shift=False,
                is_new_session=is_new_session,
                retrieved_topic_sessions=[],
                selected_qas=[],
                transition_notes=[],
                rendered="No active topic inferred yet.",
            )

        matched_topic_session, matched_score = self._find_best_topic_session(
            state.topic_sessions if state else [],
            analysis=analysis,
            query_text=user_message or analysis.topic,
        )
        matched_session_id = matched_topic_session.topic_session_id if matched_topic_session else None
        topic_shift = bool(
            state
            and state.current_topic_session_id
            and matched_session_id
            and state.current_topic_session_id != matched_session_id
        )

        if is_new_session:
            transition_notes.extend(
                [
                    "new_session_detected",
                    "load_agent_context",
                    "load_related_topic_sessions",
                ]
            )
        elif matched_topic_session:
            if state and state.current_topic_session_id == matched_topic_session.topic_session_id:
                transition_notes.append("continue_current_topic_session")
            else:
                transition_notes.append(
                    f"switch_to_topic_session={matched_topic_session.topic_session_id}"
                )
            transition_notes.append(f"topic_session_match_score={matched_score:.3f}")
            transition_notes.append("load_related_topic_sessions")
        else:
            transition_notes.append("prepare_new_topic_session")
            transition_notes.append("load_related_topic_sessions")

        retrieved_topic_sessions = self.retrieve_topic_sessions(
            analysis=analysis,
            exclude_session_id=None,
            current_session_state=state,
            current_topic_session_id=matched_session_id,
        )
        selected_qas = self.retrieve_topic_qas(
            analysis=analysis,
            user_message=user_message or analysis.topic,
            topic_sessions=retrieved_topic_sessions,
            limit=6,
        )
        rendered = self._render_prompt_context(
            analysis=analysis,
            topic_shift=topic_shift,
            is_new_session=is_new_session,
            topic_sessions=retrieved_topic_sessions,
            selected_qas=selected_qas,
            transition_notes=transition_notes,
        )

        return TopicPromptContext(
            analysis=analysis,
            topic_shift=topic_shift,
            is_new_session=is_new_session,
            retrieved_topic_sessions=retrieved_topic_sessions,
            selected_qas=selected_qas,
            transition_notes=transition_notes,
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
        """Compute a lightweight semantic similarity based on overlap."""
        keyword_overlap = self._jaccard(set(left.keywords), set(right.keywords))
        intent_overlap = self._jaccard(
            self._effective_intents(left.intents),
            self._effective_intents(right.intents),
        )
        domain_score = 1.0 if left.domain == right.domain else 0.0
        topic_score = 1.0 if left.topic == right.topic else self._soft_topic_overlap(left.topic, right.topic)
        return round(
            keyword_overlap * 0.45
            + intent_overlap * 0.2
            + domain_score * 0.15
            + topic_score * 0.2,
            3,
        )

    def retrieve_topic_sessions(
        self,
        analysis: TopicIntent,
        exclude_session_id: Optional[str] = None,
        current_session_state: Optional[SessionTopicState] = None,
        current_topic_session_id: Optional[str] = None,
        limit: int = 4,
    ) -> List[TopicSession]:
        """Retrieve topic sessions from all saved conversations with hybrid search."""
        topic_sessions: Dict[str, TopicSession] = {}
        documents: List[RetrievalDocument] = []

        for session_id, topic_session in self._iter_all_topic_sessions(
            exclude_session_id=exclude_session_id,
            current_session_state=current_session_state,
        ):
            doc_id = f"{session_id}:{topic_session.topic_session_id}"
            topic_sessions[doc_id] = topic_session
            text = self._build_topic_session_document(topic_session)
            documents.append(
                RetrievalDocument(
                    doc_id=doc_id,
                    text=text,
                    tokens=self._tokenize(text),
                    metadata={
                        "session_id": session_id,
                        "topic_session_id": topic_session.topic_session_id,
                    },
                )
            )

        if not documents:
            return []

        query_text = self._build_retrieval_query(analysis, analysis.topic)
        fused = self._hybrid_rank(documents, query_text=query_text)
        ordered: List[TopicSession] = []
        seen: Set[str] = set()
        current_doc_id = None
        if current_session_state and current_topic_session_id:
            current_doc_id = f"{current_session_state.session_id}:{current_topic_session_id}"

        for doc_id, _ in fused:
            if doc_id in seen:
                continue
            topic_session = topic_sessions.get(doc_id)
            if not topic_session:
                continue
            if self._score_topic_session(topic_session, analysis) <= 0.16 and doc_id != current_doc_id:
                continue
            ordered.append(topic_session)
            seen.add(doc_id)
            if len(ordered) >= limit:
                break

        if current_doc_id and current_doc_id not in seen:
            current_session = topic_sessions.get(current_doc_id)
            if current_session:
                ordered.insert(0, current_session)

        return ordered[:limit]

    def retrieve_topic_qas(
        self,
        analysis: TopicIntent,
        user_message: str,
        topic_sessions: Sequence[TopicSession],
        limit: int = 6,
    ) -> List[TopicQA]:
        """Retrieve the most relevant QAs inside the matched topic sessions."""
        documents: List[RetrievalDocument] = []
        qas_by_id: Dict[str, TopicQA] = {}

        for topic_session in topic_sessions:
            for qa in topic_session.qas:
                text = self._build_topic_qa_document(topic_session, qa)
                doc_id = f"{topic_session.topic_session_id}:{qa.qa_id}"
                qas_by_id[doc_id] = qa
                documents.append(
                    RetrievalDocument(
                        doc_id=doc_id,
                        text=text,
                        tokens=self._tokenize(text),
                        metadata={"topic_session_id": topic_session.topic_session_id},
                    )
                )

        if not documents:
            return []

        query_text = self._build_retrieval_query(analysis, user_message)
        fused = self._hybrid_rank(documents, query_text=query_text)
        ordered: List[TopicQA] = []
        for doc_id, _ in fused:
            qa = qas_by_id.get(doc_id)
            if not qa:
                continue
            ordered.append(qa)
            if len(ordered) >= limit:
                break
        return ordered

    def remember_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_reply: str,
        turn_messages: Sequence[Message],
    ) -> TopicTurnUpdate:
        """Persist the new turn into topic sessions and long-term topic cards."""
        state = self.load_session_state(session_id)
        previous = state.current_topic
        analysis = self.analyze_message(user_message, previous=previous)
        if not analysis:
            return TopicTurnUpdate(
                analysis=previous,
                topic_shift=False,
                topic_session_id=state.current_topic_session_id,
                topic_session_status=None,
            )

        turn_summary = self._summarize_turn(user_message, assistant_reply, turn_messages)
        topic_session, match_score = self._find_best_topic_session(
            state.topic_sessions,
            analysis=analysis,
            query_text=user_message,
        )

        if topic_session is None or match_score < 0.32:
            topic_session = TopicSession(
                topic_session_id=f"{analysis.topic_id}-{len(state.topic_sessions) + 1}",
                topic_id=analysis.topic_id,
                domain=analysis.domain,
                topic=analysis.topic,
                intents=list(analysis.intents),
                keywords=list(analysis.keywords),
                status="waiting_user",
                summary=turn_summary,
                qa_count=0,
            )
            state.topic_sessions.append(topic_session)

        topic_shift = bool(
            state.current_topic_session_id
            and state.current_topic_session_id != topic_session.topic_session_id
        )

        qa = self._build_topic_qa(
            analysis=analysis,
            turn_messages=turn_messages,
            summary=turn_summary,
        )
        topic_session.qas.append(qa)
        topic_session.qa_count += 1
        topic_session.intents = self._merge_unique(topic_session.intents, analysis.intents, limit=8)
        topic_session.keywords = self._merge_unique(topic_session.keywords, analysis.keywords, limit=12)
        topic_session.summary = self._merge_summaries(topic_session.summary, turn_summary)
        topic_session.status = self._infer_topic_session_status(
            user_message=user_message,
            assistant_reply=assistant_reply,
            turn_messages=turn_messages,
        )
        topic_session.updated_at = datetime.now().isoformat()

        state.current_topic = analysis
        state.current_topic_session_id = topic_session.topic_session_id
        state.updated_at = datetime.now().isoformat()
        self.save_session_state(state)

        return TopicTurnUpdate(
            analysis=analysis,
            topic_shift=topic_shift,
            topic_session_id=topic_session.topic_session_id,
            topic_session_status=topic_session.status,
        )

    def load_session_state(self, session_id: Optional[str]) -> SessionTopicState:
        """Load the topic state for a session, migrating legacy slice payloads when needed."""
        if not session_id:
            return SessionTopicState(session_id="")

        path = os.path.join(self.session_topics_dir, f"{session_id}.json")
        if not os.path.exists(path):
            return SessionTopicState(session_id=session_id)

        payload = self._load_json(path, {})
        if payload.get("version") == 2 or "topic_sessions" in payload:
            return self._load_v2_session_state(payload, session_id=session_id)
        return self._migrate_legacy_state(payload, session_id=session_id)

    def save_session_state(self, state: SessionTopicState):
        """Persist per-session topic sessions and current topic state."""
        if not state.session_id:
            return

        payload = {
            "version": 2,
            "session_id": state.session_id,
            "current_topic": asdict(state.current_topic) if state.current_topic else None,
            "current_topic_session_id": state.current_topic_session_id,
            "topic_sessions": [asdict(item) for item in state.topic_sessions],
            "updated_at": state.updated_at,
        }
        path = os.path.join(self.session_topics_dir, f"{state.session_id}.json")
        self._save_json(path, payload)

    def _tokenize(self, text: str) -> List[str]:
        raw_tokens = re.findall(
            r"[A-Za-z][A-Za-z0-9_+-]*(?:\u80a1|\u5e02|\u6307\u6570)?|"
            r"[\u4e00-\u9fff]{2,}|"
            r"[0-9]{3,}",
            text,
        )
        tokens: List[str] = []
        for token in raw_tokens:
            normalized = self._normalize_candidate_token(token)
            if not normalized:
                continue
            if normalized in ENGLISH_STOPWORDS or normalized in CHINESE_STOPWORDS:
                continue
            if len(normalized) == 1:
                continue
            tokens.append(normalized)
        return tokens

    def _normalize_candidate_token(self, token: str) -> str:
        normalized = token.strip().lower()
        if not normalized:
            return ""

        changed = True
        while changed and normalized:
            changed = False
            for prefix in GENERIC_TOKEN_PREFIXES:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):].strip()
                    changed = True
            for suffix in GENERIC_TOKEN_SUFFIXES:
                if normalized.endswith(suffix):
                    normalized = normalized[:-len(suffix)].strip()
                    changed = True

        return normalized

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
        is_indicator = 1 if token.upper() in {"MACD", "KDJ", "RSI", "BOLL", "EMA", "MA"} else 0
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
            score = len(token_set & keywords)
            score += sum(1 for keyword in keywords if keyword in normalized_message)
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

    def _extract_known_keywords(self, normalized_message: str) -> List[str]:
        keywords: List[str] = []
        known_terms = sorted(
            {
                term.lower()
                for group in (DOMAIN_KEYWORDS, INTENT_KEYWORDS)
                for terms in group.values()
                for term in terms
            },
            key=lambda item: (-len(item), item),
        )
        for term in known_terms:
            if term in GENERIC_RETRIEVAL_TERMS:
                continue
            if term in normalized_message and term not in keywords:
                keywords.append(term)
        return keywords

    def _score_topic_session(self, topic_session: TopicSession, analysis: TopicIntent) -> float:
        keyword_score = self._jaccard(set(topic_session.keywords), set(analysis.keywords))
        intent_score = self._jaccard(
            self._effective_intents(topic_session.intents),
            self._effective_intents(analysis.intents),
        )
        domain_score = self._domain_match_score(topic_session.domain, analysis.domain)
        topic_score = self._soft_topic_overlap(topic_session.topic, analysis.topic)
        summary_score = self._soft_topic_overlap(topic_session.summary, analysis.topic)
        if (
            analysis.domain == "general_dialog"
            and keyword_score == 0
            and topic_score < 0.35
            and summary_score < 0.2
        ):
            return 0.0
        return round(
            keyword_score * 0.35
            + intent_score * 0.15
            + domain_score * 0.15
            + topic_score * 0.2
            + summary_score * 0.15,
            3,
        )

    def _effective_intents(self, intents: Sequence[str]) -> Set[str]:
        return {intent for intent in intents if intent and intent != "general"}

    def _domain_match_score(self, left_domain: str, right_domain: str) -> float:
        if left_domain != right_domain:
            return 0.0
        if left_domain == "general_dialog":
            return 0.2
        return 1.0

    def _render_prompt_context(
        self,
        analysis: TopicIntent,
        topic_shift: bool,
        is_new_session: bool,
        topic_sessions: Sequence[TopicSession],
        selected_qas: Sequence[TopicQA],
        transition_notes: Sequence[str],
    ) -> str:
        lines = [
            "## Current Topic Analysis",
            f"- domain: {analysis.domain}",
            f"- topic: {analysis.topic}",
            f"- intents: {', '.join(analysis.intents)}",
            f"- keywords: {', '.join(analysis.keywords)}",
            f"- new_session_detected: {'yes' if is_new_session else 'no'}",
            f"- topic_shift_detected: {'yes' if topic_shift else 'no'}",
            f"- prompt_flow_actions: {', '.join(transition_notes) or 'none'}",
            "",
            "## Retrieved Topic Sessions",
        ]

        if not topic_sessions:
            lines.append("- No related topic sessions found.")
        else:
            for topic_session in topic_sessions:
                lines.append(
                    f"- [{topic_session.domain}] {topic_session.topic} "
                    f"(qas={topic_session.qa_count}, status={topic_session.status})"
                )
                if topic_session.summary:
                    lines.append(f"  summary: {topic_session.summary[:220]}")

        lines.extend(["", "## Retrieved Topic QAs"])
        if not selected_qas:
            lines.append("- No strongly related topic QAs found.")
        else:
            for qa in selected_qas[:6]:
                lines.append(
                    f"- [{qa.domain}] {qa.topic}: {qa.summary[:240]}"
                )

        return "\n".join(lines).strip()

    def _is_new_session(
        self,
        state: Optional[SessionTopicState],
        history: Sequence[Message],
    ) -> bool:
        if not history:
            return True
        if not state:
            return True
        return not state.topic_sessions and state.current_topic is None

    def _summarize_turn(
        self,
        user_message: str,
        assistant_reply: str,
        turn_messages: Sequence[Message],
    ) -> str:
        user_snippet = " ".join(user_message.split())[:180]
        assistant_snippet = " ".join((assistant_reply or "").split())[:220]
        tools = [msg.name for msg in turn_messages if msg.role == "tool" and msg.name]
        tool_names = self._merge_unique([], tools, limit=4)
        tool_summary = f"; tools={', '.join(tool_names)}" if tool_names else ""
        return f"user={user_snippet}; assistant={assistant_snippet}{tool_summary}"

    def _merge_summaries(self, existing: str, new_summary: str) -> str:
        if not existing:
            return new_summary[:420]
        if new_summary in existing:
            return existing[:420]
        return f"{existing} | {new_summary}"[:420]

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
        left_tokens = set(self._tokenize(left)) or set(left.split(" / "))
        right_tokens = set(self._tokenize(right)) or set(right.split(" / "))
        return self._jaccard(left_tokens, right_tokens)

    def _jaccard(self, left: Set[str], right: Set[str]) -> float:
        if not left or not right:
            return 0.0
        union = len(left | right)
        if union == 0:
            return 0.0
        return len(left & right) / union

    def _load_json(self, path: str, default):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _save_json(self, path: str, payload):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _load_v2_session_state(self, payload: Dict[str, Any], session_id: str) -> SessionTopicState:
        current_topic_payload = payload.get("current_topic")
        current_topic = TopicIntent(**current_topic_payload) if current_topic_payload else None
        topic_sessions = [
            self._load_topic_session(item)
            for item in payload.get("topic_sessions", [])
            if isinstance(item, dict)
        ]
        return SessionTopicState(
            session_id=payload.get("session_id", session_id),
            version=int(payload.get("version", 2)),
            current_topic=current_topic,
            current_topic_session_id=payload.get("current_topic_session_id"),
            topic_sessions=topic_sessions,
            updated_at=payload.get("updated_at", datetime.now().isoformat()),
        )

    def _load_topic_session(self, payload: Dict[str, Any]) -> TopicSession:
        qas = [
            self._load_topic_qa(item, topic_session_payload=payload)
            for item in payload.get("qas", [])
            if isinstance(item, dict)
        ]
        return TopicSession(
            topic_session_id=str(payload.get("topic_session_id", "")),
            topic_id=str(payload.get("topic_id", "")),
            domain=str(payload.get("domain", "general_dialog")),
            topic=str(payload.get("topic", "general topic")),
            intents=[str(item) for item in payload.get("intents", [])],
            keywords=[str(item) for item in payload.get("keywords", [])],
            status=str(payload.get("status", "resolved")),
            summary=str(payload.get("summary", "")),
            qa_count=int(payload.get("qa_count", len(qas))),
            qas=qas,
            created_at=str(payload.get("created_at", datetime.now().isoformat())),
            updated_at=str(payload.get("updated_at", datetime.now().isoformat())),
        )

    def _load_topic_qa(
        self,
        payload: Dict[str, Any],
        topic_session_payload: Optional[Dict[str, Any]] = None,
    ) -> TopicQA:
        topic_session_payload = topic_session_payload or {}
        return TopicQA(
            qa_id=str(payload.get("qa_id", "")),
            domain=str(
                payload.get("domain")
                or topic_session_payload.get("domain")
                or "general_dialog"
            ),
            topic=str(
                payload.get("topic")
                or topic_session_payload.get("topic")
                or "general topic"
            ),
            intents=[str(item) for item in payload.get("intents", [])]
            or [str(item) for item in topic_session_payload.get("intents", [])],
            summary=str(payload.get("summary", "")),
            messages=[
                item for item in payload.get("messages", []) if isinstance(item, dict)
            ],
            tool_names=[str(item) for item in payload.get("tool_names", [])],
            created_at=str(payload.get("created_at", datetime.now().isoformat())),
            updated_at=str(payload.get("updated_at", datetime.now().isoformat())),
        )

    def _migrate_legacy_state(self, payload: Dict[str, Any], session_id: str) -> SessionTopicState:
        current_topic_payload = payload.get("current_topic")
        current_topic = TopicIntent(**current_topic_payload) if current_topic_payload else None
        topic_sessions: List[TopicSession] = []

        for index, legacy_slice in enumerate(payload.get("slices", []), start=1):
            if not isinstance(legacy_slice, dict):
                continue
            qa_summary = str(
                legacy_slice.get("finalized_summary")
                or legacy_slice.get("summary")
                or "Migrated legacy topic summary."
            )
            qa = TopicQA(
                qa_id=f"migrated-qa-{index}",
                domain=str(legacy_slice.get("domain", "general_dialog")),
                topic=str(legacy_slice.get("topic", "general topic")),
                intents=[str(item) for item in legacy_slice.get("intents", [])],
                summary=qa_summary,
                messages=[
                    {
                        "role": "assistant",
                        "content": f"[Migrated Topic Summary]\n{qa_summary}",
                        "timestamp": legacy_slice.get("updated_at", ""),
                        "tool_calls": None,
                        "name": None,
                        "tool_call_id": None,
                    }
                ],
                tool_names=[],
                created_at=str(legacy_slice.get("started_at", datetime.now().isoformat())),
                updated_at=str(legacy_slice.get("updated_at", datetime.now().isoformat())),
            )
            topic_sessions.append(
                TopicSession(
                    topic_session_id=str(legacy_slice.get("slice_id", f"migrated-topic-{index}")),
                    topic_id=str(legacy_slice.get("topic_id", f"migrated-topic-{index}")),
                    domain=str(legacy_slice.get("domain", "general_dialog")),
                    topic=str(legacy_slice.get("topic", "general topic")),
                    intents=[str(item) for item in legacy_slice.get("intents", [])],
                    keywords=[str(item) for item in legacy_slice.get("keywords", [])],
                    status="waiting_user"
                    if legacy_slice.get("status") == "active"
                    else "resolved",
                    summary=qa_summary,
                    qa_count=1,
                    qas=[qa],
                    created_at=str(legacy_slice.get("started_at", datetime.now().isoformat())),
                    updated_at=str(legacy_slice.get("updated_at", datetime.now().isoformat())),
                )
            )

        current_topic_session_id = payload.get("current_topic_session_id")
        if not current_topic_session_id and topic_sessions:
            current_topic_session_id = topic_sessions[-1].topic_session_id

        return SessionTopicState(
            session_id=payload.get("session_id", session_id),
            version=2,
            current_topic=current_topic,
            current_topic_session_id=current_topic_session_id,
            topic_sessions=topic_sessions,
            updated_at=payload.get("updated_at", datetime.now().isoformat()),
        )

    def _find_best_topic_session(
        self,
        topic_sessions: Sequence[TopicSession],
        analysis: TopicIntent,
        query_text: str,
    ) -> Tuple[Optional[TopicSession], float]:
        if not topic_sessions:
            return None, 0.0

        query_follow_up = any(hint in query_text for hint in FOLLOW_UP_HINTS)
        scored: List[Tuple[float, TopicSession]] = []
        for topic_session in topic_sessions:
            score = self._score_topic_session(topic_session, analysis)
            if query_follow_up:
                score += 0.12 if topic_session.status == "waiting_user" else 0.0
                if topic_session.qas:
                    score += 0.04
            if topic_session.topic_id == analysis.topic_id:
                score += 0.08
            scored.append((round(min(1.0, score), 3), topic_session))

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_session = scored[0]
        if best_score < 0.18:
            return None, best_score
        return best_session, best_score

    def _build_topic_qa(
        self,
        analysis: TopicIntent,
        turn_messages: Sequence[Message],
        summary: str,
    ) -> TopicQA:
        now = datetime.now().isoformat()
        tool_names = [msg.name for msg in turn_messages if msg.role == "tool" and msg.name]
        return TopicQA(
            qa_id=f"qa-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            domain=analysis.domain,
            topic=analysis.topic,
            intents=list(analysis.intents),
            summary=summary,
            messages=[asdict(msg) for msg in turn_messages],
            tool_names=self._merge_unique([], tool_names, limit=8),
            created_at=now,
            updated_at=now,
        )

    def _infer_topic_session_status(
        self,
        user_message: str,
        assistant_reply: str,
        turn_messages: Sequence[Message],
    ) -> str:
        normalized_reply = (assistant_reply or "").strip().lower()
        if not normalized_reply:
            return "resolved"

        if any(hint in normalized_reply for hint in BLOCKED_HINTS):
            return "blocked"

        tool_errors = [
            msg
            for msg in turn_messages
            if msg.role == "tool" and "error" in (msg.content or "").lower()
        ]
        if tool_errors:
            return "blocked"

        if any(hint in assistant_reply for hint in WAITING_USER_HINTS):
            return "waiting_user"

        if any(hint in user_message for hint in FOLLOW_UP_HINTS):
            return "waiting_user"

        return "resolved"

    def _iter_all_topic_sessions(
        self,
        exclude_session_id: Optional[str],
        current_session_state: Optional[SessionTopicState],
    ) -> List[Tuple[str, TopicSession]]:
        collected: List[Tuple[str, TopicSession]] = []
        seen_sessions: Set[str] = set()

        if current_session_state and current_session_state.session_id:
            seen_sessions.add(current_session_state.session_id)
            if not exclude_session_id or current_session_state.session_id != exclude_session_id:
                for topic_session in current_session_state.topic_sessions:
                    collected.append((current_session_state.session_id, topic_session))

        for filename in os.listdir(self.session_topics_dir):
            if not filename.endswith(".json"):
                continue
            session_id = filename[:-5]
            if session_id in seen_sessions:
                continue
            if exclude_session_id and session_id == exclude_session_id:
                continue
            state = self.load_session_state(session_id)
            for topic_session in state.topic_sessions:
                collected.append((session_id, topic_session))

        return collected

    def _build_topic_session_document(self, topic_session: TopicSession) -> str:
        qa_summaries = "\n".join(qa.summary for qa in topic_session.qas[-5:] if qa.summary)
        return "\n".join(
            [
                topic_session.domain,
                topic_session.topic,
                " ".join(topic_session.intents),
                " ".join(topic_session.keywords),
                topic_session.summary,
                qa_summaries,
                " ".join(topic_session.status.split("_")),
            ]
        ).strip()

    def _build_topic_qa_document(self, topic_session: TopicSession, qa: TopicQA) -> str:
        message_text = self._flatten_message_payloads(qa.messages)
        return "\n".join(
            [
                qa.domain or topic_session.domain,
                qa.topic or topic_session.topic,
                " ".join(qa.intents),
                qa.summary,
                " ".join(qa.tool_names),
                message_text,
            ]
        ).strip()

    def _flatten_message_payloads(self, messages: Sequence[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for payload in messages:
            role = str(payload.get("role", ""))
            content = str(payload.get("content", ""))
            name = str(payload.get("name", ""))
            label = f"{role}:{name}" if name else role
            snippet = " ".join(content.split())
            if snippet:
                parts.append(f"{label} {snippet[:240]}")
        return "\n".join(parts)

    def _build_retrieval_query(self, analysis: TopicIntent, user_message: str) -> str:
        return "\n".join(
            [
                analysis.domain,
                analysis.topic,
                " ".join(analysis.intents),
                " ".join(analysis.keywords),
                user_message,
            ]
        ).strip()

    def _hybrid_rank(
        self,
        documents: Sequence[RetrievalDocument],
        query_text: str,
        limit: int = 20,
    ) -> List[Tuple[str, float]]:
        if not documents:
            return []

        bm25_scores = self._bm25_scores(documents, query_text=query_text)
        vector_scores = self._vector_scores(documents, query_text=query_text)
        ranked_sources = []

        ranked_bm25 = [
            doc_id
            for doc_id, score in sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)
            if score > 0
        ]
        ranked_vector = [
            doc_id
            for doc_id, score in sorted(vector_scores.items(), key=lambda item: item[1], reverse=True)
            if score > 0
        ]
        if ranked_bm25:
            ranked_sources.append(ranked_bm25)
        if ranked_vector:
            ranked_sources.append(ranked_vector)
        if not ranked_sources:
            ranked_sources.append([doc.doc_id for doc in documents])

        fused = self._rrf_merge(ranked_sources)
        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        return ordered[:limit]

    def _bm25_scores(
        self,
        documents: Sequence[RetrievalDocument],
        query_text: str,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> Dict[str, float]:
        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return {doc.doc_id: 0.0 for doc in documents}

        doc_freqs: Counter[str] = Counter()
        doc_token_counts: Dict[str, Counter[str]] = {}
        total_doc_len = 0

        for doc in documents:
            counter = Counter(doc.tokens)
            doc_token_counts[doc.doc_id] = counter
            total_doc_len += len(doc.tokens)
            for token in counter:
                doc_freqs[token] += 1

        avg_doc_len = total_doc_len / max(len(documents), 1)
        scores: Dict[str, float] = {}
        doc_count = len(documents)
        unique_query_tokens = list(dict.fromkeys(query_tokens))

        for doc in documents:
            score = 0.0
            doc_len = max(len(doc.tokens), 1)
            term_counts = doc_token_counts[doc.doc_id]
            for token in unique_query_tokens:
                freq = term_counts.get(token, 0)
                if freq == 0:
                    continue
                df = doc_freqs.get(token, 0)
                idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1.0))
                score += idf * numerator / max(denominator, 1e-9)
            scores[doc.doc_id] = round(score, 6)

        return scores

    def _vector_scores(
        self,
        documents: Sequence[RetrievalDocument],
        query_text: str,
    ) -> Dict[str, float]:
        embedding_scores = self._embedding_scores(documents, query_text=query_text)
        if embedding_scores:
            return embedding_scores
        return self._local_vector_scores(documents, query_text=query_text)

    def _embedding_scores(
        self,
        documents: Sequence[RetrievalDocument],
        query_text: str,
    ) -> Dict[str, float]:
        if self._embedder is None:
            return {}

        try:
            query_vector = self._embed_text(query_text)
            missing_texts = []
            missing_keys = []
            for doc in documents:
                cache_key = self._embedding_cache_key(doc.text)
                if cache_key not in self._embedding_cache:
                    missing_keys.append(cache_key)
                    missing_texts.append(doc.text)

            if missing_texts:
                vectors = self._embedder.embed_documents(missing_texts)
                for cache_key, vector in zip(missing_keys, vectors):
                    self._embedding_cache[cache_key] = [float(value) for value in vector]

            scores: Dict[str, float] = {}
            for doc in documents:
                doc_vector = self._embedding_cache.get(self._embedding_cache_key(doc.text))
                if not doc_vector:
                    continue
                scores[doc.doc_id] = round(self._cosine_similarity(query_vector, doc_vector), 6)
            return scores
        except Exception:
            self._embedder = None
            return {}

    def _local_vector_scores(
        self,
        documents: Sequence[RetrievalDocument],
        query_text: str,
    ) -> Dict[str, float]:
        query_vector = self._build_local_vector(query_text)
        if not query_vector:
            return {doc.doc_id: 0.0 for doc in documents}
        scores: Dict[str, float] = {}
        for doc in documents:
            doc_vector = self._build_local_vector(doc.text)
            scores[doc.doc_id] = round(self._sparse_cosine_similarity(query_vector, doc_vector), 6)
        return scores

    def _build_local_vector(self, text: str) -> Dict[str, float]:
        tokens = self._tokenize(text)
        normalized = re.sub(r"\s+", "", text.lower())
        char_terms = [
            normalized[index:index + 2]
            for index in range(max(0, len(normalized) - 1))
            if normalized[index:index + 2].strip()
        ]
        features = Counter(tokens + char_terms)
        return {term: float(value) for term, value in features.items()}

    def _sparse_cosine_similarity(
        self,
        left: Dict[str, float],
        right: Dict[str, float],
    ) -> float:
        if not left or not right:
            return 0.0
        shared = set(left) & set(right)
        numerator = sum(left[token] * right[token] for token in shared)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _rrf_merge(
        self,
        ranked_sources: Sequence[Sequence[str]],
        k: int = 60,
    ) -> Dict[str, float]:
        fused: Dict[str, float] = {}
        for ranked_ids in ranked_sources:
            for rank, doc_id in enumerate(ranked_ids, start=1):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
        return fused

    def _create_embedder(self):
        if OpenAIEmbeddings is None:
            return None
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None

        kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "model": os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
        }
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        if base_url:
            kwargs["base_url"] = base_url
        try:
            return OpenAIEmbeddings(**kwargs)
        except Exception:
            return None

    def _embed_text(self, text: str) -> List[float]:
        cache_key = self._embedding_cache_key(text)
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            return cached
        if self._embedder is None:
            return []
        vector = [float(value) for value in self._embedder.embed_query(text)]
        self._embedding_cache[cache_key] = vector
        return vector

    def _embedding_cache_key(self, text: str) -> str:
        return _slugify(text[:120] + str(len(text)))

    def _cosine_similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)
