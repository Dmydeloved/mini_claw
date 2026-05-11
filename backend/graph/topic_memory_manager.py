"""
Topic-oriented three-layer memory management for Mini-OpenClaw.

This module keeps the original agent/session architecture intact while
replacing the old topic-session memory with a simpler MVP:

Experience -> Segment -> QA

SegmentRelation stores graph links between segments, and RuntimeState
stores per-conversation execution state.
"""

import hashlib
import json
import math
import os
import re
import sqlite3
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from .memory import Message

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:  # pragma: no cover - optional dependency behavior
    OpenAIEmbeddings = None  # type: ignore


QAStatus = Literal["active", "archived"]
SegmentStatus = Literal["open", "completed", "interrupted", "inherited", "archived"]
ExperienceStatus = Literal["in_progress", "completed", "archived"]
DomainType = Literal[
    "general",
    "weather",
    "travel",
    "finance_analysis",
    "software_engineering",
    "research_writing",
    "product_planning",
]
IntentType = Literal[
    "general_analysis",
    "weather_lookup",
    "retrieval",
    "planning",
    "implementation",
    "explanation",
    "summarization",
    "comparison",
    "troubleshooting",
]
RelationToPreviousType = Literal["new_topic", "follow_up", "resume_previous", "switch_topic"]
SegmentRelationType = Literal[
    "continue_from",
    "refines",
    "contradicts",
]
TurnActionType = Literal[
    "append_current_segment",
    "fork_new_segment",
    "resume_historical_segment",
    "switch_experience",
]

DOMAIN_ENUM_VALUES: Tuple[DomainType, ...] = (
    "general",
    "weather",
    "travel",
    "finance_analysis",
    "software_engineering",
    "research_writing",
    "product_planning",
)
INTENT_ENUM_VALUES: Tuple[IntentType, ...] = (
    "general_analysis",
    "weather_lookup",
    "retrieval",
    "planning",
    "implementation",
    "explanation",
    "summarization",
    "comparison",
    "troubleshooting",
)
RELATION_ENUM_VALUES: Tuple[RelationToPreviousType, ...] = (
    "new_topic",
    "follow_up",
    "resume_previous",
    "switch_topic",
)
FINALIZED_SEGMENT_STATUSES: Set[str] = {
    "completed",
    "interrupted",
    "inherited",
    "archived",
}

SEGMENT_SUMMARY_QA_INTERVAL = 3
EXPERIENCE_SUMMARY_QA_MILESTONES: Tuple[int, ...] = (6, 12, 20, 40)
EXPERIENCE_SUMMARY_FINALIZED_SEGMENT_INTERVAL = 2
EXPERIENCE_SUMMARY_RELATION_INTERVAL = 2


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
    "memory",
    "topic",
    "segment",
    "experience",
    "session",
    "conversation",
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
    "系统",
    "会话",
    "主题",
    "上下文",
    "记忆",
    "经验",
    "信息",
    "情况",
    "帮忙",
}

GENERIC_PREFIXES = (
    "帮我",
    "请帮我",
    "麻烦帮我",
    "我想",
    "想",
    "请问",
    "可以",
    "能不能",
    "帮忙",
)

GENERIC_SUFFIXES = (
    "一下",
    "一下子",
    "看看",
    "内容",
    "信息",
    "情况",
    "问题",
)

RESUME_HINTS = (
    "继续刚才",
    "回到之前",
    "接着上次",
    "刚才那个继续",
    "继续前面的",
    "继续刚刚",
    "继续一下",
    "继续",
    "刚才",
    "上次",
    "接着",
    "前面的",
)

PAUSE_HINTS = (
    "先不聊这个",
    "换个话题",
    "先看另一个问题",
    "等一下",
    "先暂停",
)

BRANCH_HINTS = (
    "分支",
    "探索",
    "另一种方案",
    "另一个方案",
    "备选方案",
)

DETAIL_HINTS = (
    "明天",
    "后天",
    "今天",
    "下周",
    "周末",
    "预算",
    "偏好",
    "时间",
    "两天",
    "三天",
    "一天",
    "住宿",
    "交通",
    "出发",
    "返回",
    "自然风光",
    "亲子",
    "老人",
    "美食",
    "历史",
)

QUESTION_HINTS = (
    "?",
    "？",
    "请确认",
    "是否需要",
    "需要我",
    "你希望",
    "您希望",
    "请告诉我",
)

CONTRADICTION_HINTS = (
    "不要",
    "不需要",
    "取消",
    "改成",
    "改为",
    "相反",
    "而不是",
    "不是",
)

REFINE_HINTS = (
    "细化",
    "展开",
    "实现",
    "验证",
    "补充",
    "进一步",
    "详细",
    "具体",
    "排查",
)

WEATHER_TERMS = {"天气", "气温", "温度", "forecast", "weather", "降雨", "晴", "阴", "多云"}
TRAVEL_TERMS = {"旅游", "旅行", "攻略", "行程", "景点", "住宿", "自由行"}
FINANCE_TERMS = {
    "a股",
    "港股",
    "美股",
    "股票",
    "基金",
    "证券",
    "大盘",
    "行情",
    "指数",
    "财报",
}
SOFTWARE_TERMS = {
    "python",
    "java",
    "golang",
    "typescript",
    "javascript",
    "bug",
    "debug",
    "代码",
    "后端",
    "前端",
    "接口",
    "数据库",
    "脚本",
}
RESEARCH_TERMS = {
    "论文",
    "研究",
    "综述",
    "摘要",
    "框架",
    "方法",
    "实验",
    "写作",
    "报告",
    "课题",
}
PRODUCT_TERMS = {
    "prd",
    "需求",
    "原型",
    "功能",
    "流程",
    "页面",
    "交互",
    "产品",
    "设计稿",
    "规划",
}

DOMAIN_DESCRIPTIONS: Dict[DomainType, str] = {
    "general": "通用问题，不明显属于某个垂直领域。",
    "weather": "天气、气温、降雨、气象预测相关任务。",
    "travel": "旅行、旅游攻略、行程安排、景点住宿相关任务。",
    "finance_analysis": "股票、基金、行情、财报、技术分析相关任务。",
    "software_engineering": "代码、接口、脚本、调试、系统实现相关任务。",
    "research_writing": "论文、研究、报告、综述、学术写作相关任务。",
    "product_planning": "需求、PRD、功能方案、交互流程、产品规划相关任务。",
}

INTENT_DESCRIPTIONS: Dict[IntentType, str] = {
    "general_analysis": "分析、判断、评估、泛化问题求解。",
    "weather_lookup": "天气查询与天气结果说明。",
    "retrieval": "查找、检索、搜索已有信息。",
    "planning": "制定方案、计划、攻略、设计路线。",
    "implementation": "实现、修改、修复、开发具体产出。",
    "explanation": "解释、说明、讲解概念或原因。",
    "summarization": "总结、归纳、提炼内容。",
    "comparison": "比较多个候选项、方案或对象。",
    "troubleshooting": "定位故障、排查问题、找原因。",
}

TOOL_ROLE_MAP = {
    "get_weather": "根据用户指定的地点和时间获取天气数据",
    "fetch_url": "抓取网页内容并提取可读文本",
    "python_repl": "执行 Python 代码用于计算、解析和验证结果",
    "read_file": "读取本地文件、技能说明或项目内容",
    "write_file": "写入或更新本地文件",
    "terminal": "执行命令行操作、检查环境或运行脚本",
    "search_knowledge_base": "从本地知识库检索相关资料",
}

ALLOWED_RELATION_EXPANSION = {
    "continue_from",
    "refines",
    "contradicts",
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", value.lower())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized[:96] or "general"


def _make_id(prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def summarize_tools_from_qas(qas: List["QA"]) -> str:
    ordered: List[ToolCallRecord] = []
    seen: Set[Tuple[str, str]] = set()

    for qa in qas:
        for tool in qa.tools:
            key = (tool.tool_name, tool.role)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(tool)

    if not ordered:
        return "本阶段未记录外部工具调用。"

    parts = [
        f"{tool.tool_name}：{tool.role}"
        for tool in ordered
    ]
    return "工具使用：" + "；".join(parts) + "。"


@dataclass
class ToolCallRecord:
    tool_name: str
    role: str


@dataclass
class QA:
    qa_id: str
    conversation_id: str
    timestamp: str
    user_input: str
    assistant_output: str
    tools: List[ToolCallRecord]
    domain: DomainType
    topic: str
    intent: IntentType
    goal: str
    entities: List[str]
    facts: List[str]
    constraints: List[str]
    segment_id: str
    experience_id: str
    status: QAStatus = "active"
    confidence: float = 1.0


@dataclass
class SegmentMemory:
    facts: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)


@dataclass
class Segment:
    segment_id: str
    experience_id: str
    domain: DomainType
    topic: str
    intent: IntentType
    goal: str
    qa_ids: List[str]
    status: SegmentStatus
    summary: str
    memory: SegmentMemory
    created_at: str
    updated_at: str
    last_summarized_qa_count: int = 0
    last_summarized_at: str = ""
    version: int = 1


@dataclass
class ExperienceMemory:
    decisions: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)


@dataclass
class ExperienceSummary:
    short: str
    long: str


@dataclass
class Experience:
    experience_id: str
    domain: DomainType
    topic: str
    goal: str
    segment_ids: List[str]
    current_segment_id: Optional[str]
    main_segment_path: List[str]
    summary: ExperienceSummary
    memory: ExperienceMemory
    status: ExperienceStatus
    created_at: str
    updated_at: str
    last_summarized_segment_count: int = 0
    last_summarized_qa_count: int = 0
    last_summarized_relation_count: int = 0
    last_summarized_at: str = ""
    version: int = 1


@dataclass
class SegmentRelation:
    relation_id: str
    experience_id: str
    from_segment_id: str
    to_segment_id: str
    relation_type: SegmentRelationType
    reason: str
    weight: float
    created_at: str


@dataclass
class RuntimeState:
    conversation_id: str
    current_experience_id: Optional[str]
    current_segment_id: Optional[str]
    latest_qa_id: Optional[str]
    recent_segment_ids: List[str]
    updated_at: str


@dataclass
class TopicIntent:
    domain: DomainType
    topic: str
    intents: List[IntentType]
    keywords: List[str]
    confidence: float
    goal: str = ""
    relation_to_previous: RelationToPreviousType = "new_topic"

    @property
    def topic_id(self) -> str:
        return _slugify(f"{self.domain}-{self.topic}") or "general-topic"

    @property
    def primary_intent(self) -> str:
        return self.intents[0] if self.intents else "general_analysis"


@dataclass
class TopicPromptContext:
    analysis: Optional[TopicIntent]
    topic_shift: bool
    is_new_session: bool
    retrieved_experiences: List[Experience]
    core_segments: List[Segment]
    expanded_segments: List[Segment]
    selected_qas: List[QA]
    transition_notes: List[str]
    rendered: str


@dataclass
class TopicTurnUpdate:
    analysis: Optional[TopicIntent]
    topic_shift: bool
    topic_session_id: Optional[str]
    topic_session_status: Optional[str]


@dataclass
class TurnActionDecision:
    action: TurnActionType
    target_experience: Experience
    target_segment: Optional[Segment] = None
    resume_source_segment: Optional[Segment] = None


@dataclass
class RetrievalDocument:
    doc_id: str
    object_type: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TopicMemoryManager:
    """Manage the topic-based three-layer memory MVP."""

    def __init__(
        self,
        memory_dir: str,
        sessions_dir: str,
        extractor_llm: Optional[Any] = None,
    ):
        self.memory_dir = os.path.abspath(memory_dir)
        self.sessions_dir = os.path.abspath(sessions_dir)
        self._extractor_llm = extractor_llm
        self._prepared_analysis_cache: Dict[str, TopicIntent] = {}
        self.store_dir = os.path.join(self.memory_dir, "topic_memory_store")
        self.index_file = os.path.join(self.memory_dir, "topic_memory_store.json")
        self.memory_snapshot_file = os.path.join(self.memory_dir, "MEMORY.md")
        self.sqlite_db_path = os.path.join(self.store_dir, "topic_memory.db")

        self.qas_dir = os.path.join(self.store_dir, "qas")
        self.segments_dir = os.path.join(self.store_dir, "segments")
        self.experiences_dir = os.path.join(self.store_dir, "experiences")
        self.relations_dir = os.path.join(self.store_dir, "relations")
        self.runtime_dir = os.path.join(self.store_dir, "runtime_states")

        for path in (
            self.memory_dir,
            self.store_dir,
            self.qas_dir,
            self.segments_dir,
            self.experiences_dir,
            self.relations_dir,
            self.runtime_dir,
        ):
            os.makedirs(path, exist_ok=True)

        self._embedder = self._create_embedder()
        self._initialize_sqlite()
        self._migrate_legacy_monolith_if_needed()
        self._repair_topic_memory_store()
        self._refresh_store_artifacts()

    def analyze_message(
        self,
        message: Optional[str],
        previous: Optional[TopicIntent] = None,
        recent_turn: Optional[Dict[str, str]] = None,
    ) -> Optional[TopicIntent]:
        """Infer topic/domain/intent metadata using LLM-first extraction with rule fallback."""
        if not message or not str(message).strip():
            return previous

        llm_analysis = self._llm_analyze_message(
            message=str(message).strip(),
            previous=previous,
            recent_turn=recent_turn,
        )
        if llm_analysis is not None:
            return llm_analysis

        return self._heuristic_analyze_message(message=str(message).strip(), previous=previous)

    def _heuristic_analyze_message(
        self,
        message: str,
        previous: Optional[TopicIntent] = None,
    ) -> TopicIntent:
        raw = str(message).strip()
        lowered = raw.lower()
        tokens = self._tokenize(raw)
        keywords = self._merge_unique(
            self._extract_known_keywords(lowered),
            self._select_keywords(tokens),
            limit=8,
        )
        domain = self._infer_domain(tokens, lowered, previous)
        intent = self._infer_intent(tokens, lowered, domain)
        topic = self._infer_topic(raw, domain, keywords, previous)
        goal = raw
        relation: RelationToPreviousType = "new_topic"

        if previous and self._contains_any(raw, RESUME_HINTS):
            topic = previous.topic
            goal = previous.goal or previous.topic
            keywords = self._merge_unique(keywords, previous.keywords, limit=8)
            relation = "resume_previous"
        elif previous and self._looks_like_detail_message(raw):
            topic = previous.topic
            goal = previous.goal or goal
            keywords = self._merge_unique(previous.keywords, keywords, limit=8)
            relation = "follow_up"
        elif previous and domain != previous.domain and self._soft_text_overlap(topic, previous.topic) < 0.24:
            relation = "switch_topic"

        confidence = min(0.98, 0.35 + len(keywords) * 0.06 + (0.12 if previous else 0.0))
        return TopicIntent(
            domain=domain,
            topic=topic,
            intents=[intent],
            keywords=keywords,
            confidence=round(confidence, 2),
            goal=goal,
            relation_to_previous=relation,
        )

    def _llm_analyze_message(
        self,
        message: str,
        previous: Optional[TopicIntent],
        recent_turn: Optional[Dict[str, str]],
    ) -> Optional[TopicIntent]:
        if self._extractor_llm is None:
            return None

        fallback = self._heuristic_analyze_message(message=message, previous=previous)
        try:
            response = self._extractor_llm.invoke(
                self._build_extraction_messages(
                    message=message,
                    previous=previous,
                    recent_turn=recent_turn,
                )
            )
        except Exception:
            return None

        payload = self._extract_json_object(self._response_to_text(getattr(response, "content", response)))
        if not isinstance(payload, dict):
            return None

        domain = self._normalize_domain_choice(payload.get("domain")) or fallback.domain
        intent = self._normalize_intent_choice(payload.get("intent")) or fallback.primary_intent
        relation = self._normalize_relation_choice(payload.get("relation_to_previous"))
        if relation is None:
            relation = fallback.relation_to_previous

        topic = self._sanitize_topic_text(payload.get("topic"), fallback=fallback.topic)
        goal = self._sanitize_goal_text(payload.get("goal"), fallback=fallback.goal)
        keywords = self._normalize_keyword_list(payload.get("keywords"), fallback=fallback.keywords)

        confidence = payload.get("confidence", fallback.confidence)
        try:
            confidence_value = float(confidence)
        except Exception:
            confidence_value = fallback.confidence
        confidence_value = max(0.0, min(1.0, round(confidence_value, 2)))

        if relation == "follow_up" and previous:
            topic = previous.topic
            goal = previous.goal or goal
            keywords = self._merge_unique(previous.keywords, keywords, limit=8)
        elif relation == "resume_previous" and previous:
            keywords = self._merge_unique(previous.keywords, keywords, limit=8)

        return TopicIntent(
            domain=domain,
            topic=topic,
            intents=[intent],
            keywords=keywords,
            confidence=confidence_value,
            goal=goal,
            relation_to_previous=relation,
        )

    def _build_extraction_messages(
        self,
        message: str,
        previous: Optional[TopicIntent],
        recent_turn: Optional[Dict[str, str]],
    ) -> List[Any]:
        domain_doc = "\n".join(
            f"- {domain}: {DOMAIN_DESCRIPTIONS[domain]}"
            for domain in DOMAIN_ENUM_VALUES
        )
        intent_doc = "\n".join(
            f"- {intent}: {INTENT_DESCRIPTIONS[intent]}"
            for intent in INTENT_ENUM_VALUES
        )
        system_prompt = (
            "你是三层主题记忆系统的字段抽取器。\n"
            "请根据“当前用户消息 + 最近一轮完整对话 + 上一主题状态”抽取结构化字段。\n"
            "严格输出 JSON 对象，不要输出 Markdown、不要加解释、不要使用代码块。\n\n"
            "字段要求：\n"
            "- domain: 必须从下面枚举中选择一个值。\n"
            f"{domain_doc}\n"
            "- intent: 必须从下面枚举中选择一个值。\n"
            f"{intent_doc}\n"
            "- topic: 提取后的稳定主题名，短语即可，不要复制整句，不要带“帮我/请/一下”等口语。\n"
            "- goal: 提取后的任务目标，描述用户想完成什么；如果当前消息只是补充条件或延续上一轮，要把上一轮目标补全后再输出。\n"
            "- keywords: 2 到 8 个关键词，优先实体、对象、地点、代码对象、约束条件。\n"
            "- relation_to_previous: 只能是 new_topic、follow_up、resume_previous、switch_topic 之一。\n"
            "- confidence: 0 到 1 的小数。\n\n"
            "判定规则：\n"
            "1. 短消息、细节补充、参数补充、时间/预算/偏好补充，通常是 follow_up。\n"
            "2. 明确表示“继续刚才/接着上次/回到之前”的，通常是 resume_previous。\n"
            "3. 明确表示“先不聊这个/换个话题/先暂停”的，通常是 switch_topic。\n"
            "4. topic 表示主题，intent 表示动作类型，goal 表示当前要达成的任务目标。\n"
            "5. 如果当前消息依赖上一轮上下文才能完整理解，必须利用最近一轮对话补全 topic 和 goal。\n"
            "6. domain 和 intent 绝不能输出枚举之外的值。"
        )
        human_payload = {
            "previous_topic_state": {
                "domain": previous.domain,
                "topic": previous.topic,
                "intent": previous.primary_intent,
                "goal": previous.goal,
                "keywords": previous.keywords,
            }
            if previous
            else None,
            "recent_completed_turn": recent_turn or None,
            "current_user_message": message,
            "required_json_schema": {
                "domain": DOMAIN_ENUM_VALUES[0],
                "intent": INTENT_ENUM_VALUES[0],
                "topic": "提取后的主题短语",
                "goal": "提取后的任务目标",
                "keywords": ["关键词1", "关键词2"],
                "relation_to_previous": RELATION_ENUM_VALUES[0],
                "confidence": 0.85,
            },
        }
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(human_payload, ensure_ascii=False, indent=2)),
        ]

    def _response_to_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(str(text))
                elif hasattr(item, "text"):
                    text = getattr(item, "text", "")
                    if text:
                        parts.append(str(text))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(content)

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            payload = json.loads(cleaned)
            return payload if isinstance(payload, dict) else None
        except Exception:
            pass

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _normalize_domain_choice(self, value: Any) -> Optional[DomainType]:
        normalized = str(value or "").strip().lower()
        aliases = {
            "software": "software_engineering",
            "engineering": "software_engineering",
            "code": "software_engineering",
            "finance": "finance_analysis",
            "investment": "finance_analysis",
            "travel_planning": "travel",
            "research": "research_writing",
            "writing": "research_writing",
            "product": "product_planning",
        }
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in DOMAIN_ENUM_VALUES else None

    def _normalize_intent_choice(self, value: Any) -> Optional[IntentType]:
        normalized = str(value or "").strip().lower()
        aliases = {
            "analysis": "general_analysis",
            "analyze": "general_analysis",
            "lookup": "retrieval",
            "search": "retrieval",
            "find": "retrieval",
            "build": "implementation",
            "fix": "troubleshooting",
            "debug": "troubleshooting",
            "plan": "planning",
            "design": "planning",
            "explain": "explanation",
            "summary": "summarization",
        }
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in INTENT_ENUM_VALUES else None

    def _normalize_relation_choice(self, value: Any) -> Optional[RelationToPreviousType]:
        normalized = str(value or "").strip().lower()
        aliases = {
            "continue_previous": "follow_up",
            "continue_current": "follow_up",
            "followup": "follow_up",
            "resume": "resume_previous",
            "resume_old_topic": "resume_previous",
            "switch": "switch_topic",
        }
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in RELATION_ENUM_VALUES else None

    def _sanitize_topic_text(self, value: Any, fallback: str) -> str:
        topic = re.sub(r"\s+", " ", str(value or "").strip())
        topic = topic.strip("。；;，,：:[]{}()（）")
        if not topic:
            return fallback
        return topic[:48]

    def _sanitize_goal_text(self, value: Any, fallback: str) -> str:
        goal = re.sub(r"\s+", " ", str(value or "").strip())
        goal = goal.strip("。；;，,")
        if not goal:
            return fallback
        return goal[:160]

    def _normalize_keyword_list(self, value: Any, fallback: Sequence[str]) -> List[str]:
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
            if items:
                return self._merge_unique([], items, limit=8)
        if isinstance(value, str) and value.strip():
            parts = re.split(r"[,，、\s]+", value.strip())
            items = [part for part in parts if part]
            if items:
                return self._merge_unique([], items, limit=8)
        return list(fallback)

    def _analysis_cache_key(
        self,
        session_id: Optional[str],
        message: str,
        runtime: Optional[RuntimeState],
    ) -> str:
        anchor = runtime.latest_qa_id if runtime and runtime.latest_qa_id else "none"
        digest = hashlib.sha1(message.strip().encode("utf-8")).hexdigest()
        return f"{session_id or 'none'}:{anchor}:{digest}"

    def _store_prepared_analysis(
        self,
        session_id: Optional[str],
        message: str,
        runtime: Optional[RuntimeState],
        analysis: TopicIntent,
    ):
        key = self._analysis_cache_key(session_id, message, runtime)
        self._prepared_analysis_cache[key] = analysis
        if len(self._prepared_analysis_cache) > 64:
            oldest_keys = list(self._prepared_analysis_cache.keys())[:-64]
            for oldest_key in oldest_keys:
                self._prepared_analysis_cache.pop(oldest_key, None)

    def _consume_prepared_analysis(
        self,
        session_id: Optional[str],
        message: str,
        runtime: Optional[RuntimeState],
    ) -> Optional[TopicIntent]:
        key = self._analysis_cache_key(session_id, message, runtime)
        return self._prepared_analysis_cache.pop(key, None)

    def _extract_recent_turn_from_history(
        self,
        history: Sequence[Message],
    ) -> Optional[Dict[str, str]]:
        last_assistant: Optional[Message] = None
        last_user: Optional[Message] = None
        for message in reversed(list(history)):
            if message.role == "assistant" and last_assistant is None and (message.content or "").strip():
                last_assistant = message
                continue
            if last_assistant and message.role == "user" and (message.content or "").strip():
                last_user = message
                break
        if not last_user or not last_assistant:
            return None
        return {
            "user_input": last_user.content.strip(),
            "assistant_output": last_assistant.content.strip(),
        }

    def _extract_recent_turn_from_runtime(
        self,
        runtime: Optional[RuntimeState],
    ) -> Optional[Dict[str, str]]:
        if not runtime or not runtime.latest_qa_id:
            return None
        qa = self.load_qa(runtime.latest_qa_id)
        if qa is None:
            return None
        return {
            "user_input": qa.user_input,
            "assistant_output": qa.assistant_output,
        }

    def _resolve_recent_turn_context(
        self,
        history: Optional[Sequence[Message]],
        runtime: Optional[RuntimeState],
    ) -> Optional[Dict[str, str]]:
        if history:
            recent_turn = self._extract_recent_turn_from_history(history)
            if recent_turn:
                return recent_turn
        return self._extract_recent_turn_from_runtime(runtime)

    def build_prompt_context(
        self,
        session_id: Optional[str],
        user_message: Optional[str],
        history: Optional[Sequence[Message]] = None,
        prepare_state: bool = False,
    ) -> TopicPromptContext:
        """Retrieve topic memory following Experience -> Segment -> Relation -> QA."""
        runtime = self.load_runtime_state(session_id)
        current_segment = self.load_segment(runtime.current_segment_id) if runtime else None
        current_experience = (
            self.load_experience(runtime.current_experience_id)
            if runtime and runtime.current_experience_id
            else None
        )
        previous = self._analysis_from_state(current_segment, current_experience, runtime)
        recent_turn = self._resolve_recent_turn_context(history, runtime)

        analysis = self.analyze_message(
            user_message,
            previous=previous,
            recent_turn=recent_turn,
        )
        if analysis and current_segment and analysis.relation_to_previous == "follow_up":
            analysis.topic = current_segment.topic
            analysis.goal = current_segment.goal
            analysis.keywords = self._merge_unique(
                self._tokenize(current_segment.goal),
                analysis.keywords,
                limit=8,
            )
        if analysis is None:
            analysis = previous
        elif prepare_state and session_id and user_message:
            self._store_prepared_analysis(session_id, user_message, runtime, analysis)

        is_new_session = self._is_new_session(runtime, history or [])
        if not analysis:
            rendered = self._render_prompt_context(
                analysis=None,
                runtime=runtime,
                topic_shift=False,
                is_new_session=is_new_session,
                transition_notes=["no_active_topic"],
                experiences=[],
                core_segments=[],
                expanded_segments=[],
                qas=[],
            )
            return TopicPromptContext(
                analysis=None,
                topic_shift=False,
                is_new_session=is_new_session,
                retrieved_experiences=[],
                core_segments=[],
                expanded_segments=[],
                selected_qas=[],
                transition_notes=["no_active_topic"],
                rendered=rendered,
            )

        query_text = (user_message or analysis.goal or analysis.topic).strip()
        topic_shift = self.is_topic_shift(previous, analysis, current_segment, query_text)
        experiences = self.retrieve_experiences(analysis, query_text, runtime=runtime, limit=3)
        core_segments = self.retrieve_segments(
            analysis,
            query_text,
            experiences,
            runtime=runtime,
            limit=3,
        )
        expanded_segments = self.expand_segment_relations(core_segments, query_text)
        qas = self.retrieve_qas(
            analysis,
            query_text,
            core_segments=core_segments,
            expanded_segments=expanded_segments,
            limit=10,
        )
        transition_notes = self._build_transition_notes(
            user_message=user_message or "",
            analysis=analysis,
            runtime=runtime,
            current_segment=current_segment,
            experiences=experiences,
            core_segments=core_segments,
            is_new_session=is_new_session,
            topic_shift=topic_shift,
        )
        rendered = self._render_prompt_context(
            analysis=analysis,
            runtime=runtime,
            topic_shift=topic_shift,
            is_new_session=is_new_session,
            transition_notes=transition_notes,
            experiences=experiences,
            core_segments=core_segments,
            expanded_segments=expanded_segments,
            qas=qas,
        )

        return TopicPromptContext(
            analysis=analysis,
            topic_shift=topic_shift,
            is_new_session=is_new_session,
            retrieved_experiences=experiences,
            core_segments=core_segments,
            expanded_segments=expanded_segments,
            selected_qas=qas,
            transition_notes=transition_notes,
            rendered=rendered,
        )

    def remember_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_reply: str,
        turn_messages: Sequence[Message],
    ) -> TopicTurnUpdate:
        """Persist one completed QA turn into the new three-layer topic memory."""
        runtime = self.load_runtime_state(session_id)
        current_segment = (
            self.load_segment(runtime.current_segment_id)
            if runtime and runtime.current_segment_id
            else None
        )
        current_experience = (
            self.load_experience(runtime.current_experience_id)
            if runtime and runtime.current_experience_id
            else None
        )
        previous = self._analysis_from_state(
            current_segment,
            current_experience,
            runtime,
        )
        analysis = self._consume_prepared_analysis(session_id, user_message, runtime)
        if analysis is None:
            analysis = self.analyze_message(
                user_message,
                previous=previous,
                recent_turn=self._extract_recent_turn_from_runtime(runtime),
            )

        if analysis is None:
            return TopicTurnUpdate(
                analysis=previous,
                topic_shift=False,
                topic_session_id=runtime.current_segment_id if runtime else None,
                topic_session_status=current_segment.status if current_segment else None,
            )

        if current_segment and analysis.relation_to_previous == "follow_up":
            analysis.topic = current_segment.topic
            analysis.goal = current_segment.goal

        active_runtime = runtime or RuntimeState(
            conversation_id=session_id,
            current_experience_id=None,
            current_segment_id=None,
            latest_qa_id=None,
            recent_segment_ids=[],
            updated_at=_now_iso(),
        )

        now = _now_iso()
        decision = self._resolve_turn_action(
            analysis=analysis,
            user_message=user_message,
            current_experience=current_experience,
            current_segment=current_segment,
        )
        print(f"decision==={decision}")
        target_experience = decision.target_experience
        target_segment = decision.target_segment
        experiences_to_recompute: Set[str] = set()
        new_segment_created = False

        if decision.action in {"fork_new_segment", "resume_historical_segment"} and current_segment:
            self._close_segment_for_switch(
                current_segment,
                interrupted=False,
                user_message=user_message,
                analysis=analysis,
            )
            experiences_to_recompute.add(current_segment.experience_id)

        if decision.action == "switch_experience" and current_segment:
            self._close_segment_for_switch(
                current_segment,
                interrupted=True,
                user_message=user_message,
                analysis=analysis,
            )
            experiences_to_recompute.add(current_segment.experience_id)

        if decision.action == "append_current_segment":
            if target_segment is None:
                raise RuntimeError("append_current_segment requires a resolved current segment.")
        elif decision.action == "resume_historical_segment":
            if decision.resume_source_segment is None:
                raise RuntimeError("resume_historical_segment requires a source segment.")
            analysis.domain = decision.resume_source_segment.domain
            analysis.topic = decision.resume_source_segment.topic
            analysis.intents = [decision.resume_source_segment.intent]
            analysis.goal = decision.resume_source_segment.goal
            analysis.keywords = self._merge_unique(
                analysis.keywords,
                self._select_keywords(
                    self._tokenize(
                        " ".join(
                            [
                                decision.resume_source_segment.topic,
                                decision.resume_source_segment.goal,
                            ]
                        )
                    )
                ),
                limit=8,
            )
            self._finalize_segment(
                decision.resume_source_segment,
                "inherited",
                user_message=user_message,
                analysis=analysis,
            )
            experiences_to_recompute.add(decision.resume_source_segment.experience_id)
            target_segment = self._create_segment(
                experience_id=target_experience.experience_id,
                analysis=analysis,
            )
            new_segment_created = True
            self._add_or_update_relation(
                experience_id=target_experience.experience_id,
                from_segment_id=target_segment.segment_id,
                to_segment_id=decision.resume_source_segment.segment_id,
                relation_type="continue_from",
                reason="同一 experience 下恢复历史 segment，因此新 segment 继承旧 segment 的主题与意图。",
                weight=0.95,
            )
        elif decision.action == "fork_new_segment":
            target_segment = self._create_segment(
                experience_id=target_experience.experience_id,
                analysis=analysis,
            )
            new_segment_created = True
            if current_segment and current_segment.experience_id == target_experience.experience_id:
                relation_spec = self._determine_segment_relation(
                    current_segment=target_segment,
                    previous_segment=current_segment,
                )
                if relation_spec and relation_spec[0] in {"refines", "contradicts"}:
                    relation_type, reason, weight = relation_spec
                    self._add_or_update_relation(
                        experience_id=target_experience.experience_id,
                        from_segment_id=target_segment.segment_id,
                        to_segment_id=current_segment.segment_id,
                        relation_type=relation_type,
                        reason=reason,
                        weight=weight,
                    )
        else:
            if target_segment is not None:
                analysis.domain = target_segment.domain
                analysis.topic = target_segment.topic
                analysis.intents = [target_segment.intent]
                analysis.goal = target_segment.goal
                analysis.keywords = self._merge_unique(
                    analysis.keywords,
                    self._select_keywords(
                        self._tokenize(
                            " ".join(
                                [
                                    target_segment.topic,
                                    target_segment.goal,
                                ]
                            )
                        )
                    ),
                    limit=8,
                )
                target_current_segment = (
                    self.load_segment(target_experience.current_segment_id)
                    if target_experience.current_segment_id
                    else None
                )
                if (
                    target_current_segment
                    and target_current_segment.segment_id != target_segment.segment_id
                    and target_current_segment.experience_id == target_experience.experience_id
                    and self._normalize_segment_status(target_current_segment.status) == "open"
                ):
                    self._close_segment_for_switch(
                        target_current_segment,
                        interrupted=True,
                        user_message=user_message,
                        analysis=analysis,
                    )
                    experiences_to_recompute.add(target_current_segment.experience_id)
            else:
                target_segment = self._create_segment(
                    experience_id=target_experience.experience_id,
                    analysis=analysis,
                )
                new_segment_created = True

        if target_segment is None:
            raise RuntimeError("Failed to resolve experience or segment for topic memory update.")

        return self._persist_turn_update(
            session_id=session_id,
            analysis=analysis,
            user_message=user_message,
            assistant_reply=assistant_reply,
            turn_messages=turn_messages,
            target_experience=target_experience,
            target_segment=target_segment,
            active_runtime=active_runtime,
            now=now,
            experiences_to_recompute=experiences_to_recompute,
            topic_shift=decision.action != "append_current_segment",
            new_segment_created=new_segment_created,
        )

    def is_topic_shift(
        self,
        previous: Optional[TopicIntent],
        current: Optional[TopicIntent],
        current_segment: Optional[Segment] = None,
        query_text: str = "",
    ) -> bool:
        """Detect whether the user likely switched away from the active topic."""
        if current is None:
            return False
        if previous is None:
            return False
        if current.relation_to_previous == "follow_up":
            return False
        if current.relation_to_previous == "switch_topic":
            return True
        if self._contains_any(query_text, PAUSE_HINTS):
            return True

        similarity = self.topic_similarity(previous, current)
        if current_segment and current_segment.topic == current.topic:
            return False
        if previous.domain != current.domain and similarity < 0.44:
            return True
        return similarity < 0.26

    def topic_similarity(self, left: TopicIntent, right: TopicIntent) -> float:
        keyword_overlap = self._jaccard(set(left.keywords), set(right.keywords))
        intent_overlap = self._jaccard(set(left.intents), set(right.intents))
        domain_score = 1.0 if left.domain == right.domain else 0.0
        topic_score = self._soft_text_overlap(left.topic, right.topic)
        return round(
            keyword_overlap * 0.35
            + intent_overlap * 0.15
            + domain_score * 0.15
            + topic_score * 0.35,
            3,
        )

    def retrieve_experiences(
        self,
        analysis: TopicIntent,
        query_text: str,
        runtime: Optional[RuntimeState] = None,
        limit: int = 3,
    ) -> List[Experience]:
        experiences = self.list_experiences()
        if not experiences:
            return []

        docs = [
            RetrievalDocument(
                doc_id=experience.experience_id,
                object_type="experience",
                text=self._build_experience_document(experience),
                metadata={"updated_at": experience.updated_at},
            )
            for experience in experiences
        ]
        hybrid_scores = self._hybrid_scores(
            table_name="experience_fts",
            documents=docs,
            query_text=query_text,
        )
        recent_experience_id = runtime.current_experience_id if runtime else None
        scored: List[Tuple[float, Experience]] = []
        for experience in experiences:
            base_score = hybrid_scores.get(experience.experience_id, 0.0)
            experience_topic = self._experience_topic_key(experience.domain, experience.topic)
            analysis_topic = self._experience_topic_key(analysis.domain, analysis.topic)
            topic_bonus = 0.32 if experience_topic == analysis_topic else 0.22 * self._soft_text_overlap(
                experience_topic,
                analysis_topic,
            )
            domain_bonus = 0.14 if experience.domain == analysis.domain else 0.0
            goal_bonus = 0.18 * self._soft_text_overlap(
                " ".join([experience.goal, experience.summary.short, experience.summary.long]),
                query_text,
            )
            memory_bonus = 0.12 * self._soft_text_overlap(
                self._experience_memory_text(experience.memory),
                query_text,
            )
            recency_bonus = self._recency_bonus(experience.updated_at)
            runtime_bonus = 0.08 if recent_experience_id == experience.experience_id else 0.0
            score = round(
                base_score
                + topic_bonus
                + domain_bonus
                + goal_bonus
                + memory_bonus
                + recency_bonus
                + runtime_bonus,
                4,
            )
            if score > 0.08:
                scored.append((score, experience))

        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return [item[1] for item in scored[:limit]]

    def retrieve_segments(
        self,
        analysis: TopicIntent,
        query_text: str,
        experiences: Sequence[Experience],
        runtime: Optional[RuntimeState] = None,
        limit: int = 3,
    ) -> List[Segment]:
        if not experiences:
            return []

        candidate_ids = {experience.experience_id for experience in experiences}
        segments = [
            segment
            for segment in self.list_segments()
            if segment.experience_id in candidate_ids
        ]
        if not segments:
            return []

        docs = [
            RetrievalDocument(
                doc_id=segment.segment_id,
                object_type="segment",
                text=self._build_segment_document(segment),
                metadata={"updated_at": segment.updated_at},
            )
            for segment in segments
        ]
        hybrid_scores = self._hybrid_scores(
            table_name="segment_fts",
            documents=docs,
            query_text=query_text,
        )
        follow_up = self._contains_any(query_text, RESUME_HINTS) or (
            analysis.relation_to_previous == "resume_previous"
        )
        recent_segment_ids = runtime.recent_segment_ids if runtime else []

        scored: List[Tuple[float, Segment]] = []
        for segment in segments:
            base_score = hybrid_scores.get(segment.segment_id, 0.0)
            topic_bonus = 0.18 if segment.topic == analysis.topic else 0.12 * self._soft_text_overlap(
                segment.topic,
                analysis.topic,
            )
            intent_bonus = 0.1 if segment.intent == analysis.primary_intent else 0.0
            goal_bonus = 0.14 * self._soft_text_overlap(segment.goal, query_text)
            summary_bonus = 0.12 * self._soft_text_overlap(segment.summary, query_text)
            recent_bonus = 0.10 if segment.segment_id in recent_segment_ids else 0.0
            follow_up_bonus = 0.0
            if follow_up:
                if segment.segment_id in recent_segment_ids:
                    follow_up_bonus += 0.08

            score = round(
                base_score
                + topic_bonus
                + intent_bonus
                + goal_bonus
                + summary_bonus
                + recent_bonus
                + follow_up_bonus
                + self._recency_bonus(segment.updated_at),
                4,
            )
            if score > 0.04:
                scored.append((score, segment))

        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return [item[1] for item in scored[:limit]]

    def expand_segment_relations(
        self,
        core_segments: Sequence[Segment],
        query_text: str,
        limit: int = 4,
    ) -> List[Segment]:
        if not core_segments:
            return []

        core_ids = {segment.segment_id for segment in core_segments}
        relation_pairs = self._related_relation_pairs(core_ids)
        scored: List[Tuple[float, Segment]] = []
        seen: Set[str] = set(core_ids)
        for relation, related_segment in relation_pairs:
            if related_segment.segment_id in seen:
                continue
            seen.add(related_segment.segment_id)
            relation_bonus = 0.18 * self._soft_text_overlap(relation.reason, query_text)
            summary_bonus = 0.14 * self._soft_text_overlap(related_segment.summary, query_text)
            goal_bonus = 0.1 * self._soft_text_overlap(related_segment.goal, query_text)
            time_bonus = max(
                self._recency_bonus(relation.created_at),
                self._recency_bonus(related_segment.updated_at),
            )
            score = round(
                relation_bonus
                + summary_bonus
                + goal_bonus
                + time_bonus
                + max(min(relation.weight, 1.0), 0.0) * 0.08,
                4,
            )
            scored.append((score, related_segment))

        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return [item[1] for item in scored[:limit]]

    def retrieve_qas(
        self,
        analysis: TopicIntent,
        query_text: str,
        core_segments: Sequence[Segment],
        expanded_segments: Sequence[Segment],
        limit: int = 10,
    ) -> List[QA]:
        core_ids = [segment.segment_id for segment in core_segments]
        expanded_ids = [segment.segment_id for segment in expanded_segments]
        ordered_segments = core_ids + [segment_id for segment_id in expanded_ids if segment_id not in core_ids]

        qas: List[QA] = []
        for segment_id in ordered_segments:
            segment = self.load_segment(segment_id)
            if not segment:
                continue
            qas.extend(self.load_qas(segment.qa_ids))

        if not qas:
            return []

        segment_lookup = {
            segment.segment_id: segment
            for segment in list(core_segments) + list(expanded_segments)
        }
        related_relations = self._related_relations(core_ids)
        relation_text_by_segment: Dict[str, str] = {}
        for relation in related_relations:
            relation_text_by_segment[relation.from_segment_id] = self._merge_text_parts(
                relation_text_by_segment.get(relation.from_segment_id, ""),
                relation.reason,
            )
            relation_text_by_segment[relation.to_segment_id] = self._merge_text_parts(
                relation_text_by_segment.get(relation.to_segment_id, ""),
                relation.reason,
            )

        experience_summaries: List[str] = []
        seen_experience_ids: Set[str] = set()
        for segment in segment_lookup.values():
            if segment.experience_id in seen_experience_ids:
                continue
            seen_experience_ids.add(segment.experience_id)
            experience = self.load_experience(segment.experience_id)
            if not experience:
                continue
            experience_summaries.extend(
                [experience.summary.short.strip(), experience.summary.long.strip()]
            )
        retrieval_query_text = self._merge_text_parts(
            query_text,
            *experience_summaries,
            *[segment.summary for segment in core_segments],
            *[relation.reason for relation in related_relations],
        )

        docs = [
            RetrievalDocument(
                doc_id=qa.qa_id,
                object_type="qa",
                text=self._build_qa_document(qa),
                metadata={
                    "segment_id": qa.segment_id,
                    "timestamp": qa.timestamp,
                },
            )
            for qa in qas
        ]
        hybrid_scores = self._hybrid_scores(
            table_name="qa_fts",
            documents=docs,
            query_text=retrieval_query_text,
        )

        scored: List[Tuple[float, QA]] = []
        for qa in qas:
            base_score = hybrid_scores.get(qa.qa_id, 0.0)
            core_bonus = 0.12 if qa.segment_id in core_ids else 0.0
            topic_bonus = 0.08 if qa.topic == analysis.topic else 0.05 * self._soft_text_overlap(
                qa.topic,
                analysis.topic,
            )
            segment = segment_lookup.get(qa.segment_id)
            summary_bonus = 0.08 * self._soft_text_overlap(
                segment.summary if segment else "",
                retrieval_query_text,
            )
            relation_bonus = 0.06 * self._soft_text_overlap(
                relation_text_by_segment.get(qa.segment_id, ""),
                retrieval_query_text,
            )
            score = round(
                base_score
                + core_bonus
                + topic_bonus
                + summary_bonus
                + relation_bonus
                + self._recency_bonus(qa.timestamp),
                4,
            )
            if score > 0.02:
                scored.append((score, qa))

        scored.sort(key=lambda item: (item[0], item[1].timestamp), reverse=True)
        selected = [item[1] for item in scored[:limit]]
        selected.sort(key=lambda qa: qa.timestamp)
        return selected

    def get_current_topic_summary(self, session_id: Optional[str]) -> str:
        runtime = self.load_runtime_state(session_id)
        if not runtime or not runtime.current_segment_id:
            return "No active topic tracked for this session."

        segment = self.load_segment(runtime.current_segment_id)
        if not segment:
            return "No active topic tracked for this session."

        return (
            f"domain={segment.domain}; topic={segment.topic}; intent={segment.intent}; "
            f"goal={segment.goal}"
        )

    def _related_relations(self, segment_ids: Sequence[str]) -> List[SegmentRelation]:
        segment_id_set = set(segment_ids)
        relations: List[SegmentRelation] = []
        seen_relation_ids: Set[str] = set()
        for relation in self.list_relations():
            normalized_type = self._normalize_relation_type(relation.relation_type)
            if normalized_type not in ALLOWED_RELATION_EXPANSION:
                continue
            if relation.from_segment_id not in segment_id_set and relation.to_segment_id not in segment_id_set:
                continue
            if relation.relation_id in seen_relation_ids:
                continue
            relation.relation_type = normalized_type
            seen_relation_ids.add(relation.relation_id)
            relations.append(relation)
        return relations

    def _related_relation_pairs(
        self,
        segment_ids: Sequence[str],
    ) -> List[Tuple[SegmentRelation, Segment]]:
        segment_id_set = set(segment_ids)
        related_pairs: List[Tuple[SegmentRelation, Segment]] = []
        for relation in self._related_relations(segment_ids):
            related_segment_id: Optional[str] = None
            if relation.from_segment_id in segment_id_set:
                related_segment_id = relation.to_segment_id
            elif relation.to_segment_id in segment_id_set:
                related_segment_id = relation.from_segment_id
            if not related_segment_id:
                continue
            related_segment = self.load_segment(related_segment_id)
            if not related_segment:
                continue
            related_pairs.append((relation, related_segment))
        return related_pairs

    def get_store_overview(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        runtime = self.load_runtime_state(conversation_id) if conversation_id else self._latest_runtime_state()
        experiences = self.list_experiences()
        segments = self.list_segments()
        qas = self.list_qas()
        relations = self.list_relations()

        latest_experience = (
            self.load_experience(runtime.current_experience_id)
            if runtime and runtime.current_experience_id
            else self._latest_by_updated_at(experiences)
        )
        latest_segment = (
            self.load_segment(runtime.current_segment_id)
            if runtime and runtime.current_segment_id
            else self._latest_by_updated_at(segments)
        )

        return {
            "version": 1,
            "updated_at": _now_iso(),
            "store_dir": os.path.relpath(self.store_dir, start=os.path.dirname(self.memory_dir)),
            "index_db": os.path.relpath(self.sqlite_db_path, start=os.path.dirname(self.memory_dir)),
            "counts": {
                "experiences": len(experiences),
                "segments": len(segments),
                "qas": len(qas),
                "relations": len(relations),
            },
            "current_runtime_state": asdict(runtime) if runtime else None,
            "latest_experience": self._experience_overview_payload(latest_experience),
            "latest_segment": self._segment_overview_payload(latest_segment),
            "recent_experiences": [
                self._experience_overview_payload(item)
                for item in self._recent_items(experiences, limit=5)
            ],
            "recent_segments": [
                self._segment_overview_payload(item)
                for item in self._recent_items(segments, limit=6)
            ],
            "object_dirs": {
                "qas": self.qas_dir,
                "segments": self.segments_dir,
                "experiences": self.experiences_dir,
                "relations": self.relations_dir,
                "runtime_states": self.runtime_dir,
            },
        }

    def load_runtime_state(self, conversation_id: Optional[str]) -> Optional[RuntimeState]:
        if not conversation_id:
            return None
        path = os.path.join(self.runtime_dir, f"{conversation_id}.json")
        if not os.path.exists(path):
            return None
        payload = self._load_json(path, {})
        if not payload:
            return None
        return RuntimeState(
            conversation_id=str(payload.get("conversation_id", conversation_id)),
            current_experience_id=payload.get("current_experience_id"),
            current_segment_id=payload.get("current_segment_id"),
            latest_qa_id=payload.get("latest_qa_id"),
            recent_segment_ids=[str(item) for item in payload.get("recent_segment_ids", [])],
            updated_at=str(payload.get("updated_at", _now_iso())),
        )

    def save_runtime_state(self, runtime: RuntimeState):
        path = os.path.join(self.runtime_dir, f"{runtime.conversation_id}.json")
        self._save_json(path, asdict(runtime))

    def list_runtime_states(self) -> List[RuntimeState]:
        return self._load_all(
            self.runtime_dir,
            lambda payload: RuntimeState(
                conversation_id=str(payload.get("conversation_id", "")),
                current_experience_id=payload.get("current_experience_id"),
                current_segment_id=payload.get("current_segment_id"),
                latest_qa_id=payload.get("latest_qa_id"),
                recent_segment_ids=[str(item) for item in payload.get("recent_segment_ids", [])],
                updated_at=str(payload.get("updated_at", _now_iso())),
            ),
        )

    def load_qa(self, qa_id: Optional[str]) -> Optional[QA]:
        if not qa_id:
            return None
        path = os.path.join(self.qas_dir, f"{qa_id}.json")
        if not os.path.exists(path):
            return None
        payload = self._load_json(path, {})
        if not payload:
            return None
        return self._deserialize_qa(payload)

    def load_qas(self, qa_ids: Sequence[str]) -> List[QA]:
        qas = []
        for qa_id in qa_ids:
            qa = self.load_qa(qa_id)
            if qa:
                qas.append(qa)
        return qas

    def list_qas(self) -> List[QA]:
        return self._load_all(self.qas_dir, self._deserialize_qa)

    def save_qa(self, qa: QA):
        path = os.path.join(self.qas_dir, f"{qa.qa_id}.json")
        self._save_json(path, self._serialize_qa(qa))

    def load_segment(self, segment_id: Optional[str]) -> Optional[Segment]:
        if not segment_id:
            return None
        path = os.path.join(self.segments_dir, f"{segment_id}.json")
        if not os.path.exists(path):
            return None
        payload = self._load_json(path, {})
        if not payload:
            return None
        return self._deserialize_segment(payload)

    def list_segments(self) -> List[Segment]:
        return self._load_all(self.segments_dir, self._deserialize_segment)

    def save_segment(self, segment: Segment):
        path = os.path.join(self.segments_dir, f"{segment.segment_id}.json")
        self._save_json(path, self._serialize_segment(segment))

    def load_experience(self, experience_id: Optional[str]) -> Optional[Experience]:
        if not experience_id:
            return None
        path = os.path.join(self.experiences_dir, f"{experience_id}.json")
        if not os.path.exists(path):
            return None
        payload = self._load_json(path, {})
        if not payload:
            return None
        return self._deserialize_experience(payload)

    def list_experiences(self) -> List[Experience]:
        return self._load_all(self.experiences_dir, self._deserialize_experience)

    def save_experience(self, experience: Experience):
        path = os.path.join(self.experiences_dir, f"{experience.experience_id}.json")
        self._save_json(path, self._serialize_experience(experience))

    def load_relation(self, relation_id: Optional[str]) -> Optional[SegmentRelation]:
        if not relation_id:
            return None
        path = os.path.join(self.relations_dir, f"{relation_id}.json")
        if not os.path.exists(path):
            return None
        payload = self._load_json(path, {})
        if not payload:
            return None
        return self._deserialize_relation(payload)

    def list_relations(self) -> List[SegmentRelation]:
        return self._load_all(self.relations_dir, self._deserialize_relation)

    def save_relation(self, relation: SegmentRelation):
        path = os.path.join(self.relations_dir, f"{relation.relation_id}.json")
        self._save_json(path, asdict(relation))

    def _resolve_experience(
        self,
        analysis: TopicIntent,
        runtime: RuntimeState,
        current_experience: Optional[Experience],
        current_segment: Optional[Segment],
        query_text: str,
    ) -> Optional[Experience]:
        if (
            current_experience
            and current_segment
            and not self.is_topic_shift(
                self._analysis_from_state(current_segment, current_experience, runtime),
                analysis,
                current_segment,
                query_text,
            )
            and self._experience_matches_analysis(current_experience, analysis)
        ):
            return current_experience

        if (
            current_experience
            and analysis.relation_to_previous != "switch_topic"
            and self._experience_matches_analysis(current_experience, analysis)
        ):
            return current_experience

        experiences = self.retrieve_experiences(analysis, query_text, runtime=runtime, limit=3)
        for experience in experiences:
            if self._experience_matches_analysis(experience, analysis):
                return experience
        return None

    def _resolve_turn_action(
        self,
        analysis: TopicIntent,
        user_message: str,
        current_experience: Optional[Experience],
        current_segment: Optional[Segment],
    ) -> TurnActionDecision:
        if current_experience and self._experience_matches_analysis(current_experience, analysis):
            experience_segments = self.load_segments_for_experience(current_experience.experience_id)
            effective_current_segment = self._resolve_effective_current_segment(
                current_experience=current_experience,
                current_segment=current_segment,
                experience_segments=experience_segments,
            )
            if effective_current_segment is None:
                return TurnActionDecision(
                    action="fork_new_segment",
                    target_experience=current_experience,
                )

            llm_action, llm_resume_source = self._llm_resolve_turn_action(
                analysis=analysis,
                user_message=user_message,
                current_experience=current_experience,
                current_segment=effective_current_segment,
                experience_segments=experience_segments,
            )
            if llm_action == "append_current_segment":
                return TurnActionDecision(
                    action=llm_action,
                    target_experience=current_experience,
                    target_segment=effective_current_segment,
                )
            if llm_action == "resume_historical_segment" and llm_resume_source is not None:
                return TurnActionDecision(
                    action=llm_action,
                    target_experience=current_experience,
                    resume_source_segment=llm_resume_source,
                )
            if llm_action == "fork_new_segment":
                return TurnActionDecision(
                    action=llm_action,
                    target_experience=current_experience,
                )

            if (
                effective_current_segment.intent == analysis.primary_intent
                and self._goals_align(effective_current_segment.goal, analysis.goal, "")
            ):
                return TurnActionDecision(
                    action="append_current_segment",
                    target_experience=current_experience,
                    target_segment=effective_current_segment,
                )

            resume_source_segment = self._find_best_matching_segment_for_analysis(
                experience_id=current_experience.experience_id,
                analysis=analysis,
                exclude_segment_ids=[effective_current_segment.segment_id],
            )
            if resume_source_segment is not None:
                return TurnActionDecision(
                    action="resume_historical_segment",
                    target_experience=current_experience,
                    resume_source_segment=resume_source_segment,
                )

            return TurnActionDecision(
                action="fork_new_segment",
                target_experience=current_experience,
            )

        target_experience = self._find_best_matching_experience(
            analysis=analysis,
            exclude_experience_ids=[
                experience.experience_id
                for experience in [current_experience]
                if experience is not None
            ],
        )
        if target_experience is None:
            return TurnActionDecision(
                action="switch_experience",
                target_experience=self._create_experience(analysis),
            )

        target_segment = self._find_best_matching_segment_for_analysis(
            experience_id=target_experience.experience_id,
            analysis=analysis,
        )
        return TurnActionDecision(
            action="switch_experience",
            target_experience=target_experience,
            target_segment=target_segment,
        )

    def _llm_resolve_turn_action(
        self,
        analysis: TopicIntent,
        user_message: str,
        current_experience: Experience,
        current_segment: Segment,
        experience_segments: Sequence[Segment],
    ) -> Tuple[Optional[TurnActionType], Optional[Segment]]:
        if self._extractor_llm is None:
            return None, None

        historical_segments = [
            segment
            for segment in experience_segments
            if segment.segment_id != current_segment.segment_id
        ]

        system_prompt = (
            "你是 topic memory 的 turn action 分类器。\n"
            "已知当前 analysis 与 current_experience 的主题一致或相近。\n"
            "请根据当前 analysis、current_segment，以及同一 experience 下的全部 segments，"
            "判断本轮在当前 experience 内的写入动作。\n"
            "只能输出 JSON 对象，不要输出 Markdown，不要解释。\n"
            'action 只能是 "append_current_segment"、"resume_historical_segment"、"fork_new_segment" 之一。\n'
            "判定规则：\n"
            "1. append_current_segment: current_segment 的 intent 与 analysis 一致，且 goal 相近。\n"
            "2. resume_historical_segment: current_segment 不适合 append，但同一 experience 下存在其他 segment，"
            "其 intent 与 analysis 一致且 goal 相近。\n"
            "3. fork_new_segment: current_segment 不适合 append，且同一 experience 下也没有合适的历史 segment 可 resume。\n"
            "如果选择 resume_historical_segment，必须返回 resume_source_segment_id；否则返回 null。\n"
            "输出字段：action, resume_source_segment_id, reason, confidence。"
        )
        payload = {
            "analysis": {
                "domain": analysis.domain,
                "topic": analysis.topic,
                "intent": analysis.primary_intent,
                "goal": analysis.goal,
                "relation_to_previous": analysis.relation_to_previous,
                "keywords": analysis.keywords,
            },
            "current_experience": {
                "experience_id": current_experience.experience_id,
                "topic": current_experience.topic,
                "goal": current_experience.goal,
                "current_segment_id": current_experience.current_segment_id,
            },
            "current_segment": {
                "segment_id": current_segment.segment_id,
                "topic": current_segment.topic,
                "intent": current_segment.intent,
                "goal": current_segment.goal,
                "status": current_segment.status,
                "summary": current_segment.summary,
            },
            "historical_segments": [
                {
                    "segment_id": segment.segment_id,
                    "topic": segment.topic,
                    "intent": segment.intent,
                    "goal": segment.goal,
                    "status": segment.status,
                    "summary": segment.summary,
                    "updated_at": segment.updated_at,
                }
                for segment in historical_segments[-10:]
            ],
            "current_user_message": user_message,
            "required_json_schema": {
                "action": "append_current_segment",
                "resume_source_segment_id": None,
                "reason": "一句话说明原因",
                "confidence": 0.82,
            },
        }

        try:
            response = self._extractor_llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
                ]
            )
        except Exception:
            return None, None

        parsed = self._extract_json_object(
            self._response_to_text(getattr(response, "content", response))
        )
        if not isinstance(parsed, dict):
            return None, None

        action = self._normalize_turn_action_choice(parsed.get("action"))
        if action is None:
            return None, None

        try:
            confidence = float(parsed.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        if confidence < 0.65:
            return None, None

        if action == "append_current_segment":
            return action, current_segment
        if action == "fork_new_segment":
            return action, None
        if action != "resume_historical_segment":
            return None, None

        resume_source_id = str(parsed.get("resume_source_segment_id") or "").strip()
        if not resume_source_id:
            return None, None

        historical_map = {
            segment.segment_id: segment
            for segment in historical_segments
        }
        return action, historical_map.get(resume_source_id)

    def _resolve_effective_current_segment(
        self,
        current_experience: Experience,
        current_segment: Optional[Segment],
        experience_segments: Optional[Sequence[Segment]] = None,
    ) -> Optional[Segment]:
        if (
            current_segment is not None
            and current_segment.experience_id == current_experience.experience_id
        ):
            return current_segment

        preferred_segment = self.load_segment(current_experience.current_segment_id)
        if (
            preferred_segment is not None
            and preferred_segment.experience_id == current_experience.experience_id
        ):
            return preferred_segment

        segments = list(experience_segments or [])
        if not segments:
            segments = self.load_segments_for_experience(current_experience.experience_id)
        return self._latest_by_updated_at(segments)

    def _find_best_matching_experience(
        self,
        analysis: TopicIntent,
        exclude_experience_ids: Optional[Sequence[str]] = None,
    ) -> Optional[Experience]:
        excluded = {experience_id for experience_id in (exclude_experience_ids or []) if experience_id}
        scored: List[Tuple[float, Experience]] = []
        analysis_topic = self._experience_topic_key(analysis.domain, analysis.topic)

        for experience in self.list_experiences():
            if experience.experience_id in excluded:
                continue
            if not self._experience_matches_analysis(experience, analysis):
                continue

            experience_topic = self._experience_topic_key(experience.domain, experience.topic)
            topic_score = self._soft_text_overlap(experience_topic, analysis_topic)
            if experience_topic == analysis_topic:
                topic_score = max(topic_score, 1.0)
            goal_score = self._soft_text_overlap(experience.goal, analysis.goal)
            segment_bonus = 0.1 if self._find_best_matching_segment_for_analysis(
                experience_id=experience.experience_id,
                analysis=analysis,
            ) else 0.0
            score = round(
                topic_score * 0.56
                + goal_score * 0.24
                + segment_bonus
                + self._recency_bonus(experience.updated_at),
                4,
            )
            scored.append((score, experience))

        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return scored[0][1] if scored else None

    def _find_best_matching_segment_for_analysis(
        self,
        experience_id: str,
        analysis: TopicIntent,
        exclude_segment_ids: Optional[Sequence[str]] = None,
    ) -> Optional[Segment]:
        excluded = {segment_id for segment_id in (exclude_segment_ids or []) if segment_id}
        scored: List[Tuple[float, Segment]] = []

        for segment in self.load_segments_for_experience(experience_id):
            if segment.segment_id in excluded:
                continue
            if segment.intent != analysis.primary_intent:
                continue
            if not self._goals_align(segment.goal, analysis.goal, ""):
                continue

            goal_score = self._soft_text_overlap(segment.goal, analysis.goal)
            topic_score = self._soft_text_overlap(segment.topic, analysis.topic)
            if self._topics_equal(segment.topic, analysis.topic):
                topic_score = max(topic_score, 1.0)
            status_bonus = 0.06 if self._normalize_segment_status(segment.status) == "open" else 0.0
            score = round(
                goal_score * 0.58
                + topic_score * 0.18
                + status_bonus
                + self._recency_bonus(segment.updated_at),
                4,
            )
            scored.append((score, segment))

        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return scored[0][1] if scored else None

    def _normalize_turn_action_choice(self, value: Any) -> Optional[TurnActionType]:
        normalized = str(value or "").strip().lower()
        aliases = {
            "append": "append_current_segment",
            "append_segment": "append_current_segment",
            "fork": "fork_new_segment",
            "fork_new_segment_same_experience": "fork_new_segment",
            "resume": "resume_historical_segment",
            "resume_historical_segment_same_experience": "resume_historical_segment",
            "resume_previous_segment": "resume_historical_segment",
            "switch": "switch_experience",
            "new_experience": "switch_experience",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in {
            "append_current_segment",
            "fork_new_segment",
            "resume_historical_segment",
            "switch_experience",
        }:
            return normalized  # type: ignore[return-value]
        return None

    def _resolve_turn_target_experience(
        self,
        analysis: TopicIntent,
        runtime: RuntimeState,
        current_experience: Optional[Experience],
        query_text: str,
    ) -> Optional[Experience]:
        if current_experience and self._experience_matches_analysis(current_experience, analysis):
            return current_experience

        experiences = self.retrieve_experiences(analysis, query_text, runtime=runtime, limit=3)
        for experience in experiences:
            if self._experience_matches_analysis(experience, analysis):
                return experience
        for experience in experiences:
            open_segment = self._load_open_segment_for_experience(experience)
            if self._can_append_to_segment(open_segment, analysis, query_text):
                return experience
            if self._find_resume_candidate_segment(experience.experience_id, analysis):
                return experience
        return None

    def _can_append_to_segment(
        self,
        segment: Optional[Segment],
        analysis: TopicIntent,
        user_message: str,
    ) -> bool:
        if segment is None:
            return False
        if self._normalize_segment_status(segment.status) != "open":
            return False
        if self._topics_equal(segment.topic, analysis.topic):
            return True

        topic_overlap = self._soft_text_overlap(segment.topic, analysis.topic)
        if topic_overlap < 0.32:
            return False
        if segment.intent == analysis.primary_intent and self._goals_align(
            segment.goal,
            analysis.goal,
            user_message,
        ):
            return True
        if analysis.relation_to_previous == "follow_up" and topic_overlap >= 0.5:
            return True
        return False

    def _find_resume_candidate_segment(
        self,
        experience_id: str,
        analysis: TopicIntent,
        exclude_segment_ids: Optional[Sequence[str]] = None,
    ) -> Optional[Segment]:
        excluded = {segment_id for segment_id in (exclude_segment_ids or []) if segment_id}
        candidates = [
            segment
            for segment in self.load_segments_for_experience(experience_id)
            if segment.segment_id not in excluded
            and self._normalize_segment_status(segment.status) != "open"
            and self._topics_equal(segment.topic, analysis.topic)
            and segment.intent == analysis.primary_intent
        ]
        if not candidates:
            return None

        scored: List[Tuple[float, Segment]] = []
        for segment in candidates:
            goal_score = 0.22 * self._soft_text_overlap(segment.goal, analysis.goal)
            recency_score = self._recency_bonus(segment.updated_at)
            score = round(0.78 + goal_score + recency_score, 4)
            if score >= 0.8:
                scored.append((score, segment))

        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return scored[0][1] if scored else None

    def _select_relation_anchor_segment(
        self,
        experience_id: str,
        analysis: TopicIntent,
        recent_segment_ids: Sequence[str],
        closed_anchor_segment: Optional[Segment] = None,
        exclude_segment_ids: Optional[Sequence[str]] = None,
    ) -> Optional[Segment]:
        excluded = {segment_id for segment_id in (exclude_segment_ids or []) if segment_id}
        if (
            closed_anchor_segment
            and closed_anchor_segment.experience_id == experience_id
            and closed_anchor_segment.segment_id not in excluded
        ):
            return closed_anchor_segment

        closest_historical = self._find_relation_anchor_segment(
            experience_id=experience_id,
            analysis=analysis,
            exclude_segment_ids=list(excluded),
        )
        if closest_historical:
            return closest_historical

        return self._find_recent_related_segment(
            experience_id=experience_id,
            analysis=analysis,
            recent_segment_ids=recent_segment_ids,
            exclude_segment_ids=list(excluded),
        )

    def _find_recent_related_segment(
        self,
        experience_id: str,
        analysis: TopicIntent,
        recent_segment_ids: Sequence[str],
        exclude_segment_ids: Optional[Sequence[str]] = None,
    ) -> Optional[Segment]:
        excluded = {segment_id for segment_id in (exclude_segment_ids or []) if segment_id}
        recent_rank = {
            segment_id: max(len(recent_segment_ids) - index, 1)
            for index, segment_id in enumerate(recent_segment_ids)
        }
        scored: List[Tuple[float, Segment]] = []
        for segment in self.load_segments_for_experience(experience_id):
            if segment.segment_id in excluded:
                continue
            if self._normalize_segment_status(segment.status) == "open":
                continue
            topic_overlap = self._soft_text_overlap(segment.topic, analysis.topic)
            goal_overlap = self._soft_text_overlap(segment.goal, analysis.goal)
            intent_bonus = 0.12 if segment.intent == analysis.primary_intent else 0.0
            recent_bonus = 0.18 if segment.segment_id in recent_rank else 0.0
            score = round(
                topic_overlap * 0.32
                + goal_overlap * 0.18
                + intent_bonus
                + recent_bonus
                + self._recency_bonus(segment.updated_at),
                4,
            )
            if segment.segment_id in recent_rank or score >= 0.18:
                scored.append((score, segment))

        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return scored[0][1] if scored else None

    def _create_experience(self, analysis: TopicIntent) -> Experience:
        now = _now_iso()
        experience = Experience(
            experience_id=_make_id("exp"),
            domain=analysis.domain,
            topic=self._experience_topic_key(analysis.domain, analysis.topic),
            goal=analysis.goal,
            segment_ids=[],
            current_segment_id=None,
            main_segment_path=[],
            summary=ExperienceSummary(short="", long=""),
            memory=ExperienceMemory(),
            status="in_progress",
            created_at=now,
            updated_at=now,
            last_summarized_segment_count=0,
            last_summarized_qa_count=0,
            last_summarized_relation_count=0,
            last_summarized_at="",
            version=1,
        )
        self.save_experience(experience)
        return experience

    def _experience_topic_key(self, domain: DomainType, topic: str) -> str:
        normalized = re.sub(r"\s+", " ", str(topic or "").strip())
        if domain == "weather":
            return "天气查询"
        return normalized or "general topic"

    def _experience_matches_analysis(
        self,
        experience: Experience,
        analysis: TopicIntent,
    ) -> bool:
        if experience.domain != analysis.domain:
            return False

        experience_topic = self._experience_topic_key(experience.domain, experience.topic)
        analysis_topic = self._experience_topic_key(analysis.domain, analysis.topic)
        if experience_topic == analysis_topic:
            return True

        topic_overlap = self._soft_text_overlap(experience_topic, analysis_topic)
        if topic_overlap >= 0.24:
            return True

        goal_overlap = self._soft_text_overlap(experience.goal, analysis.goal)
        return topic_overlap >= 0.14 and goal_overlap >= 0.12

    def _create_segment(self, experience_id: str, analysis: TopicIntent) -> Segment:
        now = _now_iso()
        segment = Segment(
            segment_id=_make_id("seg"),
            experience_id=experience_id,
            domain=analysis.domain,
            topic=analysis.topic,
            intent=analysis.primary_intent,
            goal=analysis.goal,
            qa_ids=[],
            status="open",
            summary="",
            memory=SegmentMemory(),
            created_at=now,
            updated_at=now,
            last_summarized_qa_count=0,
            last_summarized_at="",
            version=1,
        )
        self.save_segment(segment)
        return segment

    def _build_qa(
        self,
        session_id: str,
        analysis: TopicIntent,
        user_message: str,
        assistant_reply: str,
        turn_messages: Sequence[Message],
        segment_id: str,
        experience_id: str,
    ) -> QA:
        timestamp = _now_iso()
        tools = [
            ToolCallRecord(
                tool_name=tool_name,
                role=TOOL_ROLE_MAP.get(tool_name, f"调用工具 `{tool_name}` 辅助完成任务"),
            )
            for tool_name in self._extract_tool_sequence(turn_messages)
        ]
        entities = self._extract_entities(user_message, assistant_reply)
        facts = self._extract_facts(assistant_reply)
        constraints = self._extract_constraints(user_message, assistant_reply)
        return QA(
            qa_id=_make_id("qa"),
            conversation_id=session_id,
            timestamp=timestamp,
            user_input=user_message.strip(),
            assistant_output=assistant_reply.strip(),
            tools=tools,
            domain=analysis.domain,
            topic=analysis.topic,
            intent=analysis.primary_intent,
            goal=analysis.goal,
            entities=entities,
            facts=facts,
            constraints=constraints,
            segment_id=segment_id,
            experience_id=experience_id,
            status="active",
            confidence=analysis.confidence,
        )

    def _persist_turn_update(
        self,
        session_id: str,
        analysis: TopicIntent,
        user_message: str,
        assistant_reply: str,
        turn_messages: Sequence[Message],
        target_experience: Experience,
        target_segment: Segment,
        active_runtime: RuntimeState,
        now: str,
        experiences_to_recompute: Set[str],
        topic_shift: bool,
        new_segment_created: bool,
    ) -> TopicTurnUpdate:
        qa = self._build_qa(
            session_id=session_id,
            analysis=analysis,
            user_message=user_message,
            assistant_reply=assistant_reply,
            turn_messages=turn_messages,
            segment_id=target_segment.segment_id,
            experience_id=target_experience.experience_id,
        )
        self.save_qa(qa)

        target_segment.qa_ids = self._merge_unique(target_segment.qa_ids, [qa.qa_id], limit=500)
        target_segment.domain = target_segment.domain or qa.domain
        target_segment.topic = target_segment.topic or qa.topic
        target_segment.intent = target_segment.intent or qa.intent
        target_segment.goal = target_segment.goal or qa.goal
        previous_target_status = self._normalize_segment_status(target_segment.status)
        target_segment.status = "open"
        target_segment.updated_at = now
        target_segment.version += 1
        segment_qas = self.load_qas(target_segment.qa_ids)
        target_segment.memory = self._build_segment_memory(segment_qas)
        self._refresh_segment_summary(
            segment=target_segment,
            qas=segment_qas,
            user_message=user_message,
            previous_status=previous_target_status,
            analysis=analysis,
            force=False,
        )
        self.save_segment(target_segment)

        target_experience.segment_ids = self._merge_unique(
            target_experience.segment_ids,
            [target_segment.segment_id],
            limit=500,
        )
        target_experience.current_segment_id = target_segment.segment_id
        target_experience.domain = target_experience.domain or qa.domain
        target_experience.topic = self._experience_topic_key(
            target_experience.domain,
            target_experience.topic or qa.topic,
        )
        target_experience.goal = target_experience.goal or qa.goal
        target_experience.updated_at = now
        target_experience.version += 1
        if new_segment_created:
            target_experience.main_segment_path = self._merge_unique(
                target_experience.main_segment_path,
                [target_segment.segment_id],
                limit=500,
            )
        self.save_experience(target_experience)

        experience_ids_to_refresh = set(experiences_to_recompute)
        experience_ids_to_refresh.add(target_experience.experience_id)
        for experience_id in sorted(experience_ids_to_refresh):
            self._recompute_experience(
                experience_id=experience_id,
                user_message=user_message,
                analysis=analysis,
            )

        active_runtime.current_experience_id = target_experience.experience_id
        active_runtime.current_segment_id = target_segment.segment_id
        active_runtime.latest_qa_id = qa.qa_id
        active_runtime.recent_segment_ids = self._prepend_recent_segment(
            active_runtime.recent_segment_ids,
            target_segment.segment_id,
        )
        active_runtime.updated_at = now
        self.save_runtime_state(active_runtime)

        self._repair_topic_memory_store(preferred_experience_id=target_experience.experience_id)
        self._refresh_store_artifacts(conversation_id=session_id)

        refreshed_segment = self.load_segment(target_segment.segment_id)
        return TopicTurnUpdate(
            analysis=analysis,
            topic_shift=topic_shift,
            topic_session_id=target_segment.segment_id,
            topic_session_status=refreshed_segment.status if refreshed_segment else "open",
        )

    def _close_segment_for_switch(
        self,
        segment: Segment,
        interrupted: bool,
        user_message: str = "",
        analysis: Optional[TopicIntent] = None,
    ):
        if segment.status != "open":
            return
        next_status: SegmentStatus = "interrupted" if interrupted else "completed"
        self._finalize_segment(
            segment=segment,
            next_status=next_status,
            user_message=user_message,
            analysis=analysis,
        )

    def _finalize_segment(
        self,
        segment: Segment,
        next_status: SegmentStatus,
        user_message: str = "",
        analysis: Optional[TopicIntent] = None,
    ):
        previous_status = self._normalize_segment_status(segment.status)
        segment_qas = self.load_qas(segment.qa_ids)
        segment.memory = self._build_segment_memory(segment_qas)
        segment.status = next_status
        segment.updated_at = _now_iso()
        self._refresh_segment_summary(
            segment=segment,
            qas=segment_qas,
            user_message=user_message,
            previous_status=previous_status,
            analysis=analysis,
            force=next_status in FINALIZED_SEGMENT_STATUSES,
        )
        segment.version += 1
        self.save_segment(segment)

    def _recompute_experience(
        self,
        experience_id: str,
        user_message: str = "",
        analysis: Optional[TopicIntent] = None,
        force_summary: bool = False,
    ):
        experience = self.load_experience(experience_id)
        if not experience:
            return

        segments = [
            segment
            for segment in self.load_segments_for_experience(experience.experience_id)
        ]
        segment_id_set = {segment.segment_id for segment in segments}
        experience.segment_ids = [segment.segment_id for segment in segments]
        experience.topic = self._experience_topic_key(experience.domain, experience.topic)
        experience.goal = self._recompute_experience_goal(experience, segments)
        experience.main_segment_path = [
            segment_id
            for segment_id in experience.main_segment_path
            if segment_id in segment_id_set
        ]
        if not experience.main_segment_path:
            experience.main_segment_path = [segment.segment_id for segment in segments]
        qas: List[QA] = []
        for segment in segments:
            qas.extend(self.load_qas(segment.qa_ids))

        experience.memory = self._build_experience_memory(segments)
        finalized_segments = self._summarizable_experience_segments(segments)
        finalized_qas: List[QA] = []
        for segment in finalized_segments:
            finalized_qas.extend(self.load_qas(segment.qa_ids))
        relation_count = self._count_experience_relations(experience.experience_id)
        all_qas: List[QA] = []
        for segment in segments:
            all_qas.extend(self.load_qas(segment.qa_ids))
        summary_segments = finalized_segments or list(segments)
        summary_qas = finalized_qas or list(all_qas)
        if self._should_summarize_experience(
            experience=experience,
            segments=segments,
            qas=all_qas,
            relation_count=relation_count,
            user_message=user_message,
            analysis=analysis,
            force=force_summary,
        ):
            experience.summary = self._summarize_experience(
                experience,
                summary_segments,
                summary_qas,
            )
            experience.last_summarized_segment_count = len(finalized_segments)
            experience.last_summarized_qa_count = len(all_qas)
            experience.last_summarized_relation_count = relation_count
            experience.last_summarized_at = _now_iso()
        elif not experience.summary.short and not experience.summary.long and len(finalized_segments) >= 2:
            experience.summary = self._summarize_experience(
                experience,
                summary_segments,
                summary_qas,
            )
            experience.last_summarized_segment_count = len(finalized_segments)
            experience.last_summarized_qa_count = len(all_qas)
            experience.last_summarized_relation_count = relation_count
            experience.last_summarized_at = _now_iso()
        experience.status = self._infer_experience_status(segments)
        if experience.current_segment_id and experience.current_segment_id not in {
            segment.segment_id for segment in segments
        }:
            experience.current_segment_id = segments[-1].segment_id if segments else None
        experience.updated_at = _now_iso()
        self.save_experience(experience)

    def _recompute_experience_goal(
        self,
        experience: Experience,
        segments: Sequence[Segment],
    ) -> str:
        goals = self._merge_unique([], [segment.goal for segment in segments], limit=32)
        if experience.domain == "weather" and len(goals) > 1:
            return "围绕天气查询相关需求进行持续跟进与结果说明"
        if experience.goal.strip():
            return experience.goal
        return goals[-1] if goals else ""

    def _repair_topic_memory_store(
        self,
        preferred_experience_id: Optional[str] = None,
    ) -> Dict[str, str]:
        for experience in self.list_experiences():
            normalized_topic = self._experience_topic_key(experience.domain, experience.topic)
            if experience.topic == normalized_topic:
                continue
            experience.topic = normalized_topic
            self.save_experience(experience)

        return self._consolidate_overlapping_experiences(
            preferred_experience_id=preferred_experience_id,
        )

    def _consolidate_overlapping_experiences(
        self,
        preferred_experience_id: Optional[str] = None,
    ) -> Dict[str, str]:
        experiences = sorted(
            self.list_experiences(),
            key=lambda item: (item.updated_at, item.created_at, item.experience_id),
            reverse=True,
        )
        if len(experiences) < 2:
            return {}

        groups: List[List[Experience]] = []
        for experience in experiences:
            placed = False
            for group in groups:
                if any(
                    self._experiences_should_merge(candidate, experience)
                    for candidate in group
                ):
                    group.append(experience)
                    placed = True
                    break
            if not placed:
                groups.append([experience])

        merged_into: Dict[str, str] = {}
        for group in groups:
            if len(group) < 2:
                continue
            primary = self._select_primary_experience(group, preferred_experience_id)
            duplicates = [
                experience
                for experience in group
                if experience.experience_id != primary.experience_id
            ]
            if not duplicates:
                continue
            self._merge_experience_group(primary, duplicates)
            for duplicate in duplicates:
                merged_into[duplicate.experience_id] = primary.experience_id

        return merged_into

    def _experiences_should_merge(
        self,
        left: Experience,
        right: Experience,
    ) -> bool:
        if left.experience_id == right.experience_id:
            return False
        if left.domain != right.domain:
            return False

        left_topic = self._experience_topic_key(left.domain, left.topic)
        right_topic = self._experience_topic_key(right.domain, right.topic)
        if left_topic == right_topic:
            return True

        topic_overlap = self._soft_text_overlap(left_topic, right_topic)
        if topic_overlap >= 0.28:
            return True

        goal_overlap = self._soft_text_overlap(left.goal, right.goal)
        return topic_overlap >= 0.16 and goal_overlap >= 0.1

    def _select_primary_experience(
        self,
        group: Sequence[Experience],
        preferred_experience_id: Optional[str],
    ) -> Experience:
        if preferred_experience_id:
            for experience in group:
                if experience.experience_id == preferred_experience_id:
                    return experience
        return max(
            group,
            key=lambda item: (item.updated_at, item.created_at, item.experience_id),
        )

    def _merge_experience_group(
        self,
        primary: Experience,
        duplicates: Sequence[Experience],
    ):
        moved_segment_ids: Set[str] = set()
        merged_path = list(primary.main_segment_path)

        for duplicate in duplicates:
            duplicate_segments = self.load_segments_for_experience(duplicate.experience_id)
            merged_path = self._merge_unique(
                merged_path,
                duplicate.main_segment_path or [segment.segment_id for segment in duplicate_segments],
                limit=500,
            )

            for segment in duplicate_segments:
                segment.experience_id = primary.experience_id
                self.save_segment(segment)
                moved_segment_ids.add(segment.segment_id)
                for qa in self.load_qas(segment.qa_ids):
                    qa.experience_id = primary.experience_id
                    self.save_qa(qa)

            for relation in self.list_relations():
                if relation.experience_id != duplicate.experience_id:
                    continue
                relation.experience_id = primary.experience_id
                self.save_relation(relation)

            duplicate_path = os.path.join(self.experiences_dir, f"{duplicate.experience_id}.json")
            if os.path.exists(duplicate_path):
                os.remove(duplicate_path)

        primary = self.load_experience(primary.experience_id) or primary
        primary.topic = self._experience_topic_key(primary.domain, primary.topic)
        primary.main_segment_path = merged_path
        if moved_segment_ids and (
            not primary.current_segment_id or primary.current_segment_id in moved_segment_ids
        ):
            latest_segment = self._latest_by_updated_at(
                self.load_segments_for_experience(primary.experience_id),
            )
            primary.current_segment_id = latest_segment.segment_id if latest_segment else primary.current_segment_id
        self.save_experience(primary)

        for runtime in self.list_runtime_states():
            changed = False
            if runtime.current_experience_id in {item.experience_id for item in duplicates}:
                runtime.current_experience_id = primary.experience_id
                changed = True
            if runtime.current_segment_id in moved_segment_ids:
                runtime.current_experience_id = primary.experience_id
                changed = True
            if changed:
                runtime.updated_at = _now_iso()
                self.save_runtime_state(runtime)

        self._recompute_experience(primary.experience_id)

    def _summarizable_experience_segments(self, segments: Sequence[Segment]) -> List[Segment]:
        return [
            segment
            for segment in segments
            if segment.status in FINALIZED_SEGMENT_STATUSES and bool(segment.summary.strip())
        ]

    def load_segments_for_experience(self, experience_id: str) -> List[Segment]:
        segments = [
            segment
            for segment in self.list_segments()
            if segment.experience_id == experience_id
        ]
        segments.sort(key=lambda item: item.updated_at)
        return segments

    def _infer_experience_status(self, segments: Sequence[Segment]) -> ExperienceStatus:
        if any(self._normalize_segment_status(segment.status) == "open" for segment in segments):
            return "in_progress"
        if segments and all(
            self._normalize_segment_status(segment.status) == "archived"
            for segment in segments
        ):
            return "archived"
        if segments:
            return "completed"
        return "archived"

    def _add_or_update_relation(
        self,
        experience_id: str,
        from_segment_id: str,
        to_segment_id: str,
        relation_type: SegmentRelationType,
        reason: str,
        weight: float,
    ):
        normalized_relation_type = self._normalize_relation_type(relation_type)
        if normalized_relation_type is None:
            return
        for relation in self.list_relations():
            if (
                relation.experience_id == experience_id
                and relation.from_segment_id == from_segment_id
                and relation.to_segment_id == to_segment_id
                and self._normalize_relation_type(relation.relation_type) == normalized_relation_type
            ):
                relation.relation_type = normalized_relation_type
                relation.reason = reason
                relation.weight = weight
                self.save_relation(relation)
                return

        relation = SegmentRelation(
            relation_id=_make_id("rel"),
            experience_id=experience_id,
            from_segment_id=from_segment_id,
            to_segment_id=to_segment_id,
            relation_type=normalized_relation_type,
            reason=reason,
            weight=round(weight, 3),
            created_at=_now_iso(),
        )
        self.save_relation(relation)

    def _normalize_relation_type(self, value: Any) -> Optional[SegmentRelationType]:
        normalized = str(value or "").strip()
        if normalized == "continues_from":
            normalized = "continue_from"
        if normalized in {"continue_from", "refines", "contradicts"}:
            return normalized  # type: ignore[return-value]
        return None

    def _normalize_segment_status(self, value: Any) -> SegmentStatus:
        normalized = str(value or "").strip()
        if normalized == "continued":
            normalized = "inherited"
        if normalized in {"open", "completed", "interrupted", "inherited", "archived"}:
            return normalized  # type: ignore[return-value]
        return "open"

    def _normalized_topic_label(self, value: Any) -> str:
        normalized = re.sub(r"\s+", " ", str(value or "").strip())
        normalized = normalized.strip("。；;，,：:[]{}()（）")
        return normalized[:48]

    def _topics_equal(self, left: Any, right: Any) -> bool:
        left_topic = self._normalized_topic_label(left)
        right_topic = self._normalized_topic_label(right)
        return bool(left_topic and right_topic and left_topic == right_topic)

    def _determine_segment_relation(
        self,
        current_segment: Segment,
        previous_segment: Segment,
    ) -> Optional[Tuple[SegmentRelationType, str, float]]:
        if current_segment.segment_id == previous_segment.segment_id:
            return None
        if current_segment.experience_id != previous_segment.experience_id:
            return None
        llm_status, llm_relation = self._llm_determine_segment_relation(
            current_segment=current_segment,
            previous_segment=previous_segment,
        )
        if llm_status == "resolved" and llm_relation is not None:
            return llm_relation
        if llm_status == "no_relation":
            return None
        return self._heuristic_determine_segment_relation(
            current_segment=current_segment,
            previous_segment=previous_segment,
        )

    def _llm_determine_segment_relation(
        self,
        current_segment: Segment,
        previous_segment: Segment,
    ) -> Tuple[str, Optional[Tuple[SegmentRelationType, str, float]]]:
        if self._extractor_llm is None:
            return "error", None

        system_prompt = (
            "你是 segment relation 分类器。\n"
            "请判断 current_segment 相对于 previous_segment 的关系。\n"
            "只能输出 JSON 对象，不要输出 Markdown，不要解释。\n"
            'relation_type 只能是 "continue_from"、"refines"、"contradicts"、"none" 之一。\n'
            "判定规则：\n"
            "1. continue_from: 当前 segment 承接旧 segment，尤其是同主题、同意图、同目标的延续。\n"
            "2. refines: 当前 segment 是对旧 segment 的扩展、细化、实现、验证或补充。\n"
            "3. contradicts: 当前 segment 与旧 segment 的目标、约束、结论存在明显冲突。\n"
            "4. none: 不存在明确关系，或证据不足。\n"
            "输出字段：relation_type, reason, confidence。"
        )
        payload = {
            "previous_segment": {
                "segment_id": previous_segment.segment_id,
                "topic": previous_segment.topic,
                "intent": previous_segment.intent,
                "goal": previous_segment.goal,
                "status": previous_segment.status,
                "summary": previous_segment.summary,
            },
            "current_segment": {
                "segment_id": current_segment.segment_id,
                "topic": current_segment.topic,
                "intent": current_segment.intent,
                "goal": current_segment.goal,
                "status": current_segment.status,
                "summary": current_segment.summary,
            },
            "required_json_schema": {
                "relation_type": "none",
                "reason": "一句话说明原因",
                "confidence": 0.72,
            },
        }

        try:
            response = self._extractor_llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
                ]
            )
        except Exception:
            return "error", None

        parsed = self._extract_json_object(
            self._response_to_text(getattr(response, "content", response))
        )
        if not isinstance(parsed, dict):
            return "error", None

        relation_value = str(parsed.get("relation_type", "")).strip().lower()
        if relation_value in {"", "none", "null", "no_relation"}:
            return "no_relation", None

        relation_type = self._normalize_relation_type(relation_value)
        if relation_type is None:
            return "error", None

        try:
            confidence = float(parsed.get("confidence", 0.0))
        except Exception:
            return "error", None
        confidence = max(0.0, min(1.0, round(confidence, 3)))
        if confidence < 0.65:
            return "no_relation", None

        reason = re.sub(r"\s+", " ", str(parsed.get("reason", "")).strip())
        if not reason:
            reason = "LLM 判断这两个 segment 存在明确的同 experience 内语义关系。"

        return "resolved", (relation_type, reason[:240], confidence)

    def _heuristic_determine_segment_relation(
        self,
        current_segment: Segment,
        previous_segment: Segment,
    ) -> Optional[Tuple[SegmentRelationType, str, float]]:
        if current_segment.domain != previous_segment.domain:
            return None

        topic_overlap = self._soft_text_overlap(current_segment.topic, previous_segment.topic)
        goal_overlap = self._soft_text_overlap(current_segment.goal, previous_segment.goal)
        same_topic = self._topics_equal(current_segment.topic, previous_segment.topic) or topic_overlap >= 0.3

        if same_topic and current_segment.intent == previous_segment.intent and goal_overlap >= 0.08:
            return (
                "continue_from",
                "同一 experience 下新建 segment，主题、意图与目标高度一致，判定为承接关系。",
                0.88,
            )
        if same_topic and self._segments_contradict(current_segment, previous_segment):
            return (
                "contradicts",
                "同一 experience 下两个 segment 的目标或约束表达存在明显冲突。",
                0.8,
            )
        if same_topic and self._segments_refine(current_segment, previous_segment):
            return (
                "refines",
                "同一 experience 下新 segment 对旧 segment 形成了补充、展开或实现。",
                0.72,
            )
        return None

    def _find_relation_anchor_segment(
        self,
        experience_id: str,
        analysis: TopicIntent,
        exclude_segment_ids: Optional[Sequence[str]] = None,
    ) -> Optional[Segment]:
        excluded = {segment_id for segment_id in (exclude_segment_ids or []) if segment_id}
        candidates = [
            segment
            for segment in self.load_segments_for_experience(experience_id)
            if segment.segment_id not in excluded
            and self._normalize_segment_status(segment.status) != "open"
        ]
        if not candidates:
            return None

        scored: List[Tuple[float, Segment]] = []
        for segment in candidates:
            topic_score = self._soft_text_overlap(segment.topic, analysis.topic)
            if self._topics_equal(segment.topic, analysis.topic):
                topic_score = max(topic_score, 1.0)
            intent_bonus = 0.24 if segment.intent == analysis.primary_intent else 0.0
            goal_score = 0.18 * self._soft_text_overlap(segment.goal, analysis.goal)
            time_score = self._recency_bonus(segment.updated_at)
            score = round(topic_score * 0.5 + intent_bonus + goal_score + time_score, 4)
            if score > 0.3:
                scored.append((score, segment))

        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return scored[0][1] if scored else None

    def _segment_matches_analysis(
        self,
        segment: Optional[Segment],
        analysis: TopicIntent,
        user_message: str,
    ) -> bool:
        return self._can_append_to_segment(segment, analysis, user_message)

    def _should_append_to_current_segment(
        self,
        current_segment: Optional[Segment],
        analysis: TopicIntent,
        explicit_pause: bool,
        explicit_resume: bool,
    ) -> bool:
        return self._can_append_to_segment(current_segment, analysis, "")

    def _load_open_segment_for_experience(self, experience: Optional[Experience]) -> Optional[Segment]:
        if experience is None:
            return None

        preferred_segment = self.load_segment(experience.current_segment_id)
        if preferred_segment and preferred_segment.experience_id == experience.experience_id:
            preferred_segment.status = self._normalize_segment_status(preferred_segment.status)
            if preferred_segment.status == "open":
                return preferred_segment

        open_segments = [
            segment
            for segment in self.load_segments_for_experience(experience.experience_id)
            if self._normalize_segment_status(segment.status) == "open"
        ]
        return self._latest_by_updated_at(open_segments)

    def _find_inherited_source_segment(
        self,
        experience_id: str,
        analysis: TopicIntent,
        exclude_segment_ids: Optional[Sequence[str]] = None,
        force: bool = False,
    ) -> Optional[Segment]:
        candidate = self._find_resume_candidate_segment(
            experience_id=experience_id,
            analysis=analysis,
            exclude_segment_ids=exclude_segment_ids,
        )
        if candidate is not None or not force:
            return candidate
        return self._find_relation_anchor_segment(
            experience_id=experience_id,
            analysis=analysis,
            exclude_segment_ids=exclude_segment_ids,
        )

    def _is_summary_request(
        self,
        user_message: str,
        analysis: Optional[TopicIntent] = None,
    ) -> bool:
        lowered = str(user_message or "").lower()
        if analysis and analysis.primary_intent == "summarization":
            return True
        return any(term in lowered for term in ("总结", "归纳", "汇总", "summary", "summarize"))

    def _should_summarize_segment(
        self,
        segment: Segment,
        qas: Sequence[QA],
        user_message: str,
        previous_status: Optional[SegmentStatus] = None,
        analysis: Optional[TopicIntent] = None,
    ) -> bool:
        normalized_status = self._normalize_segment_status(segment.status)
        if (
            previous_status == "open"
            and normalized_status in FINALIZED_SEGMENT_STATUSES
        ):
            return True
        if normalized_status in FINALIZED_SEGMENT_STATUSES and not segment.summary.strip():
            return True
        if (
            normalized_status == "open"
            and len(qas) - segment.last_summarized_qa_count >= SEGMENT_SUMMARY_QA_INTERVAL
        ):
            return True
        return self._is_summary_request(user_message, analysis=analysis)

    def _refresh_segment_summary(
        self,
        segment: Segment,
        qas: Sequence[QA],
        user_message: str,
        previous_status: Optional[SegmentStatus] = None,
        analysis: Optional[TopicIntent] = None,
        force: bool = False,
    ):
        if not qas:
            return
        if not force and not self._should_summarize_segment(
            segment=segment,
            qas=qas,
            user_message=user_message,
            previous_status=previous_status,
            analysis=analysis,
        ):
            return

        segment.summary = self._summarize_segment(segment, qas)
        segment.last_summarized_qa_count = len(qas)
        segment.last_summarized_at = _now_iso()

    def _count_experience_relations(self, experience_id: str) -> int:
        return sum(
            1
            for relation in self.list_relations()
            if relation.experience_id == experience_id
        )

    def _crossed_summary_milestone(
        self,
        previous_count: int,
        current_count: int,
        milestones: Sequence[int],
    ) -> bool:
        return any(previous_count < milestone <= current_count for milestone in milestones)

    def _should_summarize_experience(
        self,
        experience: Experience,
        segments: Sequence[Segment],
        qas: Sequence[QA],
        relation_count: int,
        user_message: str,
        analysis: Optional[TopicIntent] = None,
        force: bool = False,
    ) -> bool:
        if force or self._is_summary_request(user_message, analysis=analysis):
            return True

        finalized_segments = [
            segment
            for segment in segments
            if self._normalize_segment_status(segment.status) in FINALIZED_SEGMENT_STATUSES
        ]
        if (
            len(finalized_segments) - experience.last_summarized_segment_count
            >= EXPERIENCE_SUMMARY_FINALIZED_SEGMENT_INTERVAL
        ):
            return True
        if self._crossed_summary_milestone(
            previous_count=experience.last_summarized_qa_count,
            current_count=len(qas),
            milestones=EXPERIENCE_SUMMARY_QA_MILESTONES,
        ):
            return True
        if not any(self._normalize_segment_status(segment.status) == "open" for segment in segments):
            if len(qas) > experience.last_summarized_qa_count:
                return True
        if (
            relation_count - experience.last_summarized_relation_count
            >= EXPERIENCE_SUMMARY_RELATION_INTERVAL
        ):
            return True
        return False

    def _segments_contradict(
        self,
        current_segment: Segment,
        previous_segment: Segment,
    ) -> bool:
        goal_overlap = self._soft_text_overlap(current_segment.goal, previous_segment.goal)
        if goal_overlap < 0.18:
            return False
        return self._contains_any(current_segment.goal, CONTRADICTION_HINTS)

    def _segments_refine(
        self,
        current_segment: Segment,
        previous_segment: Segment,
    ) -> bool:
        goal_overlap = self._soft_text_overlap(current_segment.goal, previous_segment.goal)
        if current_segment.intent != previous_segment.intent and goal_overlap >= 0.12:
            return True
        return self._contains_any(current_segment.goal, REFINE_HINTS) and goal_overlap >= 0.08

    def _build_segment_memory(self, qas: Sequence[QA]) -> SegmentMemory:
        decisions: List[str] = []
        open_questions: List[str] = []
        for qa in qas:
            decisions.extend(self._extract_decisions(qa.assistant_output))
            open_questions.extend(self._extract_open_questions(qa.assistant_output))
        return SegmentMemory(
            facts=self._merge_unique([], [item for qa in qas for item in qa.facts], limit=8),
            constraints=self._merge_unique([], [item for qa in qas for item in qa.constraints], limit=8),
            decisions=self._merge_unique([], decisions, limit=8),
            open_questions=self._merge_unique([], open_questions, limit=6),
        )

    def _summarize_segment(self, segment: Segment, qas: Sequence[QA]) -> str:
        fact_text = self._render_numbered_items(segment.memory.facts, fallback="暂无关键事实。")
        constraint_text = self._render_numbered_items(
            segment.memory.constraints,
            fallback="暂无显式约束。",
        )
        return (
            f"目标：{segment.goal}。"
            f"QA数量：{len(qas)}。"
            f"关键facts：{fact_text}"
            f"关键constraints：{constraint_text}"
            f"{summarize_tools_from_qas(list(qas))}"
        )

    def _build_experience_memory(self, segments: Sequence[Segment]) -> ExperienceMemory:
        decisions = [item for segment in segments for item in segment.memory.decisions]
        constraints = [item for segment in segments for item in segment.memory.constraints]
        open_questions = [item for segment in segments for item in segment.memory.open_questions]
        return ExperienceMemory(
            decisions=self._merge_unique([], decisions, limit=10),
            constraints=self._merge_unique([], constraints, limit=10),
            open_questions=self._merge_unique([], open_questions, limit=10),
        )

    def _summarize_experience(
        self,
        experience: Experience,
        segments: Sequence[Segment],
        qas: Sequence[QA],
    ) -> ExperienceSummary:
        short = f"{experience.topic} / {experience.goal}"[:220]
        segment_lookup = {segment.segment_id: segment for segment in segments}
        ordered_path = [
            segment_lookup[segment_id]
            for segment_id in experience.main_segment_path
            if segment_id in segment_lookup
        ]
        if not ordered_path:
            ordered_path = list(segments)

        path_parts = []
        for segment in ordered_path[:8]:
            path_parts.append(
                f"{segment.segment_id}（{segment.status}，{len(segment.qa_ids)} 轮QA）：{segment.summary}"
            )
        path_text = "；".join(path_parts) if path_parts else "暂无路径推进。"

        tool_experience = self._summarize_tool_experience(qas)
        decisions_text = self._render_numbered_items(
            experience.memory.decisions,
            fallback="暂无关键决策。",
        )
        constraints_text = self._render_numbered_items(
            experience.memory.constraints,
            fallback="暂无关键约束。",
        )
        questions_text = self._render_numbered_items(
            experience.memory.open_questions,
            fallback="暂无待解问题。",
        )
        long = (
            f"主题目标：{experience.goal}。"
            f"主路径推进：{path_text}。"
            f"关键decisions：{decisions_text}"
            f"关键constraints：{constraints_text}"
            f"open_questions：{questions_text}"
            f"{tool_experience}"
        )
        return ExperienceSummary(short=short, long=long[:3000])

    def _display_segment_summary(self, segment: Segment) -> str:
        if segment.summary.strip():
            return segment.summary
        if segment.status == "open":
            return f"阶段进行中：目标={segment.goal}；当前已累计 {len(segment.qa_ids)} 轮 QA。"
        return "该阶段尚未生成总结。"

    def _render_tool_calls_for_prompt(self, tools: Sequence[ToolCallRecord]) -> str:
        if not tools:
            return "无"
        return "；".join(
            f"{tool.tool_name}：{tool.role}"
            for tool in tools
        )

    def _render_qa_prompt_block(self, qa: QA) -> List[str]:
        return [
            f"- 用户输入：{qa.user_input[:180]}",
            f"  工具调用：{self._render_tool_calls_for_prompt(qa.tools)}",
            f"  模型回答：{qa.assistant_output[:260]}",
        ]

    def _summarize_tool_experience(self, qas: Sequence[QA]) -> str:
        tools: List[ToolCallRecord] = []
        seen: Set[Tuple[str, str]] = set()
        for qa in qas:
            for tool in qa.tools:
                key = (tool.tool_name, tool.role)
                if key in seen:
                    continue
                seen.add(key)
                tools.append(tool)
        if not tools:
            return "未形成明确的工具调用经验。"
        if len(tools) == 1:
            tool = tools[0]
            return f"工具使用经验：通常使用 {tool.tool_name}（{tool.role}）完成任务。"

        ordered = []
        for index, tool in enumerate(tools):
            if index == 0:
                ordered.append(f"先使用 {tool.tool_name}（{tool.role}）")
            elif index == len(tools) - 1:
                ordered.append(f"最后使用 {tool.tool_name}（{tool.role}）")
            else:
                ordered.append(f"再使用 {tool.tool_name}（{tool.role}）")
        return "工具使用经验：" + "，".join(ordered) + "。"

    def _render_prompt_context(
        self,
        analysis: Optional[TopicIntent],
        runtime: Optional[RuntimeState],
        topic_shift: bool,
        is_new_session: bool,
        transition_notes: Sequence[str],
        experiences: Sequence[Experience],
        core_segments: Sequence[Segment],
        expanded_segments: Sequence[Segment],
        qas: Sequence[QA],
    ) -> str:
        lines = ["## Current Topic Analysis"]
        if analysis:
            lines.extend(
                [
                    f"- domain: {analysis.domain}",
                    f"- topic: {analysis.topic}",
                    f"- intent: {analysis.primary_intent}",
                    f"- goal: {analysis.goal}",
                    f"- relation_to_previous: {analysis.relation_to_previous}",
                    f"- keywords: {', '.join(analysis.keywords) or 'n/a'}",
                ]
            )
        else:
            lines.append("- analysis: no active topic")

        lines.extend(
            [
                f"- new_session_detected: {'yes' if is_new_session else 'no'}",
                f"- topic_shift_detected: {'yes' if topic_shift else 'no'}",
                f"- prompt_flow_actions: {', '.join(transition_notes) or 'none'}",
                "",
                "## Runtime State",
            ]
        )
        if not runtime:
            lines.append("- No runtime state yet.")
        else:
            lines.append(f"- current_experience_id: {runtime.current_experience_id or 'none'}")
            lines.append(f"- current_segment_id: {runtime.current_segment_id or 'none'}")
            lines.append(f"- latest_qa_id: {runtime.latest_qa_id or 'none'}")
            lines.append(
                f"- recent_segment_ids: {', '.join(runtime.recent_segment_ids) or 'none'}"
            )

        lines.extend(["", "## Retrieved Experiences"])
        if not experiences:
            lines.append("- No related experiences found.")
        else:
            for experience in experiences:
                lines.append(
                    f"- [{experience.domain}] {experience.topic} "
                    f"(status={experience.status}, segments={len(experience.segment_ids)})"
                )
                if experience.summary.short:
                    lines.append(f"  summary: {experience.summary.short[:240]}")

        lines.extend(["", "## Core Segments"])
        if not core_segments:
            lines.append("- No strongly related segments found.")
        else:
            for segment in core_segments:
                lines.append(
                    f"- {segment.segment_id} [{segment.status}] "
                    f"{segment.topic} / {segment.intent}"
                )
                lines.append(f"  summary: {self._display_segment_summary(segment)[:240]}")

        lines.extend(["", "## Related Segments"])
        if not expanded_segments:
            lines.append("- No related segment expansions.")
        else:
            for segment in expanded_segments:
                lines.append(
                    f"- {segment.segment_id} [{segment.status}] "
                    f"{segment.topic} / {segment.intent}"
                )
                lines.append(f"  summary: {self._display_segment_summary(segment)[:220]}")

        relation_segments = list(core_segments) + list(expanded_segments)
        related_relations = self._related_relations(
            [segment.segment_id for segment in relation_segments]
        )
        lines.extend(["", "## Related Segment Relations"])
        if not related_relations:
            lines.append("- No related segment relations found.")
        else:
            for relation in related_relations:
                lines.append(f"- {relation.relation_type}: {relation.reason[:220]}")

        lines.extend(["", "## Retrieved QAs"])
        if not qas:
            lines.append("- No strongly related QA memories found.")
        else:
            for qa in qas[:10]:
                lines.extend(self._render_qa_prompt_block(qa))

        return "\n".join(lines).strip()

    def _build_transition_notes(
        self,
        user_message: str,
        analysis: TopicIntent,
        runtime: Optional[RuntimeState],
        current_segment: Optional[Segment],
        experiences: Sequence[Experience],
        core_segments: Sequence[Segment],
        is_new_session: bool,
        topic_shift: bool,
    ) -> List[str]:
        notes: List[str] = []
        if is_new_session:
            notes.append("new_session_detected")
        if analysis.relation_to_previous == "resume_previous" or self._contains_any(user_message, RESUME_HINTS):
            notes.append("resume_historical_topic")
        if self._contains_any(user_message, PAUSE_HINTS):
            notes.append("interrupt_current_segment")
        if topic_shift or analysis.relation_to_previous == "switch_topic":
            notes.append("switch_topic")
        if current_segment and current_segment.topic == analysis.topic and not topic_shift:
            notes.append("continue_current_segment")
        if experiences:
            notes.append("load_related_experiences")
        if core_segments:
            notes.append("load_related_segments")
        if runtime and runtime.recent_segment_ids:
            notes.append("use_runtime_recent_segments")
        return notes or ["prepare_new_experience"]

    def _analysis_from_state(
        self,
        current_segment: Optional[Segment],
        current_experience: Optional[Experience],
        runtime: Optional[RuntimeState],
    ) -> Optional[TopicIntent]:
        if current_segment:
            return TopicIntent(
                domain=current_segment.domain,
                topic=current_segment.topic,
                intents=[current_segment.intent],
                keywords=self._select_keywords(
                    self._tokenize(" ".join([current_segment.topic, current_segment.goal]))
                ),
                confidence=0.85,
                goal=current_segment.goal,
                relation_to_previous="follow_up",
            )
        if current_experience:
            return TopicIntent(
                domain=current_experience.domain,
                topic=current_experience.topic,
                intents=["general_analysis"],
                keywords=self._select_keywords(
                    self._tokenize(" ".join([current_experience.topic, current_experience.goal]))
                ),
                confidence=0.72,
                goal=current_experience.goal,
                relation_to_previous="follow_up",
            )
        if runtime and runtime.latest_qa_id:
            latest_qa = self.load_qa(runtime.latest_qa_id)
            if latest_qa:
                return TopicIntent(
                    domain=latest_qa.domain,
                    topic=latest_qa.topic,
                    intents=[latest_qa.intent],
                    keywords=self._select_keywords(
                        self._tokenize(" ".join([latest_qa.topic, latest_qa.goal]))
                    ),
                    confidence=0.7,
                    goal=latest_qa.goal,
                    relation_to_previous="follow_up",
                )
        return None

    def _find_resume_segment(
        self,
        analysis: TopicIntent,
        query_text: str,
        runtime: Optional[RuntimeState],
        exclude_segment_id: Optional[str],
        force: bool = False,
    ) -> Optional[Segment]:
        if not force and not self._contains_any(query_text, RESUME_HINTS):
            return None

        segments = self.list_segments()
        if not segments:
            return None

        recent_segment_ids = runtime.recent_segment_ids if runtime else []
        scored: List[Tuple[float, Segment]] = []
        for segment in segments:
            if exclude_segment_id and segment.segment_id == exclude_segment_id:
                continue
            score = 0.16 * self._soft_text_overlap(segment.topic, analysis.topic)
            score += 0.16 * self._soft_text_overlap(segment.goal, query_text)
            if segment.intent == analysis.primary_intent:
                score += 0.08
            if segment.segment_id in recent_segment_ids:
                score += 0.16
            score += self._recency_bonus(segment.updated_at)
            if score > 0.22:
                scored.append((score, segment))

        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return scored[0][1] if scored else None

    def _build_experience_document(self, experience: Experience) -> str:
        parts = [
            experience.domain,
            experience.topic,
            experience.goal,
            experience.summary.short,
            experience.summary.long,
            self._experience_memory_text(experience.memory),
            experience.status,
        ]
        return "\n".join(part for part in parts if part).strip()

    def _build_segment_document(self, segment: Segment) -> str:
        parts = [
            segment.domain,
            segment.topic,
            segment.intent,
            segment.goal,
            segment.summary,
            self._segment_memory_text(segment.memory),
        ]
        return "\n".join(part for part in parts if part).strip()

    def _merge_text_parts(self, *parts: str) -> str:
        return " ".join(part.strip() for part in parts if str(part).strip())

    def _build_qa_document(self, qa: QA) -> str:
        tool_text = " ".join(
            f"{tool.tool_name} {tool.role}"
            for tool in qa.tools
        )
        parts = [
            qa.domain,
            qa.topic,
            qa.intent,
            qa.goal,
            qa.user_input,
            qa.assistant_output,
            " ".join(qa.entities),
            " ".join(qa.facts),
            " ".join(qa.constraints),
            tool_text,
            qa.timestamp,
        ]
        return "\n".join(part for part in parts if part).strip()

    def _segment_memory_text(self, memory: SegmentMemory) -> str:
        return " ".join(
            memory.facts
            + memory.constraints
            + memory.decisions
            + memory.open_questions
        )

    def _experience_memory_text(self, memory: ExperienceMemory) -> str:
        return " ".join(memory.decisions + memory.constraints + memory.open_questions)

    def _hybrid_scores(
        self,
        table_name: str,
        documents: Sequence[RetrievalDocument],
        query_text: str,
    ) -> Dict[str, float]:
        if not documents:
            return {}

        candidate_ids = [doc.doc_id for doc in documents]
        bm25_rank = self._sqlite_rank(table_name, query_text, candidate_ids)
        vector_rank = self._vector_rank(documents, query_text)
        fused = self._rrf_merge([bm25_rank, vector_rank])
        return fused

    def _sqlite_rank(
        self,
        table_name: str,
        query_text: str,
        candidate_ids: Sequence[str],
        limit: int = 50,
    ) -> List[str]:
        tokens = self._tokenize(query_text)
        if not tokens:
            return []

        query = " OR ".join(f'"{token.replace(chr(34), " ")}"' for token in tokens[:12])
        if not query:
            return []

        candidate_set = set(candidate_ids)
        with sqlite3.connect(self.sqlite_db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT doc_id, bm25({table_name}) AS score
                FROM {table_name}
                WHERE {table_name} MATCH ?
                ORDER BY score ASC
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()

        ranked: List[str] = []
        for row in rows:
            doc_id = str(row["doc_id"])
            if doc_id in candidate_set:
                ranked.append(doc_id)
        return ranked

    def _vector_rank(
        self,
        documents: Sequence[RetrievalDocument],
        query_text: str,
    ) -> List[str]:
        embedding_scores = self._embedding_scores(documents, query_text)
        if embedding_scores:
            return [
                doc_id
                for doc_id, score in sorted(
                    embedding_scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                if score > 0
            ]

        local_scores = self._local_vector_scores(documents, query_text)
        return [
            doc_id
            for doc_id, score in sorted(
                local_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if score > 0
        ]

    def _embedding_scores(
        self,
        documents: Sequence[RetrievalDocument],
        query_text: str,
    ) -> Dict[str, float]:
        if self._embedder is None:
            return {}

        try:
            query_vector = self._embed_query(query_text)
            if not query_vector:
                return {}
            scores: Dict[str, float] = {}
            for doc in documents:
                doc_vector = self._embed_document(doc.doc_id, doc.object_type, doc.text)
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
            scores[doc.doc_id] = round(
                self._sparse_cosine_similarity(query_vector, doc_vector),
                6,
            )
        return scores

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

    def _build_tokenized_content(self, text: str) -> str:
        return " ".join(self._tokenize(text))

    def _initialize_sqlite(self):
        with sqlite3.connect(self.sqlite_db_path) as conn:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS experience_fts
                USING fts5(doc_id UNINDEXED, content)
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS segment_fts
                USING fts5(doc_id UNINDEXED, content)
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS qa_fts
                USING fts5(doc_id UNINDEXED, content)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    doc_id TEXT PRIMARY KEY,
                    object_type TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _refresh_store_artifacts(self, conversation_id: Optional[str] = None):
        self._rebuild_sqlite_indexes()
        overview = self.get_store_overview(conversation_id=conversation_id)
        self._save_json(self.index_file, overview)
        snapshot = self.render_memory_snapshot(conversation_id=conversation_id)
        with open(self.memory_snapshot_file, "w", encoding="utf-8") as f:
            f.write(snapshot)

    def _rebuild_sqlite_indexes(self):
        with sqlite3.connect(self.sqlite_db_path) as conn:
            conn.execute("DELETE FROM experience_fts")
            conn.execute("DELETE FROM segment_fts")
            conn.execute("DELETE FROM qa_fts")

            for experience in self.list_experiences():
                conn.execute(
                    "INSERT INTO experience_fts (doc_id, content) VALUES (?, ?)",
                    (
                        experience.experience_id,
                        self._build_tokenized_content(self._build_experience_document(experience)),
                    ),
                )

            for segment in self.list_segments():
                conn.execute(
                    "INSERT INTO segment_fts (doc_id, content) VALUES (?, ?)",
                    (
                        segment.segment_id,
                        self._build_tokenized_content(self._build_segment_document(segment)),
                    ),
                )

            for qa in self.list_qas():
                conn.execute(
                    "INSERT INTO qa_fts (doc_id, content) VALUES (?, ?)",
                    (
                        qa.qa_id,
                        self._build_tokenized_content(self._build_qa_document(qa)),
                    ),
                )

            conn.commit()

    def render_memory_snapshot(self, conversation_id: Optional[str] = None) -> str:
        overview = self.get_store_overview(conversation_id=conversation_id)
        runtime = overview.get("current_runtime_state") or {}
        latest_experience = overview.get("latest_experience") or {}
        latest_segment = overview.get("latest_segment") or {}

        lines = [
            "# Topic Memory Snapshot",
            "",
            "此文件由系统根据三层主题记忆自动生成，请勿手工编辑。",
            "",
            "## Runtime State",
            "",
            f"- conversation_id: {runtime.get('conversation_id', 'none')}",
            f"- current_experience_id: {runtime.get('current_experience_id', 'none')}",
            f"- current_segment_id: {runtime.get('current_segment_id', 'none')}",
            f"- latest_qa_id: {runtime.get('latest_qa_id', 'none')}",
            "- recent_segment_ids: "
            + (", ".join(runtime.get("recent_segment_ids", [])) or "none"),
            f"- updated_at: {runtime.get('updated_at', overview.get('updated_at', ''))}",
            "",
            "## Counts",
            "",
            f"- experiences: {overview['counts']['experiences']}",
            f"- segments: {overview['counts']['segments']}",
            f"- qas: {overview['counts']['qas']}",
            f"- relations: {overview['counts']['relations']}",
            "",
            "## Latest Experience",
            "",
            f"- experience_id: {latest_experience.get('experience_id', 'none')}",
            f"- topic: {latest_experience.get('topic', 'none')}",
            f"- goal: {latest_experience.get('goal', 'none')}",
            f"- summary.short: {latest_experience.get('summary_short', '')}",
            f"- summary.long: {latest_experience.get('summary_long', '')}",
            "",
            "## Latest Segment",
            "",
            f"- segment_id: {latest_segment.get('segment_id', 'none')}",
            f"- topic: {latest_segment.get('topic', 'none')}",
            f"- intent: {latest_segment.get('intent', 'none')}",
            f"- status: {latest_segment.get('status', 'none')}",
            f"- summary: {latest_segment.get('summary', '')}",
            "",
            "## Recent Experiences",
            "",
        ]

        recent_experiences = overview.get("recent_experiences", [])
        if not recent_experiences:
            lines.append("暂无经验记录。")
        else:
            for index, item in enumerate(recent_experiences, start=1):
                lines.extend(
                    [
                        f"{index}. topic: {item.get('topic', 'none')}",
                        f"   goal: {item.get('goal', 'none')}",
                        f"   summary: {item.get('summary_short', '')}",
                    ]
                )

        lines.extend(["", "## Recent Segments", ""])
        recent_segments = overview.get("recent_segments", [])
        if not recent_segments:
            lines.append("暂无阶段记录。")
        else:
            for index, item in enumerate(recent_segments, start=1):
                lines.extend(
                    [
                        f"{index}. segment_id: {item.get('segment_id', 'none')}",
                        f"   topic: {item.get('topic', 'none')}",
                        f"   intent: {item.get('intent', 'none')}",
                        f"   status: {item.get('status', 'none')}",
                        f"   summary: {item.get('summary', '')}",
                    ]
                )
        return "\n".join(lines).strip() + "\n"

    def _migrate_legacy_monolith_if_needed(self):
        if not os.path.exists(self.index_file):
            return
        if any(os.listdir(directory) for directory in (self.qas_dir, self.segments_dir, self.experiences_dir)):
            return

        payload = self._load_json(self.index_file, {})
        if not isinstance(payload, dict):
            return
        if not any(key in payload for key in ("qas", "segments", "experiences", "relations", "runtime_state")):
            return

        for item in payload.get("qas", []):
            if not isinstance(item, dict):
                continue
            try:
                self.save_qa(self._deserialize_qa(item))
            except Exception:
                continue

        for item in payload.get("segments", []):
            if not isinstance(item, dict):
                continue
            try:
                self.save_segment(self._deserialize_segment(item))
            except Exception:
                continue

        for item in payload.get("experiences", []):
            if not isinstance(item, dict):
                continue
            try:
                self.save_experience(self._deserialize_experience(item))
            except Exception:
                continue

        for item in payload.get("relations", []):
            if not isinstance(item, dict):
                continue
            try:
                self.save_relation(self._deserialize_relation(item))
            except Exception:
                continue

        runtime_state = payload.get("runtime_state")
        if isinstance(runtime_state, dict):
            try:
                self.save_runtime_state(
                    RuntimeState(
                        conversation_id=str(runtime_state.get("conversation_id", "")),
                        current_experience_id=runtime_state.get("current_experience_id"),
                        current_segment_id=runtime_state.get("current_segment_id"),
                        latest_qa_id=runtime_state.get("latest_qa_id"),
                        recent_segment_ids=[
                            str(item) for item in runtime_state.get("recent_segment_ids", [])
                        ],
                        updated_at=str(runtime_state.get("updated_at", _now_iso())),
                    )
                )
            except Exception:
                pass

    def _serialize_qa(self, qa: QA) -> Dict[str, Any]:
        payload = asdict(qa)
        payload["tools"] = [asdict(item) for item in qa.tools]
        return payload

    def _deserialize_qa(self, payload: Dict[str, Any]) -> QA:
        return QA(
            qa_id=str(payload.get("qa_id", "")),
            conversation_id=str(payload.get("conversation_id", "")),
            timestamp=str(payload.get("timestamp", _now_iso())),
            user_input=str(payload.get("user_input", "")),
            assistant_output=str(payload.get("assistant_output", "")),
            tools=[
                ToolCallRecord(
                    tool_name=str(item.get("tool_name", "")),
                    role=str(item.get("role", "")),
                )
                for item in payload.get("tools", [])
                if isinstance(item, dict)
            ],
            domain=str(payload.get("domain", "general")),
            topic=str(payload.get("topic", "general")),
            intent=str(payload.get("intent", "general_analysis")),
            goal=str(payload.get("goal", "")),
            entities=[str(item) for item in payload.get("entities", [])],
            facts=[str(item) for item in payload.get("facts", [])],
            constraints=[str(item) for item in payload.get("constraints", [])],
            segment_id=str(payload.get("segment_id", "")),
            experience_id=str(payload.get("experience_id", "")),
            status=payload.get("status", "active"),
            confidence=float(payload.get("confidence", 1.0)),
        )

    def _serialize_segment(self, segment: Segment) -> Dict[str, Any]:
        payload = asdict(segment)
        payload["memory"] = asdict(segment.memory)
        return payload

    def _deserialize_segment(self, payload: Dict[str, Any]) -> Segment:
        memory_payload = payload.get("memory", {}) or {}
        memory = SegmentMemory(
            facts=[str(item) for item in memory_payload.get("facts", [])],
            constraints=[str(item) for item in memory_payload.get("constraints", [])],
            decisions=[str(item) for item in memory_payload.get("decisions", [])],
            open_questions=[str(item) for item in memory_payload.get("open_questions", [])],
        )
        return Segment(
            segment_id=str(payload.get("segment_id", "")),
            experience_id=str(payload.get("experience_id", "")),
            domain=str(payload.get("domain", "general")),
            topic=str(payload.get("topic", "general")),
            intent=str(payload.get("intent", "general_analysis")),
            goal=str(payload.get("goal", "")),
            qa_ids=[str(item) for item in payload.get("qa_ids", [])],
            status=self._normalize_segment_status(payload.get("status")),
            summary=str(payload.get("summary", "")),
            memory=memory,
            created_at=str(payload.get("created_at", _now_iso())),
            updated_at=str(payload.get("updated_at", _now_iso())),
            last_summarized_qa_count=int(payload.get("last_summarized_qa_count", 0)),
            last_summarized_at=str(payload.get("last_summarized_at", "")),
            version=int(payload.get("version", 1)),
        )

    def _serialize_experience(self, experience: Experience) -> Dict[str, Any]:
        payload = asdict(experience)
        payload["summary"] = asdict(experience.summary)
        payload["memory"] = asdict(experience.memory)
        return payload

    def _deserialize_experience(self, payload: Dict[str, Any]) -> Experience:
        summary_payload = payload.get("summary", {}) or {}
        memory_payload = payload.get("memory", {}) or {}
        return Experience(
            experience_id=str(payload.get("experience_id", "")),
            domain=str(payload.get("domain", "general")),
            topic=str(payload.get("topic", "general")),
            goal=str(payload.get("goal", "")),
            segment_ids=[str(item) for item in payload.get("segment_ids", [])],
            current_segment_id=payload.get("current_segment_id"),
            main_segment_path=[str(item) for item in payload.get("main_segment_path", [])],
            summary=ExperienceSummary(
                short=str(summary_payload.get("short", "")),
                long=str(summary_payload.get("long", "")),
            ),
            memory=ExperienceMemory(
                decisions=[str(item) for item in memory_payload.get("decisions", [])],
                constraints=[str(item) for item in memory_payload.get("constraints", [])],
                open_questions=[str(item) for item in memory_payload.get("open_questions", [])],
            ),
            status=payload.get("status", "in_progress"),
            created_at=str(payload.get("created_at", _now_iso())),
            updated_at=str(payload.get("updated_at", _now_iso())),
            last_summarized_segment_count=int(payload.get("last_summarized_segment_count", 0)),
            last_summarized_qa_count=int(payload.get("last_summarized_qa_count", 0)),
            last_summarized_relation_count=int(payload.get("last_summarized_relation_count", 0)),
            last_summarized_at=str(payload.get("last_summarized_at", "")),
            version=int(payload.get("version", 1)),
        )

    def _deserialize_relation(self, payload: Dict[str, Any]) -> SegmentRelation:
        normalized_relation_type = self._normalize_relation_type(payload.get("relation_type")) or "refines"
        return SegmentRelation(
            relation_id=str(payload.get("relation_id", "")),
            experience_id=str(payload.get("experience_id", "")),
            from_segment_id=str(payload.get("from_segment_id", "")),
            to_segment_id=str(payload.get("to_segment_id", "")),
            relation_type=normalized_relation_type,
            reason=str(payload.get("reason", "")),
            weight=float(payload.get("weight", 0.0)),
            created_at=str(payload.get("created_at", _now_iso())),
        )

    def _load_all(self, directory: str, loader) -> List[Any]:
        items: List[Any] = []
        if not os.path.exists(directory):
            return items
        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(directory, filename)
            payload = self._load_json(path, {})
            if not payload:
                continue
            try:
                items.append(loader(payload))
            except Exception:
                continue
        return items

    def _tokenize(self, text: str) -> List[str]:
        raw_tokens = re.findall(
            r"[A-Za-z][A-Za-z0-9_+-]{1,}|[0-9]{2,}|[\u4e00-\u9fff]{2,}",
            text,
        )
        tokens: List[str] = []
        for token in raw_tokens:
            normalized = self._normalize_token(token)
            if not normalized:
                continue
            if normalized in ENGLISH_STOPWORDS or normalized in CHINESE_STOPWORDS:
                continue
            if len(normalized) == 1:
                continue
            tokens.append(normalized)
        return tokens

    def _normalize_token(self, token: str) -> str:
        normalized = token.strip().lower()
        if not normalized:
            return ""

        changed = True
        while changed and normalized:
            changed = False
            for prefix in GENERIC_PREFIXES:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):].strip()
                    changed = True
            for suffix in GENERIC_SUFFIXES:
                if normalized.endswith(suffix):
                    normalized = normalized[:-len(suffix)].strip()
                    changed = True
        return normalized

    def _select_keywords(self, tokens: Sequence[str], limit: int = 8) -> List[str]:
        seen: Set[str] = set()
        keywords: List[str] = []
        for token in sorted(tokens, key=self._keyword_sort_key):
            if token in seen:
                continue
            seen.add(token)
            keywords.append(token)
            if len(keywords) >= limit:
                break
        return keywords

    def _keyword_sort_key(self, token: str) -> Tuple[int, int, str]:
        is_numeric = 1 if token.isdigit() else 0
        return (-len(token), -is_numeric, token)

    def _extract_known_keywords(self, lowered_message: str) -> List[str]:
        terms = sorted(
            WEATHER_TERMS
            | TRAVEL_TERMS
            | FINANCE_TERMS
            | SOFTWARE_TERMS
            | RESEARCH_TERMS
            | PRODUCT_TERMS,
            key=lambda item: (-len(item), item),
        )
        return [term for term in terms if term in lowered_message]

    def _infer_domain(
        self,
        tokens: Sequence[str],
        lowered_message: str,
        previous: Optional[TopicIntent],
    ) -> str:
        token_set = set(tokens)
        buckets = {
            "weather": WEATHER_TERMS,
            "travel": TRAVEL_TERMS,
            "finance_analysis": FINANCE_TERMS,
            "software_engineering": SOFTWARE_TERMS,
            "research_writing": RESEARCH_TERMS,
            "product_planning": PRODUCT_TERMS,
        }
        best_domain = "general"
        best_score = 0
        for domain, terms in buckets.items():
            score = len(token_set & terms) + sum(1 for term in terms if term in lowered_message)
            if score > best_score:
                best_domain = domain
                best_score = score
        if best_score == 0 and previous:
            return previous.domain
        return best_domain

    def _infer_intent(
        self,
        tokens: Sequence[str],
        lowered_message: str,
        domain: str,
    ) -> str:
        token_set = set(tokens)
        if domain == "weather":
            return "weather_lookup"
        if any(term in lowered_message for term in ("排查", "报错", "异常", "故障", "debug", "troubleshoot")):
            return "troubleshooting"
        if any(term in lowered_message for term in ("对比", "比较", "区别", "compare")):
            return "comparison"
        if any(term in lowered_message for term in ("解释", "说明", "why", "explain")):
            return "explanation"
        if any(term in lowered_message for term in ("修复", "实现", "修改", "fix", "implement", "build")):
            return "implementation"
        if any(term in lowered_message for term in ("总结", "归纳", "summary", "summarize")):
            return "summarization"
        if any(term in lowered_message for term in ("方案", "规划", "攻略", "plan", "design")):
            return "planning"
        if any(term in lowered_message for term in ("查询", "检索", "查找", "search", "find")):
            return "retrieval"
        if token_set & {"旅游", "旅行", "攻略"}:
            return "planning"
        return "general_analysis"

    def _infer_topic(
        self,
        message: str,
        domain: str,
        keywords: Sequence[str],
        previous: Optional[TopicIntent],
    ) -> str:
        normalized = " ".join(message.strip().split())

        if domain == "weather":
            return "天气查询"

        city = self._extract_location_phrase(message)
        if domain == "travel" and city:
            return f"{city}旅游"
        if domain == "travel":
            return previous.topic if previous and previous.domain == "travel" else "旅行规划"

        if previous and self._looks_like_detail_message(message):
            return previous.topic

        if keywords:
            return " / ".join(list(keywords)[:2])
        if normalized:
            return normalized[:48]
        if previous:
            return previous.topic
        return "general topic"

    def _extract_location_phrase(self, message: str) -> str:
        match = re.search(r"([\u4e00-\u9fffA-Za-z]{2,10})(?:的)?(?:天气|旅游|旅行|攻略)", message)
        if match:
            candidate = match.group(1)
            candidate = re.sub(r"^(帮我|请帮我|我想去|我想|去)", "", candidate).strip()
            if candidate:
                return candidate
        if re.fullmatch(r"[\u4e00-\u9fffA-Za-z]{2,10}", message.strip()):
            return message.strip()
        return ""

    def _extract_tool_sequence(self, turn_messages: Sequence[Message]) -> List[str]:
        tools: List[str] = []
        for msg in turn_messages:
            if msg.role == "tool" and msg.name:
                tools.append(msg.name)
        return tools

    def _extract_entities(self, user_message: str, assistant_reply: str) -> List[str]:
        phrases = re.findall(
            r"[\u4e00-\u9fffA-Za-z0-9]{2,18}",
            "\n".join([user_message, assistant_reply]),
        )
        normalized = [phrase.strip() for phrase in phrases if phrase.strip()]
        return self._merge_unique([], normalized, limit=10)

    def _extract_facts(self, assistant_reply: str) -> List[str]:
        lines = [
            line.strip(" -\t")
            for line in re.split(r"[\n\r]+", assistant_reply)
            if line.strip()
        ]
        facts: List[str] = []
        for line in lines:
            if re.match(r"^(\d+[\.\)]|[-*])", line):
                facts.append(re.sub(r"^(\d+[\.\)]|[-*])\s*", "", line))
                continue
            if "：" in line or ":" in line:
                facts.append(line)
        if not facts and assistant_reply.strip():
            facts.append(assistant_reply.strip().splitlines()[0][:180])
        return self._merge_unique([], facts, limit=6)

    def _extract_constraints(self, user_message: str, assistant_reply: str) -> List[str]:
        constraints: List[str] = []
        for text in (user_message, assistant_reply):
            snippets = re.split(r"[，,。；;\n]", text)
            for snippet in snippets:
                snippet = snippet.strip()
                if not snippet:
                    continue
                if any(
                    term in snippet
                    for term in (
                        "预算",
                        "偏好",
                        "时间",
                        "明天",
                        "今天",
                        "两天",
                        "三天",
                        "不要",
                        "必须",
                        "只能",
                        "先",
                        "暂停",
                    )
                ):
                    constraints.append(snippet)
        return self._merge_unique([], constraints, limit=6)

    def _extract_decisions(self, assistant_reply: str) -> List[str]:
        decisions: List[str] = []
        for line in re.split(r"[\n\r]+", assistant_reply):
            cleaned = line.strip(" -\t")
            if not cleaned:
                continue
            if any(term in cleaned for term in ("建议", "推荐", "安排", "方案", "计划", "步骤")):
                decisions.append(cleaned)
        return self._merge_unique([], decisions, limit=6)

    def _extract_open_questions(self, assistant_reply: str) -> List[str]:
        questions: List[str] = []
        for sentence in re.split(r"[。\n]", assistant_reply):
            snippet = sentence.strip()
            if not snippet:
                continue
            if self._contains_any(snippet, QUESTION_HINTS):
                questions.append(snippet.rstrip("。") + "。")
        return self._merge_unique([], questions, limit=4)

    def _render_numbered_items(self, items: Sequence[str], fallback: str) -> str:
        if not items:
            return fallback
        return "".join(f"{index}. {item}" for index, item in enumerate(items[:6], start=1))

    def _prepend_recent_segment(self, segment_ids: Sequence[str], segment_id: str, limit: int = 8) -> List[str]:
        ordered = [segment_id]
        for item in segment_ids:
            if item == segment_id:
                continue
            ordered.append(item)
            if len(ordered) >= limit:
                break
        return ordered

    def _goals_align(self, current_goal: str, candidate_goal: str, user_message: str) -> bool:
        if current_goal == candidate_goal:
            return True
        if self._looks_like_detail_message(user_message):
            return True
        overlap = self._soft_text_overlap(current_goal, candidate_goal)
        return overlap >= 0.32

    def _looks_like_detail_message(self, message: str) -> bool:
        lowered = (message or "").strip().lower()
        if not lowered:
            return False
        if self._contains_any(lowered, RESUME_HINTS) or self._contains_any(lowered, PAUSE_HINTS):
            return False
        if any(hint in lowered for hint in DETAIL_HINTS):
            return True
        return len(lowered) <= 24 and not any(
            hint in lowered
            for hint in ("换个话题", "先不聊", "继续刚才", "重新开始")
        )

    def _contains_any(self, text: str, terms: Iterable[str]) -> bool:
        return any(term in (text or "") for term in terms)

    def _soft_text_overlap(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        left_tokens = set(self._tokenize(left))
        right_tokens = set(self._tokenize(right))
        if left_tokens and right_tokens:
            return self._jaccard(left_tokens, right_tokens)

        left_chars = set(left.lower())
        right_chars = set(right.lower())
        return self._jaccard(left_chars, right_chars)

    def _jaccard(self, left: Set[str], right: Set[str]) -> float:
        if not left or not right:
            return 0.0
        union = len(left | right)
        if union == 0:
            return 0.0
        return len(left & right) / union

    def _merge_unique(
        self,
        existing: Sequence[str],
        incoming: Sequence[str],
        limit: int,
    ) -> List[str]:
        merged = [str(item).strip() for item in existing if str(item).strip()]
        for item in incoming:
            normalized = str(item).strip()
            if not normalized or normalized in merged:
                continue
            merged.append(normalized)
        return merged[-limit:]

    def _recency_bonus(self, iso_value: str) -> float:
        try:
            delta = datetime.now() - datetime.fromisoformat(iso_value)
        except Exception:
            return 0.0
        hours = max(delta.total_seconds() / 3600.0, 0.0)
        return round(max(0.0, 0.12 - min(hours / 240.0, 0.12)), 4)

    def _is_new_session(
        self,
        runtime: Optional[RuntimeState],
        history: Sequence[Message],
    ) -> bool:
        if not history:
            return True
        if not runtime:
            return True
        return runtime.latest_qa_id is None and runtime.current_segment_id is None

    def _latest_runtime_state(self) -> Optional[RuntimeState]:
        runtimes = self._load_all(
            self.runtime_dir,
            lambda payload: RuntimeState(
                conversation_id=str(payload.get("conversation_id", "")),
                current_experience_id=payload.get("current_experience_id"),
                current_segment_id=payload.get("current_segment_id"),
                latest_qa_id=payload.get("latest_qa_id"),
                recent_segment_ids=[str(item) for item in payload.get("recent_segment_ids", [])],
                updated_at=str(payload.get("updated_at", _now_iso())),
            ),
        )
        return self._latest_by_updated_at(runtimes)

    def _recent_items(self, items: Sequence[Any], limit: int) -> List[Any]:
        return sorted(items, key=lambda item: getattr(item, "updated_at", ""), reverse=True)[:limit]

    def _latest_by_updated_at(self, items: Sequence[Any]) -> Optional[Any]:
        ordered = self._recent_items(items, limit=1)
        return ordered[0] if ordered else None

    def _experience_overview_payload(self, experience: Optional[Experience]) -> Optional[Dict[str, Any]]:
        if not experience:
            return None
        return {
            "experience_id": experience.experience_id,
            "domain": experience.domain,
            "topic": experience.topic,
            "goal": experience.goal,
            "status": experience.status,
            "current_segment_id": experience.current_segment_id,
            "summary_short": experience.summary.short,
            "summary_long": experience.summary.long[:800],
            "updated_at": experience.updated_at,
        }

    def _segment_overview_payload(self, segment: Optional[Segment]) -> Optional[Dict[str, Any]]:
        if not segment:
            return None
        return {
            "segment_id": segment.segment_id,
            "experience_id": segment.experience_id,
            "topic": segment.topic,
            "intent": segment.intent,
            "goal": segment.goal,
            "status": segment.status,
            "summary": self._display_segment_summary(segment)[:800],
            "updated_at": segment.updated_at,
        }

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

    def _embed_query(self, text: str) -> List[float]:
        if self._embedder is None:
            return []
        return [float(value) for value in self._embedder.embed_query(text)]

    def _embed_document(self, doc_id: str, object_type: str, text: str) -> List[float]:
        text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
        with sqlite3.connect(self.sqlite_db_path) as conn:
            row = conn.execute(
                """
                SELECT text_hash, vector_json
                FROM embedding_cache
                WHERE doc_id = ?
                """,
                (doc_id,),
            ).fetchone()
            if row and row[0] == text_hash:
                try:
                    return [float(value) for value in json.loads(row[1])]
                except Exception:
                    pass

        if self._embedder is None:
            return []

        vector = [float(value) for value in self._embedder.embed_documents([text])[0]]
        with sqlite3.connect(self.sqlite_db_path) as conn:
            conn.execute(
                """
                INSERT INTO embedding_cache (doc_id, object_type, text_hash, vector_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    object_type = excluded.object_type,
                    text_hash = excluded.text_hash,
                    vector_json = excluded.vector_json,
                    updated_at = excluded.updated_at
                """,
                (
                    doc_id,
                    object_type,
                    text_hash,
                    json.dumps(vector),
                    _now_iso(),
                ),
            )
            conn.commit()
        return vector

    def _cosine_similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _load_json(self, path: str, default):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _save_json(self, path: str, payload):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
