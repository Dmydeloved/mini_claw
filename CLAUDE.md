# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**Mini-OpenClaw** is a lightweight, transparent AI Agent system built with Python, designed as a local-first digital assistant with "file-as-memory" architecture. It replicates and optimizes the core experience of OpenClaw (formerly Moltbot/Clawdbot).

**Core Philosophy**:
- **File-first Memory**: All conversations and reflections stored as human-readable Markdown/JSON files
- **Skills as Plugins**: Follows Anthropic's Agent Skills paradigm with folder-based capability management
- **Transparent & Controllable**: All system prompts, tool calls, and memory operations are fully visible

---

## Technology Stack

### Backend (Port 8002)
- **Language**: Python 3.10+ (mandatory Type Hinting)
- **Framework**: FastAPI (async RESTful API)
- **Agent Engine**: LangChain 1.x with `create_agent` API (NOT `AgentExecutor` or old `create_react_agent`)
- **RAG Engine**: LlamaIndex Core (Hybrid Search: BM25 + Vector)
- **LLM Interface**: OpenAI-compatible API (supports OpenRouter, DeepSeek, Claude)
- **Storage**: Local file system only (no MySQL/Redis)

### Frontend (Port 3000)
- **Framework**: Next.js 14+ (App Router)
- **UI Library**: shadcn/ui + Tailwind CSS
- **Code Editor**: Monaco Editor (for SKILL.md editing)
- **State Management**: React hooks + Context API

---

## Project Structure

```
mini-openclaw/
├── backend/
│   ├── app.py                    # FastAPI entry point (port 8002)
│   ├── memory/
│   │   ├── logs/                 # Daily conversation logs (YYYY-MM-DD.md)
│   │   └── MEMORY.md             # Core persistent memory
│   ├── sessions/                 # JSON session records
│   ├── skills/                   # Agent Skills folder
│   │   └── get_weather/
│   │       └── SKILL.md          # Skill definition with frontmatter
│   ├── workspace/                # System prompts
│   │   └── SOUL.md               # Agent personality definition
│   ├── tools/                    # Core tools implementation
│   │   ├── terminal.py           # ShellTool wrapper
│   │   ├── python_repl.py        # PythonREPLTool wrapper
│   │   ├── fetch_url.py          # RequestsGetTool + HTML cleaning
│   │   ├── read_file.py          # ReadFileTool wrapper
│   │   └── rag_search.py         # LlamaIndex hybrid search
│   ├── graph/                    # LangGraph state machine
│   │   └── agent_graph.py        # Agent workflow definition
│   └── requirements.txt
│
└── frontend/
    ├── src/
    │   ├── app/
    │   │   ├── page.tsx          # Main chat interface
    │   │   └── layout.tsx
    │   ├── components/
    │   │   ├── chat/
    │   │   │   ├── ChatWindow.tsx
    │   │   │   └── MessageList.tsx
    │   │   └── editor/
    │   │       └── SkillEditor.tsx  # Monaco-based SKILL.md editor
    │   └── lib/
    │       └── api.ts            # Fetch wrapper for backend
    └── package.json
```

---

## Core Architecture

### 1. Agent Skills System

**Skills are instruction-following, not function-calling**. Each skill is a folder containing `SKILL.md` with:

**Frontmatter** (YAML):
```yaml
---
name: get_weather
description: Get current weather for a city
trigger: weather|天气|forecast
enabled: true
---
```

**Body**: Step-by-step instructions teaching the Agent how to use base tools (Python/Terminal) to accomplish tasks.

**Loading Flow**:
1. On startup, scan `backend/skills/` for all `SKILL.md` files
2. Parse frontmatter to extract metadata
3. Generate skill list and inject into system prompt
4. When user message matches `trigger` regex, inject full skill body into context

### 2. Core Tools (5 Built-in)

All tools use LangChain native implementations:

1. **terminal** - `langchain_community.tools.ShellTool` (sandboxed with `root_dir`)
2. **python_repl** - `langchain_experimental.tools.PythonREPLTool`
3. **fetch_url** - `langchain_community.tools.RequestsGetTool` (wrapped with BeautifulSoup/html2text)
4. **read_file** - `langchain_community.tools.file_management.ReadFileTool`
5. **search_knowledge_base** - LlamaIndex hybrid search (BM25 + Vector)

### 3. Memory System

**Three-layer architecture**:

1. **MEMORY.md** (Core Memory)
   - Persistent facts about user (name, preferences, habits)
   - Always loaded into system prompt
   - Updated via Agent reflection after conversations

2. **Daily Logs** (`memory/logs/YYYY-MM-DD.md`)
   - Chronological conversation records
   - One file per day
   - Used for context retrieval

3. **Session JSON** (`sessions/{session_id}.json`)
   - Complete message history for active sessions
   - Includes tool calls and results
   - Used for session resumption

**Memory Update Flow**:
- After each conversation, Agent calls internal `update_memory` function
- Extracts key information and appends to MEMORY.md
- Writes full conversation to daily log

### 4. System Prompt Assembly

Final prompt structure:
```
[SOUL.md - Agent personality]
[MEMORY.md - User facts]
[Skill List - Available skills]
[Active Skill Body - If triggered]
[Conversation History]
[User Message]
```

---

## Key Commands

### Backend Development

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app:app --reload --port 8002

# Run with specific model
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=https://api.openrouter.ai/v1  # Optional
python app.py
```

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install
# or
pnpm install

# Run development server
npm run dev
# Access: http://localhost:3000

# Build for production
npm run build
npm run start
```

---

## API Endpoints

### Backend (FastAPI)

**POST /chat**
- Request: `{ session_id: string, message: string }`
- Response: `{ reply: string, session_id: string }`
- Handles Agent conversation with tool calling

**GET /skills**
- Response: `{ skills: Array<{ name, description, enabled }> }`
- Lists all available skills

**POST /skills/{skill_name}**
- Request: `{ content: string }`
- Updates SKILL.md content

**GET /memory**
- Response: `{ content: string }`
- Returns MEMORY.md content

**POST /memory**
- Request: `{ content: string }`
- Updates MEMORY.md

---

## Critical Implementation Notes

### LangChain Agent Creation

**MUST USE**:
```python
from langchain.agents import create_agent

agent = create_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template
)
```

**DO NOT USE**:
- `AgentExecutor` (deprecated)
- `create_react_agent` (old chain-based structure)

### Tool Safety

- **ShellTool**: Set `root_dir` to project directory, blacklist dangerous commands (`rm -rf /`, `sudo`, etc.)
- **ReadFileTool**: Restrict `root_dir` to prevent system file access
- **PythonREPLTool**: Runs in isolated environment, but still monitor for malicious code

### RAG Implementation

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import BM25Retriever

# Build index from knowledge/ directory
documents = SimpleDirectoryReader("knowledge/").load_data()
index = VectorStoreIndex.from_documents(documents)

# Hybrid search: BM25 + Vector
retriever = index.as_retriever(similarity_top_k=5)
```

---

## Development Workflow

1. **Adding a new skill**:
   - Create folder in `backend/skills/{skill_name}/`
   - Write `SKILL.md` with frontmatter and instructions
   - Restart backend to reload skills

2. **Modifying memory**:
   - Edit `backend/workspace/memory/MEMORY.md` directly
   - Or use frontend editor (syncs via API)

3. **Testing Agent**:
   - Use frontend chat interface
   - Check `backend/workspace/memory/logs/` for conversation records
   - Monitor FastAPI logs for tool calls

---

## Environment Variables

Create `.env` in `backend/`:

```bash
# Required
OPENAI_API_KEY=your_api_key

# Optional
OPENAI_BASE_URL=https://api.openrouter.ai/v1  # For OpenRouter
MODEL_NAME=anthropic/claude-3.5-sonnet         # Model identifier

# Paths
MEMORY_DIR=./workspace/memory
SKILLS_DIR=./skills
WORKSPACE_DIR=./workspace
```

---

## Design Principles

1. **Transparency First**: All Agent decisions must be traceable through logs
2. **File-based Storage**: No hidden databases, everything in readable files
3. **Minimal Dependencies**: Avoid heavy frameworks, keep it lightweight
4. **Local-first**: Runs entirely on user's machine, no cloud dependencies
5. **Instruction-following**: Skills teach Agent how to use tools, not pre-built functions

---

**Last Updated**: 2026-03-18
