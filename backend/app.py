"""
FastAPI Application for Mini-OpenClaw
RESTful API server for agent interactions and file editing.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph.agent_graph import MiniOpenClawAgent

BACKEND_ROOT = Path(__file__).resolve().parent
load_dotenv(BACKEND_ROOT / ".env", override=True)

app = FastAPI(title="Mini-OpenClaw API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("MODEL_NAME", "gpt-4")
default_headers_raw = os.getenv("OPENAI_DEFAULT_HEADERS", "").strip()
use_responses_api = os.getenv("OPENAI_USE_RESPONSES_API", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

default_headers: Dict[str, str] = {}
if default_headers_raw:
    try:
        parsed_headers = json.loads(default_headers_raw)
        if isinstance(parsed_headers, dict):
            default_headers = {str(key): str(value) for key, value in parsed_headers.items()}
        else:
            raise ValueError("OPENAI_DEFAULT_HEADERS must be a JSON object")
    except Exception as exc:
        raise ValueError(f"Invalid OPENAI_DEFAULT_HEADERS: {exc}") from exc

agent = MiniOpenClawAgent(
    api_key=api_key,
    base_url=base_url,
    model_name=model_name,
    default_headers=default_headers or None,
    use_responses_api=use_responses_api,
    root_dir=str(BACKEND_ROOT),
)

print("✓ Mini-OpenClaw Agent initialized")
print(f"✓ Model: {model_name}")
print(f"✓ Base URL: {base_url or 'https://api.openai.com/v1'}")
print(f"✓ Use Responses API: {use_responses_api}")
print(f"✓ Default Headers: {list(default_headers.keys()) if default_headers else 'none'}")
print(f"✓ Skills loaded: {len(agent.get_skills())}")


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    stream: bool = False


class ChatResponse(BaseModel):
    reply: str
    session_id: str


class SkillInfo(BaseModel):
    name: str
    description: str
    trigger: str
    enabled: bool
    location: str


class FileUpdateRequest(BaseModel):
    path: str
    content: str


class SessionInfo(BaseModel):
    session_id: str
    title: str
    updated_at: str


class CreateSessionResponse(BaseModel):
    session_id: str


class RawMessage(BaseModel):
    role: str
    content: str


class RawMessagesResponse(BaseModel):
    session_id: str
    updated_at: str
    message_count: int
    messages: List[RawMessage]


def _resolve_project_file(relative_path: str) -> Path:
    """Resolve and validate a file path under backend/."""
    if not relative_path:
        raise HTTPException(status_code=400, detail="Path is required")

    backend_root = Path(agent.root_dir)
    target = (backend_root / relative_path).resolve()

    try:
        target.relative_to(backend_root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Path is outside backend root") from exc

    return target


@app.get("/")
@app.get("/api")
async def root():
    """Health check endpoint."""
    return {"status": "running", "service": "Mini-OpenClaw", "version": "1.0.0"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint."""
    try:
        result = agent.chat(message=request.message, session_id=request.session_id)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/skills", response_model=List[SkillInfo])
async def get_skills():
    """Get all available skills."""
    try:
        return [SkillInfo(**skill) for skill in agent.get_skills()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/skills/{skill_name}")
async def get_skill_content(skill_name: str):
    """Get the full content of a specific skill."""
    try:
        skill = agent.skill_manager.get_skill_by_name(skill_name)
        if not skill:
            raise HTTPException(status_code=404, detail="Skill not found")

        with open(skill.file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return {"name": skill_name, "content": content, "path": skill.relative_path.lstrip("./")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/files")
async def get_file(path: str = Query(..., description="Relative file path under backend/")):
    """Read a backend file."""
    try:
        target = _resolve_project_file(path)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        with open(target, "r", encoding="utf-8") as f:
            return {"path": path, "content": f.read()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/files")
async def save_file(request: FileUpdateRequest):
    """Save a backend file."""
    try:
        target = _resolve_project_file(request.path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with open(target, "w", encoding="utf-8") as f:
            f.write(request.content)

        if target.name == "SKILL.md":
            agent.skill_manager.reload_skills()
            agent.skill_manager.write_snapshot(agent.workspace_dir)
        elif target.name == "MEMORY.md":
            agent.update_memory(request.content)

        return {"status": "success", "path": request.path}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions", response_model=List[SessionInfo])
async def get_sessions():
    """Get all saved sessions."""
    try:
        return [SessionInfo(**item) for item in agent.list_sessions()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions", response_model=CreateSessionResponse)
async def create_session():
    """Create a fresh empty session."""
    try:
        return CreateSessionResponse(**agent.create_session())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}/raw-messages", response_model=RawMessagesResponse)
async def get_raw_messages(session_id: str):
    """Get the latest model payload for a session."""
    try:
        payload = agent.get_raw_messages(session_id)
        return RawMessagesResponse(**payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/raw-messages/preview", response_model=RawMessagesResponse)
async def preview_raw_messages(session_id: Optional[str] = None):
    """Preview the current model payload without requiring a new user message."""
    try:
        payload = agent.preview_raw_messages(session_id=session_id)
        return RawMessagesResponse(**payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
