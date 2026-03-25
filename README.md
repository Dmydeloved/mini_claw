# Mini-OpenClaw

一个基于 Python 的轻量级、透明的 AI Agent 系统，具有真实的文件记忆功能。

## 项目特点

- **文件即记忆**: 所有对话和反思以人类可读的 Markdown/JSON 文件存储
- **技能即插件**: 遵循 Anthropic 的 Agent Skills 范式，文件夹结构管理能力
- **透明可控**: 所有系统提示词、工具调用、记忆操作完全透明
- **本地优先**: 完全运行在本地，无需云服务依赖

## 技术栈

### 后端 (Port 8002)
- Python 3.10+ (Type Hinting)
- FastAPI (异步 RESTful API)
- LangChain 1.x (使用 `create_agent` API)
- LlamaIndex (混合检索: BM25 + Vector)
- 支持 OpenAI 兼容 API (OpenRouter, DeepSeek, Claude)

### 前端 (Port 3000)
- Next.js 14+ (App Router)
- TypeScript
- Tailwind CSS
- Monaco Editor (技能编辑器)

## 快速开始

### 1. 后端设置

```bash
cd backend

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key

# 启动后端服务
python app.py
# 或使用 uvicorn
uvicorn app:app --reload --port 8002
```

### 2. 前端设置

```bash
cd frontend

# 安装依赖
npm install
# 或使用 pnpm
pnpm install

# 启动开发服务器
npm run dev
# 访问 http://localhost:3000
```

## 项目结构

```
mini-openclaw/
├── backend/
│   ├── app.py              # FastAPI 入口
│   ├── memory/             # 记忆存储
│   │   ├── logs/           # 每日对话日志
│   │   └── MEMORY.md       # 核心记忆
│   ├── sessions/           # 会话记录
│   ├── skills/             # Agent 技能
│   │   └── get_weather/
│   │       └── SKILL.md
│   ├── workspace/          # 系统提示词
│   │   └── SOUL.md         # Agent 人格定义
│   ├── tools/              # 核心工具实现
│   └── graph/              # Agent 图和状态管理
│
└── frontend/
    ├── src/
    │   ├── app/            # Next.js 页面
    │   ├── components/     # React 组件
    │   └── lib/            # API 客户端
    └── package.json
```

## 核心功能

### 1. 五大内置工具

- **terminal**: 沙箱化的 Shell 命令执行
- **python_repl**: Python 代码解释器
- **fetch_url**: 网页内容获取（自动清洗 HTML）
- **read_file**: 本地文件读取
- **search_knowledge_base**: RAG 混合检索

### 2. Agent Skills 系统

技能以文件夹形式存在，每个技能包含 `SKILL.md` 文件：

```yaml
---
name: get_weather
description: 获取城市天气信息
trigger: weather|天气|forecast
enabled: true
---

# 技能说明
...
```

### 3. 三层记忆系统

1. **MEMORY.md**: 核心持久化事实（用户偏好、重要信息）
2. **Daily Logs**: 按日期存储的对话记录
3. **Session JSON**: 活跃会话的完整历史

## API 端点

### 后端 API (http://localhost:8002)

- `POST /chat` - 发送消息给 Agent
- `GET /skills` - 获取所有技能列表
- `GET /skills/{name}` - 获取特定技能内容
- `POST /skills/{name}` - 更新技能内容
- `GET /memory` - 获取 MEMORY.md 内容
- `POST /memory` - 更新 MEMORY.md 内容

## 环境变量配置

在 `backend/.env` 中配置：

```bash
# 必需
OPENAI_API_KEY=your_api_key_here

# 可选 - 使用 OpenRouter 或其他提供商
OPENAI_BASE_URL=https://api.openrouter.ai/v1
MODEL_NAME=anthropic/claude-3.5-sonnet

# 路径配置
MEMORY_DIR=./memory
SKILLS_DIR=./skills
WORKSPACE_DIR=./workspace
KNOWLEDGE_DIR=./knowledge
SESSIONS_DIR=./sessions

# 服务器
PORT=8002
```

## 开发指南

### 添加新技能

1. 在 `backend/skills/` 创建新文件夹
2. 创建 `SKILL.md` 文件，包含 frontmatter 和说明
3. 重启后端服务加载新技能

### 修改 Agent 人格

编辑 `backend/workspace/SOUL.md` 文件，定义 Agent 的性格和行为方式。

### 添加知识库文档

将 PDF/MD/TXT 文件放入 `backend/knowledge/` 目录，系统会自动构建索引。

## 注意事项

- 确保 Python 3.10+ 版本
- 首次运行会自动创建必要的目录和文件
- 所有工具都有安全限制，防止访问系统关键文件
- 前端通过 Next.js rewrites 代理后端 API

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

---

**Mini-OpenClaw** - 透明、可控、本地优先的 AI Agent 系统
