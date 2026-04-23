# Skill Candidate: finance_analysis:行情 大盘近一周的整体行情:terminal

## Summary
- Description: Candidate reusable workflow for 行情 / 大盘近一周的整体行情 using terminal -> read_file -> fetch_url.
- Domain: finance_analysis
- Topic: 行情 / 大盘近一周的整体行情
- Intents: general
- Tool sequence: terminal -> read_file -> fetch_url
- Occurrence count: 1
- Confidence: 0.65

## Why This Candidate Exists
This candidate was generated because the agent repeated a similar tool-use pattern.

## Evidence
- Session: 20260419_213953
  user: 大盘近一周的整体行情
  assistant: 刚查询了上证指数的近一周行情数据，结果显示没有返回K线数据，可能是由于请求参数或接口限制。为了更准确帮您查询大盘近一周的整体行情，我用东方财富金融数据技能做具体接口查询。

请您稍等，我立即调整方式帮您获取最新的近一周大盘行情。
为了更准确地帮您获取大盘近一周的整体行情，我可以用东方财富金融数据技能里接口标准查询。请问您关注的是哪些具体指数？比如：
- 上证指数（代码：000001）
- 深证成指（代码：399001）
- 沪深300指数（代码：000300）

或者您需要我同时查询这几个主要指数？这样我能帮您更精

## Proposed Skill Prototype
```md
---
name: "finance_analysis:行情 大盘近一周的整体行情:terminal"
description: "Candidate reusable workflow for 行情 / 大盘近一周的整体行情 using terminal -> read_file -> fetch_url."
trigger: "行情 大盘近一周的整体行情|general"
enabled: false
---

# Goal
Handle recurring workflow for 行情 / 大盘近一周的整体行情.

# Recommended Steps
1. Use `terminal` when its prerequisite inputs are available.
2. Use `read_file` when its prerequisite inputs are available.
3. Use `fetch_url` when its prerequisite inputs are available.

# Validation Checklist
1. Confirm the trigger matches a genuinely repeatable task.
2. Confirm each tool call is necessary and ordered correctly.
3. Confirm the workflow generalizes beyond one session.
```
