# Skill Candidate: finance_analysis:a股 大盘 大盘情况如何:read_file

## Summary
- Description: Candidate reusable workflow for a股 / 大盘 / 大盘情况如何 using read_file -> read_file -> terminal -> read_file -> terminal -> terminal -> terminal -> fetch_url -> fetch_url -> fetch_url.
- Domain: finance_analysis
- Topic: a股 / 大盘 / 大盘情况如何
- Intents: general
- Tool sequence: read_file -> read_file -> terminal -> read_file -> terminal -> terminal -> terminal -> fetch_url -> fetch_url -> fetch_url
- Occurrence count: 1
- Confidence: 0.95

## Why This Candidate Exists
This candidate was generated because the agent repeated a similar tool-use pattern.

## Evidence
- Session: 20260423_215045
  user: 今天A股大盘情况如何
  assistant: 今天A股大盘主要指数的名称分别是：
- 上证指数（000001）
- 深证成指（399001）
- 沪深300指数（000300）

当前接口返回的是指数代码和名称字段，未包含涨跌幅具体数值字段。可能需要调整请求的字段参数，或者使用其他接口字段。

请确认您需要查询今日大盘涨跌幅的具体指标，我可以帮您确认正确的接口字段名称后再次查询。您也可以告诉我是否需要成交量、资金流向等其他大盘指标。

## Proposed Skill Prototype
```md
---
name: "finance_analysis:a股 大盘 大盘情况如何:read_file"
description: "Candidate reusable workflow for a股 / 大盘 / 大盘情况如何 using read_file -> read_file -> terminal -> read_file -> terminal -> terminal -> terminal -> fetch_url -> fetch_url -> fetch_url."
trigger: "a股 大盘 大盘情况如何|general"
enabled: false
---

# Goal
Handle recurring workflow for a股 / 大盘 / 大盘情况如何.

# Recommended Steps
1. Use `read_file` when its prerequisite inputs are available.
2. Use `read_file` when its prerequisite inputs are available.
3. Use `terminal` when its prerequisite inputs are available.
4. Use `read_file` when its prerequisite inputs are available.
5. Use `terminal` when its prerequisite inputs are available.
6. Use `terminal` when its prerequisite inputs are available.
7. Use `terminal` when its prerequisite inputs are available.
8. Use `fetch_url` when its prerequisite inputs are available.
9. Use `fetch_url` when its prerequisite inputs are available.
10. Use `fetch_url` when its prerequisite inputs are available.

# Validation Checklist
1. Confirm the trigger matches a genuinely repeatable task.
2. Confirm each tool call is necessary and ordered correctly.
3. Confirm the workflow generalizes beyond one session.
```
