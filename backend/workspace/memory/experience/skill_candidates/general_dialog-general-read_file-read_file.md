# Skill Candidate: general_dialog:我想了解下近一周的 股信息:read_file

## Summary
- Description: Candidate reusable workflow for 我想了解下近一周的 / 股信息 using read_file -> read_file.
- Domain: general_dialog
- Topic: 我想了解下近一周的 / 股信息
- Intents: general
- Tool sequence: read_file -> read_file
- Occurrence count: 2
- Confidence: 0.75

## Why This Candidate Exists
This candidate was generated because the agent repeated a similar tool-use pattern.

## Evidence
- Session: 20260418_230151
  user: 帮我查询下天气
  assistant: 请告诉我您想查询哪个城市的天气？
- Session: 20260419_213953
  user: 我想了解下近一周的A股信息
  assistant: 您想了解近一周的A股信息，我可以帮您查询A股的行情类数据，比如实时行情、涨跌幅，资金流向等，也可以查询财务相关指标。请问您具体想了解什么类型的信息？比如：

- 某支股票近一周的走势数据
- 大盘近一周的整体行情
- 最近一周的资金流向情况
- 某个板块或行业的表现

请告诉我更具体的需求，我好帮您查询。

## Proposed Skill Prototype
```md
---
name: "general_dialog:我想了解下近一周的 股信息:read_file"
description: "Candidate reusable workflow for 我想了解下近一周的 / 股信息 using read_file -> read_file."
trigger: "我想了解下近一周的 股信息|general"
enabled: false
---

# Goal
Handle recurring workflow for 我想了解下近一周的 / 股信息.

# Recommended Steps
1. Use `read_file` when its prerequisite inputs are available.
2. Use `read_file` when its prerequisite inputs are available.

# Validation Checklist
1. Confirm the trigger matches a genuinely repeatable task.
2. Confirm each tool call is necessary and ordered correctly.
3. Confirm the workflow generalizes beyond one session.
```
