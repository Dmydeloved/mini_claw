# Skill Candidate: finance_analysis:在尝试下:read_file

## Summary
- Description: Candidate reusable workflow for 在尝试下 using read_file -> read_file -> terminal -> terminal -> terminal -> terminal -> terminal -> terminal.
- Domain: finance_analysis
- Topic: 在尝试下
- Intents: general
- Tool sequence: read_file -> read_file -> terminal -> terminal -> terminal -> terminal -> terminal -> terminal
- Occurrence count: 1
- Confidence: 0.90

## Why This Candidate Exists
This candidate was generated because the agent repeated a similar tool-use pattern.

## Evidence
- Session: 20260419_220344
  user: 在尝试下
  assistant: 我尝试了多种格式执行命令都失败，推测当前环境调用系统命令的接口有问题。

我改用查阅技能文档，看看技能内部是否提供了调用接口或示例API调用方式，我将通过技能内部接口调用查询，请稍等。
由于系统命令调用失败，我准备直接调用“eastmoney-financial-data”技能查询平安银行最新股价。我先确认技能的调用方法。

## Proposed Skill Prototype
```md
---
name: "finance_analysis:在尝试下:read_file"
description: "Candidate reusable workflow for 在尝试下 using read_file -> read_file -> terminal -> terminal -> terminal -> terminal -> terminal -> terminal."
trigger: "在尝试下|general"
enabled: false
---

# Goal
Handle recurring workflow for 在尝试下.

# Recommended Steps
1. Use `read_file` when its prerequisite inputs are available.
2. Use `read_file` when its prerequisite inputs are available.
3. Use `terminal` when its prerequisite inputs are available.
4. Use `terminal` when its prerequisite inputs are available.
5. Use `terminal` when its prerequisite inputs are available.
6. Use `terminal` when its prerequisite inputs are available.
7. Use `terminal` when its prerequisite inputs are available.
8. Use `terminal` when its prerequisite inputs are available.

# Validation Checklist
1. Confirm the trigger matches a genuinely repeatable task.
2. Confirm each tool call is necessary and ordered correctly.
3. Confirm the workflow generalizes beyond one session.
```
