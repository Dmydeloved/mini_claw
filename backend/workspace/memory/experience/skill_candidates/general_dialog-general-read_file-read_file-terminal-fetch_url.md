# Skill Candidate: general_dialog:哈尔滨:read_file

## Summary
- Description: Candidate reusable workflow for 哈尔滨 using read_file -> read_file -> terminal -> fetch_url.
- Domain: general_dialog
- Topic: 哈尔滨
- Intents: general
- Tool sequence: read_file -> read_file -> terminal -> fetch_url
- Occurrence count: 2
- Confidence: 0.85

## Why This Candidate Exists
This candidate was generated because the agent repeated a similar tool-use pattern.

## Evidence
- Session: 20260419_193029
  user: 沈阳
  assistant: 沈阳当前的天气是：晴，气温约22°C。如果您需要更详细的天气信息或者未来的天气预报，请告诉我。
- Session: 20260419_211501
  user: 哈尔滨
  assistant: 哈尔滨当前的天气是多云，气温约17°C，风速约26公里/小时。如需更详细天气信息或未来预报，请告诉我。

## Proposed Skill Prototype
```md
---
name: "general_dialog:哈尔滨:read_file"
description: "Candidate reusable workflow for 哈尔滨 using read_file -> read_file -> terminal -> fetch_url."
trigger: "哈尔滨|general"
enabled: false
---

# Goal
Handle recurring workflow for 哈尔滨.

# Recommended Steps
1. Use `read_file` when its prerequisite inputs are available.
2. Use `read_file` when its prerequisite inputs are available.
3. Use `terminal` when its prerequisite inputs are available.
4. Use `fetch_url` when its prerequisite inputs are available.

# Validation Checklist
1. Confirm the trigger matches a genuinely repeatable task.
2. Confirm each tool call is necessary and ordered correctly.
3. Confirm the workflow generalizes beyond one session.
```
