# Skill Candidate: finance_analysis:是:fetch_url

## Summary
- Description: Candidate reusable workflow for 是 using fetch_url -> fetch_url -> fetch_url.
- Domain: finance_analysis
- Topic: 是
- Intents: general
- Tool sequence: fetch_url -> fetch_url -> fetch_url
- Occurrence count: 1
- Confidence: 0.65

## Why This Candidate Exists
This candidate was generated because the agent repeated a similar tool-use pattern.

## Evidence
- Session: 20260419_213953
  user: 是
  assistant: 抱歉，当前请求接口在获取近一周的K线数据时遇到了网络或接口响应问题，导致上证指数、深证成指和沪深300这三个主要指数的近一周行情数据未能成功获取。

不过，这些返回结果显示接口正常响应，只是未返回具体的K线数据，可能接口时间参数或调用方式有误。

您是否需要我尝试用其他方式或工具帮您总结近一周大盘的行情走势？或者您有其他特定股票或指数想了解近一周的走势？

## Proposed Skill Prototype
```md
---
name: "finance_analysis:是:fetch_url"
description: "Candidate reusable workflow for 是 using fetch_url -> fetch_url -> fetch_url."
trigger: "是|general"
enabled: false
---

# Goal
Handle recurring workflow for 是.

# Recommended Steps
1. Use `fetch_url` when its prerequisite inputs are available.
2. Use `fetch_url` when its prerequisite inputs are available.
3. Use `fetch_url` when its prerequisite inputs are available.

# Validation Checklist
1. Confirm the trigger matches a genuinely repeatable task.
2. Confirm each tool call is necessary and ordered correctly.
3. Confirm the workflow generalizes beyond one session.
```
