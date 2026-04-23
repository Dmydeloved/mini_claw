# Skill Candidate: finance_analysis:大盘 涨跌 大盘涨跌幅:terminal

## Summary
- Description: Candidate reusable workflow for 大盘 / 涨跌 / 大盘涨跌幅 using terminal -> read_file -> fetch_url -> fetch_url -> fetch_url.
- Domain: finance_analysis
- Topic: 大盘 / 涨跌 / 大盘涨跌幅
- Intents: general
- Tool sequence: terminal -> read_file -> fetch_url -> fetch_url -> fetch_url
- Occurrence count: 1
- Confidence: 0.75

## Why This Candidate Exists
This candidate was generated because the agent repeated a similar tool-use pattern.

## Evidence
- Session: 20260420_215653
  user: 大盘涨跌幅
  assistant: 我查询了今日A股三大主要指数的涨跌情况，结果如下：

- 上证指数（000001）成交额约 408213（单位可能为手，具体单位技能接口未明确）
- 深证成指（399001）成交额约 1496675
- 沪深300指数（000300）成交额约 475744

当前接口返回的是成交额数据，未包含明确涨跌幅百分比字段。请问您是否需要我继续尝试查找涨跌幅？或者需要查询其他相关指标？

## Proposed Skill Prototype
```md
---
name: "finance_analysis:大盘 涨跌 大盘涨跌幅:terminal"
description: "Candidate reusable workflow for 大盘 / 涨跌 / 大盘涨跌幅 using terminal -> read_file -> fetch_url -> fetch_url -> fetch_url."
trigger: "大盘 涨跌 大盘涨跌幅|general"
enabled: false
---

# Goal
Handle recurring workflow for 大盘 / 涨跌 / 大盘涨跌幅.

# Recommended Steps
1. Use `terminal` when its prerequisite inputs are available.
2. Use `read_file` when its prerequisite inputs are available.
3. Use `fetch_url` when its prerequisite inputs are available.
4. Use `fetch_url` when its prerequisite inputs are available.
5. Use `fetch_url` when its prerequisite inputs are available.

# Validation Checklist
1. Confirm the trigger matches a genuinely repeatable task.
2. Confirm each tool call is necessary and ordered correctly.
3. Confirm the workflow generalizes beyond one session.
```
