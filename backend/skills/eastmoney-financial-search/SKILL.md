---
name: eastmoney-financial-search
description: 基于东方财富妙想搜索能力获取金融资讯，用于新闻、公告、研报、政策、交易规则、事件及影响分析等金融场景信息检索。触发场景包括：查询个股最新研报/机构观点、板块/主题新闻、政策解读、宏观风险分析、大盘异动原因、北向资金流向等需要时效性或特定事件信息的查询。
trigger: 东方财富|eastmoney|金融资讯|研报|公告|政策|新闻
enabled: true
---

# 东方财富资讯搜索 (eastmoney-financial-search)

根据用户问句搜索相关金融资讯，获取与问句相关的资讯信息（如研报、新闻、解读等）。

## API 调用

```bash
curl -X POST --location 'https://mkapi2.dfcfs.com/finskillshub/api/claw/news-search' \
--header 'Content-Type: application/json' \
--header 'apikey: ${EASTMONEY_APIKEY}' \
--data '{"query":"<用户问句>","variables":{}}'
```

**环境变量**: 将 apikey 存到环境变量 `EASTMONEY_APIKEY`

## 问句示例

| 类型 | 示例问句 |
|------|----------|
| 个股资讯 | 格力电器最新研报、贵州茅台机构观点 |
| 板块/主题 | 商业航天板块近期新闻、新能源政策解读 |
| 宏观/风险 | A股具备自然对冲优势的公司 汇率风险、美联储加息对A股影响 |
| 综合解读 | 今日大盘异动原因、北向资金流向解读 |

## 返回字段说明

| 字段路径 | 释义 |
|----------|------|
| `title` | 信息标题，高度概括核心内容 |
| `secuList` | 关联证券列表 |
| `secuList[].secuCode` | 证券代码（如 002475） |
| `secuList[].secuName` | 证券名称（如立讯精密） |
| `secuList[].secuType` | 证券类型（股票/债券） |
| `trunk` | 信息核心正文/结构化数据块 |

## 使用流程

1. 从东方财富妙想hub获取 apikey
2. 设置环境变量 `EASTMONEY_APIKEY`
3. 将用户问句作为 `query` 参数调用 API
4. 解析返回结果，提取 title、secuList、trunk 等字段
5. 将结果以可读格式返回给用户
