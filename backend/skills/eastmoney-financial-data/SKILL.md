---
name: eastmoney-financial-data
description: 基于东方财富数据库查询金融数据，支持行情类数据（实时行情、资金流向、估值）、财务类数据（基本面、财务指标、高管信息）、关系与经营类数据。触发场景包括：查询股票最新价、涨跌幅、财务指标、股东信息、资金流向等具体数据问题。
trigger: 东方财富金融数据|股票最新价|涨跌幅|财务指标|股东信息|资金流向
enabled: true
---

# 东方财富金融数据 (eastmoney_financial_data)

通过文本输入查询金融相关数据（股票、板块、指数等），接口返回 JSON 格式内容。

## API 调用

```bash
curl -X POST --location 'https://mkapi2.dfcfs.com/finskillshub/api/claw/query' \
--header 'Content-Type: application/json' \
--header 'apikey: ${EASTMONEY_APIKEY}' \
--data '{"toolQuery":"<查询内容>"}'
```

**API Key 已配置**: 存于 `~/.openclaw/workspace/.env.eastmoney`

## 数据类型

| 类型 | 示例 |
|------|------|
| 行情类 | 股票最新价、涨跌幅、资金流向、估值等 |
| 财务类 | 基本信息、财务指标、高管信息、主营业务、股东结构 |
| 关系类 | 关联关系数据企业经营数据 |

## 查询示例

- "平安银行最新价"
- "贵州茅台市盈率"
- "立讯精密主营业务"
- "招商银行前十大股东"

## 返回字段核心路径

### 一级：`data.dataTableDTOList[]`

| 字段 | 释义 |
|------|------|
| `code` | 证券代码（含市场标识，如 000001.SZ） |
| `entityName` | 证券全称 |
| `title` | 指标标题 |
| `table` | 标准化表格数据（键=指标编码，值=数值数组） |
| `nameMap` | 列名映射（指标编码→中文名） |
| `field.returnName` | 指标业务名称 |
| `entityTagDTO.fullName` | 证券完整中文名 |

### 查询条件

| 字段 | 释义 |
|------|------|
| `condition.search_data_task_0` | 原始查询条件 |

## 注意事项

- **大数据范围谨慎查询**：如查询多年日线数据可能导致返回内容过多，引发上下文爆炸
- **结果为空**：提示用户到东方财富妙想AI查询

## 使用流程

1. 从东方财富妙想hub获取 apikey
2. 设置环境变量 `EASTMONEY_APIKEY`
3. 将查询内容作为 `toolQuery` 参数调用 API
4. 解析返回结果，提取 table、nameMap、entityName 等字段
5. 将结果以可读格式返回给用户
