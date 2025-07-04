# 字段名不一致问题总结

## 🔍 问题发现

在TAT-QA数据集中，我们发现同一个数据集的不同文件使用了不同的字段名来表示相同的内容：

### 📊 字段名对比

| 文件 | 字段名 | 用途 |
|------|--------|------|
| `tatqa_knowledge_base.jsonl` | `text` | 知识库内容 |
| `tatqa_eval_enhanced.jsonl` | `context` | 评估数据内容 |

### 🔍 具体示例

**知识库文件** (`tatqa_knowledge_base.jsonl`):
```json
{
  "text": "Table ID: e78f8b29-6085-43de-b32f-be1a68641be3\nHeaders: 2019 % | 2018 % | 2017 %\n...",
  "doc_id": "e78f8b29608543deb32fbe1a68641be3",
  "source_type": "train"
}
```

**评估文件** (`tatqa_eval_enhanced.jsonl`):
```json
{
  "query": "What method did the company use when Topic 606 in fiscal 2019 was adopted?",
  "context": "Paragraph ID: 4202457313786d975b89fabc695c3efb\nWe utilized a comprehensive approach...",
  "answer": "the modified retrospective method",
  "doc_id": "4202457313786d975b89fabc695c3efb",
  "answer_from": "text"
}
```

## ⚠️ 问题影响

1. **数据加载混乱**: 数据加载器需要支持多种字段名
2. **代码维护困难**: 需要记住不同文件的字段名
3. **容易出错**: 开发时容易混淆字段名

## 🛠️ 解决方案

### 方案1: 统一字段名（推荐）
将所有数据文件的字段名统一为 `context`

### 方案2: 智能数据加载器（当前采用）
数据加载器自动检测并支持多种字段名：
- `text`
- `context` 
- `original_context`

## 📋 建议

1. **立即**: 使用当前的数据加载器（已支持多种字段名）
2. **短期**: 创建统一格式的数据文件
3. **长期**: 建立数据格式标准，避免类似问题

## 🎯 根本原因

这个问题出现的原因是：
- 不同阶段的数据处理使用了不同的字段名
- 缺乏统一的数据格式标准
- 没有在数据生成时进行字段名统一

**建议**: 在数据预处理阶段就统一字段名，避免后续的混乱。 