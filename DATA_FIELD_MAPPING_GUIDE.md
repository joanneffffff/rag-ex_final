# 数据字段映射指南

## 🔍 问题背景

在RAG系统中，我们发现不同数据源使用了不同的字段名来表示相同的内容，这导致了数据加载问题。

## 📊 当前数据字段映射情况

### 1. AlphaFin中文数据
**文件**: `data/alphafin/alphafin_cleaned.json`
**格式**: JSON数组
**字段映射**:
```json
{
  "original_context": "实际的财务新闻内容",
  "original_answer": "生成的答案",
  "summary": "摘要内容",
  "generated_question": "生成的问题",
  "company_name": "公司名称",
  "stock_code": "股票代码",
  "report_date": "报告日期"
}
```

### 2. TAT-QA英文数据
**文件**: `evaluate_mrr/tatqa_knowledge_base.jsonl`
**格式**: JSONL（每行一个JSON对象）
**字段映射**:
```json
{
  "doc_id": "文档ID",
  "source_type": "数据源类型",
  "text": "实际的表格或段落内容"
}
```

### 3. TAT-QA评估数据
**文件**: `evaluate_mrr/tatqa_eval_enhanced.jsonl`
**格式**: JSONL（每行一个JSON对象）
**字段映射**:
```json
{
  "query": "查询问题",
  "context": "实际的表格或段落内容",
  "answer": "标准答案",
  "doc_id": "文档ID",
  "relevant_doc_ids": ["相关文档ID列表"],
  "answer_from": "答案来源类型"
}
```

## ⚠️ 问题分析

### 问题1: 字段名不一致
- **AlphaFin**: 使用 `original_context` 字段
- **TAT-QA知识库**: 使用 `text` 字段
- **TAT-QA评估数据**: 使用 `context` 字段
- **数据加载器**: 只查找 `context` 字段

### 问题2: 数据格式不一致
- **AlphaFin**: JSON数组格式
- **TAT-QA**: JSONL格式（每行一个对象）

### 问题3: 内容结构不一致
- **AlphaFin**: 包含模板化内容（已清理）
- **TAT-QA知识库**: 包含Table ID和Paragraph ID标识
- **TAT-QA评估数据**: 包含Table ID和Paragraph ID标识

## 🛠️ 解决方案

### 方案1: 统一字段名（推荐）

#### 1.1 修改TAT-QA知识库数据格式
将TAT-QA知识库数据的`text`字段重命名为`context`：

```bash
# 创建转换脚本
python -c "
import json
with open('evaluate_mrr/tatqa_knowledge_base.jsonl', 'r') as f_in:
    with open('evaluate_mrr/tatqa_knowledge_base_unified.jsonl', 'w') as f_out:
        for line in f_in:
            item = json.loads(line.strip())
            # 将text字段重命名为context
            if 'text' in item:
                item['context'] = item.pop('text')
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
"
```

#### 1.2 修改AlphaFin数据格式
将AlphaFin数据的`original_context`字段重命名为`context`：

```bash
# 创建转换脚本
python -c "
import json
with open('data/alphafin/alphafin_cleaned.json', 'r') as f:
    data = json.load(f)
    
for item in data:
    # 将original_context字段重命名为context
    if 'original_context' in item:
        item['context'] = item.pop('original_context')
    
with open('data/alphafin/alphafin_unified.json', 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
"
```

### 方案2: 修改数据加载器（当前采用）

修改`xlm/utils/dual_language_loader.py`中的数据加载函数，使其支持多种字段名：

```python
def load_tatqa_context_only(self, file_path: str) -> List[DocumentWithMetadata]:
    """加载TAT-QA英文数据（支持多种字段名）"""
    documents = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    # 支持多种字段名：text, context, original_context
                    context = (item.get('text', '') or 
                             item.get('context', '') or 
                             item.get('original_context', '')).strip()
                    
                    if context:
                        metadata = DocumentMetadata(
                            source="tatqa",
                            language="english",
                            doc_id=f"tatqa_{idx}"
                        )
                        document = DocumentWithMetadata(
                            content=context,
                            metadata=metadata
                        )
                        documents.append(document)
                except Exception as e:
                    print(f"跳过第{idx+1}行，原因: {e}")
        
        print(f"加载了 {len(documents)} 个TAT-QA文档")
        return documents
        
    except Exception as e:
        print(f"错误: 加载TAT-QA数据失败: {e}")
        return []
```

**关键发现**：
- **TAT-QA知识库** (`tatqa_knowledge_base.jsonl`): 使用 `text` 字段
- **TAT-QA评估数据** (`tatqa_eval_enhanced.jsonl`): 使用 `context` 字段
- 同一个数据集的不同用途文件使用了不同的字段名！

## 📋 推荐的数据标准

### 统一的数据格式标准

```json
{
  "doc_id": "唯一文档标识符",
  "context": "文档内容（表格、段落或混合内容）",
  "question": "相关问题（可选）",
  "answer": "标准答案（可选）",
  "source_type": "数据源类型（train/test/dev）",
  "language": "语言标识（chinese/english）",
  "metadata": {
    "company_name": "公司名称（中文数据）",
    "stock_code": "股票代码（中文数据）",
    "report_date": "报告日期（中文数据）",
    "table_id": "表格ID（英文数据）",
    "paragraph_id": "段落ID（英文数据）"
  }
}
```

### 文件格式标准
- **统一使用JSONL格式**：每行一个JSON对象
- **统一字段名**：使用`context`作为内容字段
- **统一编码**：UTF-8编码

## 🔧 实施步骤

### 步骤1: 创建统一数据转换脚本

```python
#!/usr/bin/env python3
"""
统一数据格式转换脚本
"""

import json
from pathlib import Path

def convert_alphafin_to_unified(input_path: str, output_path: str):
    """转换AlphaFin数据为统一格式"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            unified_item = {
                "doc_id": f"alphafin_{item.get('company_name', 'unknown')}",
                "context": item.get('original_context', ''),
                "question": item.get('generated_question', ''),
                "answer": item.get('original_answer', ''),
                "source_type": "train",
                "language": "chinese",
                "metadata": {
                    "company_name": item.get('company_name', ''),
                    "stock_code": item.get('stock_code', ''),
                    "report_date": item.get('report_date', '')
                }
            }
            f.write(json.dumps(unified_item, ensure_ascii=False) + '\n')

def convert_tatqa_to_unified(input_path: str, output_path: str):
    """转换TAT-QA数据为统一格式"""
    with open(input_path, 'r', encoding='utf-8') as f:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line in f:
                item = json.loads(line.strip())
                # 支持多种字段名：text, context
                context = item.get('text', '') or item.get('context', '')
                unified_item = {
                    "doc_id": item.get('doc_id', 'unknown'),
                    "context": context,
                    "source_type": item.get('source_type', 'train'),
                    "language": "english",
                    "metadata": {}
                }
                f_out.write(json.dumps(unified_item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    # 转换AlphaFin数据
    convert_alphafin_to_unified(
        "data/alphafin/alphafin_cleaned.json",
        "data/alphafin/alphafin_unified.jsonl"
    )
    
    # 转换TAT-QA知识库数据
    convert_tatqa_to_unified(
        "evaluate_mrr/tatqa_knowledge_base.jsonl",
        "evaluate_mrr/tatqa_knowledge_base_unified.jsonl"
    )
    
    # 转换TAT-QA评估数据
    convert_tatqa_to_unified(
        "evaluate_mrr/tatqa_eval_enhanced.jsonl",
        "evaluate_mrr/tatqa_eval_enhanced_unified.jsonl"
    )
    
    print("✅ 数据格式统一完成！")
```

### 步骤2: 更新配置文件

```python
# config/parameters.py
@dataclass
class DataConfig:
    # 使用统一格式的数据文件
    chinese_data_path: str = "data/alphafin/alphafin_unified.jsonl"
    english_data_path: str = "evaluate_mrr/tatqa_knowledge_base_unified.jsonl"
```

### 步骤3: 简化数据加载器

```python
def load_unified_data(self, file_path: str, language: str) -> List[DocumentWithMetadata]:
    """加载统一格式的数据"""
    documents = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                context = item.get('context', '').strip()
                
                if context:
                    metadata = DocumentMetadata(
                        source=item.get('source_type', 'unknown'),
                        language=language,
                        doc_id=item.get('doc_id', f'doc_{idx}')
                    )
                    
                    document = DocumentWithMetadata(
                        content=context,
                        metadata=metadata
                    )
                    documents.append(document)
                    
            except Exception as e:
                print(f"跳过第{idx+1}行，原因: {e}")
    
    return documents
```

## 🎯 总结

### 当前状态
- ✅ **问题已识别**：字段名不一致导致数据加载失败
- ✅ **临时修复**：数据加载器支持多种字段名
- ⏳ **长期方案**：统一数据格式标准

### 建议
1. **立即**：使用当前修复的数据加载器
2. **短期**：创建统一格式的数据文件
3. **长期**：建立数据格式标准，避免类似问题

### 验证方法
```bash
# 测试数据加载
python test_english_data_loading.py
python test_chinese_data_loading.py

# 测试RAG系统
python run_optimized_ui.py
```

通过统一数据格式，我们可以：
- 简化数据加载逻辑
- 提高系统可维护性
- 避免字段名不一致问题
- 便于后续数据扩展 