# TAT-QA 知识库合并总结

## 🎯 合并目标

将优化版本的 TAT-QA 训练和评估数据合并为统一的知识库，用于上下文分离功能的测试和评估。

## 📊 合并过程

### 输入文件
- **训练数据**：`evaluate_mrr/tatqa_train_qc_enhanced_optimized.jsonl`
- **评估数据**：`evaluate_mrr/tatqa_eval_enhanced_optimized.jsonl`

### 输出文件
- **合并知识库**：`data/unified/tatqa_knowledge_base_combined.jsonl`

## 📈 合并统计

### 原始数据统计
- **训练文档数**：14,883 个
- **评估文档数**：1,663 个
- **总计**：16,546 个

### 去重后统计
- **唯一文档数**：10,067 个
- **训练文档**：9,012 个
- **评估文档**：1,055 个
- **去重率**：39.1% (6,479 个重复文档被移除)

### 文件信息
- **文件大小**：11.1 MB
- **格式**：JSONL (每行一个 JSON 对象)
- **编码**：UTF-8

## 🔍 数据结构

### 文档格式
```json
{
    "doc_id": "train_optimized_0",
    "content": "Table ID: e78f8b29-6085-43de-b32f-be1a68641be3\nTable columns: Weighted average actuarial assumptions used at 31 March1:, 2019 %, 2018 %, 2017 %.\nFor Rate of inflation2: Weighted average actuarial assumptions used at 31 March1: is Rate of inflation2, 2019 % is 2.9, 2018 % is 2.9, 2017 % is 3.0...",
    "source": "tatqa_train_optimized",
    "language": "english",
    "created_at": "",
    "author": ""
}
```

### 字段说明
- **doc_id**：文档唯一标识符
  - 训练文档：`train_optimized_{id}`
  - 评估文档：`eval_optimized_{id}`
- **content**：文档内容（包含 Table ID 和 Paragraph ID）
- **source**：数据来源
  - `tatqa_train_optimized`：训练数据
  - `tatqa_eval_optimized`：评估数据
- **language**：语言（english）
- **created_at**：创建时间（空）
- **author**：作者（空）

## ✅ 优化特性

### 1. 表格文本化优化
合并的知识库使用了优化版本的表格文本化，具有以下改进：

- **更自然的语言表达**：使用 "For"、"is" 等自然语言结构
- **更清晰的数值表达**：去除货币符号，使用 "a negative" 表达负数
- **更好的结构描述**：`Table columns:` 替代 `Headers:`
- **统一的单位管理**：单独声明单位信息

### 2. 上下文分离友好
知识库内容包含清晰的标识符：
- **Table ID**：表格数据标识
- **Paragraph ID**：文本数据标识
- 支持 1个 Table ID + 多个 Paragraph ID 的复杂结构

## 🧪 使用示例

### 1. 测试上下文分离功能
```bash
python test_context_separation_simple.py
```

### 2. 使用评估系统
```bash
python comprehensive_evaluation_enhanced.py --data_path data/unified/tatqa_knowledge_base_combined.jsonl --sample_size 10
```

### 3. 编程使用
```python
from xlm.utils.context_separator import context_separator

# 加载知识库
with open("data/unified/tatqa_knowledge_base_combined.jsonl", "r") as f:
    for line in f:
        doc = json.loads(line)
        context = doc["content"]
        
        # 分离上下文
        separated = context_separator.separate_context(context)
        print(f"文档 {doc['doc_id']}: {separated.context_type}")
```

## 📋 样本内容预览

### 样本1：表格数据
```
Table ID: e78f8b29-6085-43de-b32f-be1a68641be3
Table columns: Weighted average actuarial assumptions used at 31 March1:, 2019 %, 2018 %, 2017 %.
For Rate of inflation2: Weighted average actuarial assumptions used at 31 March1: is Rate of inflation2, 2019 % is 2.9, 2018 % is 2.9, 2017 % is 3.0.
```

### 样本2：文本数据
```
Paragraph ID: ddf26912-5783-4b3b-b351-87e91b4a5f5b
Internally developed software is capitalised at cost less accumulated amortisation. Amortisation is calculated using the straight-line basis over the asset's useful economic life which is generally two to three years.
```

### 样本3：混合数据
```
Table ID: 991d23d7-f32d-4954-8e1d-87ad22470fcf
Table columns: 2019, 2018.
All monetary amounts are in thousands of USD.
For Drinkable Kefir other than ProBugs: 2019 is 71822, 2018 is 78523.

Paragraph ID: a4d3952f-4390-4ab2-b6f3-460d14653c10
Drinkable Kefir, sold in a variety of organic and non-organic sizes, flavors, and types...
```

## 🎉 合并完成

✅ **成功合并**：10,067 个唯一文档
✅ **优化文本化**：使用改进的表格文本化格式
✅ **上下文分离**：支持 table_context 和 text_context 分离
✅ **去重处理**：移除 6,479 个重复文档
✅ **格式统一**：标准化的 JSONL 格式

现在您可以使用这个合并的知识库来测试上下文分离功能，它将为 LLM 提供更清晰、更易理解的数据结构，从而提高问答的准确性。 