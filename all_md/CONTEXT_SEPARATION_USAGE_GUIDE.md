# 上下文分离功能使用指南

## 🎯 功能概述

上下文分离功能已经成功集成到 `comprehensive_evaluation_enhanced.py` 中，能够将 TATQA 数据中的 `table_context` 和 `text_context` 分开传递给 Prompt，让 LLM 更清晰地理解不同类型信息的来源和结构。

## ✅ 功能验证

从测试结果可以看到，系统已经成功检测到上下文分离功能：

```bash
$ python comprehensive_evaluation_enhanced.py --test
✅ 使用RAG系统的LocalLLMGenerator
✅ 使用上下文分离功能
```

## 🚀 使用方法

### 1. 直接使用（推荐）

```bash
# 使用现有的 TATQA 评估数据
python comprehensive_evaluation_enhanced.py --data_path data/unified/tatqa_eval_enhanced_unified.jsonl --sample_size 10

# 使用完整的评估数据
python comprehensive_evaluation_enhanced.py --data_path data/unified/tatqa_eval_enhanced_unified.jsonl
```

### 2. 使用测试数据

```bash
# 使用我们创建的测试样本
python comprehensive_evaluation_enhanced.py --data_path test_context_separation_samples.jsonl --sample_size 3
```

### 3. 编程使用

```python
from xlm.utils.context_separator import context_separator
from comprehensive_evaluation_enhanced import get_final_prompt

# 分离上下文
separated = context_separator.separate_context(context)

# 生成 prompt
messages = get_final_prompt(context, query)
```

## 📊 支持的上下文类型

### 1. 纯文本数据
```
Paragraph ID: 4202457313786d975b89fabc695c3efb
We utilized a comprehensive approach to evaluate and document...
```

### 2. 纯表格数据
```
Table ID: e78f8b29-6085-43de-b32f-be1a68641be3
Headers: 2019 % | 2018 % | 2017 %
Rate of inflation2: 2019 % is $2.9; 2018 % is $2.9; 2017 % is $3.0
```

### 3. 混合数据（1个 Table ID + 多个 Paragraph ID）
```
Table ID: 991d23d7-f32d-4954-8e1d-87ad22470fcf
Headers: 2019 | 2018
Drinkable Kefir other than ProBugs:  is $ 71,822; 2019 is 77%

Paragraph ID: a4d3952f-4390-4ab2-b6f3-460d14653c10
Drinkable Kefir, sold in a variety of organic and non-organic sizes...

Paragraph ID: d623137a-e787-4204-952a-af9d4ed3a2db
European-style soft cheeses, including farmer cheese...
```

## 🔧 工作原理

### 1. 上下文分离流程
```
原始上下文 → 模式识别 → 分离处理 → 格式化输出
     ↓           ↓          ↓          ↓
混合数据    识别Table ID   分离为      table_context
         识别Paragraph ID  table_context  text_context
                         text_context
```

### 2. Prompt 生成流程
```
分离的上下文 → 模板选择 → 参数格式化 → 最终 Prompt
     ↓           ↓          ↓           ↓
table_context  混合决策   替换占位符   发送给 LLM
text_context   确定模板   {table_context}
                {text_context}
```

### 3. 兼容性保证
- **自动检测：** 系统自动检测是否可以使用上下文分离功能
- **优雅回退：** 如果分离失败，自动回退到原始方式
- **向后兼容：** 不影响现有的模板和 API

## 📈 预期效果

### 1. 提高答案准确性
- LLM 能更清晰地理解数据结构
- 减少表格数据和文本描述的混淆
- 提高数值计算的准确性

### 2. 增强推理能力
- 更精确地定位相关信息
- 减少错误推理
- 提高答案的一致性

### 3. 改善用户体验
- 更快的响应速度
- 更准确的答案
- 更好的可解释性

## 🧪 测试验证

### 1. 快速测试
```bash
python quick_test_context_separation.py
```

### 2. 详细测试
```bash
python test_context_separation_simple.py
```

### 3. 集成测试
```bash
python test_integration.py
```

## 📋 测试样本

我们创建了包含 3 种类型数据的测试样本：

1. **纯文本样本：** 测试文本数据的处理
2. **混合数据样本：** 测试 1个 Table ID + 多个 Paragraph ID 的复杂结构
3. **纯表格样本：** 测试表格数据的处理

每个样本都包含：
- `query`: 问题
- `context`: 上下文数据
- `answer`: 期望答案
- `answer_from`: 答案来源类型

## 🔍 故障排除

### 1. 导入错误
```
❌ 上下文分离器导入失败
```
**解决方案：** 检查 `xlm/utils/context_separator.py` 文件是否存在

### 2. 模板错误
```
❌ 模板文件未找到
```
**解决方案：** 检查 `data/prompt_templates/` 目录下的模板文件

### 3. 回退到原始方式
```
⚠️ 上下文分离失败，回退到原始方式
```
**解决方案：** 检查上下文数据格式是否正确

## 📝 总结

上下文分离功能已经成功集成到评估系统中，能够：

1. **智能识别** TATQA 数据中的 Table ID 和 Paragraph ID
2. **自动分离** table_context 和 text_context
3. **保持兼容** 现有模板和 API
4. **提供回退** 机制确保系统稳定性

通过这种分离方式，预期能够显著提高 LLM 在 TATQA 数据集上的表现，特别是在处理复杂的混合数据时，能够生成更准确、更符合预期的答案。

现在您可以直接使用 `comprehensive_evaluation_enhanced.py` 来评估模型性能，系统会自动使用上下文分离功能来优化 Prompt 生成。 