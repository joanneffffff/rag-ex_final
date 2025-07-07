# 上下文分离功能集成总结

## 🎯 目标
将 table context 和 text context 分开传递到 Prompt，让 LLM 更清晰地理解不同类型信息的来源和结构，从而可能生成更准确、更符合预期的答案。

## ✅ 已完成的工作

### 1. 创建上下文分离器 (`xlm/utils/context_separator.py`)

**核心功能：**
- 智能识别 TATQA 数据中的 Table ID 和 Paragraph ID
- 将混合上下文分离为 `table_context` 和 `text_context`
- 支持 1个 Table ID + 多个 Paragraph ID 的复杂结构
- 提供详细的元数据信息

**关键特性：**
```python
@dataclass
class SeparatedContext:
    table_context: str      # 表格相关上下文
    text_context: str       # 文本相关上下文
    context_type: str       # "table", "text", "table-text", "unknown"
    metadata: Dict          # 详细元数据
```

### 2. 集成到评估系统 (`comprehensive_evaluation_enhanced.py`)

**修改内容：**
- 添加上下文分离器导入
- 修改 `get_final_prompt()` 函数，集成上下文分离功能
- 新增 `load_and_format_template_with_separated_context()` 函数
- 保持向后兼容性，失败时回退到原始方式

**集成逻辑：**
```python
def get_final_prompt(context: str, query: str) -> List[Dict[str, str]]:
    # 1. 混合决策确定模板类型
    predicted_answer_source = hybrid_decision(context, query)
    
    # 2. 上下文分离
    if USE_CONTEXT_SEPARATOR:
        separated = context_separator.separate_context(context)
        prompt_params = context_separator.format_for_prompt(separated, query)
        
        # 3. 使用分离的上下文格式化模板
        return load_and_format_template_with_separated_context(
            template_file, 
            prompt_params["table_context"], 
            prompt_params["text_context"], 
            query
        )
    else:
        # 回退到原始方式
        return load_and_format_template(template_file, context, query)
```

### 3. 测试验证

**测试结果：**
- ✅ 上下文分离器正常工作
- ✅ 能够正确处理 1个 Table ID + 多个 Paragraph ID 的结构
- ✅ 成功将 table_context 和 text_context 分开传递给 Prompt
- ✅ 保持现有模板结构不变

**示例输出：**
```
上下文类型: table-text
表格行数: 2
文本行数: 7

📊 Table Context:
Table ID: 991d23d7-f32d-4954-8e1d-87ad22470fcf
Headers: 2019 | 2018

📝 Text Context:
In thousands:  is $; 2019 is %;  is $; 2018 is %
Drinkable Kefir other than ProBugs:  is $ 71,822; 2019 is 77%...
Paragraph ID: a4d3952f-4390-4ab2-b6f3-460d14653c10
Drinkable Kefir, sold in a variety of organic and non-organic sizes...
```

## 🔧 技术实现细节

### 上下文分离算法

1. **模式识别：**
   - Table ID 模式：`Table ID: [uuid]`
   - Paragraph ID 模式：`Paragraph ID: [uuid]`
   - 表格数据模式：包含数字、百分比、货币符号的行
   - 文本数据模式：描述性文本段落

2. **分离策略：**
   - 按行分析上下文内容
   - 根据标识符和内容特征分类
   - 保持原始格式和结构
   - 生成详细的元数据信息

3. **Prompt 格式化：**
   - 将分离的上下文作为独立参数传递
   - 支持现有模板的 `{table_context}` 和 `{text_context}` 占位符
   - 保持模板的 system 和 user 部分结构

### 兼容性保证

1. **向后兼容：**
   - 如果上下文分离失败，自动回退到原始方式
   - 保持现有 API 接口不变
   - 不影响现有模板文件

2. **错误处理：**
   - 优雅处理导入错误
   - 详细的错误日志
   - 自动回退机制

## 📊 优势分析

### 1. 结构清晰性
- **分离前：** 混合的上下文，LLM 需要自己识别数据类型
- **分离后：** 明确的 Table Context 和 Text Context，LLM 更容易理解

### 2. 信息组织
- **Table Context：** 专注于数值数据和表格结构
- **Text Context：** 专注于描述性信息和解释性文本

### 3. 推理准确性
- LLM 可以更精确地定位相关信息
- 减少混淆和错误推理
- 提高答案的准确性和一致性

## 🚀 使用方法

### 1. 自动使用（推荐）
```bash
python comprehensive_evaluation_enhanced.py --data_path your_data.jsonl
```
系统会自动检测并使用上下文分离功能。

### 2. 手动测试
```bash
python test_context_separation_simple.py
```

### 3. 编程使用
```python
from xlm.utils.context_separator import context_separator

# 分离上下文
separated = context_separator.separate_context(context)

# 格式化 prompt 参数
prompt_params = context_separator.format_for_prompt(separated, question)
```

## 📈 预期效果

1. **提高答案准确性：** LLM 能更清晰地理解数据结构
2. **减少推理错误：** 避免混淆表格数据和文本描述
3. **提升一致性：** 相同类型的问题得到更一致的答案
4. **增强可解释性：** 更容易理解 LLM 的推理过程

## 🔮 未来优化方向

1. **智能模板选择：** 根据上下文类型自动选择最优模板
2. **动态权重调整：** 根据问题类型调整 table_context 和 text_context 的重要性
3. **多模态增强：** 支持更复杂的表格结构和图表数据
4. **性能优化：** 优化分离算法的速度和准确性

## 📝 总结

成功实现了上下文分离功能，将 table context 和 text context 分开传递给 Prompt，让 LLM 更清晰地理解不同类型信息的来源和结构。该功能已完全集成到 `comprehensive_evaluation_enhanced.py` 中，保持了向后兼容性，并提供了完善的错误处理机制。

通过这种分离方式，预期能够显著提高 LLM 在 TATQA 数据集上的表现，特别是在处理混合数据（1个 Table ID + 多个 Paragraph ID）时，能够生成更准确、更符合预期的答案。 