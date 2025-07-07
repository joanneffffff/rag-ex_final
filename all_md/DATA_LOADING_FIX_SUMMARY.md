# 数据加载问题修复总结

## 问题描述

在运行 `comprehensive_evaluation_enhanced.py` 时遇到以下错误：

1. **TypeError: string indices must be integers, not 'str'**
2. **AttributeError: 'str' object has no attribute 'get'**

## 根本原因

脚本的数据加载逻辑有问题：
- 脚本首先尝试作为JSONL格式加载数据（逐行解析）
- 但 `evaluate_mrr/tatqa_test_15_samples.json` 是**标准JSON数组格式**
- 导致 `eval_data` 变成了字符串列表，而不是字典列表
- 后续代码假设每个元素是字典，导致 `sample.get(...)` 报错

## 解决方案

### 1. 修复了 `comprehensive_evaluation_enhanced.py`

**修改前的问题逻辑：**
```python
# 首先尝试作为JSONL格式加载（逐行解析）
jsonl_success = False
with open(args.data_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if line:
            try:
                eval_data.append(json.loads(line))  # 错误：把JSON数组当作JSONL处理
            except json.JSONDecodeError as e:
                print(f"❌ 第{line_num}行JSON解析失败: {e}")
                continue
```

**修改后的智能检测逻辑：**
```python
# 智能检测文件格式并加载
with open(args.data_path, 'r', encoding='utf-8') as f:
    first_char = f.read(1)
    f.seek(0)  # 重置文件指针
    
    if first_char == '[':
        # 标准JSON数组格式
        data = json.load(f)
        if isinstance(data, list):
            eval_data = data
    else:
        # JSONL格式（每行一个JSON对象）
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                eval_data.append(json.loads(line))
```

### 2. 创建了通用数据加载工具 `utils/data_loader.py`

提供了智能的数据加载功能：

```python
from utils.data_loader import load_json_or_jsonl, sample_data

# 智能加载JSON或JSONL格式
data = load_json_or_jsonl("path/to/data.json")

# 数据采样
sampled_data = sample_data(data, sample_size=10)

# 格式验证
validate_data_format(data, required_fields=["query", "answer"])

# 格式转换
convert_format("input.json", "output.jsonl", "jsonl")
```

### 3. 更新了 `comprehensive_evaluation_enhanced.py`

使用新的数据加载工具：

```python
try:
    from utils.data_loader import load_json_or_jsonl, sample_data
    eval_data = load_json_or_jsonl(args.data_path)
    
    # 采样
    if args.sample_size and args.sample_size < len(eval_data):
        eval_data = sample_data(eval_data, args.sample_size, 42)
        
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    return
```

## 测试结果

修复后的脚本能够正确运行：

```bash
python comprehensive_evaluation_enhanced.py --data_path evaluate_mrr/tatqa_test_15_samples.json --sample_size 2
```

**输出：**
```
✅ 成功加载为JSON数组，样本数: 15
✅ 随机采样 2 个样本
✅ 随机采样 2 个样本进行评估。
🔄 加载模型...
✅ 模型加载完成
🔍 评估样本: 100%|██████████████████████████████████████████████| 2/2 [05:37<00:00, 168.99s/个]
✅ 评估完成，总耗时: 337.99秒
```

## 支持的数据格式

### 1. 标准JSON数组格式
```json
[
  {
    "query": "What is the total assets?",
    "answer": "948,578",
    "context": "..."
  },
  {
    "query": "What are the fiscal years?",
    "answer": "2019; 2018",
    "context": "..."
  }
]
```

### 2. JSONL格式（每行一个JSON对象）
```jsonl
{"query": "What is the total assets?", "answer": "948,578", "context": "..."}
{"query": "What are the fiscal years?", "answer": "2019; 2018", "context": "..."}
```

## 工具特性

1. **智能格式检测**：自动识别JSON和JSONL格式
2. **错误处理**：详细的错误信息和异常处理
3. **数据验证**：验证数据格式和必需字段
4. **采样功能**：支持随机采样
5. **格式转换**：支持JSON和JSONL之间的转换
6. **通用性**：可在其他脚本中复用

## 使用建议

1. **统一使用工具函数**：在其他脚本中也使用 `utils.data_loader` 中的函数
2. **数据格式标准化**：建议统一使用JSON数组格式，便于处理
3. **错误处理**：始终包含适当的错误处理逻辑
4. **数据验证**：在加载后验证数据格式和必需字段

## 相关文件

- `comprehensive_evaluation_enhanced.py` - 修复后的评估脚本
- `utils/data_loader.py` - 通用数据加载工具
- `utils/__init__.py` - 工具包初始化文件
- `evaluate_mrr/tatqa_test_15_samples.json` - 测试数据文件 