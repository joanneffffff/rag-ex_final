# 增强扰动评估系统

## 概述

本系统实现了对扰动结果的增强评估，包括：

1. **LLM-Judge评估**: 使用`llm_comparison/chinese_llm_judge.py`对扰动结果进行评估
2. **F1和EM计算**: 使用`llm_comparison/chinese_llm_evaluation.py`中的逻辑计算：
   - 扰动答案 vs 期望答案
   - 扰动答案 vs 原始答案

## 功能特性

### 1. 多维度评估指标

- **F1分数对比**:
  - 原始答案 vs 期望答案
  - 扰动答案 vs 期望答案  
  - 扰动答案 vs 原始答案

- **精确匹配(EM)对比**:
  - 原始答案 vs 期望答案
  - 扰动答案 vs 期望答案
  - 扰动答案 vs 原始答案

- **F1改进**: 扰动答案相对于原始答案的F1分数改进

### 2. LLM Judge评估

- 准确性评分
- 简洁性评分  
- 专业性评分
- 综合评分
- 评估推理过程

### 3. 统计摘要

- 各指标的平均值统计
- 按扰动器分组的统计
- 详细的评估报告

## 文件结构

```
├── llm_comparison/
│   ├── enhanced_perturbation_evaluation.py  # 主要评估脚本
│   ├── chinese_llm_evaluation.py           # F1/EM计算逻辑
│   └── chinese_llm_judge.py               # LLM Judge评估
├── run_enhanced_perturbation_evaluation.py  # 运行示例
├── perturbation_results_incremental.json    # 扰动结果文件
└── data/alphafin/
    └── alphafin_eval_samples_updated.jsonl # 期望答案数据
```

## 使用方法

### 方法1: 使用命令行脚本

```bash
python llm_comparison/enhanced_perturbation_evaluation.py \
    --perturbation_file perturbation_results_incremental.json \
    --alphafin_data data/alphafin/alphafin_eval_samples_updated.jsonl \
    --output_file enhanced_results.json \
    --judge_model Qwen3-8B \
    --judge_device cuda:1
```

### 方法2: 使用示例脚本

```bash
python run_enhanced_perturbation_evaluation.py
```

### 方法3: 在代码中使用

```python
from llm_comparison.enhanced_perturbation_evaluation import EnhancedPerturbationEvaluator

# 创建评估器
evaluator = EnhancedPerturbationEvaluator()

# 加载期望答案
evaluator.load_expected_answers("data/alphafin/alphafin_eval_samples_updated.jsonl")

# 初始化LLM Judge (可选)
evaluator.initialize_llm_judge("Qwen3-8B", "cuda:1")

# 执行评估
evaluator.evaluate_perturbation_results(
    "perturbation_results_incremental.json",
    "enhanced_results.json"
)
```

## 输入文件格式

### 扰动结果文件 (JSON)

```json
[
  {
    "sample_id": "sample_xxx",
    "question": "问题内容",
    "context": "上下文信息",
    "expected_answer": "",
    "perturber_name": "term",
    "perturbation_detail": {
      "original_text": "原始文本",
      "perturbed_text": "扰动后文本",
      "perturbation_type": "term",
      "confidence": 1.0
    },
    "original_answer": "原始答案",
    "perturbed_answer": "扰动后答案",
    "similarity_score": 0.0,
    "importance_score": 0.0,
    "f1_score": 0.0,
    "em_score": 0.0,
    "timestamp": "2025-07-16T05:01:48.478052"
  }
]
```

### AlphaFin数据文件 (JSONL)

```jsonl
{"generated_question": "问题1", "answer": "期望答案1"}
{"generated_question": "问题2", "answer": "期望答案2"}
```

## 输出文件格式

增强评估结果包含原始数据加上新的评估指标：

```json
[
  {
    // 原始数据
    "sample_id": "sample_xxx",
    "question": "问题内容",
    "original_answer": "原始答案",
    "perturbed_answer": "扰动后答案",
    
    // 新增的评估指标
    "f1_original_vs_expected": 0.5,
    "em_original_vs_expected": 0.0,
    "f1_perturbed_vs_expected": 0.6,
    "em_perturbed_vs_expected": 0.0,
    "f1_perturbed_vs_original": 0.8,
    "em_perturbed_vs_original": 0.0,
    "f1_improvement": 0.1,
    
    // LLM Judge评估结果
    "llm_judge_scores": {
      "accuracy": 8.5,
      "conciseness": 7.0,
      "professionalism": 8.0,
      "overall_score": 7.8,
      "reasoning": "评估推理过程...",
      "raw_output": "原始输出..."
    },
    
    "expected_answer": "期望答案",
    "evaluation_timestamp": "2025-01-16 10:30:00"
  }
]
```

## 评估指标说明

### F1分数
- 基于词重叠的F1分数计算
- 使用jieba进行中文分词
- 范围: 0-1，越高越好

### 精确匹配(EM)
- 完全匹配检查
- 范围: 0或1，1表示完全匹配

### LLM Judge评分
- **准确性**: 答案的准确程度 (0-10)
- **简洁性**: 答案的简洁程度 (0-10)  
- **专业性**: 答案的专业程度 (0-10)
- **综合评分**: 加权平均分数 (0-10)

## 统计摘要

评估完成后会显示详细的统计摘要：

```
============================================================
📊 增强扰动评估摘要
============================================================
📈 评估样本总数: 100
📊 F1分数统计:
   原始答案 vs 期望答案: 平均 0.4500
   扰动答案 vs 期望答案: 平均 0.5200
   扰动答案 vs 原始答案: 平均 0.7800
📊 EM分数统计:
   原始答案 vs 期望答案: 平均 0.1000
   扰动答案 vs 期望答案: 平均 0.1200
   扰动答案 vs 原始答案: 平均 0.8500
🤖 LLM Judge评分: 平均 7.50
🔄 扰动器统计:
   term: 50个样本, F1改进: 0.0700, Judge评分: 7.80
   year: 50个样本, F1改进: 0.0500, Judge评分: 7.20
============================================================
```

## 依赖要求

- Python 3.8+
- jieba (中文分词)
- transformers (LLM Judge)
- torch (GPU支持)

## 注意事项

1. **LLM Judge初始化**: 如果GPU内存不足或模型不可用，可以跳过LLM Judge评估
2. **期望答案匹配**: 系统会根据问题内容匹配期望答案，确保问题格式一致
3. **中文处理**: 系统专门针对中文文本优化，使用jieba进行分词
4. **内存使用**: 处理大量数据时注意内存使用情况

## 故障排除

### 常见问题

1. **LLM Judge初始化失败**
   - 检查GPU内存是否充足
   - 确认模型文件是否存在
   - 可以跳过LLM Judge评估继续执行

2. **期望答案匹配失败**
   - 检查问题格式是否一致
   - 确认AlphaFin数据文件格式正确

3. **内存不足**
   - 分批处理数据
   - 减少并发评估数量

## 扩展功能

系统设计为可扩展的，可以轻松添加：

- 新的评估指标
- 不同的LLM Judge模型
- 自定义评估逻辑
- 可视化报告生成 