# RAG系统多语言端到端测试

## 概述

本测试框架提供了完整的RAG系统多语言端到端测试功能，支持中文和英文数据集的评估。测试框架模拟真实用户与RAG系统的完整交互流程，评估系统的整体性能。

## 功能特性

### 🌍 多语言支持
- **中文数据集**: 支持AlphaFin金融数据集评估
- **英文数据集**: 支持TAT-QA数据集评估
- **自动语言检测**: 根据数据文件路径自动检测语言

### 📊 评估指标
- **F1-score**: 基于token级别的精确率和召回率
- **Exact Match**: 完全匹配率
- **处理时间**: 查询处理性能
- **成功率**: 系统稳定性指标

### 🔧 配置选项
- **重排序器**: 可启用/禁用重排序功能
- **股票预测**: 可启用/禁用股票预测模式
- **采样数量**: 支持数据采样以加快测试

## 数据集格式

### 中文数据集 (AlphaFin)
```json
{
  "query": "这个股票的下月最终收益结果是？",
  "answer": "这个股票的下月最终收益结果是:'涨',上涨/下跌概率:极大"
}
```

### 英文数据集 (TAT-QA)
```json
{
  "question": "What is the revenue in 2019?",
  "answer": "The revenue in 2019 is $1,234,567"
}
```

## 使用方法

### 1. 命令行运行

#### 基础多语言测试
```bash
python test_rag_system_e2e_multilingual.py \
    --chinese_data_path data/alphafin/alphafin_eval_samples_updated.jsonl \
    --english_data_path evaluate_mrr/tatqa_eval_balanced_100.jsonl \
    --output_dir e2e_test_results \
    --sample_size 50
```

#### 启用股票预测模式
```bash
python test_rag_system_e2e_multilingual.py \
    --chinese_data_path data/alphafin/alphafin_eval_samples_updated.jsonl \
    --english_data_path evaluate_mrr/tatqa_eval_balanced_100.jsonl \
    --enable_stock_prediction
```

#### 禁用重排序器
```bash
python test_rag_system_e2e_multilingual.py \
    --chinese_data_path data/alphafin/alphafin_eval_samples_updated.jsonl \
    --english_data_path evaluate_mrr/tatqa_eval_balanced_100.jsonl \
    --disable_reranker
```

### 2. Python脚本运行

#### 运行示例脚本
```bash
python run_multilingual_e2e_test_example.py
```

#### 在代码中使用
```python
from test_rag_system_e2e_multilingual import run_multilingual_e2e_test

# 运行多语言端到端测试
combined_summary = run_multilingual_e2e_test(
    chinese_data_path="data/alphafin/alphafin_eval_samples_updated.jsonl",
    english_data_path="evaluate_mrr/tatqa_eval_balanced_100.jsonl",
    output_dir="e2e_test_results",
    sample_size=100,
    enable_reranker=True,
    enable_stock_prediction=False
)

# 查看结果
print(f"加权平均F1-score: {combined_summary['weighted_f1_score']:.4f}")
print(f"加权平均Exact Match: {combined_summary['weighted_exact_match']:.4f}")
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--chinese_data_path` | str | `data/alphafin/alphafin_eval_samples_updated.jsonl` | 中文测试数据文件路径 |
| `--english_data_path` | str | `evaluate_mrr/tatqa_eval_balanced_100.jsonl` | 英文测试数据文件路径 |
| `--output_dir` | str | `e2e_test_results` | 结果输出目录 |
| `--sample_size` | int | `None` | 采样数量 (默认使用全部数据) |
| `--disable_reranker` | flag | `False` | 禁用重排序器 |
| `--enable_stock_prediction` | flag | `False` | 启用股票预测模式 |

## 测试结果

### 输出文件

1. **chinese_results.json**: 中文测试详细结果
2. **english_results.json**: 英文测试详细结果
3. **combined_results.json**: 综合测试结果

### 控制台输出示例

```
================================================================================
🎯 多语言端到端测试最终结果
================================================================================
📊 总体指标:
   总样本数: 200
   成功样本数: 195
   整体成功率: 97.50%
   加权平均F1-score: 0.8234
   加权平均Exact Match: 0.7123
   总处理时间: 1234.56秒

🌍 分语言指标:
   chinese:
     样本数: 100
     成功率: 98.00%
     平均F1-score: 0.8567
     平均Exact Match: 0.7432
     平均处理时间: 6.23秒
   english:
     样本数: 100
     成功率: 97.00%
     平均F1-score: 0.7901
     平均Exact Match: 0.6814
     平均处理时间: 6.11秒
================================================================================
```

## 测试场景

### 1. 基础功能测试
- 验证RAG系统的基本问答功能
- 评估检索和生成的准确性
- 测试系统稳定性

### 2. 股票预测模式测试
- 验证股票预测功能的准确性
- 测试专业指令的转换效果
- 评估预测格式的规范性

### 3. 多语言性能评估
- 对比中文和英文处理的性能差异
- 分析不同语言的处理策略
- 评估多语言支持的效果

## 性能优化

### 1. 数据采样
```bash
# 使用采样加快测试
python test_rag_system_e2e_multilingual.py --sample_size 50
```

### 2. 使用现有索引
```python
# 在适配器中启用现有索引
self.rag_ui = OptimizedRagUI(
    enable_reranker=self.enable_reranker,
    use_existing_embedding_index=True  # 加快初始化
)
```

## 故障排除

### 常见问题

1. **数据文件不存在**
   ```
   ❌ 中文数据文件不存在: data/alphafin/alphafin_eval_samples_updated.jsonl
   ```
   解决：检查文件路径是否正确，确保数据文件存在

2. **RAG系统初始化失败**
   ```
   ❌ 多语言RAG系统初始化失败: [错误信息]
   ```
   解决：检查RAG系统配置，确保所有依赖组件正常

3. **内存不足**
   ```
   MemoryError: [错误信息]
   ```
   解决：减少sample_size，或增加系统内存

### 调试模式

启用详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

查看详细处理过程：
```bash
python test_rag_system_e2e_multilingual.py --sample_size 5
```

---

**注意**: 本测试框架需要完整的RAG系统环境支持，请确保所有依赖组件已正确安装和配置。 