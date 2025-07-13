# RAG系统单数据集端到端测试

## 概述

本测试框架支持分别测试中文和英文数据集，每个数据集独立测试，不进行比较。支持数据采样和多种配置选项。**每处理10个数据就保存一次原始数据**，包括查询、上下文、答案、期望答案、EM和F1分数。

## 功能特性

### 🌍 数据集支持
- **中文数据集**: AlphaFin金融数据集
- **英文数据集**: TAT-QA数据集
- **自动语言检测**: 根据文件路径自动识别语言

### 📊 评估指标
- **F1-score**: 基于token级别的精确率和召回率
- **Exact Match**: 完全匹配率
- **处理时间**: 查询处理性能
- **成功率**: 系统稳定性指标

### 🔧 配置选项
- **重排序器**: 可启用/禁用重排序功能
- **股票预测**: 可启用/禁用股票预测模式
- **采样数量**: 支持数据采样以加快测试

### 💾 原始数据保存
- **每10个数据保存一次**: 自动分批保存原始数据
- **完整记录**: 包含查询、上下文、答案、期望答案、EM、F1等
- **JSON格式**: 便于后续分析和处理

## 原始数据格式

### 保存的原始数据包含以下字段：

```json
{
  "sample_id": 0,
  "query": "这个股票的下月最终收益结果是？",
  "context": "<div>检索到的上下文信息...</div>",
  "answer": "这个股票的下月最终收益结果是:'涨',上涨/下跌概率:极大",
  "expected_answer": "这个股票的下月最终收益结果是:'涨',上涨/下跌概率:极大",
  "em": 1.0,
  "f1": 1.0,
  "processing_time": 6.23,
  "success": true,
  "language": "chinese"
}
```

### 保存位置和文件结构：

```
raw_data_alphafin_eval_samples_updated/
├── batch_001.json  # 第1-10个数据
├── batch_002.json  # 第11-20个数据
└── batch_003.json  # 第21-25个数据（最后一批）

raw_data_tatqa_eval_balanced_100/
├── batch_001.json  # 第1-10个数据
├── batch_002.json  # 第11-20个数据
└── batch_003.json  # 第21-25个数据（最后一批）
```

## 使用方法

### 1. 命令行运行

#### 测试中文数据集
```bash
python test_rag_system_e2e_multilingual.py \
    --data_path data/alphafin/alphafin_eval_samples_updated.jsonl \
    --sample_size 20
```

#### 测试英文数据集
```bash
python test_rag_system_e2e_multilingual.py \
    --data_path evaluate_mrr/tatqa_eval_balanced_100.jsonl \
    --sample_size 20
```

#### 启用股票预测模式
```bash
python test_rag_system_e2e_multilingual.py \
    --data_path data/alphafin/alphafin_eval_samples_updated.jsonl \
    --enable_stock_prediction
```

#### 禁用重排序器
```bash
python test_rag_system_e2e_multilingual.py \
    --data_path data/alphafin/alphafin_eval_samples_updated.jsonl \
    --disable_reranker
```

### 2. Python脚本运行

#### 运行示例脚本
```bash
python test_single_dataset_example.py
```

#### 运行原始数据保存测试
```bash
python test_with_raw_data_saving.py
```

#### 在代码中使用
```python
from test_rag_system_e2e_multilingual import test_single_dataset

# 测试中文数据集
chinese_summary = test_single_dataset(
    data_path="data/alphafin/alphafin_eval_samples_updated.jsonl",
    sample_size=20,
    enable_reranker=True,
    enable_stock_prediction=False
)

# 测试英文数据集
english_summary = test_single_dataset(
    data_path="evaluate_mrr/tatqa_eval_balanced_100.jsonl",
    sample_size=20,
    enable_reranker=True,
    enable_stock_prediction=False
)
```

## 命令行参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--data_path` | str | 是 | 测试数据文件路径 |
| `--sample_size` | int | 否 | 采样数量 (默认使用全部数据) |
| `--disable_reranker` | flag | 否 | 禁用重排序器 |
| `--enable_stock_prediction` | flag | 否 | 启用股票预测模式 |

## 测试结果

### 控制台输出示例

```
================================================================================
🎯 数据集测试结果: data/alphafin/alphafin_eval_samples_updated.jsonl
================================================================================
📊 测试指标:
   数据路径: data/alphafin/alphafin_eval_samples_updated.jsonl
   语言: chinese
   总样本数: 20
   成功样本数: 19
   成功率: 95.00%
   平均F1-score: 0.8234
   平均Exact Match: 0.7123
   平均处理时间: 6.23秒
   总处理时间: 124.56秒
   重排序器: 启用
   股票预测: 禁用
================================================================================
```

### 原始数据保存日志

```
📁 保存原始数据批次 1 到: raw_data_alphafin_eval_samples_updated/batch_001.json
📁 保存原始数据批次 2 到: raw_data_alphafin_eval_samples_updated/batch_002.json
```

### 返回结果格式

```python
{
    "data_path": "data/alphafin/alphafin_eval_samples_updated.jsonl",
    "language": "chinese",
    "total_samples": 20,
    "successful_samples": 19,
    "success_rate": 0.95,
    "average_f1_score": 0.8234,
    "average_exact_match": 0.7123,
    "average_processing_time": 6.23,
    "total_processing_time": 124.56,
    "enable_reranker": True,
    "enable_stock_prediction": False,
    "detailed_results": [...]
}
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

### 3. 性能优化测试
- 对比不同配置的性能差异
- 分析处理时间的变化
- 评估系统资源使用

### 4. 原始数据分析
- 分析查询和答案的对应关系
- 评估上下文信息的质量
- 研究EM和F1分数的分布

## 性能优化

### 1. 数据采样
```bash
# 使用采样加快测试
python test_rag_system_e2e_multilingual.py --data_path data/alphafin/alphafin_eval_samples_updated.jsonl --sample_size 50
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
   ❌ 数据文件不存在: data/alphafin/alphafin_eval_samples_updated.jsonl
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

4. **原始数据保存失败**
   ```
   PermissionError: [错误信息]
   ```
   解决：检查目录权限，确保有写入权限

### 调试模式

启用详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

查看详细处理过程：
```bash
python test_rag_system_e2e_multilingual.py --data_path data/alphafin/alphafin_eval_samples_updated.jsonl --sample_size 5
```

## 原始数据分析

### 分析保存的原始数据

```python
import json
from pathlib import Path

# 读取原始数据
with open("raw_data_alphafin_eval_samples_updated/batch_001.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 分析数据
for record in raw_data:
    print(f"查询: {record['query']}")
    print(f"答案: {record['answer']}")
    print(f"期望: {record['expected_answer']}")
    print(f"F1: {record['f1']:.4f}, EM: {record['em']:.4f}")
    print("-" * 50)
```

---

**注意**: 本测试框架需要完整的RAG系统环境支持，请确保所有依赖组件已正确安装和配置。 