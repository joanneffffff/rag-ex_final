# RAG系统端到端测试

## 概述

这个端到端测试脚本模拟真实用户与RAG系统的完整交互流程，评估整个系统的性能。测试包括检索、生成和评估三个主要环节。

## 核心组件

### 1. RagSystemAdapter
RAG系统适配器，提供统一的接口来测试整个RAG系统：

```python
class RagSystemAdapter:
    def __init__(self, enable_reranker=True, enable_stock_prediction=False):
        # 初始化适配器
        
    def initialize(self):
        # 初始化RAG系统
        
    def process_query(self, query: str) -> Dict[str, Any]:
        # 处理用户查询，返回完整响应
```

### 2. 端到端测试流程

1. **输入用户问题**：从评估数据集中获取用户问题
2. **RAG系统处理**：
   - 检索模块：根据问题从知识库中检索相关上下文
   - 生成模块：将检索到的上下文和问题输入给Fin-R1或Qwen3-8B模型
3. **获取系统答案**：RAG系统返回最终生成的答案
4. **与标准答案比较**：计算F1-score、Exact Match等指标
5. **记录性能指标**：记录处理时间、Token数量等

## 使用方法

### 1. 命令行运行

```bash
# 基础测试
python test_rag_system_e2e.py --data_path data/alphafin/alphafin_eval_samples_updated.jsonl --sample_size 10

# 启用股票预测模式
python test_rag_system_e2e.py --data_path data/alphafin/alphafin_eval_samples_updated.jsonl --enable_stock_prediction

# 禁用重排序器
python test_rag_system_e2e.py --data_path data/alphafin/alphafin_eval_samples_updated.jsonl --disable_reranker

# 完整测试
python test_rag_system_e2e.py --data_path data/alphafin/alphafin_eval_samples_updated.jsonl --output_path results.json --sample_size 50
```

### 2. 使用示例脚本

```bash
# 运行示例测试
python run_e2e_test_example.py
```

### 3. 编程方式使用

```python
from test_rag_system_e2e import run_e2e_test

# 运行端到端测试
test_summary = run_e2e_test(
    data_path="data/alphafin/alphafin_eval_samples_updated.jsonl",
    output_path="results.json",
    sample_size=20,
    enable_reranker=True,
    enable_stock_prediction=False
)

# 查看结果
print(f"平均F1-score: {test_summary['overall_metrics']['average_f1_score']:.4f}")
print(f"平均Exact Match: {test_summary['overall_metrics']['average_exact_match']:.4f}")
print(f"成功率: {test_summary['overall_metrics']['success_rate']:.2%}")
```

## 测试配置

### 命令行参数

- `--data_path`: 测试数据文件路径（必需）
- `--output_path`: 结果输出文件路径（默认：e2e_test_results.json）
- `--sample_size`: 采样数量（默认：使用全部数据）
- `--disable_reranker`: 禁用重排序器
- `--enable_stock_prediction`: 启用股票预测模式

### 数据格式

测试数据支持JSON和JSONL格式，每个样本应包含：

```json
{
    "query": "用户问题",
    "answer": "标准答案",
    "question": "用户问题（备用字段）",
    "expected_answer": "标准答案（备用字段）"
}
```

## 评估指标

### 1. F1-score
使用jieba分词计算中文F1-score：

```python
def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = set(get_tokens_chinese(normalize_answer_chinese(prediction)))
    gt_tokens = set(get_tokens_chinese(normalize_answer_chinese(ground_truth)))
    # 计算F1-score
```

### 2. Exact Match
计算完全匹配率：

```python
def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    pred_normalized = normalize_answer_chinese(prediction)
    gt_normalized = normalize_answer_chinese(ground_truth)
    return 1.0 if pred_normalized == gt_normalized else 0.0
```

### 3. 性能指标
- 处理时间
- 成功率
- Token数量（需要从RAG系统日志中解析）

## 输出结果

### 1. 控制台输出
```
🚀 开始端到端测试
📂 加载测试数据集: data/alphafin/alphafin_eval_samples_updated.jsonl
✅ 加载完成，共 100 个测试样本
🔄 正在初始化RAG系统...
✅ RAG系统初始化完成
🔄 开始处理测试样本...
处理测试样本: 100%|██████████| 100/100 [02:30<00:00,  1.50s/it]
📊 进度: 10/100, 平均F1: 0.8234, 平均EM: 0.4567, 平均时间: 1.45s
...
🎉 端到端测试完成！
📊 测试摘要:
   总样本数: 100
   成功样本数: 98
   成功率: 98.00%
   平均F1-score: 0.8234
   平均Exact Match: 0.4567
   平均处理时间: 1.45秒
   总处理时间: 145.23秒
```

### 2. JSON结果文件
```json
{
  "test_config": {
    "data_path": "data/alphafin/alphafin_eval_samples_updated.jsonl",
    "sample_size": 100,
    "enable_reranker": true,
    "enable_stock_prediction": false
  },
  "overall_metrics": {
    "total_samples": 100,
    "successful_samples": 98,
    "success_rate": 0.98,
    "average_f1_score": 0.8234,
    "average_exact_match": 0.4567,
    "average_processing_time": 1.45,
    "total_processing_time": 145.23
  },
  "detailed_results": [
    {
      "sample_id": 0,
      "query": "用户问题",
      "ground_truth": "标准答案",
      "predicted_answer": "系统生成的答案",
      "f1_score": 0.8234,
      "exact_match": 0.0,
      "processing_time": 1.45,
      "success": true,
      "performance_metrics": {...}
    }
  ]
}
```

## 测试场景

### 1. 基础功能测试
- 验证RAG系统的基本检索和生成功能
- 评估整体性能和准确性

### 2. 股票预测测试
- 测试股票预测模式的特殊功能
- 验证格式化输出是否正确

### 3. 重排序器对比测试
- 比较启用/禁用重排序器的效果
- 评估重排序器对性能的影响

### 4. 性能基准测试
- 测试不同配置下的性能表现
- 建立性能基准和优化目标

## 注意事项

1. **数据文件**：确保测试数据文件存在且格式正确
2. **系统资源**：端到端测试需要较多计算资源，建议使用GPU
3. **内存管理**：大量测试样本可能导致内存不足，建议分批测试
4. **日志记录**：测试过程会生成详细日志，便于调试和分析

## 故障排除

### 常见问题

1. **数据文件不存在**
   ```
   ❌ 数据文件不存在: data/alphafin/alphafin_eval_samples_updated.jsonl
   ```
   解决：检查文件路径是否正确

2. **RAG系统初始化失败**
   ```
   ❌ RAG系统初始化失败: ...
   ```
   解决：检查模型文件、配置文件等依赖

3. **内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决：减少sample_size或使用CPU模式

4. **性能指标解析失败**
   ```
   Warning: 无法解析性能指标
   ```
   解决：检查RAG系统日志格式

### 调试建议

1. 使用小样本测试：`--sample_size 5`
2. 查看详细日志：检查`e2e_test.log`文件
3. 分步测试：先测试单个查询，再批量测试
4. 资源监控：监控GPU内存和CPU使用率

## 扩展功能

### 1. 自定义评估指标
可以添加更多评估指标，如BLEU、ROUGE等：

```python
def calculate_bleu_score(prediction: str, ground_truth: str) -> float:
    # 实现BLEU评分
    pass
```

### 2. 多模型对比
可以扩展支持多个生成模型的对比测试：

```python
def run_multi_model_comparison():
    models = ["Fin-R1", "Qwen3-8B", "Qwen2-7B"]
    results = {}
    for model in models:
        results[model] = run_e2e_test(model=model)
    return results
```

### 3. 实时性能监控
可以添加实时性能监控功能：

```python
def monitor_performance():
    # 实时监控GPU使用率、内存使用等
    pass
```

## 总结

这个端到端测试框架提供了：

1. **完整的测试流程**：从数据加载到结果评估
2. **灵活的配置选项**：支持多种测试场景
3. **详细的性能指标**：F1-score、Exact Match、处理时间等
4. **易于使用**：命令行和编程接口
5. **可扩展性**：支持自定义评估指标和测试场景

通过这个框架，您可以全面评估RAG系统的性能，为系统优化提供数据支持。 