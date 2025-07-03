# TatQA数据集MRR评估指南

## 数据集差异说明

### 训练数据集对比

**tatqa_train_qc.jsonl** (原始版本):
```json
{
  "query": "How was internally developed software capitalised?",
  "context": "Internally developed software is capitalised at cost less accumulated amortisation...",
  "answer": "at cost less accumulated amortisation."
}
```

**tatqa_train_qc_enhanced.jsonl** (增强版本):
```json
{
  "query": "How was internally developed software capitalised?",
  "context": "Internally developed software is capitalised at cost less accumulated amortisation...",
  "answer": "at cost less accumulated amortisation.",
  "doc_id": "chunk_1",
  "relevant_doc_ids": ["ddf26912-5783-4b3b-b351-87e91b4a5f5b"]
}
```

**主要差异**:
- `enhanced`版本增加了`doc_id`和`relevant_doc_ids`字段
- 这些字段用于更精确的文档匹配和评估

### 评估数据集对比

**tatqa_eval_enhanced.jsonl** (推荐使用):
```json
{
  "query": "What method did the company use when Topic 606 in fiscal 2019 was adopted?",
  "context": "We utilized a comprehensive approach to evaluate...",
  "answer": "the modified retrospective method",
  "doc_id": "chunk_1",
  "relevant_doc_ids": ["4202457313786d975b89fabc695c3efb"]
}
```

**tatqa_eval_upgraded.jsonl**:
```json
{
  "query": "How was internally developed software capitalised?",
  "context": "Internally developed software is capitalised...",
  "answer": "at cost less accumulated amortisation.",
  "relevant_doc_ids": ["doc_0_para_6"]
}
```

**推荐使用 `tatqa_eval_enhanced.jsonl`**，因为：
1. 与训练数据格式一致
2. 包含完整的`doc_id`和`relevant_doc_ids`信息
3. 支持更精确的匹配策略

## GPU版本MRR评估

### 新功能特性

1. **GPU加速**: 自动检测并使用GPU，大幅提升评估速度
2. **仅英文数据集**: 专注于TatQA英文数据集评估
3. **内存优化**: 分批处理，适应GPU内存限制
4. **灵活参数**: 支持多种配置选项

### 使用方法

#### 1. 快速开始
```bash
# 完整评估
python run_tatqa_mrr_gpu.py

# 快速测试（只评估前100个样本）
python run_tatqa_mrr_gpu.py --quick

# CPU模式（用于调试）
python run_tatqa_mrr_gpu.py --cpu
```

#### 2. 直接调用评估脚本
```bash
python encoder_finetune_evaluate/evaluate_encoder_reranker_mrr_gpu.py \
    --encoder_model_name models/finetuned_finbert_tatqa \
    --reranker_model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --eval_jsonl evaluate_mrr/tatqa_eval_enhanced.jsonl \
    --base_raw_data_path data/tatqa_dataset_raw/ \
    --top_k_retrieval 100 \
    --top_k_rerank 10 \
    --batch_size 32
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--encoder_model_name` | `models/finetuned_finbert_tatqa` | Encoder模型路径 |
| `--reranker_model_name` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker模型 |
| `--eval_jsonl` | `evaluate_mrr/tatqa_eval_enhanced.jsonl` | 评估数据文件 |
| `--base_raw_data_path` | `data/tatqa_dataset_raw/` | TatQA原始数据路径 |
| `--top_k_retrieval` | `100` | Encoder检索的top-k数量 |
| `--top_k_rerank` | `10` | Reranker重排序后的top-k数量 |
| `--batch_size` | `32` | 批处理大小 |
| `--force_cpu` | `False` | 强制使用CPU |
| `--max_eval_samples` | `None` | 最大评估样本数 |

### 评估指标

脚本会输出以下指标：
- **MRR (Mean Reciprocal Rank)**: 平均倒数排名
- **Hit@1**: 正确答案排在第1位的比例
- **Hit@3**: 正确答案排在前3位的比例
- **Hit@5**: 正确答案排在前5位的比例
- **Hit@10**: 正确答案排在前10位的比例

### 系统要求

- Python 3.8+
- PyTorch with CUDA support
- sentence-transformers
- 至少8GB GPU内存（推荐16GB+）

### 性能优化建议

1. **GPU内存不足时**:
   - 减小`--batch_size`（如16或8）
   - 减小`--top_k_retrieval`（如50）
   - 使用`--max_eval_samples`限制样本数

2. **快速测试**:
   - 使用`--quick`参数
   - 设置`--max_eval_samples 100`

3. **调试模式**:
   - 使用`--force_cpu`参数
   - 设置较小的参数值

## 文件结构

```
├── encoder_finetune_evaluate/
│   ├── evaluate_encoder_reranker_mrr_gpu.py  # GPU版本评估脚本
│   └── evaluate_encoder_reranker_mrr.py      # 原始CPU版本
├── evaluate_mrr/
│   ├── tatqa_eval_enhanced.jsonl             # 推荐评估数据
│   ├── tatqa_eval_upgraded.jsonl             # 升级版评估数据
│   ├── tatqa_train_qc_enhanced.jsonl         # 增强版训练数据
│   └── tatqa_train_qc.jsonl                  # 原始训练数据
├── run_tatqa_mrr_gpu.py                      # 运行脚本
└── TATQA_MRR_EVALUATION_README.md            # 本说明文件
```

## 注意事项

1. **数据格式**: 确保使用`tatqa_eval_enhanced.jsonl`作为评估数据
2. **模型路径**: 确保Encoder模型路径正确
3. **GPU内存**: 监控GPU内存使用情况，必要时调整参数
4. **原始数据**: 确保TatQA原始数据文件存在于指定路径

## 故障排除

### 常见问题

1. **GPU内存不足**:
   ```
   RuntimeError: CUDA out of memory
   ```
   解决：减小batch_size或top_k_retrieval参数

2. **模型加载失败**:
   ```
   OSError: Can't load tokenizer
   ```
   解决：检查模型路径是否正确

3. **数据文件不存在**:
   ```
   FileNotFoundError: [Errno 2] No such file or directory
   ```
   解决：检查数据文件路径

### 调试技巧

1. 使用`--force_cpu`参数进行调试
2. 使用`--max_eval_samples 10`进行小规模测试
3. 检查GPU内存使用：`nvidia-smi`
4. 查看详细错误信息：添加`--verbose`参数 