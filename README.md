# RAG系统检索评测文档

## 系统概述

这是一个基于RAG（Retrieval-Augmented Generation）的金融问答系统，支持多种检索模式和评测功能。

## 系统架构

### 核心组件

1. **编码器（Encoder）**
   - 中文：`models/alphafin_encoder_finetuned_1epoch`
   - 英文：`models/finetuned_tatqa_mixed_enhanced`
   - 设备：`cuda:0`
   - 批处理大小：64

2. **重排序器（Reranker）**
   - 模型：`Qwen/Qwen3-Reranker-0.6B`
   - 设备：`cuda:1`
   - 批处理大小：1（避免内存不足）
   - 量化：4bit量化以节省GPU内存

3. **生成器（Generator）**
   - 模型：`Qwen/Qwen2-0.5B-Instruct`
   - 设备：`cuda:1`
   - 量化：4bit量化

## 检索模式

### 1. Baseline模式
- **预过滤**：不使用
- **重排序器**：不使用
- **流程**：纯FAISS检索
- **适用场景**：基础检索评测

### 2. Prefilter模式
- **预过滤**：使用（自动启用股票代码和公司名称映射）
- **重排序器**：不使用
- **流程**：元数据过滤 + FAISS检索
- **适用场景**：需要精确匹配的检索

### 3. Reranker模式
- **预过滤**：使用（自动启用股票代码和公司名称映射）
- **重排序器**：使用
- **流程**：元数据过滤 + FAISS检索 + 重排序
- **适用场景**：高质量检索，需要重排序优化

### 4. Reranker_no_prefilter模式
- **预过滤**：不使用
- **重排序器**：使用
- **流程**：FAISS检索 + 重排序
- **适用场景**：全量检索 + 重排序优化

## 评测指标

### MRR（Mean Reciprocal Rank）
```python
# 计算逻辑
if found_rank is not None:
    mrr_total += 1.0 / found_rank
else:
    # 如果没有找到相关文档，MRR贡献为0
    pass

# 最终MRR
mrr = mrr_total / total_samples
```

### Hit@k
```python
# 计算逻辑
hit = False
for doc_id in retrieved_doc_ids[:top_k]:
    if doc_id in ground_truth_doc_ids:
        hit = True
        break
if hit:
    hitk_total += 1

# 最终Hit@k
hitk = hitk_total / total_samples
```

## 配置文件

### 主要配置参数

```python
# 编码器配置
encoder:
  chinese_model_path: "models/alphafin_encoder_finetuned_1epoch"
  english_model_path: "models/finetuned_tatqa_mixed_enhanced"
  device: "cuda:0"
  batch_size: 64

# 重排序器配置
reranker:
  model_name: "Qwen/Qwen3-Reranker-0.6B"
  device: "cuda:1"
  batch_size: 1
  use_quantization: True
  quantization_type: "4bit"

# 检索器配置
retriever:
  retrieval_top_k: 20  # FAISS检索的top-k
  rerank_top_k: 10     # 重排序后的top-k
  use_prefilter: True  # 是否使用预过滤
  batch_size: 64

# 生成器配置
generator:
  model_name: "Qwen/Qwen2-0.5B-Instruct"
  device: "cuda:1"
  use_quantization: True
  quantization_type: "4bit"
```

## 数据流程

### 中文查询流程
1. **关键词提取**：使用`extract_stock_info_with_mapping`提取公司名称和股票代码
2. **元数据过滤**：根据`use_prefilter`参数决定是否使用预过滤
3. **FAISS检索**：在过滤后的文档中进行检索
4. **重排序**：根据`reranker_checkbox`决定是否应用重排序器

### 英文查询流程
1. **FAISS检索**：直接进行FAISS检索
2. **重排序**：根据`reranker_checkbox`决定是否应用重排序器

### 文档映射
```python
# 使用doc_id进行正确的文档映射
doc_id_to_original_map = {}
for doc in retrieved_documents:
    doc_id = getattr(doc.metadata, 'doc_id', None)
    if doc_id is None:
        doc_id = hashlib.md5(doc.content.encode('utf-8')).hexdigest()[:16]
    doc_id_to_original_map[doc_id] = doc
```

## 错误处理

### 重排序器加载失败
```python
if self.reranker is None:
    print("⚠️ 重排序器加载失败，将禁用重排序功能")
    self.enable_reranker = False
```

### GPU内存不足
```python
# 检查GPU内存
if free_memory < 2 * 1024**3:  # 2GB
    print(f"- GPU {gpu_id} 内存不足，回退到CPU")
    device = "cpu"
```

### 异常处理
每个关键步骤都有完整的try-catch异常处理机制。

## 使用说明

### 运行评测
```bash
# 快速测试
python alphafin_data_process/run_retrieval_evaluation_background.py \
    --eval_data_path evaluate_mrr/alphafin_eval.jsonl \
    --output_dir results \
    --modes reranker reranker_no_prefilter \
    --max_samples 10

# 完整评测
python alphafin_data_process/run_retrieval_evaluation_background.py \
    --eval_data_path evaluate_mrr/alphafin_eval.jsonl \
    --output_dir results \
    --modes baseline prefilter reranker reranker_no_prefilter
```

### 后台运行
```bash
# 使用nohup
nohup python alphafin_data_process/run_retrieval_evaluation_background.py \
    --eval_data_path evaluate_mrr/alphafin_eval.jsonl \
    --output_dir results \
    --modes reranker reranker_no_prefilter > evaluation.log 2>&1 &

# 使用screen
screen -S evaluation
python alphafin_data_process/run_retrieval_evaluation_background.py \
    --eval_data_path evaluate_mrr/alphafin_eval.jsonl \
    --output_dir results \
    --modes reranker reranker_no_prefilter
# Ctrl+A+D 分离screen
```

## 性能优化

### 内存管理
- 批处理大小设为1避免内存不足
- 使用4bit量化节省GPU内存
- 定期清理GPU内存

### 设备分配
- 编码器：`cuda:0`
- 重排序器：`cuda:1`
- 生成器：`cuda:1`

### 缓存策略
- 使用现有embedding索引：`use_existing_embedding_index: True`
- 限制AlphaFin数据chunk数量：`max_alphafin_chunks: 1000000`

## 日志和监控

### 日志文件
- 位置：`results/evaluation_log_YYYYMMDD_HHMMSS.txt`
- 包含：检索过程、错误信息、性能指标

### 中间结果保存
- 自动保存中间结果防止数据丢失
- 进度信息：`results/progress_YYYYMMDD_HHMMSS.json`

## 常见问题

### 1. 重排序器加载失败
- 检查GPU内存是否充足
- 确认模型路径是否正确
- 查看错误日志

### 2. 评测进程停止
- 检查GPU内存使用情况
- 确认配置文件参数正确
- 查看进程状态

### 3. MRR计算结果异常
- 确认数据格式正确
- 检查doc_id映射
- 验证评测数据完整性

## 代码质量

### 架构设计
- 模块化设计，职责分离
- 配置驱动，易于调整
- 错误处理完善

### 性能优化
- 批处理优化
- 内存管理
- GPU资源合理分配

### 可维护性
- 代码注释详细
- 日志记录完整
- 配置参数集中管理

## 版本信息

- **Python版本**：3.8+
- **主要依赖**：torch, transformers, faiss-cpu, gradio
- **GPU要求**：CUDA 11.0+
- **内存要求**：16GB+ RAM

## 联系方式

如有问题，请查看日志文件或联系开发团队。