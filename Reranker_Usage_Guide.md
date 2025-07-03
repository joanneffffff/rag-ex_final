# Reranker使用指南 - 中英文数据集兼容性

## 问题背景

RAG系统中，中文数据集（AlphaFin）和英文数据集（TatQA）使用Reranker的方式不同：

- **中文数据集**: 使用 `rerank_with_metadata()` 方法，需要元数据
- **英文数据集**: 使用 `rerank()` 方法，只需要纯文本

## 解决方案

### 1. 中文数据集（AlphaFin）- 已修复

**调用方式**:
```python
# 准备带元数据的文档
docs_for_rerank = []
for doc in retrieved_docs:
    docs_for_rerank.append({
        'content': doc.content,
        'metadata': doc.metadata.__dict__
    })

# 执行重排序
reranked_results = reranker.rerank_with_metadata(
    query=query_text,
    documents_with_metadata=docs_for_rerank,
    batch_size=2
)

# 提取重排序后的文档
final_retrieved_docs_for_ranking = []
for result in reranked_results[:top_k_rerank]:
    doc = DocumentWithMetadata(
        content=result['content'],
        metadata=DocumentMetadata(**result['metadata'])
    )
    final_retrieved_docs_for_ranking.append(doc)
```

**修复内容**: 已修复 `rerank_with_metadata()` 方法中的元数据映射bug

### 2. 英文数据集（TatQA）- 已修复

**调用方式**:
```python
# 准备纯文本文档
docs_content_for_rerank = [doc.content for doc in initial_retrieved_docs]

# 执行重排序
reranked_items = reranker.rerank(
    query=query, 
    documents=docs_content_for_rerank, 
    batch_size=4
)

# 根据重排序结果重建文档列表
content_to_original_doc_map = {doc.content: doc for doc in initial_retrieved_docs}
temp_reranked_docs = []
for doc_text, score in reranked_items[:20]:  # 取前20个结果
    original_doc = content_to_original_doc_map.get(doc_text)
    if original_doc:
        temp_reranked_docs.append(original_doc)

final_retrieved_docs_for_ranking = temp_reranked_docs
```

**修复内容**: 已修复 `eval_retrieval_mrr.py` 中的调用格式错误

## 关键修复点

### 1. 中文数据集修复
- **问题**: `rerank_with_metadata()` 方法中元数据映射错误
- **修复**: 使用文档内容作为key来正确映射元数据
- **影响**: 确保重排序结果与元数据正确对应

### 2. 英文数据集修复
- **问题**: 代码期望 `{text: ..., score: ...}` 格式，但实际返回 `List[Tuple[str, float]]`
- **修复**: 正确解包元组格式的返回值
- **影响**: 确保重排序结果能正确应用到最终排名

## 验证方法

### 1. 测试中文数据集
```bash
# 运行中文数据集评估
python encoder_finetune_evaluate/evaluate_encoder_reranker_mrr_rag_system_multi_gpu_fixed.py \
    --encoder_model_name models/finetuned_alphafin_zh \
    --reranker_model_name Qwen/Qwen3-Reranker-0.6B \
    --eval_jsonl evaluate_mrr/alphafin_eval.jsonl \
    --corpus_jsonl evaluate_mrr/alphafin_knowledge_base.jsonl \
    --max_eval_samples 100
```

### 2. 测试英文数据集
```bash
# 运行英文数据集评估
python encoder_finetune_evaluate/evaluate_encoder_reranker_mrr_rag_system_multi_gpu_fixed.py \
    --encoder_model_name models/finetuned_finbert_tatqa \
    --reranker_model_name Qwen/Qwen3-Reranker-0.6B \
    --eval_jsonl evaluate_mrr/tatqa_eval_enhanced.jsonl \
    --corpus_jsonl evaluate_mrr/tatqa_knowledge_base.jsonl \
    --max_eval_samples 100
```

## 预期结果

修复后，您应该看到：

1. **英文数据集**: MRR从0.3提升到0.4-0.5
2. **中文数据集**: MRR从0.3提升到0.4-0.6
3. **总体提升**: 所有Hit@K指标都有显著改善

## 注意事项

1. **保持RAG系统代码不变**: 所有修复都在现有代码基础上进行，没有改变核心功能
2. **向后兼容**: 修复后的代码完全兼容现有的调用方式
3. **性能提升**: 修复后Reranker应该能正常发挥作用，带来明显的性能提升

## 结论

**不应该移除Reranker**，因为问题在于实现bug，而不是Reranker本身无效。修复后，Reranker应该能带来显著的性能提升，特别是在英文数据集上。 