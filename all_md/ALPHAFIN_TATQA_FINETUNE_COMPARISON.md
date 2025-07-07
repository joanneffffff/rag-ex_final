# AlphaFin vs TAT-QA 编码器微调对比

## 📊 核心差异总结

| 方面 | AlphaFin (中文) | TAT-QA (英文) |
|------|----------------|---------------|
| **语言** | 中文 | 英文 |
| **基础模型** | `Langboat/mengzi-bert-base-fin` | `ProsusAI/finbert` |
| **微调脚本** | `finetune_encoder.py` | `finetune_encoder.py` |
| **数据处理** | 使用generated_question和summary | 直接使用文本 |
| **批次大小** | 16 | 32 |
| **训练轮数** | 3 | 2 |

## 🔧 技术实现差异

### 1. 数据处理逻辑

#### AlphaFin (中文)
```python
# 直接使用generated_question和summary
query_text = data.get('generated_question', '').strip()
doc_text = data.get('summary', '').strip()
```

#### TAT-QA (英文)
```python
# 直接使用文本，无需特殊处理
query_text = data.get('query', '').strip()
doc_text = data.get('context', '').strip()
```

### 2. 模型选择原因

#### AlphaFin: `Langboat/mengzi-bert-base-fin`
- **专门针对中文金融领域**
- **预训练数据包含财务报告、新闻等**
- **支持中文金融术语理解**

#### TAT-QA: `ProsusAI/finbert`
- **专门针对英文金融领域**
- **在金融文本上预训练**
- **适合处理英文财务数据**

### 3. 训练参数差异

| 参数 | AlphaFin | TAT-QA | 原因 |
|------|----------|--------|------|
| **批次大小** | 16 | 32 | 中文模型更大，需要更小批次 |
| **训练轮数** | 3 | 2 | 中文数据更复杂，需要更多轮数 |
| **学习率** | 2e-5 | 2e-5 | 相同，都是标准学习率 |

## 🚀 使用方式

### 通用脚本使用
```bash
# AlphaFin微调
./finetune_encoder_universal.sh alphafin

# TAT-QA微调
./finetune_encoder_universal.sh tatqa

# 快速测试模式
./finetune_encoder_universal.sh alphafin quick
./finetune_encoder_universal.sh tatqa quick
```

### 单独脚本使用
```bash
# AlphaFin专用
./finetune_alphafin_encoder.sh

# TAT-QA使用原有脚本
python encoder_finetune_evaluate/finetune_encoder.py \
    --model_name ProsusAI/finbert \
    --train_jsonl evaluate_mrr/tatqa_train_qc.jsonl \
    --output_dir models/finetuned_tatqa_encoder \
    --batch_size 32 \
    --epochs 2
```

## 📁 数据文件结构

### AlphaFin数据
```
evaluate_mrr/
├── alphafin_train_qc.jsonl           # 训练数据 (generated_question + summary)
└── alphafin_eval.jsonl               # 评估数据
```

### TAT-QA数据
```
evaluate_mrr/
├── tatqa_train_qc.jsonl              # 训练数据
├── tatqa_eval.jsonl                  # 评估数据
└── tatqa_knowledge_base.jsonl        # 原始数据
```

## 🎯 评估方式

### AlphaFin评估
```bash
# 检索评估
python alphafin_data_process/run_retrieval_evaluation_background.py \
    --eval_data_path data/alphafin/eval_data_100_from_corpus.jsonl \
    --modes baseline prefilter reranker \
    --encoder_model_path ./models/finetuned_alphafin_encoder

# 集成评估
python encoder_finetune_evaluate/evaluate_chinese_encoder_reranker_mrr.py \
    --encoder_model_name ./models/finetuned_alphafin_encoder
```

### TAT-QA评估
```bash
# 编码器评估
python encoder_finetune_evaluate/run_encoder_eval.py \
    --model_name ./models/finetuned_tatqa_encoder \
    --eval_jsonl evaluate_mrr/tatqa_eval.jsonl

# TAT-QA专用评估
python alphafin_data_process/run_tatqa_retrieval_evaluation.py \
    --mode reranker \
    --encoder_model_path ./models/finetuned_tatqa_encoder
```

## 🔄 核心逻辑相同点

1. **损失函数**: 都使用 `MultipleNegativesRankingLoss`
2. **评估指标**: 都使用 MRR (Mean Reciprocal Rank)
3. **训练策略**: 都是对比学习，query-document配对
4. **评估方式**: 都使用 `InformationRetrievalEvaluator`
5. **优化器**: 都使用 AdamW 优化器

## 💡 关键洞察

### 为什么逻辑基本相同？
1. **任务相同**: 都是信息检索任务
2. **目标相同**: 都是学习query和document的语义相似性
3. **方法相同**: 都使用对比学习方法
4. **评估相同**: 都使用检索评估指标

### 主要差异来源？
1. **语言差异**: 中文vs英文，需要不同的预训练模型
2. **数据格式**: AlphaFin使用generated_question+summary，TAT-QA直接使用文本
3. **领域特点**: 中文财务数据更复杂，需要更多训练轮数

## ✅ 结论

**是的，AlphaFin和TAT-QA的微调逻辑基本相同，主要区别在于：**

1. **模型选择**: 根据语言选择合适的基础模型
2. **数据路径**: 指向不同的数据集文件
3. **数据处理**: AlphaFin使用generated_question+summary，TAT-QA直接使用文本
4. **训练参数**: 根据数据特点调整批次大小和轮数

**核心的对比学习框架、损失函数、评估方式都是完全相同的！** 