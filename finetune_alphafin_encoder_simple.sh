#!/bin/bash

# AlphaFin中文编码器微调脚本 (简化版)
# 使用generated_question和summary，无需复杂chunking

echo "=========================================="
echo "AlphaFin中文编码器微调脚本 (简化版)"
echo "使用: generated_question + summary"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# 数据路径配置
TRAIN_DATA="evaluate_mrr/alphafin_train_qc.jsonl"
EVAL_DATA="evaluate_mrr/alphafin_eval.jsonl"

# 模型配置
BASE_MODEL="Langboat/mengzi-bert-base-fin"
OUTPUT_MODEL_PATH="./models/finetuned_alphafin_encoder_summary"

# 训练参数
BATCH_SIZE=16
EPOCHS=3
LEARNING_RATE=2e-5
MAX_SEQ_LENGTH=512

# 数据限制（用于快速测试）
LIMIT_TRAIN=0  # 0表示使用全部数据
LIMIT_EVAL=100  # 限制评估数据量以加快速度

echo "配置信息："
echo "  训练数据: $TRAIN_DATA"
echo "  评估数据: $EVAL_DATA"
echo "  基础模型: $BASE_MODEL"
echo "  输出路径: $OUTPUT_MODEL_PATH"
echo "  批次大小: $BATCH_SIZE"
echo "  训练轮数: $EPOCHS"
echo "  学习率: $LEARNING_RATE"
echo "  最大序列长度: $MAX_SEQ_LENGTH"
echo "  训练数据限制: $LIMIT_TRAIN"
echo "  评估数据限制: $LIMIT_EVAL"
echo ""

# 检查数据文件是否存在
echo "检查数据文件..."
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ 训练数据文件不存在: $TRAIN_DATA"
    exit 1
fi

if [ ! -f "$EVAL_DATA" ]; then
    echo "❌ 评估数据文件不存在: $EVAL_DATA"
    exit 1
fi

echo "✅ 所有数据文件检查通过"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_MODEL_PATH"

# 开始微调
echo "开始AlphaFin编码器微调 (简化版)..."
echo "时间: $(date)"
echo ""

# 使用简化的微调脚本，直接使用generated_question和summary
python encoder_finetune_evaluate/finetune_encoder.py \
    --model_name "$BASE_MODEL" \
    --train_jsonl "$TRAIN_DATA" \
    --eval_jsonl "$EVAL_DATA" \
    --output_dir "$OUTPUT_MODEL_PATH" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --max_samples $LIMIT_TRAIN

# 检查微调是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ AlphaFin编码器微调完成！"
    echo "模型保存路径: $OUTPUT_MODEL_PATH"
    echo "完成时间: $(date)"
    echo ""
    echo "下一步："
    echo "1. 使用微调后的模型进行检索评估："
    echo "   python alphafin_data_process/run_retrieval_evaluation_background.py \\"
    echo "       --eval_data_path data/alphafin/eval_data_100_from_corpus.jsonl \\"
    echo "       --output_dir alphafin_data_process/evaluation_results \\"
    echo "       --modes baseline prefilter reranker \\"
    echo "       --encoder_model_path $OUTPUT_MODEL_PATH"
    echo ""
    echo "2. 或者使用编码器评估："
    echo "   python encoder_finetune_evaluate/run_encoder_eval.py \\"
    echo "       --model_name $OUTPUT_MODEL_PATH \\"
    echo "       --eval_jsonl $EVAL_DATA \\"
    echo "       --max_samples 1000"
else
    echo ""
    echo "❌ AlphaFin编码器微调失败！"
    echo "请检查错误信息并重试。"
    exit 1
fi 