#!/bin/bash

# AlphaFin中文编码器微调脚本 (增强版)
# 使用generated_question作为query，summary作为context

echo "=========================================="
echo "AlphaFin中文编码器微调脚本 (增强版)"
echo "使用: generated_question -> summary"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# 数据路径配置
TRAIN_DATA="evaluate_mrr/alphafin_train_qc.jsonl"
EVAL_DATA="evaluate_mrr/alphafin_eval.jsonl"

# 模型配置
BASE_MODEL="Langboat/mengzi-bert-base-fin"
OUTPUT_MODEL_PATH="./models/alphafin_encoder_finetuned_enhanced"

# 训练参数
BATCH_SIZE=32
EPOCHS=5  # 对于2万数据，使用5个epoch
EVAL_STEPS=200  # 更频繁的评估

# 数据限制（使用全部数据）
MAX_SAMPLES=20000  # 使用全部训练数据

echo "配置信息："
echo "  训练数据: $TRAIN_DATA"
echo "  评估数据: $EVAL_DATA"
echo "  基础模型: $BASE_MODEL"
echo "  输出路径: $OUTPUT_MODEL_PATH"
echo "  批次大小: $BATCH_SIZE"
echo "  训练轮数: $EPOCHS"
echo "  评估步数: $EVAL_STEPS"
echo "  最大样本数: $MAX_SAMPLES"
echo "  使用字段: generated_question -> summary"
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
echo "开始AlphaFin编码器微调 (增强版)..."
echo "时间: $(date)"
echo ""

python finetune_alphafin_encoder_enhanced.py \
    --model_name "$BASE_MODEL" \
    --train_jsonl "$TRAIN_DATA" \
    --eval_jsonl "$EVAL_DATA" \
    --output_dir "$OUTPUT_MODEL_PATH" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --max_samples $MAX_SAMPLES \
    --eval_steps $EVAL_STEPS

# 检查微调是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ AlphaFin编码器微调 (增强版) 完成！"
    echo "📁 模型保存在: $OUTPUT_MODEL_PATH"
    echo "时间: $(date)"
else
    echo ""
    echo "❌ AlphaFin编码器微调 (增强版) 失败！"
    exit 1
fi 