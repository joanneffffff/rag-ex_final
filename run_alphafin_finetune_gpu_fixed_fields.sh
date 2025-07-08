#!/bin/bash

# AlphaFin编码器微调GPU训练脚本 (修复字段映射版)
# 直接使用generated_question和summary字段，不使用query/context映射

echo "🚀 开始AlphaFin编码器微调GPU训练 (修复字段映射版)"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

# 创建日志目录
mkdir -p logs

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/alphafin_encoder_finetune_gpu_fixed_fields_${TIMESTAMP}.log"

echo "📝 日志文件: $LOG_FILE"
echo "🕐 开始时间: $(date)"
echo "🔧 直接使用generated_question和summary字段"

# 运行微调脚本
python finetune_alphafin_encoder_enhanced.py \
    --batch_size 32 \
    --epochs 10 \
    --eval_steps 500 \
    --output_dir "models/alphafin_encoder_finetuned_fixed_fields" \
    2>&1 | tee "$LOG_FILE"

echo "✅ 训练完成！"
echo "📊 查看日志: tail -f $LOG_FILE"
echo "📊 查看MRR结果: grep -E '(MRR|Epoch|Step|loss)' $LOG_FILE"
echo "📊 查看最终MRR: tail -20 $LOG_FILE | grep 'MRR'" 