#!/bin/bash

# AlphaFin编码器微调GPU完整训练脚本
# 使用完整数据集，10个epoch进行充分训练

echo "🚀 开始AlphaFin编码器微调GPU完整训练"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# 创建日志目录
mkdir -p logs

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/alphafin_encoder_finetune_gpu_full_${TIMESTAMP}.log"

echo "📝 日志文件: $LOG_FILE"
echo "🕐 开始时间: $(date)"
echo "🔧 使用完整数据集，10个epoch充分训练，修复字段映射和MRR计算"

# 运行微调脚本
python finetune_alphafin_encoder_enhanced.py \
    --batch_size 32 \
    --epochs 10 \
    --eval_steps 500 \
    --output_dir "models/alphafin_encoder_finetuned_full_fixed" \
    2>&1 | tee "$LOG_FILE"

echo "✅ 完整训练完成！"
echo "📊 查看日志: tail -f $LOG_FILE"
echo "📊 查看MRR结果: grep -E '(MRR|Epoch|Step|loss)' $LOG_FILE"
echo "📊 查看最终MRR: tail -20 $LOG_FILE | grep 'MRR'" 