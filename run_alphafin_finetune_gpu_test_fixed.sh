#!/bin/bash

# AlphaFin编码器微调GPU测试脚本 (修复版)
# 使用修复后的数据加载器

echo "🚀 开始AlphaFin编码器微调GPU测试 (修复版)"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# 创建日志目录
mkdir -p logs

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/alphafin_encoder_finetune_gpu_test_fixed_${TIMESTAMP}.log"

echo "📝 日志文件: $LOG_FILE"
echo "🕐 开始时间: $(date)"
echo "🔧 使用修复后的数据加载器"

# 运行微调脚本
python finetune_alphafin_encoder_enhanced.py \
    --max_samples 50 \
    --batch_size 8 \
    --epochs 2 \
    --eval_steps 10 \
    --output_dir "models/alphafin_encoder_finetuned_test_fixed" \
    2>&1 | tee "$LOG_FILE"

echo "✅ 测试完成！"
echo "📊 查看日志: tail -f $LOG_FILE"
echo "📊 查看MRR结果: grep -E '(MRR|Epoch|Step|loss)' $LOG_FILE" 