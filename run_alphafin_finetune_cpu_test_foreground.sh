#!/bin/bash

# AlphaFin中文编码器微调脚本 (CPU测试版 - 前台运行)
# 使用generated_question作为query，summary作为context
# 用于验证修复后的代码是否能正常运行

echo "=========================================="
echo "AlphaFin中文编码器微调脚本 (CPU测试版 - 前台运行)"
echo "使用: generated_question -> summary"
echo "=========================================="

# 设置环境变量 - 强制使用CPU
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false

echo "开始CPU测试运行 ALPHAFIN 编码器微调..."
echo "开始时间: $(date)"
echo ""

# CPU测试运行命令 - 前台运行
python finetune_alphafin_encoder_enhanced.py \
    --model_name "Langboat/mengzi-bert-base-fin" \
    --train_jsonl "evaluate_mrr/alphafin_train_qc.jsonl" \
    --eval_jsonl "evaluate_mrr/alphafin_eval.jsonl" \
    --output_dir "./models/alphafin_encoder_finetuned_cpu_test" \
    --batch_size 8 \
    --epochs 1 \
    --max_samples 100 \
    --eval_steps 50

echo ""
echo "CPU测试完成！"
echo "结束时间: $(date)" 