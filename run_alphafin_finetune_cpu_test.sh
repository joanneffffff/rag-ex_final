#!/bin/bash

# AlphaFin中文编码器微调脚本 (CPU测试版)
# 使用generated_question作为query，summary作为context
# 用于验证修复后的代码是否能正常运行

echo "=========================================="
echo "AlphaFin中文编码器微调脚本 (CPU测试版)"
echo "使用: generated_question -> summary"
echo "=========================================="

# 创建日志目录
mkdir -p logs

# 设置日志文件路径
LOG_FILE="logs/alphafin_encoder_finetune_cpu_test_$(date +%Y%m%d_%H%M%S).log"

echo "开始CPU测试运行 ALPHAFIN 编码器微调..."
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"

# 设置环境变量 - 强制使用CPU
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false

# CPU测试运行命令
nohup python finetune_alphafin_encoder_enhanced.py \
    --model_name "Langboat/mengzi-bert-base-fin" \
    --train_jsonl "evaluate_mrr/alphafin_train_qc.jsonl" \
    --eval_jsonl "evaluate_mrr/alphafin_eval.jsonl" \
    --output_dir "./models/alphafin_encoder_finetuned_cpu_test" \
    --batch_size 8 \
    --epochs 1 \
    --max_samples 100 \
    --eval_steps 50 \
    > "$LOG_FILE" 2>&1 &

# 获取后台进程ID
PID=$!
echo "进程ID: $PID"
echo "进程ID: $PID" >> "$LOG_FILE"

echo "CPU测试任务已在后台启动，进程ID: $PID"
echo "使用以下命令查看日志:"
echo "tail -f $LOG_FILE"
echo ""
echo "使用以下命令检查进程状态:"
echo "ps aux | grep $PID"
echo ""
echo "使用以下命令停止进程:"
echo "kill $PID"
echo ""
echo "使用以下命令查看实时日志（最后50行）:"
echo "tail -f $LOG_FILE | grep -E '(MRR|Epoch|Step|loss|Error|Exception)'"
echo ""
echo "使用以下命令查看完整日志:"
echo "cat $LOG_FILE" 