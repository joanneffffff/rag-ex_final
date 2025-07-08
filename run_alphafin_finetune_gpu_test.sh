#!/bin/bash

# AlphaFin中文编码器微调脚本 (GPU测试版)
# 使用generated_question作为query，summary作为context
# 使用cuda:1设备进行快速测试

echo "=========================================="
echo "AlphaFin中文编码器微调脚本 (GPU测试版)"
echo "使用: generated_question -> summary"
echo "设备: cuda:1"
echo "=========================================="

# 创建日志目录
mkdir -p logs

# 设置日志文件路径
LOG_FILE="logs/alphafin_encoder_finetune_gpu_test_$(date +%Y%m%d_%H%M%S).log"

echo "开始GPU测试运行 ALPHAFIN 编码器微调..."
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"

# 设置环境变量 - 使用cuda:1
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

# GPU测试运行命令
nohup python finetune_alphafin_encoder_enhanced.py \
    --model_name "Langboat/mengzi-bert-base-fin" \
    --train_jsonl "evaluate_mrr/alphafin_train_qc.jsonl" \
    --eval_jsonl "evaluate_mrr/alphafin_eval.jsonl" \
    --output_dir "./models/alphafin_encoder_finetuned_gpu_test" \
    --batch_size 16 \
    --epochs 2 \
    --max_samples 500 \
    --eval_steps 100 \
    > "$LOG_FILE" 2>&1 &

# 获取后台进程ID
PID=$!
echo "进程ID: $PID"
echo "进程ID: $PID" >> "$LOG_FILE"

echo "GPU测试任务已在后台启动，进程ID: $PID"
echo "使用以下命令查看日志:"
echo "tail -f $LOG_FILE"
echo ""
echo "使用以下命令检查进程状态:"
echo "ps aux | grep $PID"
echo ""
echo "使用以下命令停止进程:"
echo "kill $PID"
echo ""
echo "使用以下命令查看GPU使用情况:"
echo "nvidia-smi"
echo ""
echo "使用以下命令查看实时日志（关键信息）:"
echo "tail -f $LOG_FILE | grep -E '(MRR|Epoch|Step|loss|Error|Exception)'"
echo ""
echo "使用以下命令查看完整日志:"
echo "cat $LOG_FILE" 