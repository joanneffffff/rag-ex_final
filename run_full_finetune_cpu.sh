#!/bin/bash

# 完整CPU训练脚本 - 无GPU环境

echo "🚀 完整CPU训练 - AlphaFin编码器微调"
echo "时间: $(date)"
echo ""

# 设置环境变量 - 强制使用CPU
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1

echo "🔧 环境配置:"
echo "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (强制CPU模式)"
echo "  - TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"
echo ""

# 训练配置（CPU优化）
FULL_EPOCHS=5
FULL_EVAL_STEPS=200
FULL_BATCH_SIZE=8  # CPU批次大小

# 数据路径
TRAIN_DATA="evaluate_mrr/alphafin_train_qc.jsonl"
EVAL_DATA="evaluate_mrr/alphafin_eval.jsonl"
FULL_OUTPUT_DIR="./models/alphafin_encoder_finetuned_cpu"

echo "📊 训练配置:"
echo "  - 训练数据: $TRAIN_DATA"
echo "  - 评估数据: $EVAL_DATA"
echo "  - 输出目录: $FULL_OUTPUT_DIR"
echo "  - 样本数: 全部数据 (无限制)"
echo "  - 轮数: $FULL_EPOCHS"
echo "  - 批次大小: $FULL_BATCH_SIZE"
echo "  - 评估步数: $FULL_EVAL_STEPS"
echo "  - 设备: CPU"
echo ""

# 检查数据文件
echo "📋 检查数据文件..."
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ 训练数据文件不存在: $TRAIN_DATA"
    exit 1
fi

if [ ! -f "$EVAL_DATA" ]; then
    echo "❌ 评估数据文件不存在: $EVAL_DATA"
    exit 1
fi

echo "✅ 数据文件检查通过"
echo ""

# 检查Python环境
echo "🐍 检查Python环境..."
python --version
echo "  PyTorch版本: $(python -c "import torch; print(torch.__version__)")"
echo "  CUDA可用: $(python -c "import torch; print(torch.cuda.is_available())")"
echo "  SentenceTransformers版本: $(python -c "import sentence_transformers; print(sentence_transformers.__version__)")"
echo ""

# 创建输出目录
mkdir -p "$FULL_OUTPUT_DIR"

# 创建日志目录
mkdir -p logs

# 开始训练
echo "🚀 开始完整CPU训练..."
echo "时间: $(date)"
echo ""

# 后台运行训练
nohup python finetune_alphafin_encoder_enhanced.py \
    --model_name "Langboat/mengzi-bert-base-fin" \
    --train_jsonl "$TRAIN_DATA" \
    --eval_jsonl "$EVAL_DATA" \
    --output_dir "$FULL_OUTPUT_DIR" \
    --batch_size $FULL_BATCH_SIZE \
    --epochs $FULL_EPOCHS \
    --eval_steps $FULL_EVAL_STEPS \
    > logs/alphafin_finetune_cpu_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 获取后台进程ID
PID=$!
echo "🔄 完整训练已在后台启动 (PID: $PID)"
echo "📝 日志文件: logs/alphafin_finetune_cpu_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "💡 监控命令:"
echo "  - 查看日志: tail -f logs/alphafin_finetune_cpu_*.log"
echo "  - 查看进程: ps aux | grep $PID"
echo "  - 停止训练: kill $PID"
echo ""
echo "⏰ 预计完成时间: 3-5小时 (CPU模式)"
echo ""
echo "📊 训练进度监控:"
echo "  - 实时日志: tail -f logs/alphafin_finetune_cpu_*.log | grep -E '(Epoch|MRR|loss)'"
echo "  - 模型文件: ls -la $FULL_OUTPUT_DIR/"
echo "  - 评估结果: tail -f $FULL_OUTPUT_DIR/mrr_eval_mrr_evaluation_results.csv" 