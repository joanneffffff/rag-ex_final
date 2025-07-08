#!/bin/bash

# 快速测试AlphaFin编码器微调
# 如果测试成功，则后台运行完整训练

echo "=========================================="
echo "AlphaFin编码器微调 - 快速测试"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1  # 添加CUDA调试

# 测试配置（小规模）
TEST_EPOCHS=1
TEST_MAX_SAMPLES=50  # 减少样本数
TEST_EVAL_STEPS=25   # 减少评估步数

# 完整训练配置
FULL_EPOCHS=5
FULL_MAX_SAMPLES=20000
FULL_EVAL_STEPS=200

# 模型配置
BASE_MODEL="Langboat/mengzi-bert-base-fin"
TEST_OUTPUT_DIR="./models/alphafin_encoder_test"
FULL_OUTPUT_DIR="./models/alphafin_encoder_finetuned_enhanced"

# 数据路径
TRAIN_DATA="evaluate_mrr/alphafin_train_qc.jsonl"
EVAL_DATA="evaluate_mrr/alphafin_eval.jsonl"

echo "🧪 快速测试配置:"
echo "  - 训练数据: $TRAIN_DATA"
echo "  - 评估数据: $EVAL_DATA"
echo "  - 基础模型: $BASE_MODEL"
echo "  - 测试输出: $TEST_OUTPUT_DIR"
echo "  - 测试样本数: $TEST_MAX_SAMPLES"
echo "  - 测试轮数: $TEST_EPOCHS"
echo "  - 评估步数: $TEST_EVAL_STEPS"
echo "  - 批次大小: 8 (减少内存使用)"
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

# 检查GPU内存
echo "🔍 检查GPU内存..."
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | while read line; do
    used=$(echo $line | cut -d',' -f1)
    total=$(echo $line | cut -d',' -f2)
    available=$((total - used))
    echo "  GPU内存: 已用 ${used}MB / 总计 ${total}MB / 可用 ${available}MB"
    
    if [ $available -lt 4000 ]; then
        echo "⚠️  警告：GPU内存不足，建议清理内存后重试"
        echo "💡 清理命令: nvidia-smi --gpu-reset"
    fi
done

echo ""

# 创建测试输出目录
mkdir -p "$TEST_OUTPUT_DIR"

# 开始快速测试
echo "🚀 开始快速测试..."
echo "时间: $(date)"
echo ""

python finetune_alphafin_encoder_enhanced.py \
    --model_name "$BASE_MODEL" \
    --train_jsonl "$TRAIN_DATA" \
    --eval_jsonl "$EVAL_DATA" \
    --output_dir "$TEST_OUTPUT_DIR" \
    --batch_size 8 \
    --epochs $TEST_EPOCHS \
    --max_samples $TEST_MAX_SAMPLES \
    --eval_steps $TEST_EVAL_STEPS

# 检查测试是否成功
TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ 快速测试成功！"
    echo "📊 测试结果:"
    
    # 检查是否生成了模型文件
    if [ -f "$TEST_OUTPUT_DIR/pytorch_model.bin" ]; then
        echo "  - 模型文件已生成"
        MODEL_GENERATED=true
    else
        echo "  - 警告：模型文件未生成"
        MODEL_GENERATED=false
    fi
    
    # 检查是否生成了MRR评估结果
    if [ -f "$TEST_OUTPUT_DIR/mrr_eval_mrr_evaluation_results.csv" ]; then
        echo "  - MRR评估结果已生成"
        echo "  - 查看结果: tail -5 $TEST_OUTPUT_DIR/mrr_eval_mrr_evaluation_results.csv"
        MRR_GENERATED=true
    else
        echo "  - 警告：MRR评估结果未生成"
        MRR_GENERATED=false
    fi
    
    # 只有当模型和MRR都生成时才继续
    if [ "$MODEL_GENERATED" = true ] && [ "$MRR_GENERATED" = true ]; then
        echo ""
        echo "🎯 开始后台完整训练..."
        echo "时间: $(date)"
        echo ""
        
        # 创建完整训练输出目录
        mkdir -p "$FULL_OUTPUT_DIR"
        
        # 后台运行完整训练
        nohup python finetune_alphafin_encoder_enhanced.py \
            --model_name "$BASE_MODEL" \
            --train_jsonl "$TRAIN_DATA" \
            --eval_jsonl "$EVAL_DATA" \
            --output_dir "$FULL_OUTPUT_DIR" \
            --batch_size 16 \
            --epochs $FULL_EPOCHS \
            --max_samples $FULL_MAX_SAMPLES \
            --eval_steps $FULL_EVAL_STEPS \
            > logs/alphafin_finetune_full_$(date +%Y%m%d_%H%M%S).log 2>&1 &
        
        # 获取后台进程ID
        PID=$!
        echo "🔄 完整训练已在后台启动 (PID: $PID)"
        echo "📝 日志文件: logs/alphafin_finetune_full_$(date +%Y%m%d_%H%M%S).log"
        echo ""
        echo "💡 监控命令:"
        echo "  - 查看日志: tail -f logs/alphafin_finetune_full_*.log"
        echo "  - 查看进程: ps aux | grep $PID"
        echo "  - 停止训练: kill $PID"
        echo ""
        echo "⏰ 预计完成时间: 1-2小时"
    else
        echo ""
        echo "❌ 测试不完整，跳过完整训练"
        echo "请检查错误信息并修复问题。"
        exit 1
    fi
    
else
    echo ""
    echo "❌ 快速测试失败！ (退出码: $TEST_EXIT_CODE)"
    echo "请检查错误信息并修复问题。"
    echo ""
    echo "💡 可能的解决方案:"
    echo "  1. 清理GPU内存: nvidia-smi --gpu-reset"
    echo "  2. 减少批次大小: 修改 --batch_size 为 4 或 8"
    echo "  3. 使用CPU训练: 设置 CUDA_VISIBLE_DEVICES=''"
    exit 1
fi 