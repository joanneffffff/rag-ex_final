#!/bin/bash

# 简化版快速测试脚本 - 专门处理CUDA内存不足问题

echo "🧪 简化版快速测试 - 处理CUDA内存不足问题"
echo "时间: $(date)"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1

# 检查GPU内存
echo "🔍 检查GPU内存..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | while read line; do
        used=$(echo $line | cut -d',' -f1)
        total=$(echo $line | cut -d',' -f2)
        available=$((total - used))
        echo "  GPU内存: 已用 ${used}MB / 总计 ${total}MB / 可用 ${available}MB"
        
        if [ $available -lt 4000 ]; then
            echo "⚠️  警告：GPU内存不足，建议清理内存"
            echo "💡 清理命令: nvidia-smi --gpu-reset"
        fi
    done
else
    echo "  nvidia-smi 不可用"
fi

echo ""

# 测试配置（极小规模）
TEST_EPOCHS=1
TEST_MAX_SAMPLES=10  # 极少量样本
TEST_EVAL_STEPS=5    # 极少评估步数
TEST_BATCH_SIZE=2    # 极小批次

# 数据路径
TRAIN_DATA="evaluate_mrr/alphafin_train_qc.jsonl"
EVAL_DATA="evaluate_mrr/alphafin_eval.jsonl"
TEST_OUTPUT_DIR="./models/alphafin_encoder_test_simple"

echo "📊 测试配置:"
echo "  - 训练数据: $TRAIN_DATA"
echo "  - 评估数据: $EVAL_DATA"
echo "  - 输出目录: $TEST_OUTPUT_DIR"
echo "  - 样本数: $TEST_MAX_SAMPLES"
echo "  - 轮数: $TEST_EPOCHS"
echo "  - 批次大小: $TEST_BATCH_SIZE"
echo "  - 评估步数: $TEST_EVAL_STEPS"
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

# 创建输出目录
mkdir -p "$TEST_OUTPUT_DIR"

# 开始测试
echo "🚀 开始简化测试..."
echo "时间: $(date)"
echo ""

# 运行测试
python finetune_alphafin_encoder_enhanced.py \
    --model_name "Langboat/mengzi-bert-base-fin" \
    --train_jsonl "$TRAIN_DATA" \
    --eval_jsonl "$EVAL_DATA" \
    --output_dir "$TEST_OUTPUT_DIR" \
    --batch_size $TEST_BATCH_SIZE \
    --epochs $TEST_EPOCHS \
    --max_samples $TEST_MAX_SAMPLES \
    --eval_steps $TEST_EVAL_STEPS

# 检查结果
TEST_EXIT_CODE=$?

echo ""
echo "📊 测试结果分析:"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ 脚本执行成功"
    
    # 检查模型文件
    if [ -f "$TEST_OUTPUT_DIR/pytorch_model.bin" ]; then
        echo "✅ 模型文件已生成"
        MODEL_OK=true
    else
        echo "❌ 模型文件未生成"
        MODEL_OK=false
    fi
    
    # 检查评估结果
    if [ -f "$TEST_OUTPUT_DIR/mrr_eval_mrr_evaluation_results.csv" ]; then
        echo "✅ MRR评估结果已生成"
        echo "📈 查看结果:"
        tail -3 "$TEST_OUTPUT_DIR/mrr_eval_mrr_evaluation_results.csv"
        MRR_OK=true
    else
        echo "❌ MRR评估结果未生成"
        MRR_OK=false
    fi
    
    if [ "$MODEL_OK" = true ] && [ "$MRR_OK" = true ]; then
        echo ""
        echo "🎉 简化测试完全成功！"
        echo "💡 现在可以运行完整训练了"
        echo ""
        echo "📝 完整训练命令:"
        echo "python finetune_alphafin_encoder_enhanced.py \\"
        echo "    --model_name \"Langboat/mengzi-bert-base-fin\" \\"
        echo "    --train_jsonl \"$TRAIN_DATA\" \\"
        echo "    --eval_jsonl \"$EVAL_DATA\" \\"
        echo "    --output_dir \"./models/alphafin_encoder_finetuned_enhanced\" \\"
        echo "    --batch_size 8 \\"
        echo "    --epochs 5 \\"
        echo "    --max_samples 20000 \\"
        echo "    --eval_steps 200"
    else
        echo ""
        echo "⚠️  测试部分成功，但有问题需要解决"
    fi
    
else
    echo "❌ 脚本执行失败 (退出码: $TEST_EXIT_CODE)"
    echo ""
    echo "🔍 可能的问题和解决方案:"
    echo ""
    echo "1. CUDA内存不足:"
    echo "   - 清理GPU内存: nvidia-smi --gpu-reset"
    echo "   - 进一步减少批次大小: --batch_size 1"
    echo "   - 使用CPU训练: export CUDA_VISIBLE_DEVICES=''"
    echo ""
    echo "2. 模型下载问题:"
    echo "   - 检查网络连接"
    echo "   - 手动下载模型到本地"
    echo ""
    echo "3. 依赖包问题:"
    echo "   - 检查PyTorch版本: pip list | grep torch"
    echo "   - 检查sentence-transformers版本: pip list | grep sentence"
    echo ""
    echo "💡 查看详细错误信息:"
    echo "   tail -20 logs/alphafin_finetune_*.log"
fi

echo ""
echo "⏰ 测试完成时间: $(date)" 