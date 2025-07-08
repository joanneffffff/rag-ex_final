#!/bin/bash

# CPU模式快速测试脚本 - 专门用于无GPU环境

echo "🧪 CPU模式快速测试 - 无GPU环境"
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

# 测试配置（CPU优化）
TEST_EPOCHS=1
TEST_MAX_SAMPLES=20  # CPU可以处理更多样本
TEST_EVAL_STEPS=10   # 更多评估步数
TEST_BATCH_SIZE=4    # CPU批次大小

# 数据路径
TRAIN_DATA="evaluate_mrr/alphafin_train_qc.jsonl"
EVAL_DATA="evaluate_mrr/alphafin_eval.jsonl"
TEST_OUTPUT_DIR="./models/alphafin_encoder_test_cpu"

echo "📊 测试配置:"
echo "  - 训练数据: $TRAIN_DATA"
echo "  - 评估数据: $EVAL_DATA"
echo "  - 输出目录: $TEST_OUTPUT_DIR"
echo "  - 样本数: $TEST_MAX_SAMPLES"
echo "  - 轮数: $TEST_EPOCHS"
echo "  - 批次大小: $TEST_BATCH_SIZE"
echo "  - 评估步数: $TEST_EVAL_STEPS"
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
mkdir -p "$TEST_OUTPUT_DIR"

# 开始测试
echo "🚀 开始CPU模式测试..."
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
    if [ -f "$TEST_OUTPUT_DIR/model.safetensors" ] || [ -f "$TEST_OUTPUT_DIR/pytorch_model.bin" ]; then
        echo "✅ 模型文件已生成"
        MODEL_OK=true
    else
        echo "❌ 模型文件未生成"
        MODEL_OK=false
    fi
    
    # 检查评估结果
    if [ -f "$TEST_OUTPUT_DIR/eval/mrr_eval_mrr_evaluation_results.csv" ]; then
        echo "✅ MRR评估结果已生成"
        echo "📈 查看结果:"
        tail -3 "$TEST_OUTPUT_DIR/eval/mrr_eval_mrr_evaluation_results.csv"
        MRR_OK=true
    else
        echo "❌ MRR评估结果未生成"
        MRR_OK=false
    fi
    
    if [ "$MODEL_OK" = true ] && [ "$MRR_OK" = true ]; then
        echo ""
        echo "🎉 CPU模式测试完全成功！"
        echo "💡 现在可以运行完整CPU训练了"
        echo ""
        echo "📝 完整CPU训练命令:"
        echo "export CUDA_VISIBLE_DEVICES=\"\""
        echo "python finetune_alphafin_encoder_enhanced.py \\"
        echo "    --model_name \"Langboat/mengzi-bert-base-fin\" \\"
        echo "    --train_jsonl \"$TRAIN_DATA\" \\"
        echo "    --eval_jsonl \"$EVAL_DATA\" \\"
        echo "    --output_dir \"./models/alphafin_encoder_finetuned_cpu\" \\"
        echo "    --batch_size 8 \\"
        echo "    --epochs 5 \\"
        echo "    --max_samples 20000 \\"
        echo "    --eval_steps 200"
        echo ""
        echo "⏰ 预计完成时间: 3-5小时 (CPU模式)"
    else
        echo ""
        echo "⚠️  测试部分成功，但有问题需要解决"
    fi
    
else
    echo "❌ 脚本执行失败 (退出码: $TEST_EXIT_CODE)"
    echo ""
    echo "🔍 可能的问题和解决方案:"
    echo ""
    echo "1. 内存不足:"
    echo "   - 减少批次大小: --batch_size 2"
    echo "   - 减少样本数: --max_samples 10"
    echo "   - 关闭其他程序释放内存"
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
    echo "   python finetune_alphafin_encoder_enhanced.py --help"
fi

echo ""
echo "⏰ 测试完成时间: $(date)" 