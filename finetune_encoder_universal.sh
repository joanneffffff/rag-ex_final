#!/bin/bash

# 通用编码器微调脚本
# 支持AlphaFin中文和TAT-QA英文数据集

echo "=========================================="
echo "通用编码器微调脚本"
echo "支持: AlphaFin(中文) | TAT-QA(英文)"
echo "=========================================="

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: $0 [alphafin|tatqa] [可选: 快速测试模式]"
    echo ""
    echo "示例:"
    echo "  $0 alphafin          # AlphaFin完整微调"
    echo "  $0 tatqa             # TAT-QA完整微调"
    echo "  $0 alphafin quick    # AlphaFin快速测试"
    echo "  $0 tatqa quick       # TAT-QA快速测试"
    exit 1
fi

DATASET=$1
QUICK_MODE=$2

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# 根据数据集配置参数
if [ "$DATASET" = "alphafin" ]; then
    echo "🎯 配置AlphaFin中文数据集..."
    
    # 数据路径
    TRAIN_DATA="evaluate_mrr/alphafin_train_qc.jsonl"
    EVAL_DATA="evaluate_mrr/alphafin_eval.jsonl"
    
    # 模型配置
    BASE_MODEL="Langboat/mengzi-bert-base-fin"
    OUTPUT_MODEL_PATH="./models/finetuned_alphafin_encoder_summary"
    
    # 训练参数
    BATCH_SIZE=16
    EPOCHS=5
    LEARNING_RATE=2e-5
    MAX_SEQ_LENGTH=512
    
    # 微调脚本
    FINETUNE_SCRIPT="encoder_finetune_evaluate/finetune_encoder.py"
    
elif [ "$DATASET" = "tatqa" ]; then
    echo "🎯 配置TAT-QA英文数据集..."
    
    # 数据路径
    TRAIN_DATA="evaluate_mrr/tatqa_train_qc.jsonl"
    EVAL_DATA="evaluate_mrr/tatqa_eval.jsonl"
    RAW_DATA="evaluate_mrr/tatqa_knowledge_base.jsonl"
    
    # 模型配置
    # BASE_MODEL="ProsusAI/finbert"
    BASE_MODEL="/users/sgjfei3/data/manually_downloaded_models/finbert"
    OUTPUT_MODEL_PATH="./models/finetuned_tatqa_encoder_table_textualized"
    
    # 训练参数
    BATCH_SIZE=32
    EPOCHS=5
    LEARNING_RATE=2e-5
    MAX_SEQ_LENGTH=512
    
    # 微调脚本
    FINETUNE_SCRIPT="encoder_finetune_evaluate/finetune_encoder.py"
    
else
    echo "❌ 错误: 不支持的数据集 '$DATASET'"
    echo "支持的数据集: alphafin, tatqa"
    exit 1
fi

# 快速测试模式配置
if [ "$QUICK_MODE" = "quick" ]; then
    echo "⚡ 快速测试模式"
    LIMIT_TRAIN=1000
    LIMIT_EVAL=50
    EPOCHS=1
    BATCH_SIZE=8
else
    echo "🚀 完整训练模式"
    LIMIT_TRAIN=0  # 使用全部数据
    LIMIT_EVAL=100
fi

EVAL_TOP_K=100

echo ""
echo "配置信息："
echo "  数据集: $DATASET"
echo "  训练数据: $TRAIN_DATA"
echo "  评估数据: $EVAL_DATA"

echo "  基础模型: $BASE_MODEL"
echo "  输出路径: $OUTPUT_MODEL_PATH"
echo "  微调脚本: $FINETUNE_SCRIPT"
echo "  批次大小: $BATCH_SIZE"
echo "  训练轮数: $EPOCHS"
echo "  学习率: $LEARNING_RATE"
echo "  最大序列长度: $MAX_SEQ_LENGTH"
echo "  评估Top-K: $EVAL_TOP_K"
echo "  训练数据限制: $LIMIT_TRAIN"
echo "  评估数据限制: $LIMIT_EVAL"
echo ""

# 检查数据文件是否存在
echo "检查数据文件..."
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ 训练数据文件不存在: $TRAIN_DATA"
    exit 1
fi

if [ ! -f "$EVAL_DATA" ]; then
    echo "❌ 评估数据文件不存在: $EVAL_DATA"
    exit 1
fi

echo "✅ 所有数据文件检查通过"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_MODEL_PATH"

# 开始微调
echo "开始${DATASET^^}编码器微调..."
echo "时间: $(date)"
echo ""

if [ "$DATASET" = "alphafin" ]; then
    # AlphaFin使用简化微调脚本
    python "$FINETUNE_SCRIPT" \
        --model_name "$BASE_MODEL" \
        --train_jsonl "$TRAIN_DATA" \
        --eval_jsonl "$EVAL_DATA" \
        --output_dir "$OUTPUT_MODEL_PATH" \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --learning_rate $LEARNING_RATE \
        --max_seq_length $MAX_SEQ_LENGTH \
        --max_samples $LIMIT_TRAIN
else
    # TAT-QA使用英文微调脚本
    python "$FINETUNE_SCRIPT" \
        --model_name "$BASE_MODEL" \
        --train_jsonl "$TRAIN_DATA" \
        --eval_jsonl "$EVAL_DATA" \
        --output_dir "$OUTPUT_MODEL_PATH" \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --max_samples $LIMIT_TRAIN \
        --eval_steps 500
fi

# 检查微调是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ ${DATASET^^}编码器微调完成！"
    echo "模型保存路径: $OUTPUT_MODEL_PATH"
    echo "完成时间: $(date)"
    echo ""
    
    # ==================== 添加评估步骤 ====================
    echo "🔍 开始模型评估..."
    echo "时间: $(date)"
    echo ""
    
    # 创建评估结果目录
    EVAL_OUTPUT_DIR="evaluation_results/${DATASET}_encoder_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$EVAL_OUTPUT_DIR"
    
    if [ "$DATASET" = "alphafin" ]; then
        echo "📊 运行AlphaFin编码器评估..."
        
        # 1. 运行MRR评估
        echo "1. 运行MRR评估..."
        python encoder_finetune_evaluate/evaluate_chinese_encoder_reranker_mrr.py \
            --encoder_model_name "$OUTPUT_MODEL_PATH" \
            --eval_jsonl "$EVAL_DATA" \
            --base_raw_data_path "data/alphafin/alphafin_final_clean.json" \
            --output_dir "$EVAL_OUTPUT_DIR" \
            --max_samples $LIMIT_EVAL
        
        # 2. 运行检索评估
        echo "2. 运行检索评估..."
        python alphafin_data_process/run_retrieval_evaluation_background.py \
            --eval_data_path "$EVAL_DATA" \
            --output_dir "$EVAL_OUTPUT_DIR/retrieval_eval" \
            --modes baseline prefilter reranker \
            --max_samples $LIMIT_EVAL
        
    else
        echo "📊 运行TAT-QA编码器评估..."
        
        # 1. 运行编码器评估
        echo "1. 运行编码器评估..."
        python encoder_finetune_evaluate/run_encoder_eval.py \
            --model_name "$OUTPUT_MODEL_PATH" \
            --eval_jsonl "$EVAL_DATA" \
            --max_samples $LIMIT_EVAL \
            --output_dir "$EVAL_OUTPUT_DIR"
        
        # 2. 运行TAT-QA检索评估
        echo "2. 运行TAT-QA检索评估..."
        python alphafin_data_process/run_tatqa_retrieval_evaluation.py \
            --mode reranker \
            --encoder_model_path "$OUTPUT_MODEL_PATH" \
            --output_dir "$EVAL_OUTPUT_DIR/tatqa_eval" \
            --max_samples $LIMIT_EVAL
    fi
    
    echo ""
    echo "✅ 评估完成！"
    echo "评估结果保存在: $EVAL_OUTPUT_DIR"
    echo "完成时间: $(date)"
    echo ""
    
    # ==================== 显示下一步建议 ====================
    echo "下一步建议："
    echo ""
    
    if [ "$DATASET" = "alphafin" ]; then
        echo "1. 查看评估结果："
        echo "   ls -la $EVAL_OUTPUT_DIR"
        echo ""
        echo "2. 使用微调后的模型进行RAG系统测试："
        echo "   python run_optimized_ui.py"
        echo ""
        echo "3. 运行完整检索评估："
        echo "   python alphafin_data_process/run_retrieval_evaluation_background.py \\"
        echo "       --eval_data_path data/alphafin/eval_data_100_from_corpus.jsonl \\"
        echo "       --output_dir alphafin_data_process/evaluation_results \\"
        echo "       --modes baseline prefilter reranker"
    else
        echo "1. 查看评估结果："
        echo "   ls -la $EVAL_OUTPUT_DIR"
        echo ""
        echo "2. 使用微调后的模型进行RAG系统测试："
        echo "   python run_optimized_ui.py"
        echo ""
        echo "3. 运行TAT-QA完整评估："
        echo "   python alphafin_data_process/run_tatqa_retrieval_evaluation.py \\"
        echo "       --mode reranker \\"
        echo "       --encoder_model_path $OUTPUT_MODEL_PATH"
    fi
    
else
    echo ""
    echo "❌ ${DATASET^^}编码器微调失败！"
    echo "请检查错误信息并重试。"
    exit 1
fi 