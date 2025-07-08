#!/bin/bash

# GPU监控和自动训练脚本
# 当GPU可用时自动运行AlphaFin编码器微调

echo "🤖 GPU监控和自动训练脚本"
echo "时间: $(date)"
echo ""

# 配置
MONITOR_INTERVAL=300  # 监控间隔(秒) - 5分钟
MAX_WAIT_TIME=86400   # 最大等待时间(秒) - 24小时
GPU_MEMORY_THRESHOLD=4000  # GPU内存阈值(MB) - 4GB
GPU_DEVICE=1  # 使用CUDA:1

# 训练配置
TRAIN_DATA="evaluate_mrr/alphafin_train_qc.jsonl"
EVAL_DATA="evaluate_mrr/alphafin_eval.jsonl"
OUTPUT_DIR="./models/alphafin_encoder_finetuned_gpu"
LOG_DIR="./logs"

# 创建目录
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# 日志文件
LOG_FILE="$LOG_DIR/gpu_monitor_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/gpu_monitor.pid"

echo "📊 监控配置:"
echo "  - 监控间隔: ${MONITOR_INTERVAL}秒 (${MONITOR_INTERVAL}/60分钟)"
echo "  - 最大等待: ${MAX_WAIT_TIME}秒 (${MAX_WAIT_TIME}/3600小时)"
echo "  - GPU内存阈值: ${GPU_MEMORY_THRESHOLD}MB"
echo "  - 训练数据: $TRAIN_DATA"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - 日志文件: $LOG_FILE"
echo ""

# 记录PID
echo $$ > "$PID_FILE"
echo "🔄 监控进程PID: $$"
echo "💡 停止监控: kill \$(cat $PID_FILE)"
echo ""

# 检查GPU状态的函数
check_gpu_status() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ nvidia-smi 不可用，无法检测GPU"
        return 1
    fi
    
    # 检查指定GPU是否可用
    if ! nvidia-smi -i $GPU_DEVICE --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits &> /dev/null; then
        echo "❌ GPU:$GPU_DEVICE 不可用或nvidia-smi出错"
        return 1
    fi
    
    # 获取指定GPU信息
    local gpu_info=$(nvidia-smi -i $GPU_DEVICE --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits)
    if [ -z "$gpu_info" ]; then
        echo "❌ 无法获取GPU信息"
        return 1
    fi
    
    # 解析GPU信息
    local gpu_name=$(echo "$gpu_info" | cut -d',' -f1 | tr -d ' ')
    local memory_used=$(echo "$gpu_info" | cut -d',' -f2 | tr -d ' ')
    local memory_total=$(echo "$gpu_info" | cut -d',' -f3 | tr -d ' ')
    local memory_available=$((memory_total - memory_used))
    
    echo "🔍 GPU:$GPU_DEVICE 状态检查:"
    echo "  - GPU名称: $gpu_name"
    echo "  - 内存使用: ${memory_used}MB / ${memory_total}MB"
    echo "  - 可用内存: ${memory_available}MB"
    
    # 检查内存是否足够
    if [ $memory_available -ge $GPU_MEMORY_THRESHOLD ]; then
        echo "✅ GPU:$GPU_DEVICE 可用！内存充足 (${memory_available}MB >= ${GPU_MEMORY_THRESHOLD}MB)"
        return 0
    else
        echo "⚠️  GPU:$GPU_DEVICE 内存不足 (${memory_available}MB < ${GPU_MEMORY_THRESHOLD}MB)"
        return 1
    fi
}

# 运行训练的函数
run_training() {
    echo ""
    echo "🚀 GPU可用！开始训练..."
    echo "时间: $(date)"
    echo ""
    
    # 设置GPU环境变量
    export CUDA_VISIBLE_DEVICES=$GPU_DEVICE
    export TOKENIZERS_PARALLELISM=false
    
    echo "🔧 训练配置:"
    echo "  - 基础模型: Langboat/mengzi-bert-base-fin"
    echo "  - 训练数据: $TRAIN_DATA"
    echo "  - 评估数据: $EVAL_DATA"
    echo "  - 输出目录: $OUTPUT_DIR"
    echo "  - 批次大小: 16"
    echo "  - 训练轮数: 5"
    echo "  - 最大样本数: 全部数据 (无限制)"
    echo "  - 评估步数: 200"
    echo "  - 设备: GPU"
    echo ""
    
    # 运行训练
    python finetune_alphafin_encoder_enhanced.py \
        --model_name "Langboat/mengzi-bert-base-fin" \
        --train_jsonl "$TRAIN_DATA" \
        --eval_jsonl "$EVAL_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 16 \
        --epochs 5 \
        --eval_steps 200
    
    local train_exit_code=$?
    
    echo ""
    echo "📊 训练完成！"
    echo "时间: $(date)"
    echo "退出码: $train_exit_code"
    
    if [ $train_exit_code -eq 0 ]; then
        echo "✅ 训练成功完成！"
        echo "📁 模型保存在: $OUTPUT_DIR"
        
        # 检查结果文件
        if [ -f "$OUTPUT_DIR/model.safetensors" ]; then
            echo "✅ 模型文件已生成"
        fi
        
        if [ -f "$OUTPUT_DIR/eval/mrr_eval_mrr_evaluation_results.csv" ]; then
            echo "✅ MRR评估结果已生成"
            echo "📈 最终MRR结果:"
            tail -1 "$OUTPUT_DIR/eval/mrr_eval_mrr_evaluation_results.csv"
        fi
    else
        echo "❌ 训练失败 (退出码: $train_exit_code)"
    fi
    
    # 清理PID文件
    rm -f "$PID_FILE"
    exit 0
}

# 主监控循环
echo "🔄 开始GPU监控循环..."
echo "⏰ 监控间隔: ${MONITOR_INTERVAL}秒"
echo "⏰ 最大等待: ${MAX_WAIT_TIME}秒"
echo ""

start_time=$(date +%s)
check_count=0

while true; do
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    check_count=$((check_count + 1))
    
    echo ""
    echo "=== 第${check_count}次检查 ==="
    echo "⏰ 当前时间: $(date)"
    echo "⏰ 已等待: ${elapsed_time}秒 ($(($elapsed_time/60))分钟)"
    echo "⏰ 剩余时间: $(($MAX_WAIT_TIME - $elapsed_time))秒 ($(($MAX_WAIT_TIME - $elapsed_time)/60))分钟)"
    
    # 检查是否超时
    if [ $elapsed_time -ge $MAX_WAIT_TIME ]; then
        echo ""
        echo "⏰ 达到最大等待时间 (${MAX_WAIT_TIME}秒)"
        echo "💡 建议手动检查GPU状态或运行CPU训练"
        echo ""
        echo "CPU训练命令:"
        echo "./run_full_finetune_cpu.sh"
        break
    fi
    
    # 检查GPU状态
    if check_gpu_status; then
        # GPU可用，运行训练
        run_training
    else
        echo ""
        echo "⏳ GPU不可用，等待${MONITOR_INTERVAL}秒后重试..."
        echo "💡 手动清理GPU内存: nvidia-smi -i $GPU_DEVICE --gpu-reset"
        echo "💡 查看GPU状态: nvidia-smi -i $GPU_DEVICE"
        echo "💡 停止监控: kill \$(cat $PID_FILE)"
        
        # 等待下次检查
        sleep $MONITOR_INTERVAL
    fi
done

# 清理PID文件
rm -f "$PID_FILE"
echo ""
echo "🔚 GPU监控结束"
echo "时间: $(date)" 