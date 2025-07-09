#!/bin/bash

# 简化的reranker nohup运行脚本
# 使用完整路径和环境变量

echo "=========================================="
echo "Reranker模式nohup运行脚本 (简化版)"
echo "时间: $(date)"
echo "=========================================="

# 获取当前目录的完整路径
CURRENT_DIR=$(pwd)
echo "当前目录: $CURRENT_DIR"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$CURRENT_DIR:$PYTHONPATH"

# 创建日志目录
mkdir -p logs

# 设置日志文件路径
LOG_FILE="logs/reranker_nohup_simple_$(date +%Y%m%d_%H%M%S).log"

echo "开始nohup运行reranker评测..."
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"
echo "Python路径: $PYTHONPATH"

# 检查Python环境
echo "检查Python环境..."
which python
python --version

# 检查数据文件
echo "检查数据文件..."
if [ ! -f "$CURRENT_DIR/data/alphafin/alphafin_eval_samples.jsonl" ]; then
    echo "❌ 评测数据文件不存在: $CURRENT_DIR/data/alphafin/alphafin_eval_samples.jsonl"
    exit 1
fi
echo "✅ 数据文件检查通过"

# 切换到项目目录
cd "$CURRENT_DIR"

# nohup运行命令 (使用完整路径)
nohup python "$CURRENT_DIR/run_reranker_quick_test.py" \
    --modes reranker reranker_no_prefilter \
    --eval_data_path "$CURRENT_DIR/data/alphafin/alphafin_eval_samples.jsonl" \
    --output_dir "$CURRENT_DIR/alphafin_data_process/evaluation_results" \
    --max_samples 100 \
    > "$LOG_FILE" 2>&1 &

# 获取后台进程ID
PID=$!
echo "进程ID: $PID"
echo "进程ID: $PID" >> "$LOG_FILE"

echo "nohup任务已在后台启动，进程ID: $PID"
echo "使用以下命令查看日志:"
echo "tail -f $LOG_FILE"
echo ""
echo "使用以下命令检查进程状态:"
echo "ps aux | grep $PID"
echo ""
echo "使用以下命令停止进程:"
echo "kill $PID" 