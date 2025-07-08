#!/bin/bash

# 启动GPU监控脚本

echo "🤖 启动GPU监控和自动训练"
echo "时间: $(date)"
echo ""

# 检查是否已有监控进程
PID_FILE="./logs/gpu_monitor.pid"
if [ -f "$PID_FILE" ]; then
    EXISTING_PID=$(cat "$PID_FILE")
    if ps -p $EXISTING_PID > /dev/null 2>&1; then
        echo "⚠️  已有监控进程在运行 (PID: $EXISTING_PID)"
        echo "💡 停止现有监控: kill $EXISTING_PID"
        echo "💡 查看监控日志: tail -f ./logs/gpu_monitor_*.log"
        exit 1
    else
        echo "🧹 清理过期的PID文件"
        rm -f "$PID_FILE"
    fi
fi

# 给监控脚本添加执行权限
chmod +x monitor_gpu_and_train.sh

# 后台运行监控脚本
echo "🚀 启动GPU监控..."
nohup ./monitor_gpu_and_train.sh > ./logs/gpu_monitor_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 获取监控进程PID
MONITOR_PID=$!
echo "✅ GPU监控已启动 (PID: $MONITOR_PID)"
echo "📝 日志文件: ./logs/gpu_monitor_$(date +%Y%m%d_%H%M%S).log"
echo ""

echo "💡 监控命令:"
echo "  - 查看监控状态: ps aux | grep $MONITOR_PID"
echo "  - 查看实时日志: tail -f ./logs/gpu_monitor_*.log"
echo "  - 停止监控: kill $MONITOR_PID"
echo "  - 查看GPU状态: nvidia-smi"
echo ""

echo "⏰ 监控配置:"
echo "  - 检查间隔: 5分钟"
echo "  - 最大等待: 24小时"
echo "  - GPU内存阈值: 4GB"
echo "  - 目标GPU: CUDA:1"
echo "  - 自动训练: 当CUDA:1可用时"
echo ""

echo "🎯 监控将自动:"
echo "  1. 每5分钟检查CUDA:1状态"
echo "  2. 当CUDA:1内存≥4GB时自动开始训练"
echo "  3. 训练完成后自动停止监控"
echo "  4. 24小时后如果CUDA:1仍不可用则停止" 