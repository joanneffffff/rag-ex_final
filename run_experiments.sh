#!/bin/bash

# 三种检索模式实验启动脚本
# 支持后台运行和完整的日志记录

set -e  # 遇到错误时退出

# 默认配置
DEFAULT_EVAL_DATA="data/alphafin/alphafin_eval_samples.jsonl"
DEFAULT_OUTPUT_DIR="experiment_results"
DEFAULT_LOG_DIR="experiment_logs"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "三种检索模式实验启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -d, --data PATH        评测数据文件路径 (默认: $DEFAULT_EVAL_DATA)"
    echo "  -o, --output DIR       输出目录 (默认: $DEFAULT_OUTPUT_DIR)"
    echo "  -l, --log-dir DIR      日志目录 (默认: $DEFAULT_LOG_DIR)"
    echo "  -b, --background       后台运行"
    echo "  -q, --quick            快速测试模式 (只运行小样本)"
    echo "  -p, --pid-file FILE    PID文件路径"
    echo "  -h, --help             显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 前台运行完整实验"
    echo "  $0 -b                                 # 后台运行完整实验"
    echo "  $0 -q                                 # 快速测试模式"
    echo "  $0 -b -p /tmp/experiment.pid         # 后台运行并保存PID"
    echo "  $0 -d custom_data.json -o results    # 使用自定义数据和输出目录"
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装"
        exit 1
    fi
    
    # 检查数据文件
    if [ ! -f "$EVAL_DATA_PATH" ]; then
        print_error "评测数据文件不存在: $EVAL_DATA_PATH"
        exit 1
    fi
    
    # 检查Python脚本
    if [ ! -f "run_three_mode_experiments.py" ]; then
        print_error "实验脚本不存在: run_three_mode_experiments.py"
        exit 1
    fi
    
    print_success "依赖检查通过"
}

# 创建目录
create_directories() {
    print_info "创建必要的目录..."
    
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$LOG_DIR"
    
    print_success "目录创建完成"
}

# 启动实验
start_experiment() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="$LOG_DIR/experiment_${timestamp}.log"
    local pid_file=""
    
    if [ "$PID_FILE" != "" ]; then
        pid_file="--pid_file $PID_FILE"
    fi
    
    print_info "启动实验..."
    print_info "评测数据: $EVAL_DATA_PATH"
    print_info "输出目录: $OUTPUT_DIR"
    print_info "日志文件: $log_file"
    
    if [ "$BACKGROUND" = true ]; then
        print_info "后台运行模式"
        nohup python3 run_three_mode_experiments.py \
            --eval_data_path "$EVAL_DATA_PATH" \
            --output_dir "$OUTPUT_DIR" \
            $pid_file \
            > "$log_file" 2>&1 &
        
        local experiment_pid=$!
        print_success "实验已启动，PID: $experiment_pid"
        
        if [ "$PID_FILE" != "" ]; then
            echo $experiment_pid > "$PID_FILE"
            print_info "PID已保存到: $PID_FILE"
        fi
        
        print_info "使用以下命令查看日志:"
        print_info "  tail -f $log_file"
        print_info "使用以下命令停止实验:"
        print_info "  kill $experiment_pid"
        
    else
        print_info "前台运行模式"
        python3 run_three_mode_experiments.py \
            --eval_data_path "$EVAL_DATA_PATH" \
            --output_dir "$OUTPUT_DIR" \
            $pid_file \
            2>&1 | tee "$log_file"
    fi
}

# 快速测试模式
run_quick_test() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="$LOG_DIR/quick_test_${timestamp}.log"
    
    print_info "启动快速测试模式..."
    print_info "日志文件: $log_file"
    
    if [ "$BACKGROUND" = true ]; then
        nohup python3 run_three_mode_experiments.py \
            --eval_data_path "$EVAL_DATA_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --quick_test \
            > "$log_file" 2>&1 &
        
        local test_pid=$!
        print_success "快速测试已启动，PID: $test_pid"
        print_info "使用以下命令查看日志:"
        print_info "  tail -f $log_file"
        
    else
        python3 run_three_mode_experiments.py \
            --eval_data_path "$EVAL_DATA_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --quick_test \
            2>&1 | tee "$log_file"
    fi
}

# 停止实验
stop_experiment() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        print_info "停止实验进程 (PID: $pid)..."
        
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            print_success "实验已停止"
        else
            print_warning "进程 $pid 不存在或已停止"
        fi
        
        rm -f "$PID_FILE"
    else
        print_error "PID文件不存在: $PID_FILE"
    fi
}

# 查看实验状态
show_status() {
    print_info "实验状态检查..."
    
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            print_success "实验正在运行 (PID: $pid)"
            
            # 显示进程信息
            ps -p "$pid" -o pid,ppid,cmd,etime
        else
            print_warning "实验进程不存在 (PID: $pid)"
        fi
    else
        print_info "未找到PID文件"
    fi
    
    # 显示最新的日志文件
    local latest_log=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        print_info "最新日志文件: $latest_log"
        print_info "最后10行日志:"
        tail -10 "$latest_log"
    fi
}

# 解析命令行参数
EVAL_DATA_PATH="$DEFAULT_EVAL_DATA"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
LOG_DIR="$DEFAULT_LOG_DIR"
BACKGROUND=false
QUICK_TEST=false
PID_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data)
            EVAL_DATA_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -l|--log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -b|--background)
            BACKGROUND=true
            shift
            ;;
        -q|--quick)
            QUICK_TEST=true
            shift
            ;;
        -p|--pid-file)
            PID_FILE="$2"
            shift 2
            ;;
        --stop)
            if [ -z "$PID_FILE" ]; then
                print_error "停止实验需要指定PID文件 (使用 -p 选项)"
                exit 1
            fi
            stop_experiment
            exit 0
            ;;
        --status)
            if [ -z "$PID_FILE" ]; then
                print_error "查看状态需要指定PID文件 (使用 -p 选项)"
                exit 1
            fi
            show_status
            exit 0
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 主程序
main() {
    print_info "=" * 60
    print_info "三种检索模式实验启动脚本"
    print_info "=" * 60
    
    # 检查依赖
    check_dependencies
    
    # 创建目录
    create_directories
    
    # 启动实验
    if [ "$QUICK_TEST" = true ]; then
        run_quick_test
    else
        start_experiment
    fi
    
    print_success "实验启动完成！"
}

# 运行主程序
main "$@" 