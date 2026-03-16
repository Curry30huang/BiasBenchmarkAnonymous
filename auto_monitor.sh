#!/bin/bash
# 监控所有正在运行的生成任务
# 用法: ./auto_monitor.sh [--follow]

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 进程ID存储文件
PID_FILE="$SCRIPT_DIR/.generate_pids.txt"
LOG_DIR="$SCRIPT_DIR/logs"

# 检查是否使用 --follow 参数
FOLLOW_MODE=false
if [[ "$1" == "--follow" ]]; then
    FOLLOW_MODE=true
fi

echo "=========================================="
echo "任务监控面板"
echo "=========================================="
echo ""

# 检查PID文件
if [ ! -f "$PID_FILE" ]; then
    echo "警告: PID文件不存在 ($PID_FILE)"
    echo "尝试通过进程名查找..."
    PIDS=$(pgrep -f "generate_rct_bias_result|generate_evidence")
    if [ -z "$PIDS" ]; then
        echo "未找到运行中的任务进程"
        exit 0
    fi
else
    # 显示PID文件中的任务
    echo "从PID文件读取的任务:"
    echo "----------------------------------------"
    printf "%-8s %-12s %-30s %-20s\n" "PID" "任务类型" "模型名称" "启动时间"
    echo "----------------------------------------"

    ACTIVE_COUNT=0
    while IFS='|' read -r pid task_type model_name start_time; do
        [[ -z "$pid" || "$pid" =~ ^# ]] && continue

        if ps -p "$pid" > /dev/null 2>&1; then
            printf "%-8s %-12s %-30s %-20s [运行中]\n" "$pid" "$task_type" "$model_name" "$start_time"
            ((ACTIVE_COUNT++))
        else
            printf "%-8s %-12s %-30s %-20s [已结束]\n" "$pid" "$task_type" "$model_name" "$start_time"
        fi
    done < "$PID_FILE"

    echo "----------------------------------------"
    echo "活跃任务数: $ACTIVE_COUNT"
    echo ""
fi

# 显示系统进程信息
echo "系统进程信息:"
echo "----------------------------------------"
ps aux | grep -E "generate_rct_bias_result|generate_evidence" | grep -v grep | awk '{printf "PID: %-8s CPU: %-6s MEM: %-6s %s\n", $2, $3"%", $4"%", substr($0, index($0,$11))}'
echo ""

# 显示日志文件信息
if [ -d "$LOG_DIR" ]; then
    echo "日志文件:"
    echo "----------------------------------------"
    for log_file in "$LOG_DIR"/*.log; do
        if [ -f "$log_file" ]; then
            size=$(du -h "$log_file" | cut -f1)
            lines=$(wc -l < "$log_file" 2>/dev/null || echo "0")
            echo "$(basename "$log_file"): $size, $lines 行"
        fi
    done
    echo ""
fi

# 如果使用 --follow 模式，持续监控
if [ "$FOLLOW_MODE" = true ]; then
    echo "=========================================="
    echo "实时监控模式 (按 Ctrl+C 退出)"
    echo "=========================================="
    echo ""

    # 使用 tail -f 监控所有日志文件
    if [ -d "$LOG_DIR" ] && [ -n "$(ls -A "$LOG_DIR"/*.log 2>/dev/null)" ]; then
        tail -f "$LOG_DIR"/*.log
    else
        echo "未找到日志文件，退出监控模式"
    fi
else
    echo "提示: 使用 './auto_monitor.sh --follow' 进入实时监控模式"
fi
