#!/bin/bash
# 停止所有正在运行的生成任务
# 用法: ./auto_stop_generate.sh

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 进程ID存储文件
PID_FILE="$SCRIPT_DIR/.generate_pids.txt"

echo "=========================================="
echo "停止所有生成任务"
echo "=========================================="
echo ""

# 检查PID文件是否存在
if [ ! -f "$PID_FILE" ]; then
    echo "警告: PID文件不存在 ($PID_FILE)"
    echo "尝试通过进程名查找并终止..."

    # 通过进程名查找并终止
    PIDS=$(pgrep -f "generate_rct_bias_result|generate_evidence")
    if [ -z "$PIDS" ]; then
        echo "未找到运行中的任务进程"
        exit 0
    else
        echo "找到以下进程: $PIDS"
        echo "$PIDS" | xargs kill -TERM 2>/dev/null
        sleep 2
        # 如果还有进程，强制终止
        REMAINING=$(pgrep -f "generate_rct_bias_result|generate_evidence")
        if [ -n "$REMAINING" ]; then
            echo "强制终止剩余进程..."
            echo "$REMAINING" | xargs kill -9 2>/dev/null
        fi
        echo "所有进程已终止"
        exit 0
    fi
fi

# 从PID文件读取并终止进程
STOPPED=0
FAILED=0

while IFS='|' read -r pid task_type model_name start_time; do
    # 跳过空行和注释
    [[ -z "$pid" || "$pid" =~ ^# ]] && continue

    # 检查进程是否还在运行
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "[$pid] 停止 $task_type 任务 (模型: $model_name, 启动时间: $start_time)"
        kill -TERM "$pid" 2>/dev/null

        # 等待进程优雅退出
        sleep 2

        # 如果进程还在运行，强制终止
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "  -> 进程未响应，强制终止..."
            kill -9 "$pid" 2>/dev/null
            sleep 1
        fi

        # 再次检查
        if ! ps -p "$pid" > /dev/null 2>&1; then
            echo "  -> 已成功停止"
            ((STOPPED++))
        else
            echo "  -> 警告: 无法停止进程 $pid"
            ((FAILED++))
        fi
    else
        echo "[$pid] 进程已不存在 (模型: $model_name)"
    fi
done < "$PID_FILE"

echo ""
echo "=========================================="
echo "停止操作完成"
echo "  - 成功停止: $STOPPED 个进程"
if [ $FAILED -gt 0 ]; then
    echo "  - 失败: $FAILED 个进程"
fi
echo "=========================================="

# 再次检查是否还有相关进程
REMAINING=$(pgrep -f "generate_rct_bias_result|generate_evidence")
if [ -n "$REMAINING" ]; then
    echo ""
    echo "警告: 仍有相关进程在运行: $REMAINING"
    echo "是否强制终止这些进程? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "$REMAINING" | xargs kill -9 2>/dev/null
        echo "已强制终止所有剩余进程"
    fi
else
    echo ""
    echo "确认: 所有相关进程已停止"
fi

# 清空PID文件
> "$PID_FILE"
echo ""
echo "PID文件已清空"
