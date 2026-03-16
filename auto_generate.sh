#!/bin/bash
# 自动批量执行RCT偏倚评估和证据提取脚本
# 用法:
#   ./auto_generate.sh              # 使用默认模式
#   MODE=cot ./auto_generate.sh     # 使用CoT模式
#   WORKER=4 MODE=cot ./auto_generate.sh  # 指定线程数和模式
#   支持的模式: default, cot, agent, tool

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 进程ID存储文件
PID_FILE="$SCRIPT_DIR/.generate_pids.txt"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# 清空旧的PID文件
> "$PID_FILE"

# Worker 线程数（默认 2）
WORKER=${WORKER:-2}

# 模式选择（默认 default，可选：default, cot, agent, tool）
MODE=${MODE:-default}

# 模型列表（优先从文件读取，如果文件不存在则使用默认列表）
MODELS_FILE="$SCRIPT_DIR/models.txt"

if [ -f "$MODELS_FILE" ]; then
    echo "从文件读取模型列表: $MODELS_FILE"
    # 读取文件，过滤掉注释和空行
    MODELS=()
    while IFS= read -r line || [ -n "$line" ]; do
        # 去除前后空格
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        # 跳过空行和注释行
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -n "$line" ]] && MODELS+=("$line")
    done < "$MODELS_FILE"
    echo "读取到 ${#MODELS[@]} 个模型"
else
    echo "模型列表文件不存在，使用默认模型列表"
    # 直接定义模型列表
    MODELS=(
        "openai/gpt-4.1"
        "openai/gpt-5.1"
        "qwen-plus"
    )
fi

echo "=========================================="
echo "开始批量执行模型评估任务"
echo "模型列表: ${MODELS[*]}"
echo "模式: $MODE"
echo "Worker 线程数: $WORKER"
echo "日志目录: $LOG_DIR"
echo "PID文件: $PID_FILE"
echo "=========================================="
echo ""

# 遍历每个模型
for model in "${MODELS[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始处理模型: $model"

    # 生成安全的日志文件名（替换特殊字符）
    safe_model_name=$(echo "$model" | sed 's/[\/\\]/_/g')

    # 执行 generate_rct_bias_result.py
    echo "  -> 启动 RCT 偏倚评估任务..."
    nohup uv run python evaluate/generate_rct_bias_result.py --model-name "$model" --mode "$MODE" --max-workers "$WORKER" \
        > "$LOG_DIR/rct_${safe_model_name}_${MODE}.log" 2>&1 &
    RCT_PID=$!
    echo "$RCT_PID|rct|$model|$(date '+%Y-%m-%d %H:%M:%S')" >> "$PID_FILE"
    echo "    RCT 任务 PID: $RCT_PID, 日志: $LOG_DIR/rct_${safe_model_name}.log"

    # 等待一段时间再启动下一个任务（可选，避免资源竞争）
    sleep 2

    # 执行 generate_evidence.py
    echo "  -> 启动证据提取任务..."
    nohup uv run python evaluate/generate_evidence.py --model-name "$model" --mode "$MODE" --max-workers "$WORKER" \
        > "$LOG_DIR/evidence_${safe_model_name}_${MODE}.log" 2>&1 &
    EVIDENCE_PID=$!
    echo "$EVIDENCE_PID|evidence|$model|$(date '+%Y-%m-%d %H:%M:%S')" >> "$PID_FILE"
    echo "    证据提取任务 PID: $EVIDENCE_PID, 日志: $LOG_DIR/evidence_${safe_model_name}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 模型 $model 的两个任务已启动"
    echo ""

    # 等待一段时间再处理下一个模型（可选）
    sleep 3
done

echo "=========================================="
echo "所有任务已启动！"
echo "=========================================="
echo ""
echo "监控命令:"
echo "  - 查看所有进程: ps aux | grep -E 'generate_rct_bias_result|generate_evidence'"
echo "  - 查看PID文件: cat $PID_FILE"
echo "  - 查看日志: tail -f $LOG_DIR/*.log"
echo "  - 停止所有任务: ./auto_stop_generate.sh"
echo ""
echo "实时监控日志（按 Ctrl+C 退出）:"
echo "  tail -f $LOG_DIR/*.log"
