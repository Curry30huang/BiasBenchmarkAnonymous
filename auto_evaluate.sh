#!/bin/bash
# 自动批量执行模型评估脚本
# 用法:
#   ./auto_evaluate.sh              # 使用默认模式
#   MODE=cot ./auto_evaluate.sh     # 使用CoT模式
#   支持的模式: default, cot, agent, tool

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 日志目录
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

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
echo "日志目录: $LOG_DIR"
echo "=========================================="
echo ""

# 定义评估脚本列表
EVALUATE_SCRIPTS=(
    "evaluate/evaluate_aggregation_consistency.py"
    "evaluate/evaluate_atomic_consistency.py"
    "evaluate/evaluate_domain_consistency.py"
    "evaluate/evaluate_evidence_result.py"
    "evaluate/evaluate_three_consistency.py"
)

# 遍历每个模型
for model in "${MODELS[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始处理模型: $model"

    # 生成安全的日志文件名（替换特殊字符）
    safe_model_name=$(echo "$model" | sed 's/[\/\\]/_/g')

    # 遍历每个评估脚本
    for script in "${EVALUATE_SCRIPTS[@]}"; do
        # 从脚本路径提取脚本名称（不含路径和扩展名）
        script_basename=$(basename "$script" .py)

        # 生成日志文件名
        log_file="$LOG_DIR/evaluate_${script_basename}_${safe_model_name}_${MODE}.log"

        echo "  -> 执行评估脚本: $script"
        echo "     日志文件: $log_file"

        # 执行评估脚本（不后台运行，顺序执行）
        uv run python "$script" --model-name "$model" --mode "$MODE" \
            > "$log_file" 2>&1

        # 检查执行结果
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "     ✓ 评估脚本执行成功: $script"
        else
            echo "     ✗ 评估脚本执行失败: $script (退出码: $exit_code)"
            echo "     请查看日志文件: $log_file"
        fi

        echo ""
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 模型 $model 的所有评估任务已完成"
    echo ""
done

echo "=========================================="
echo "所有评估任务已完成！"
echo "=========================================="
echo ""
echo "日志文件位置: $LOG_DIR"
echo "  - 聚合一致性评估: evaluate_evaluate_aggregation_consistency_*_${MODE}.log"
echo "  - 原子一致性评估: evaluate_evaluate_atomic_consistency_*_${MODE}.log"
echo "  - 模块一致性评估: evaluate_evaluate_domain_consistency_*_${MODE}.log"
echo "  - 证据结果评估: evaluate_evaluate_evidence_result_*_${MODE}.log"
echo "  - 三种一致性端到端评估: evaluate_evaluate_three_consistency_*_${MODE}.log"
echo ""
echo "查看日志命令:"
echo "  tail -f $LOG_DIR/evaluate_*.log"
