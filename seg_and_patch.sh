#!/bin/bash

# seg_and_patch.sh - WSI分割和打补丁处理脚本
# 使用方法: ./seg_and_patch.sh [数据集名称]
# 例如: ./seg_and_patch.sh GC_2300

# =============================================================================
# 数据集配置 - 在这里定义所有数据集
# =============================================================================

declare -A DATASETS
DATASETS["GC_2300"]="/mnt/usb4/jijianxin/1.GC_2300/WSI|/mnt/usb4/jijianxin/gc_coord/GC_2300"
DATASETS["GC_269"]="/mnt/usb4/jijianxin/3.HMU_GC_269/HMU-GC|/mnt/usb4/jijianxin/gc_coord/GC_269"
DATASETS["GC_400"]="/mnt/usb4/jijianxin/GC_补充|/mnt/usb4/jijianxin/gc_coord/GC_400"
DATASETS["TCGA_STAD"]="/mnt/usb4/jijianxin/TCGA_STAD|/mnt/usb4/jijianxin/gc_coord/TCGA_STAD"

# =============================================================================
# 处理命令行参数
# =============================================================================

if [ $# -eq 0 ]; then
    echo "=========================================="
    echo "WSI 分割和打补丁处理脚本"
    echo "=========================================="
    echo "使用方法: $0 [数据集名称]"
    echo ""
    echo "可用的数据集:"
    for dataset in "${!DATASETS[@]}"; do
        IFS='|' read -r source_dir output_dir <<< "${DATASETS[$dataset]}"
        echo "  $dataset"
        echo "    源目录: $source_dir"
        echo "    输出目录: $output_dir"
        echo ""
    done
    echo "示例: $0 GC_2300"
    echo "示例: $0 TCGA_STAD"
    exit 1
fi

DATASET_NAME="$1"

# 修复: 使用兼容 bash 4.2 的方式检查键是否存在
if [ -z "${DATASETS[$DATASET_NAME]+x}" ]; then
    echo "❌ 错误: 未知的数据集 '$DATASET_NAME'"
    echo "可用的数据集: ${!DATASETS[*]}"
    exit 1
fi

# 解析数据集配置
IFS='|' read -r SOURCE_DIR OUTPUT_DIR <<< "${DATASETS[$DATASET_NAME]}"

echo "=========================================="
echo "WSI 分割和打补丁处理脚本"
echo "当前时间: $(date)"
echo "处理数据集: $DATASET_NAME"
echo "=========================================="

# =============================================================================
# 其他配置参数
# =============================================================================

PYTHON_SCRIPT="create_patches_fp.py"

# 处理选项
ENABLE_SEG=true
ENABLE_PATCH=true
ENABLE_STITCH=true

# Patch参数
PATCH_SIZE=256
STEP_SIZE=256
AUTO_ADJUST_PATCH_SIZE=true
BASE_PATCH_SIZE_20X=512
PATCH_LEVEL=0

# 放大倍率检测相关 - 为每个数据集创建独立的日志文件
SAVE_UNREADABLE_PATHS=true
UNREADABLE_LOG_FILE="unreadable_magnification_${DATASET_NAME}.txt"  # 添加数据集名称前缀

# 其他选项
PROCESS_LIST=""
PRESET=""

# =============================================================================
# 脚本主体（保持原有逻辑）
# =============================================================================

# 检查必要的参数
if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ 错误: 源目录不存在: $SOURCE_DIR"
    echo "请检查数据集配置"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ 错误: Python脚本不存在: $PYTHON_SCRIPT"
    echo "请确保脚本文件在正确位置"
    exit 1
fi

# 检查源目录中是否有WSI文件
WSI_COUNT=$(find "$SOURCE_DIR" -type f \( -name "*.svs" -o -name "*.ndpi" -o -name "*.tiff" -o -name "*.tif" -o -name "*.mrxs" \) | wc -l)
if [ $WSI_COUNT -eq 0 ]; then
    echo "⚠️  警告: 在源目录中未找到WSI文件"
    echo "支持的格式: .svs, .ndpi, .tiff, .tif, .mrxs"
    read -p "是否继续执行? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消执行"
        exit 1
    fi
else
    echo "✅ 找到 $WSI_COUNT 个WSI文件"
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 显示配置信息
echo ""
echo "配置信息:"
echo "  数据集: $DATASET_NAME"
echo "  源目录: $SOURCE_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  分割: $ENABLE_SEG"
echo "  打补丁: $ENABLE_PATCH"
echo "  拼接: $ENABLE_STITCH"
echo "  自动调整patch大小: $AUTO_ADJUST_PATCH_SIZE"

if [ "$AUTO_ADJUST_PATCH_SIZE" = true ]; then
    echo "  自动调整模式:"
    echo "    20X基础大小: ${BASE_PATCH_SIZE_20X}×${BASE_PATCH_SIZE_20X}"
    echo "    40X将使用: $((BASE_PATCH_SIZE_20X * 2))×$((BASE_PATCH_SIZE_20X * 2))"
    echo "    10X将使用: $((BASE_PATCH_SIZE_20X / 2))×$((BASE_PATCH_SIZE_20X / 2))"
else
    echo "  固定patch大小: ${PATCH_SIZE}×${PATCH_SIZE}"
    echo "  固定步长: ${STEP_SIZE}×${STEP_SIZE}"
fi

echo "  保存无法读取放大倍率的文件: $SAVE_UNREADABLE_PATHS"
echo "  无法读取放大倍率日志文件: $UNREADABLE_LOG_FILE"
echo "=========================================="

# 构建命令
CMD="python $PYTHON_SCRIPT"
CMD="$CMD --source \"$SOURCE_DIR\""
CMD="$CMD --save_dir \"$OUTPUT_DIR\""
CMD="$CMD --patch_size $PATCH_SIZE"
CMD="$CMD --step_size $STEP_SIZE"
CMD="$CMD --base_patch_size_20x $BASE_PATCH_SIZE_20X"
CMD="$CMD --patch_level $PATCH_LEVEL"
CMD="$CMD --unreadable_log_file \"$UNREADABLE_LOG_FILE\""

# 添加布尔选项
if [ "$ENABLE_SEG" = true ]; then
    CMD="$CMD --seg"
fi

if [ "$ENABLE_PATCH" = true ]; then
    CMD="$CMD --patch"
fi

if [ "$ENABLE_STITCH" = true ]; then
    CMD="$CMD --stitch"
fi

if [ "$AUTO_ADJUST_PATCH_SIZE" = true ]; then
    CMD="$CMD --auto_adjust_patch_size"
else
    CMD="$CMD --no_auto_adjust_patch_size"
fi

if [ "$SAVE_UNREADABLE_PATHS" = true ]; then
    CMD="$CMD --save_unreadable_paths"
fi

# 添加可选参数
if [ -n "$PROCESS_LIST" ] && [ -f "$PROCESS_LIST" ]; then
    CMD="$CMD --process_list \"$PROCESS_LIST\""
fi

if [ -n "$PRESET" ] && [ -f "$PRESET" ]; then
    CMD="$CMD --preset \"$PRESET\""
fi

# 显示最终命令
echo "执行命令:"
echo "$CMD"
echo "=========================================="

# 显示预期的patch大小
if [ "$AUTO_ADJUST_PATCH_SIZE" = true ]; then
    echo "预期的patch大小调整:"
    echo "  10X及以下: $((BASE_PATCH_SIZE_20X / 2))×$((BASE_PATCH_SIZE_20X / 2))"
    echo "  20X: ${BASE_PATCH_SIZE_20X}×${BASE_PATCH_SIZE_20X}"
    echo "  40X及以上: $((BASE_PATCH_SIZE_20X * 2))×$((BASE_PATCH_SIZE_20X * 2))"
    echo "=========================================="
fi

# 最后确认
echo "即将开始处理数据集: $DATASET_NAME"
echo "按 Ctrl+C 可以中断..."
sleep 3

# 记录开始时间
START_TIME=$(date)
START_TIMESTAMP=$(date +%s)
echo "开始时间: $START_TIME"
echo "=========================================="

# 执行命令
eval $CMD

# 记录结束时间和结果
END_TIME=$(date)
END_TIMESTAMP=$(date +%s)
DURATION=$((END_TIMESTAMP - START_TIMESTAMP))
EXIT_CODE=$?

echo "=========================================="
echo "处理完成!"
echo "数据集: $DATASET_NAME"
echo "开始时间: $START_TIME"
echo "结束时间: $END_TIME"
echo "总耗时: $(($DURATION / 3600))小时 $((($DURATION % 3600) / 60))分钟 $(($DURATION % 60))秒"
echo "退出代码: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 处理成功完成"
    echo "输出文件位置: $OUTPUT_DIR"
    
    # 显示生成的文件统计
    if [ -d "$OUTPUT_DIR/patches" ]; then
        PATCH_COUNT=$(find "$OUTPUT_DIR/patches" -name "*.h5" | wc -l)
        echo "生成的patch文件数量: $PATCH_COUNT"
    fi
    
    if [ -d "$OUTPUT_DIR/masks" ]; then
        MASK_COUNT=$(find "$OUTPUT_DIR/masks" -name "*.jpg" | wc -l)
        echo "生成的mask文件数量: $MASK_COUNT"
    fi
    
    if [ -d "$OUTPUT_DIR/stitches" ]; then
        STITCH_COUNT=$(find "$OUTPUT_DIR/stitches" -name "*.jpg" -o -name "*.png" | wc -l)
        echo "生成的拼接图像数量: $STITCH_COUNT"
    fi
    
    # 检查磁盘空间使用情况
    OUTPUT_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    echo "输出目录大小: $OUTPUT_SIZE"
    
    # 检查无法读取放大倍率的文件 - 现在每个数据集有独立的日志文件
    if [ -f "$OUTPUT_DIR/$UNREADABLE_LOG_FILE" ]; then
        UNREADABLE_COUNT=$(grep -v "^#" "$OUTPUT_DIR/$UNREADABLE_LOG_FILE" | grep -v "^$" | wc -l)
        if [ $UNREADABLE_COUNT -gt 0 ]; then
            echo "⚠️  数据集 $DATASET_NAME 中无法读取放大倍率的文件数量: $UNREADABLE_COUNT"
            echo "详细信息请查看: $OUTPUT_DIR/$UNREADABLE_LOG_FILE"
        else
            echo "✅ 数据集 $DATASET_NAME 中所有文件的放大倍率都能正常读取"
        fi
    fi
    
    # 显示process_list_autogen.csv的信息
    if [ -f "$OUTPUT_DIR/process_list_autogen.csv" ]; then
        echo ""
        echo "📊 数据集 $DATASET_NAME 处理统计信息:"
        echo "详细处理记录请查看: $OUTPUT_DIR/process_list_autogen.csv"
        
        # 尝试显示放大倍率统计
        python3 -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('$OUTPUT_DIR/process_list_autogen.csv')
    print(f'数据集 $DATASET_NAME 总共处理文件数: {len(df)}')
    
    if 'magnification' in df.columns:
        mag_counts = df['magnification'].value_counts().sort_index()
        print('放大倍率分布:')
        for mag, count in mag_counts.items():
            print(f'  {mag}X: {count} 个文件')
    
    if 'adjusted_patch_size' in df.columns:
        patch_counts = df['adjusted_patch_size'].value_counts().sort_index()
        print('使用的patch大小分布:')
        for size, count in patch_counts.items():
            print(f'  {int(size)}×{int(size)}: {count} 个文件')
            
    if 'status' in df.columns:
        status_counts = df['status'].value_counts()
        print('处理状态分布:')
        for status, count in status_counts.items():
            print(f'  {status}: {count} 个文件')
            
except Exception as e:
    print(f'无法显示详细统计信息: {e}')
" 2>/dev/null || echo "无法显示详细统计信息(需要pandas)"
    fi
    
else
    echo "❌ 处理过程中出现错误"
    echo "请检查错误信息并重试"
    
    # 检查常见问题
    echo ""
    echo "常见问题排查:"
    echo "1. 检查Python环境和依赖包是否正确安装"
    echo "2. 检查源目录中的文件格式是否支持"
    echo "3. 检查磁盘空间是否充足"
    echo "4. 检查文件权限是否正确"
fi

echo "=========================================="
