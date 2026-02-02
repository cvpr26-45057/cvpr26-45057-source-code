#!/bin/bash

# 内存优化训练启动脚本
# 用于解决PyTorch显存管理问题

echo "========================================"
echo "开始内存优化的RelTRv3训练"
echo "========================================"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1  # 用于调试CUDA内存问题

# 激活conda环境
source /mnt/ShareDB_6TB/baitianyou/miniconda3/bin/activate pid_rec

# 进入工作目录
cd /mnt/ShareDB_6TB/baitianyou/RelTR-main

echo "检查GPU状态..."
nvidia-smi

echo ""
echo "检查PyTorch环境..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'GPU数量: {torch.cuda.device_count()}')
print(f'GPU名称: {torch.cuda.get_device_name(0)}')

# 清理GPU缓存
torch.cuda.empty_cache()
import gc
gc.collect()

# 检查内存
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
allocated = torch.cuda.memory_allocated(0) / 1024**3
reserved = torch.cuda.memory_reserved(0) / 1024**3
print(f'GPU总内存: {total:.2f}GB')
print(f'已分配: {allocated:.2f}GB')
print(f'已保留: {reserved:.2f}GB')
print(f'可用: {total-reserved:.2f}GB')
"

echo ""
echo "========================================"
echo "启动训练 (1000实体 + 2000三元组 = 3000查询)"
echo "========================================"

# 运行内存优化的训练脚本
python train_memory_optimized.py

echo ""
echo "训练完成或中断"
echo "最终GPU状态:"
nvidia-smi
