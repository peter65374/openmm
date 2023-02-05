#!/bin/bash
# 加载模块
module load anaconda/2021.05
module load cuda/11.3
module load gcc/7.3

# 激活conda env
source activate openmm1

# 缓存日志刷新
export PYTHONUNBUFFERED=1

# 模型训练
python tools/train.py \
    configs/resnet/resnet18_b32_flower.py \
    --work-dir assignment/flower
