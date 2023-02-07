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
python tools/train.py configs/codecamp/resnet50_finetune_cifar10.py --work-dir assignment/cifar10

# 计算作业command
# sbatch --gpus=1 run_codecamp_resnet50_cifar10.sh
