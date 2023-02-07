# mmclassification practice

version: mmclassification 0.25, mmcv=1.7.0, openmm-1.0

computing resources: 北京超算中心


# Assignment-1
## training scripts
data file pre-processing script: split_data.py

training config script: resnet18_b32_flower.py

base checkpoint imagenet-1k resnet18: https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth

training command script: run_codecamp_resnet18_flower.sh

北京超算job提交： sbatch --gpus=1 run_codecamp_resnet18_flower.sh

## output & logs
top-1 accuracy = 96.6783

The ouput model checkpoint file: epoch83.pth

the running log: 20230205_202511.log

the 100 epoches accuracy result log in json: 20230205_202511.log.json


# Assignment-2
## training scripts
training config script: resnet50_finetune_cifar10.py

base checkpoint imagenet-1k resnet50: https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth

training command script: run_codecamp_resnet50_cifar10.sh

北京超算job提交： sbatch --gpus=1 run_codecamp_resnet50_cifar10.sh

## output & logs
top-1 accuracy = 96.8
top-5 accuracy = 99.96

The ouput model checkpoint file: epoch29.pth
The scheduled max epoches = 200. since the accuracy is 96.8 already, the job was stopped mannually at epoch-33. And the epoch29.pth is the best checkpoints.

the running log: 20230207_185746.log

the 100 epoches accuracy result log in json: 20230207_185746.log.json
