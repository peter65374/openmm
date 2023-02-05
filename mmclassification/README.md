# mmclassification practice

version: mmclassification 0.25, mmcv=1.7.0, openmm-1.0

computing resources: 北京超算中心

## training scripts
data file pre-processing script: split_data.py

training config script: resnet18_b32_flower.py

base checkpoint imagenet-1k resnet18: https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth

training command script: run.sh

北京超算job提交： sbatch --gpus=1 run.sh

## output & logs
top-1 accuracy = 96.6783

The ouput model checkpoint file: epoch83.pth

the running log: 20230205_202511.log

the 100 epoches accuracy result log in json: 20230205_202511.log.json
