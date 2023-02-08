# OpenMMLab code camp assignment
# mmclassification advanced assignment, train cifar10 dataset
# assignment URL: https://github.com/open-mmlab/OpenMMLabCamp/issues/16
# pretrained model checkpoint: https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth
'''
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
'''

_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/cifar10_bs16.py', '../_base_/default_runtime.py'
]


# model settings
model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/HOME/scz0ap9/run/mmcls025/mmclassification/checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=10),
)

# dataset setting, cifar10 is different from imagenet format. we need modify the dataset config.
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False,
)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Resize', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='Resize', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]

data = dict(
    samples_per_gpu=128,  # 单GPU的话，base dataset config是 batch=16, 所以要修改一下，以便和下面schedule的batch=128保持一致
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

# training schedule setting
# lr is set for a batch size of 128, 因为这里_base_ cifar10 schedule config是batch=128，所以前面dataset要一致修改一下，单GPU的话，如果8GPU并行，刚好16*8=128那就不需要。
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[15])
runner = dict(type='EpochBasedRunner', max_epochs=200)
log_config = dict(interval=100)

# Load下载到本地的预训练模型
# load_from = '/HOME/scz0ap9/run/mmcls025/mmclassification/checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth'

# start training by CLI command for single GPU
# python tools/train.py configs/codecamp/resnet50_finetune_cifar.py --work-dir assignment/cifar10
