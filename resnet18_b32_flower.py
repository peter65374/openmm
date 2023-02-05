# resnet18, flower classification training open-mmlab config file
# model file addr: https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth
# base config: https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb32_in1k.py
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
    '../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
# _base_ = ['../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py', '../_base_/default_runtime.py']

# model settings
model = dict(
    head=dict(
        num_classes=5,
        topk = (1,) 
    ))

data = dict(
    samples_per_gpu = 32, # batchsize
    workers_per_gpu = 2, # worker数量
    train = dict(
        data_prefix = 'data/flower_dataset/train',
        ann_file = 'data/flower_dataset/train.txt',
        classes = 'data/flower_dataset/classes.txt'
    ),
    val = dict(
        data_prefix = 'data/flower_dataset/val',
        ann_file = 'data/flower_dataset/val.txt',
        classes = 'data/flower_dataset/classes.txt'
    ),
)

# 定义评估方法
evaluation = dict(metric_options={'topk': (1, )})

# 优化器
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001) 
optimizer_config = dict(grad_clip=None)

# 学习率策略
lr_config = dict(
    policy='step',
    step=[1])
runner = dict(type='EpochBasedRunner', max_epochs=100)

# 预训练模型
# load_from = '/HOME/shenpg/run/openmmlab/mmclassification/checkpoints/resnet18_batch256_imagenet_20200708-34ab8f90.pth' 
load_from = '/HOME/scz0ap9/run/mmcls025/mmclassification/checkpoints/resnet18_8xb32_in1k_20210831-fbbb1da6.pth'
