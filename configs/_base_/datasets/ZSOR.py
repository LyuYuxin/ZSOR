#ZSOR dataset settings.
#written by lyx.
# dataset settings
dataset_type = 'CocoDataset'# 数据集类型，这将被用来定义数据集。
data_root = './datasets/ZSOR'# 数据的根路径。
novels_path = 'datasets/ZSOR/test_data_A/Novels'
img_norm_cfg = dict(#图像归一化配置，用来归一化输入的图像。
    mean=[123.675, 116.28, 103.53], # image net 均值、
    std=[58.395, 57.12, 57.375],# image net标准差。
    to_rgb=True)#  预训练里用于预训练主干网络的图像的通道顺序。

class_names = open('./datasets/ZSOR/class_names.txt', 'r').readlines()    
class_names = tuple([name.rstrip() for name in class_names])

novels_names = open('./datasets/ZSOR/novels_names.txt', 'r').readlines()    
novels_names = tuple([name.rstrip() for name in novels_names])
train_pipeline = [# 训练时数据处理流程
    # 1.从文件路径里加载图像。
    dict(type='LoadImageFromFile'),
    
    # 2.对于当前图像，加载它的注释信息
    dict(type='LoadAnnotations', 
    with_bbox=True,
    with_mask=False,
    ),
    # 3.变化图像和其注释大小的数据增广的流程。
    dict(type='Resize', 
    img_scale=(448, 448), # 图像的最大规模。
    keep_ratio=True),

    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),# 填充图像可以被当前值整除。
    dict(type='DefaultFormatBundle'),#流程里收集数据的默认格式捆。
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),# 从数据中选取特定key传递给检测器
]
test_pipeline = [
    # 1.从文件路径里加载图像。
    dict(type='LoadImageFromFile'),
    # 2. TTA
    dict(
        type='MultiScaleFlipAug',
        img_scale=(448, 448),#测试时最大图像size
        flip=False,#不翻转图像
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'), # flip=false时不被使用
            dict(type='Normalize', **img_norm_cfg),#归一化配置
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),#将图像转化为tensor
            dict(type='Collect', keys=['img']),#选取图片传入检测器
        ])
]
data = dict(
    samples_per_gpu=4,# 每个gpu的batch size
    workers_per_gpu=1,# 每个gpu的数据加载线程数
    train=dict(#训练数据配置
        type=dataset_type,
        anno_file=data_root + '/train_annos_small.json',
        img_prefix=data_root + '/train_data/' + 'Images',
        classes=class_names,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        anno_file=data_root + '/val_annos.json',
        img_prefix=data_root + '/train_data/' + 'Images',
        classes=class_names,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        anno_file=data_root + '/test_annos.json',
        img_prefix=data_root + '/test_data_A/' + 'Images',
        classes=novels_names,#类别为novels0,novels1 ……noveln-1
        pipeline=test_pipeline))
evaluation = dict(interval=1, #验证间隔 
metric='bbox') #验证时采用的指标类型
