dataset_type = 'CocoDataset'
data_root = './datasets/ZSOR'
novels_path = 'datasets/ZSOR/test_data_A/Novels'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
class_names = ('脚踏', '防晒霜', '桔子', '核桃', '摆件', '眼镜', '喷雾', '护肤品', '钱包', '娃娃',
               '面霜', '面包', '咸鸭蛋', '盒子', '充电器', '双面胶', '刮胡刀', '眉笔', '玩具', '秒表',
               '手机壳', '口红', '饮料', '罐子', '梳子', '手机盒', '充气柱', '包', '体温枪', '雨伞',
               '速溶咖啡', '饼干', '收音机', '感冒灵', '硬币', '耳机', '盖子', '发卡', '纽扣', '薯片',
               '手机', '滤水网', '水乳', '易拉罐', '粽子', '钥匙', '相册', '洗发露', '棒棒糖', '矿泉水',
               '耳机盒', '酒瓶', '显卡', '艾灸贴', '火腿肠', '沐浴露', '牙膏', '杯子', '笔', '螺丝',
               '手机袋', '挂件', '洗脸巾', '瓶子', '笔筒', '报警灯', '打火机', '衣服', '眼镜盒', '面膜',
               '纸巾盒', '奥特曼', '啫喱水', '牙刷', '瓜子', '鼠标', '牛奶', '花', '零食', '手机支架',
               '档案袋', '夹子', '螺丝刀', '美妆蛋', '水管', '烟盒', '蓝牙音响', '手表', '订书机',
               '皮带扣', '枣子', '风扇', '洗面奶', '口香糖', '钳子', '手串', '印章', '便签')
novels_names = ('novels0', 'novels1', 'novels2', 'novels3', 'novels4',
                'novels5', 'novels6', 'novels7', 'novels8', 'novels9',
                'novels10', 'novels11')
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', img_scale=(448, 448), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(448, 448),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type='CocoDataset',
        anno_file='./datasets/ZSOR/train_annos_small.json',
        img_prefix='./datasets/ZSOR/train_data/Images',
        classes=('脚踏', '防晒霜', '桔子', '核桃', '摆件', '眼镜', '喷雾', '护肤品', '钱包', '娃娃',
                 '面霜', '面包', '咸鸭蛋', '盒子', '充电器', '双面胶', '刮胡刀', '眉笔', '玩具',
                 '秒表', '手机壳', '口红', '饮料', '罐子', '梳子', '手机盒', '充气柱', '包', '体温枪',
                 '雨伞', '速溶咖啡', '饼干', '收音机', '感冒灵', '硬币', '耳机', '盖子', '发卡',
                 '纽扣', '薯片', '手机', '滤水网', '水乳', '易拉罐', '粽子', '钥匙', '相册', '洗发露',
                 '棒棒糖', '矿泉水', '耳机盒', '酒瓶', '显卡', '艾灸贴', '火腿肠', '沐浴露', '牙膏',
                 '杯子', '笔', '螺丝', '手机袋', '挂件', '洗脸巾', '瓶子', '笔筒', '报警灯', '打火机',
                 '衣服', '眼镜盒', '面膜', '纸巾盒', '奥特曼', '啫喱水', '牙刷', '瓜子', '鼠标',
                 '牛奶', '花', '零食', '手机支架', '档案袋', '夹子', '螺丝刀', '美妆蛋', '水管',
                 '烟盒', '蓝牙音响', '手表', '订书机', '皮带扣', '枣子', '风扇', '洗面奶', '口香糖',
                 '钳子', '手串', '印章', '便签'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
            dict(type='Resize', img_scale=(448, 448), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        anno_file='./datasets/ZSOR/val_annos.json',
        img_prefix='./datasets/ZSOR/train_data/Images',
        classes=('脚踏', '防晒霜', '桔子', '核桃', '摆件', '眼镜', '喷雾', '护肤品', '钱包', '娃娃',
                 '面霜', '面包', '咸鸭蛋', '盒子', '充电器', '双面胶', '刮胡刀', '眉笔', '玩具',
                 '秒表', '手机壳', '口红', '饮料', '罐子', '梳子', '手机盒', '充气柱', '包', '体温枪',
                 '雨伞', '速溶咖啡', '饼干', '收音机', '感冒灵', '硬币', '耳机', '盖子', '发卡',
                 '纽扣', '薯片', '手机', '滤水网', '水乳', '易拉罐', '粽子', '钥匙', '相册', '洗发露',
                 '棒棒糖', '矿泉水', '耳机盒', '酒瓶', '显卡', '艾灸贴', '火腿肠', '沐浴露', '牙膏',
                 '杯子', '笔', '螺丝', '手机袋', '挂件', '洗脸巾', '瓶子', '笔筒', '报警灯', '打火机',
                 '衣服', '眼镜盒', '面膜', '纸巾盒', '奥特曼', '啫喱水', '牙刷', '瓜子', '鼠标',
                 '牛奶', '花', '零食', '手机支架', '档案袋', '夹子', '螺丝刀', '美妆蛋', '水管',
                 '烟盒', '蓝牙音响', '手表', '订书机', '皮带扣', '枣子', '风扇', '洗面奶', '口香糖',
                 '钳子', '手串', '印章', '便签'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(448, 448),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        anno_file='./datasets/ZSOR/test_annos.json',
        img_prefix='./datasets/ZSOR/test_data_A/Images',
        classes=('novels0', 'novels1', 'novels2', 'novels3', 'novels4',
                 'novels5', 'novels6', 'novels7', 'novels8', 'novels9',
                 'novels10', 'novels11'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(448, 448),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='MobileNetV2',
        out_indices=(1, 2, 4, 7),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        init_cfg=None),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 1280],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=98,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                unknown_top_k=1,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            unknown_weight=-1,
            debug=False,
            begin_clustering_loss_iter=10,
            update_every=10,
            bbox_head=dict(
                momentum=0.9, memorySize=5, clustering_loss_margin=10))),
    val_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            bbox_head=dict(isNovels=True, novel_class_num=12))))
work_dir = './work_dirs/faster_rcnn_mobilenetv2_fpn_baseline_ZSOR'
gpu_ids = [0]
