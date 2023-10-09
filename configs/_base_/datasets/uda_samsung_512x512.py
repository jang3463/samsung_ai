# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
dataset_type = 'samsung'
data_root = 'data/samsung_source/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
crop_size = (512, 512)
gta_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='GaussNoise', prob = 0.2),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(960, 540)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 540),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='SourceDataset',
            data_root='data/samsung_total/',
            img_dir='images',
            ann_dir='labels',
            pipeline=gta_train_pipeline),
        target=dict(
            type='TargetDataset',
            data_root='data/samsung_target/',
            img_dir='images',
            # ann_dir='gtFine/train',
            ann_dir=None,
            pipeline=cityscapes_train_pipeline)),
    val=dict(
        type='SourceDataset',
        data_root='data/samsung_valid/',
        img_dir='images',
        ann_dir='labels',
        pipeline=test_pipeline),
    test=dict(
        type='TargetDataset',
        data_root='data/samsung_test/',
        img_dir='test_image',
        ann_dir=None,
        pipeline=test_pipeline)
        )
