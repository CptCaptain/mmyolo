custom_imports = dict(imports=['finetunecopypaste'], allow_failed_imports=False)

_base_ = './yolov8_s_syncbn_fast_8xb16-300e_coco.py'

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               # 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
               'vase', 'scissors', 'teddy bear', 'Audi_A7', 'Audi_RS_6_Avant')

pre_transform = _base_.pre_transform
albu_train_transform = _base_.albu_train_transform
last_transform = _base_.last_transform

mosaic_affine_transform = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))
]

fcp_albu_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='Resize',
                width=image_size[1],
                height=image_size[0],
            ),
            dict(
                type='RandomResizedCrop',
                width=image_size[1],
                height=image_size[0],
            ),
        ],
        p=1.0
    ),
    dict(
        type='ShiftScaleRotate',
        scale_limit=[-0.9,0.0],
        rotate_limit=0,
        border_mode=0,
        rotate_method='ellipse',
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=128,
                sat_shift_limit=128,
                val_shift_limit=128,
                p=1.0),
        ],
        p=0.9),
    dict(type='ImageCompression', quality_lower=45, quality_upper=95, p=0.3),
    dict(type='ChannelShuffle', p=0.4),
    dict(type='GaussNoise', p=0.8),
    dict(
        type='OneOf',
        transforms=[
            dict(type='AdvancedBlur', p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0),
        ],
        p=0.4),
    dict(
        type='ColorJitter',
        ),
    dict(
        type='RandomCrop',
        width=400,
        height=400,
        p=0.2,
        ),
    dict(
        type='Flip',
        ),
    dict(
        type='Resize',
        width=image_size[1],
        height=image_size[0],
        ),
    dict(
        type='PadIfNeeded',
        min_width=image_size[1],
        min_height=image_size[0],
        border_mode=0,
        p=1.0,
        ),
]

fcp_pipeline = [
    dict(
        type='FineTuneCopyPaste', 
        max_num_pasted=100,
        copy_paste_chance=0.8,
        supl_dataset_cfg=dict(
            ann_file='/home/nils/datasets/cars/coco/train.json',
            data_root='/home/nils/datasets/cars/',
            img_prefix='raw',
            pipeline=[
                dict(type='LoadImageFromFile', file_client_args=file_client_args),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(
                    type='Albu',
                    transforms=fcp_albu_transforms,
                    bbox_params=dict(
                        type='BboxParams',
                        format='pascal_voc',
                        label_fields=['gt_labels'],
                        min_visibility=0.0,
                        filter_lost_elements=True,
                    ),
                    update_pad_shape=False,
                    skip_img_without_anno=True,
                ),
            ],
            classes=CLASSES,
        ),
    ),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

train_pipeline = [
    *pre_transform, *fcp_pipeline, *mosaic_affine_transform,
    dict(
        type='YOLOv5MixUp',
        prob=mixup_ratio,
        pre_transform=[*pre_transform, *fcp_pipeline, *mosaic_affine_transform]),
    *last_transform
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

train_pipeline_stage2 = [
    *pre_transform, *fcp_pipeline, 
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=100,
        border_val=(114, 114, 114)), *last_transform
]

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=_base_.max_epochs - 10,
        switch_pipeline=train_pipeline_stage2)
]
