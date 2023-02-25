import copy
import logging
import os

import torch

from hemat.torch.quantization import (
    HorizonCalibratedActivationFakeQuantize,
    HorizonCalibratedWeightFakeQuantize,
    HorizonCalibrationActivationFakeQuantize,
    HorizonCalibrationWeightFakeQuantize,
    HorizonPrepareCustomConfigDict,
)
import hat

training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "selfcoco_test"
num_classes = 5 
batch_size_per_gpu = 36
device_ids = [0, 1]
ckpt_dir = "./selfcoco_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
bn_kwargs = {}

# for model convert and compile
opt = "O3"
rt_input_type = "yuv444"
rt_input_layout = "NCHW"

model = dict(
    type="FCOS",
    backbone=dict(
        type="efficientnet",
        bn_kwargs=bn_kwargs,
        model_type="b0",
        num_classes=1000,
        include_top=False,
        activation="relu",
        use_se_block=False,
    ),
    neck=dict(
        type="BiFPN",
        in_strides=[2, 4, 8, 16, 32],
        out_strides=[8, 16, 32, 64, 128],
        stride2channels=dict({2: 16, 4: 24, 8: 40, 16: 112, 32: 320}),
        out_channels=64,
        num_outs=5,
        stack=3,
        start_level=2,
        end_level=-1,
        fpn_name="bifpn_sum",
    ),
    head=dict(
        type="FCOSHead",
        num_classes=num_classes,
        in_strides=[8, 16, 32, 64, 128],
        out_strides=[8, 16, 32, 64, 128],
        stride2channels=dict({8: 64, 16: 64, 32: 64, 64: 64, 128: 64}),
        upscale_bbox_pred=False,
        feat_channels=64,
        stacked_convs=4,
        int8_output=False,
        dequant_output=True,
        deepcopy_share_conv=True,
    ),
    targets=dict(
        type="DynamicFcosTarget",
        strides=[8, 16, 32, 64, 128],
        cls_out_channels=num_classes,
        background_label=num_classes,
        topK=10,
        loss_cls=dict(
            type="FocalLoss",
            loss_name="cls",
            num_classes=num_classes + 1,
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            reduction="none",
        ),
        loss_reg=dict(
            type="GIoULoss", loss_name="reg", loss_weight=2.0, reduction="none"
        ),
    ),
    post_process=dict(
        type="FCOSDecoder",
        num_classes=num_classes,
        strides=[8, 16, 32, 64, 128],
        nms_use_centerness=True,
        nms_sqrt=True,
        rescale=True,
        test_cfg=dict(
            score_thr=0.05,
            nms_pre=1000,
            nms=dict(name="nms", iou_threshold=0.6, max_per_img=100),
        ),
        upscale_bbox_pred=True,
    ),
    loss_cls=dict(
        type="FocalLoss",
        loss_name="cls",
        num_classes=num_classes + 1,
        alpha=0.25,
        gamma=2.0,
        loss_weight=1.0,
    ),
    loss_centerness=dict(
        type="CrossEntropyLossV2", loss_name="centerness", use_sigmoid=True
    ),
    loss_reg=dict(
        type="GIoULoss",
        loss_name="reg",
        loss_weight=1.0,
    ),
)

test_model = dict(
    type="FCOS",
    backbone=dict(
        type="efficientnet",
        bn_kwargs=bn_kwargs,
        model_type="b0",
        num_classes=1000,
        include_top=False,
        activation="relu",
        use_se_block=False,
    ),
    neck=dict(
        type="BiFPN",
        in_strides=[2, 4, 8, 16, 32],
        out_strides=[8, 16, 32, 64, 128],
        stride2channels=dict({2: 16, 4: 24, 8: 40, 16: 112, 32: 320}),
        out_channels=64,
        num_outs=5,
        stack=3,
        start_level=2,
        end_level=-1,
        fpn_name="bifpn_sum",
    ),
    head=dict(
        type="FCOSHead",
        num_classes=num_classes,
        in_strides=[8, 16, 32, 64, 128],
        out_strides=[8, 16, 32, 64, 128],
        stride2channels=dict({8: 64, 16: 64, 32: 64, 64: 64, 128: 64}),
        upscale_bbox_pred=False,
        feat_channels=64,
        stacked_convs=4,
        int8_output=False,
        dequant_output=True,
        deepcopy_share_conv=True,
        bbox_relu=False,
    ),
    post_process=dict(
        type="BBoxUpscaler",
        strides=[8, 16, 32, 64, 128],
        nhwc_output=True,
    ),
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path="/data/selfcoco/train_lmdb/",
        transforms=[
            dict(
                type="Resize",
                img_scale=(512, 512),
                ratio_range=(0.5, 2.0),
                keep_ratio=True,
            ),
            dict(type="RandomCrop", size=(512, 512)),
            dict(
                type="Pad",
                divisor=512,
            ),
            dict(
                type="RandomFlip",
                px=0.5,
                py=0,
            ),
            dict(type="AugmentHSV", hgain=0.015, sgain=0.7, vgain=0.4),
            dict(
                type="ToTensor",
                to_yuv=True,
            ),
            dict(
                type="Normalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    collate_fn=hat.data.collates.collate_2d,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path="/data/selfcoco/val_lmdb/",
        transforms=[
            dict(
                type="Resize",
                img_scale=(512, 512),
                keep_ratio=True,
            ),
            dict(
                type="Pad",
                size=(512, 512),
            ),
            dict(
                type="ToTensor",
                to_yuv=True,
            ),
            dict(
                type="Normalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    collate_fn=hat.data.collates.collate_2d,
)


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs.items():
        losses.append(loss)
    return losses


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=1,
    epoch_log_freq=1,
    log_prefix="loss_ " + task_name,
)

batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
)


def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=5000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=1,
)

test_inputs = dict(img=torch.randn((1, 3, 512, 512)))
ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=1,
    strict_match=True,
    mode="max",
    monitor_metric_key="mAP",
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    init_with_train_model=False,
    val_interval=1,
    val_on_train_end=True,
)

freeze_bn_callback = dict(
    type="FreezeBNStatistics",
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.14,
        momentum=0.937,
        nesterov=True,
    ),
    batch_processor=batch_processor,
    num_epochs=100,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(type="ExponentialMovingAverage"),
        dict(
            type="CosLrUpdater",
            warmup_len=2,
            warmup_by="epoch",
            step_log_interval=1,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    sync_bn=True,
    val_metrics=dict(
        type="COCODetectionMetric",
        ann_file="/data/selfcoco/annotations/instances_val2017.json",
    ),
)

float_solver = dict(
    trainer=float_trainer,
    allow_miss=True,
    ignore_extra=True,
    resume_optimizer=None,
    resume_epoch=None,
    resume_step=None,
    quantize=False,
)

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_qconfig_dict = {
    "": {
        "activation": HorizonCalibrationActivationFakeQuantize,
        "weight": HorizonCalibrationWeightFakeQuantize,
    }
}

calibration_trainer = dict(
    type="CalibratorComm",
    model=model,
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=2000,
    device=None,
    callbacks=[
        stat_callback,
        val_callback,
        ckpt_callback,
    ],
    val_metrics=dict(
        type="COCODetectionMetric",
        ann_file="/data/selfcoco/annotations/instances_val2017.json",
    ),
    log_interval=200,
)

calibration_solver = dict(
    trainer=calibration_trainer,
    quantize=False,
    allow_not_init=True,
    pre_step="float",
    pre_step_checkpoint=os.path.join(
        ckpt_dir, "float-checkpoint-best.pth.tar"
    ),
    strict_match=True,
    custom_config_dict={
        "modules": {
            "backbone": {
                "qconfig_dict": calibration_qconfig_dict,
                "prepare_custom_config_dict": HorizonPrepareCustomConfigDict,
            },
            "neck": {
                "qconfig_dict": calibration_qconfig_dict,
                "prepare_custom_config_dict": HorizonPrepareCustomConfigDict,
                "quantize_input": False,
            },
            "head": {
                "qconfig_dict": calibration_qconfig_dict,
                "prepare_custom_config_dict": HorizonPrepareCustomConfigDict,
                "quantize_input": False,
                "quantize_output": False,
            },
        }
    },
)

qat_qconfig_dict = {
    "": {
        "activation": HorizonCalibratedActivationFakeQuantize,
        "weight": HorizonCalibratedWeightFakeQuantize,
    }
}

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=4e-5)},
        lr=0.00001,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=15,
    device=None,
    callbacks=[
        freeze_bn_callback,
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[4],
            step_log_interval=500,
        ),
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    val_metrics=dict(
        type="COCODetectionMetric",
        ann_file="/data/selfcoco/annotations/instances_val2017.json",
    ),
)

qat_solver = dict(
    trainer=qat_trainer,
    quantize=True,
    check_quantize_model=True,
    pre_step="calibration",
    pre_step_checkpoint=os.path.join(
        ckpt_dir, "calibration-checkpoint-best.pth.tar"
    ),
    strict_match=True,
    custom_config_dict={
        "modules": {
            "backbone": {
                "qconfig_dict": qat_qconfig_dict,
                "prepare_custom_config_dict": HorizonPrepareCustomConfigDict,
            },
            "neck": {
                "qconfig_dict": qat_qconfig_dict,
                "prepare_custom_config_dict": HorizonPrepareCustomConfigDict,
                "quantize_input": False,
            },
            "head": {
                "qconfig_dict": qat_qconfig_dict,
                "prepare_custom_config_dict": HorizonPrepareCustomConfigDict,
                "quantize_input": False,
                "quantize_output": False,
            },
        }
    },
)

step2solver = dict(
    float=float_solver,
    calibration=calibration_solver,
    qat=qat_solver,
)
