from projects.NeRF.configs.config_nerf import (
    train,
    dataset,
    dataloader,
    graph,
    model,
    LazyCall,
    build_image_test_loader,
)
from projects.NeRF.evaluation.nerf_evaluator import NerfVisEvaluator
from libai.data.samplers import SingleRoundSampler

# NOTE: Used for generating MP4 format files
# Redefining evaluator
train.evaluation = dict(
    enabled=True,
    # evaluator for calculating psnr
    evaluator=LazyCall(NerfVisEvaluator)(
        img_wh=(400, 400) if train.dataset_type == "Blender" else (504, 378),
        pose_dir_len=40 if train.dataset_type == "Blender" else 120,
        name="blender_rendering_result"
        if train.dataset_type == "Blender"
        else "llff_rendering_result",
    ),
    eval_period=train.evaluation.eval_period,
    eval_iter=1e5,  # running steps for validation/test
    # Metrics to be used for best model checkpoint.
    eval_metric="psnr",
    eval_mode="max",
)

dataloader.test = [
    LazyCall(build_image_test_loader)(
        dataset=LazyCall(dataset)(
            split="vis",
            img_wh=(400, 400) if dataset.dataset_type == "Blender" else (504, 378),
            root_dir=train.blender_dataset_path
            if dataset.dataset_type == "Blender"
            else train.llff_dataset_path,
            spheric_poses=None if dataset.dataset_type == "Blender" else False,
            val_num=None if dataset.dataset_type == "Blender" else 1,  # Number of your GPUs
        ),
        sampler=LazyCall(SingleRoundSampler)(shuffle=False, drop_last=False),
        num_workers=0,
        test_batch_size=train.test_micro_batch_size,
    )
]

train.load_weight = "/path/to/ckpt"  # Already trained NeRF checkpoint location
