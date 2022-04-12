import sys

sys.path.append("projects/DETR")

from modeling.detr import build


detr_args = dict(
    dataset_file = "coco",
    num_queries = 1,
    aux_loss = 1,
    frozen_weights = 1,
    bbox_loss_coef = 1,
    mask_loss_coef = 1,
    dice_loss_coef = 1,
    dec_layers = 1,
    eos_coef = 1,
)

model, criterion, postprocessors = build(detr_args)