from omegaconf import OmegaConf

from modeling.build_detr import build
from modeling.post_process import PostProcess

# args for detr build_model
detr_args = OmegaConf.create()
detr_args.lr_backbone = 1e-5
detr_args.dataset_file = "coco"
detr_args.num_queries = 100
detr_args.frozen_weights = None
detr_args.bbox_loss_coef = 5
detr_args.mask_loss_coef = 1
detr_args.dice_loss_coef = 1
detr_args.eos_coef = 0.1
detr_args.device = "cuda"
detr_args.masks = False 

# Backbone
# Name of the convolutional backbone to use
detr_args.backbone = "resnet50"
# If true, we replace stride with dilation in the last convolutional block (DC5)
detr_args.dilation = False
# Type of positional embedding to use on top of the image features (sine, learned)
detr_args.position_embedding = "sine"

# Transformer
detr_args.hidden_dim = 256
detr_args.dropout = 0.1
detr_args.nheads = 8 
detr_args.dim_feedforward = 2048 
detr_args.enc_layers = 6 
detr_args.dec_layers = 6 
detr_args.pre_norm = False

# Loss
detr_args.aux_loss = True

# Matcher
# Class coefficient in the matching cost
detr_args.set_cost_class = 1
# L1 box coefficient in the matching cost
detr_args.set_cost_bbox = 5
# giou box coefficient in the matching cost
detr_args.set_cost_giou = 2

# Loss coefficients
detr_args.mask_loss_coef = 1
detr_args.dice_loss_coef = 1
detr_args.bbox_loss_coef = 1
detr_args.giou_loss_coef = 1
# Relative classification weight of the no-object class
detr_args.eos_coef = 1

model = build(args=detr_args)
postprocessors = {'bbox': PostProcess()}