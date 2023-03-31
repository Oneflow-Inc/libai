set -aux

ONEFLOW_STATE_DICT_PATH="projects/T5/output/mt5_output/model_final/model"
CONFIG_PATH="projects/T5/configs/mt5_pretrain.py"
SAVE_PATH="projects/T5/pytorch_model.bin"

python3 projects/T5/utils/weight_convert.py \
    --oneflow_state_dict_path $ONEFLOW_STATE_DICT_PATH \
    --config_path $CONFIG_PATH \
    --save_path $SAVE_PATH