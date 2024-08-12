# set visible devices
export ASCEND_RT_VISIBLE_DEVICES=1

# debug
export ONEFLOW_DEBUG=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0


# infer
python projects/Llama/pipeline.py


# # train
# python projects/Llama/utils/prepare_alpaca.py
# bash tools/train.sh tools/train_net.py projects/Llama/configs/llama_sft.py 1