set -ex
# # bash tools/args_libai_gpt2.sh model_config pre_gpu node rank master_ip mp pp fp16 activation mbsz gbsz commit

export NCCL_IB_PCI_RELAXED_ORDERING=1
export ONEFLOW_COMM_NET_IB_GID_INDEX=$NCCL_IB_GID_INDEX
export ONEFLOW_COMM_NET_IB_HCA=$NCCL_IB_HCA
#export ONEFLOW_COMM_NET_IB_MEM_BLOCK_SIZE=16777216
#export ONEFLOW_COMM_NET_IB_MEM_BLOCK_SIZE=4194304
#export ONEFLOW_COMM_NET_IB_QUEUE_DEPTH=2048
#export NCCL_NET_GDR_LEVEL=0
#export NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
#export NCCL_IB_HCA=$NCCL_IB_HCA
#export NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX
#export NCCL_IB_TIMEOU=$NCCL_IB_TIMEOU
#export NCCL_IB_DISABLE=$NCCL_IB_DISABLE
#export NCCL_IB_RETRY_CNT=$NCCL_IB_RETRY_CNT
CONFIG=$1
NNODES=${2:-1}
GPUS_PER_NODE=${3:-8}
# Change for multinode config
NODE_RANK=${4:-0}
MASTER_ADDR=${5:-"127.0.0.1"}
MASTER_PORT=12345
MP=${6:-1}
PP=${7:-1}
USE_FP16=${8:-"True"}
ACTIVATION_CHECKPOINT=${9:-"False"}
MICRO_BATCH_SIZE=${10:-4}
GLOBAL_BATCH_SIZE=${11:-4}
RUN_COMMIT=${12:-"01b1d32"}
TRAIN_ITERS=${13:-220}
LOG_PERIOD=${14:-100}

TRAN_MODEL="LibAI_gpt2"
RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")
LOG_FOLDER=test_logs/$HOSTNAME/$RUN_COMMIT/${NNODES}n${GPUS_PER_NODE}g

AMP_OR="FP32"
if [ $USE_FP16 == 'True' ]; then
    AMP_OR="FP16"
fi

export ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE=true
export MULTIHEAD_ATTN_FUSION=true
export ONEFLOW_FUSE_OPTIMIZER_UPDATE_CAST=true
export ONEFLOW_COMM_NET_IB_ENABLE=True


LOG_FILENAME=$LOG_FOLDER/${TRAN_MODEL}_nl24_nah16_hs1024_${AMP_OR}_ac${ACTIVATION_CHECKPOINT}_mp${MP}_pp${PP}_mb${MICRO_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_${NNODES}n${GPUS_PER_NODE}g_${RUN_TIME}
echo LOG_FILENAME=$LOG_FILENAME
mkdir -p $LOG_FILENAME

python3 -m oneflow.distributed.launch \
--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
tools/train_net.py \
--config-file $CONFIG \
train.train_micro_batch_size=$MICRO_BATCH_SIZE \
train.global_batch_size=$GLOBAL_BATCH_SIZE \
train.dist.tensor_parallel_size=$MP \
train.dist.pipeline_parallel_size=$PP \
train.amp.enabled=$USE_FP16 \
train.activation_checkpoint.enabled=$ACTIVATION_CHECKPOINT \
train.train_iter=$TRAIN_ITERS \
train.log_period=$LOG_PERIOD \
train.output_dir=$LOG_FILENAME 2>&1 | tee ${LOG_FILENAME}/output.log



