#!/usr/bin/env bash

clear

FILE=$1
CONFIG=$2
GPUS=$3
NODE=${NODE:-1}
NODE_RANK=${NODE_RANK:-0}
ADDR=${ADDR:-127.0.0.1}
PORT=${PORT:-12345}

export GLOG_logtostderr=1
export ONEFLOW_ACTOR_ENABLE_LIGHT_ACTOR=0 # 禁用lightweight actor

export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
# export NCCL_MAX_NCHANNELS=1
# export NCCL_MIN_NCHANNELS=1
# export NCCL_NTHREADS=64

if [ -z $RUN_TYPE ];then
    RUN_TYPE="PURE"
    # RUN_TYPE="GDB"
    # RUN_TYPE="NSYS"
fi

export ONEFLOW_ENABLE_OFCCL=1
export DISABLE_NCCL_COMPUTE_STREAM=1
export ONEFLOW_DEBUG_MODE=1
export ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=1

export GLOG_vmodule=plan_util*=1,of_collective_actor*=1,of_collective_boxing_kernels*=1,collective_backend_ofccl*=1,hierarchical_sub_task_graph_builder_impl*=1,of_request_store*=1,request_store*=1,runtime*=1,scheduler*=1,collective_manager*=1
# nn_graph*=1,
# export GLOG_v=1

echo GPUS=$GPUS
echo ONEFLOW_ENABLE_OFCCL=$ONEFLOW_ENABLE_OFCCL
echo ONEFLOW_OFCCL_SKIP_NEGO=$ONEFLOW_OFCCL_SKIP_NEGO
echo ONEFLOW_ENABLE_OFCCL=$ONEFLOW_DEBUG_MODE
echo ONEFLOW_OFCCL_SKIP_NEGO=$ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE
echo NCCL_PROTO=$NCCL_PROTO
echo NCCL_ALGO=$NCCL_ALGO
echo NCCL_MAX_NCHANNELS=$NCCL_MAX_NCHANNELS
echo NCCL_NTHREADS=$NCCL_NTHREADS
echo ONEFLOW_OFCCL_CHAIN=$ONEFLOW_OFCCL_CHAIN
echo GLOG_vmodule=$GLOG_vmodule
echo GLOG_v=$GLOG_v
echo GLOG_logtostderr=$GLOG_logtostderr

export SHOW_ALL_PREPARED_COLL=1

export TRAVERSE_TIMES=10
export DEV_TRY_ROUND=10
export CHECK_REMAINING_SQE_INTERVAL=10000
export DEBUG_FILE="/home/panlichen/work/oneflow/log/oneflow_cpu_rank_"

export NUM_ITER_ENV=20
echo NUM_ITER_ENV=$NUM_ITER_ENV

if [ $GPUS = 2 ]; then
    export CUDA_VISIBLE_DEVICES=4,5

    #pure dp
    # export BASE_CTX_SWITCH_THRESHOLD=100
    # export TOLERANT_UNPROGRESSED_CNT=2000
    # export NUM_TRY_TASKQ_HEAD=40
    
    #pure tp
    export BASE_CTX_SWITCH_THRESHOLD=120
    export TOLERANT_UNPROGRESSED_CNT=10000
    export NUM_TRY_TASKQ_HEAD=100
elif [ $GPUS = 4 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,4,5
    export ONEFLOW_OFCCL_SKIP_NEGO=0

    #pure dp
    # export BASE_CTX_SWITCH_THRESHOLD=80
    # export TOLERANT_UNPROGRESSED_CNT=10000
    # export NUM_TRY_TASKQ_HEAD=50
    
    #pure tp
    # export BASE_CTX_SWITCH_THRESHOLD=120
    # export TOLERANT_UNPROGRESSED_CNT=14000
    # export NUM_TRY_TASKQ_HEAD=120
    #pure tp-no nego
    export BASE_CTX_SWITCH_THRESHOLD=3000
    export TOLERANT_UNPROGRESSED_CNT=16000
    export NUM_TRY_TASKQ_HEAD=200
elif [  $GPUS = 8 ]; then
    export ONEFLOW_OFCCL_SKIP_NEGO=1

    #pure dp
    # export BASE_CTX_SWITCH_THRESHOLD=120
    # export TOLERANT_UNPROGRESSED_CNT=70000
    # export NUM_TRY_TASKQ_HEAD=240
    
    #pure tp no nego
    export BASE_CTX_SWITCH_THRESHOLD=4000
    export TOLERANT_UNPROGRESSED_CNT=8000
    export NUM_TRY_TASKQ_HEAD=100
fi

echo TRAVERSE_TIMES=$TRAVERSE_TIMES
echo TOLERANT_UNPROGRESSED_CNT=$TOLERANT_UNPROGRESSED_CNT
echo BASE_CTX_SWITCH_THRESHOLD=$BASE_CTX_SWITCH_THRESHOLD
echo NUM_TRY_TASKQ_HEAD=$NUM_TRY_TASKQ_HEAD
echo DEV_TRY_ROUND=$DEV_TRY_ROUND
echo CHECK_REMAINING_SQE_INTERVAL=$CHECK_REMAINING_SQE_INTERVAL
echo DEBUG_FILE=$DEBUG_FILE

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=PARALLEL
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE
# export NCCL_DEBUG=INFO

rm -rf /home/panlichen/work/libai/log
mkdir -p /home/panlichen/work/libai/log

rm -rf /home/panlichen/work/oneflow/log
mkdir -p /home/panlichen/work/oneflow/log

export ONEFLOW_FUSE_OPTIMIZER_UPDATE_CAST=true

if [ "$ONEFLOW_ENABLE_OFCCL" == "1" ]; then
    NSYS_FILE="ofccl_vit"_${HOST}_${GPUS}_card
else
    NSYS_FILE="nccl_vit"_${HOST}_${GPUS}_card
fi

if [ "$RUN_TYPE" == "PURE" ];then
    cmd="python3 -m oneflow.distributed.launch"
elif [ "$RUN_TYPE" == "GDB" ];then
    cmd="gdb -ex r --args python3 -m oneflow.distributed.launch"
elif [ "$RUN_TYPE" == "NSYS" ];then
    if [ ! -d "/home/panlichen/work/oneflow/log/nsys" ];then
        mkdir -p /home/panlichen/work/oneflow/log/nsys
    fi
    # cmd="nsys profile -f true --trace=cuda,cudnn,cublas,osrt,nvtx -o /home/panlichen/work/oneflow/log/nsys/$NSYS_FILE python3 -m oneflow.distributed.launch"
    cmd="nsys profile -f true -o /home/panlichen/work/oneflow/log/nsys/$NSYS_FILE python3 -m oneflow.distributed.launch"
fi
echo cmd=$cmd

$cmd \
  --nproc_per_node $GPUS --nnodes $NODE --node_rank $NODE_RANK --master_addr $ADDR --master_port $PORT \
  $FILE --config-file $CONFIG ${@:4} \
  > /home/panlichen/work/oneflow/log/oneflow.log 2>&1

