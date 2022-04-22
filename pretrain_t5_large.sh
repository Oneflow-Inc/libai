#!/bin/bash
#SBATCH --job-name=t5_libai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1               # number of gpus
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -o /cognitive_comp/ganruyi/experiments/t5_libai/%x-%j.log
#SBATCH -e /cognitive_comp/ganruyi/experiments/t5_libai/%x-%j.err

set -x -e
source activate libai


# to debug - add echo (it exits and prints what it would have launched)
#run_cmd="$PY_LAUNCHER $CMD"
# salloc --nodes=1 --gres=gpu:2 --cpus-per-gpu=20 -t 24:00:00
export CMD=" \
  python tools/train.sh tools/train_net.py configs/t5_large_pretrain.py 1
"
# clear; srun --nodes=1 --gres=gpu:8 --ntasks-per-node=8 —-jobid=151810 bash -c 'python $CMD'
clear; srun bash -c 'python $CMD'
