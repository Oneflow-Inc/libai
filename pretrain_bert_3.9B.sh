#!/bin/bash
#SBATCH --job-name=bert3.9B_libai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8               # number of gpus
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -o /cognitive_comp/ganruyi/experiments/t5_libai/%x-%j.log
#SBATCH -e /cognitive_comp/ganruyi/experiments/t5_libai/%x-%j.err

set -x -e
# source activate libai

# to debug - add echo (it exits and prints what it would have launched)
#run_cmd="$PY_LAUNCHER $CMD"
# salloc --nodes=1 --gres=gpu:8 --cpus-per-gpu=20 -t 24:00:00
CMD="sh tools/train.sh tools/train_net.py configs/bert_3.9B_pp_pretrain.py 8"
# clear; srun --nodes=1 --gres=gpu:8 --ntasks-per-node=8 —-jobid=151810 bash -c 'python $CMD'
clear; srun -N 1 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=20 -o %x-%j.log $CMD
