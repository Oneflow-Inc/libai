set -x
#!/bin/bash

MODEL_SIZE=7b
LOG_DIR=log
SCRIPT=llama_test.sh
mkdir -p $LOG_DIR

# 并行配置数组（DP TP PP）
parallel_configs=(
  "8 1 1"
  "4 2 1"
  "2 2 2"
  "4 1 2"
)

batch_sizes=(4 8 12 16)
auto_parallel_options=(0 1)
nll_loss_options=(0 1)

for config in "${parallel_configs[@]}"; do
  for batch in "${batch_sizes[@]}"; do
    for ap in "${auto_parallel_options[@]}"; do
      for nll in "${nll_loss_options[@]}"; do
        # 拆解并行配置
        read -r dp tp pp <<< "$config"
        total_devices=$(( dp * tp * pp ))

        # 日志文件名示例：llama_7b_dp4_tp2_pp1_bs8_ap1_nll1.log
        LOG_FILE="${LOG_DIR}/llama_${MODEL_SIZE}_dp${dp}_tp${tp}_pp${pp}_bs${batch}_ap${ap}_nll${nll}.log"

        echo "Running config: DP=${dp}, TP=${tp}, PP=${pp}, BATCH=${batch}, AP=${ap}, NLL=${nll}"

        if [ "$nll" -eq 1 ]; then
          USE_NLL_LOSS=1 bash $SCRIPT $dp $tp $pp $batch $MODEL_SIZE $ap 2>&1 | tee "$LOG_FILE"
        else
          bash $SCRIPT $dp $tp $pp $batch $MODEL_SIZE $ap 2>&1 | tee "$LOG_FILE"
        fi

        echo "Saved log to $LOG_FILE"
        echo "---------------------------------------------"
      done
    done
  done
done

set +x
