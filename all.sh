set -x
bash gpt_train_npu.sh 1 1 1 2>&1 | tee log/gpt_111.log
bash gpt_train_npu.sh 8 1 1 2>&1 | tee log/gpt_811.log
bash gpt_train_npu.sh 1 8 1 2>&1 | tee log/gpt_181.log
bash gpt_train_npu.sh 1 1 8 2>&1 | tee log/gpt_118.log
bash gpt_auto_parallel.sh 2>&1 | tee log/gpt_ap.log
bash bert_train_npu.sh 8 1 1 2>&1 | tee log/bert_811.log
bash bert_train_npu.sh 1 8 1 2>&1 | tee log/bert_181.log
bash bert_train_npu.sh 1 1 8 2>&1 | tee log/bert_118.log
bash bert_auto_parallel.sh 2>&1 | tee log/bert_ap.log
bash llama_train_npu.sh 7b 0 2>&1 | tee log/llama_7b_pp.log
bash llama_train_npu.sh 13b 0 2>&1 | tee log/llama_13b_pp.log
bash llama_train_npu.sh 7b 1 2>&1 | tee log/llama_7b_ap.log
bash llama_train_npu.sh 13b 1 2>&1 | tee log/llama_13b_ap.log
set +x
