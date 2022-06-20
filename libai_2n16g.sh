
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip install https://oneflow-staging.oss-cn-beijing.aliyuncs.com/canary/commit/111fbf5262b83ee39f28cfbba05368cc3a3a72e9/cu112/oneflow-0.8.0%2Bcu112.git.111fbf5-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


python3 -m pip install -e . --user


bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 1 1 True False 6 96 
sleep 10s 
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 2 1 True True 32 2048 
sleep 10s 
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 8 1 True True 32 512 
sleep 10s 
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 2 4 True True 32 512 
sleep 10s 
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 1 1 True False 24 384 
sleep 5s 
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 2 1 True True 128 8192 
sleep 5s 
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 8 1 True True 128 2048 
sleep 5s 
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 2 4 True True 128 4096
sleep 5s
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 2 4 True True 128 2048
sleep 5s 
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 1 4 True True 128 4096

sleep 5s 
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 1 4 True True 32 1024




# debug 
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 2 4 True True 32 64
