today=$(date +%Y-%m-%d)

export CUDA_VISIBLE_DEVICES="4,5,6,7"
# bert loss compare

# single GPU
bash tools/train.sh projects/bert_loss_compare/train_net.py \
projects/bert_loss_compare/configs/compare_loss_base.py 1 \
train.output_dir=loss_align/bert_loss_compare/${today}_base

# data parallel, 8 GPUs
bash tools/train.sh projects/bert_loss_compare/train_net.py \
projects/bert_loss_compare/configs/compare_loss_base.py 8 \
train.output_dir=loss_align/bert_loss_compare/${today}_dp8

# tensor paralle, 8 GPUs
bash tools/train.sh projects/bert_loss_compare/train_net.py \
projects/bert_loss_compare/configs/compare_loss_base.py 8 \
train.dist.tensor_parallel_size=8 \
data.make_vocab_size_divisible_by=16 \
train.output_dir=loss_align/bert_loss_compare/${today}_mp8

# data&tensor paralle, 4x2 GPUs
bash tools/train.sh projects/bert_loss_compare/train_net.py \
projects/bert_loss_compare/configs/compare_loss_base.py 8 \
train.dist.tensor_parallel_size=2 \
data.make_vocab_size_divisible_by=64 \
train.output_dir=loss_align/bert_loss_compare/${today}_dp4mp2

# data&tensor paralle, 2x4 GPUs
bash tools/train.sh projects/bert_loss_compare/train_net.py \
projects/bert_loss_compare/configs/compare_loss_base.py 8 \
train.dist.tensor_parallel_size=4 \
data.make_vocab_size_divisible_by=32 \
train.output_dir=loss_align/bert_loss_compare/${today}_dp2mp4

# data&pipeline paralle, 2x4 GPUs
bash tools/train.sh projects/bert_loss_compare/train_net.py \
projects/bert_loss_compare/configs/compare_loss_base.py 8 \
train.dist.pipeline_parallel_size=4 \
train.output_dir=loss_align/bert_loss_compare/${today}_dp2pp4

# data&pipeline paralle, 4x2 GPUs
bash tools/train.sh projects/bert_loss_compare/train_net.py \
projects/bert_loss_compare/configs/compare_loss_base.py 8 \
train.dist.pipeline_parallel_size=2 \
train.output_dir=loss_align/bert_loss_compare/${today}_dp4pp2

# tensor&pipeline paralle, 4x2 GPUs
bash tools/train.sh projects/bert_loss_compare/train_net.py \
projects/bert_loss_compare/configs/compare_loss_base.py 8 \
train.dist.tensor_parallel_size=4 \
train.dist.pipeline_parallel_size=2 \
data.make_vocab_size_divisible_by=32 \
train.output_dir=loss_align/bert_loss_compare/${today}_mp4pp2

# tensor&pipeline paralle, 2x4 GPUs
bash tools/train.sh projects/bert_loss_compare/train_net.py \
projects/bert_loss_compare/configs/compare_loss_base.py 8 \
train.dist.tensor_parallel_size=2 \
train.dist.pipeline_parallel_size=4 \
data.make_vocab_size_divisible_by=64 \
train.output_dir=loss_align/bert_loss_compare/${today}_mp2pp4

# data&tensor&pipeline paralle, 2x2x2 GPUs
bash tools/train.sh projects/bert_loss_compare/train_net.py \
projects/bert_loss_compare/configs/compare_loss_base.py 8 \
train.dist.tensor_parallel_size=2 \
train.dist.pipeline_parallel_size=2 \
data.make_vocab_size_divisible_by=64 \
train.output_dir=loss_align/bert_loss_compare/${today}_dp2mp2pp2

# draw loss curve
echo "draw bert loss curve"
python3 projects/bert_loss_compare/utils/draw_loss_curve.py --compare-item 1000
python3 projects/bert_loss_compare/utils/draw_loss_curve.py --compare-item 100
