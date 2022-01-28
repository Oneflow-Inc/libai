# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain T5"""
import oneflow
import os
os.environ['MASTER_ADDR']='localhost'
os.environ['MASTER_PORT']='12346'
os.environ['RANK']='0'
os.environ['WORLD_SIZE']='1'
# os.environ['DATA_PATH']='/home/wang/workspace/Megatron-LM/examples'
# os.environ['VOCAB_FILE']='/home/wang/data/t5/dataset/bert-base-chinese-vocab.txt'
# os.environ['CHECKPOINT_PATH']='/home/wang/workspace/Megatron-LM/examples'
os.environ['DATA_PATH']='/workspace/Megatron-LM/examples'
os.environ['VOCAB_FILE']='/workspace/data/libai_dataset/bert-base-chinese-vocab.txt'
os.environ['CHECKPOINT_PATH']='/workspace/Megatron-LM/examples'
import sys

os.environ['CHECKPOINT_PATH']="checkpoints/gpt_base"
os.environ['DATA_PATH']="/workspace/data/libai_dataset/loss_compara_content_sentence"
sys.argv.extend([
       "--num-layers", " 6"         ,
       "--hidden-size", " 1024"     ,
       "--num-attention-heads", " 16",
       "--micro-batch-size", " 4"   ,
       "--global-batch-size", " 8"  ,
       "--seq-length", " 1024"      ,
       "--max-position-embeddings", " 1024",
       "--train-iters", " 500000"   ,
       "--lr-decay-iters", " 320000",
       "--save", " $CHECKPOINT_PATH",
       "--load", " $CHECKPOINT_PATH",
       "--data-path", " $DATA_PATH" ,
       "--vocab-file", "/workspace/data/gpt_dataset/gpt2-vocab.json",
       "--merge-file", "/workspace/data/gpt_dataset/gpt2-merges.txt",
       "--data-impl", "mmap",
       "--split", " 949,50,1"       ,
       "--distributed-backend", "nccl",
       "--lr", " 0.00015"           ,
       "--min-lr", " 1.0e-5"        ,
       "--lr-decay-style", "cosine",
       "--weight-decay", "1e-2"    ,
       "--clip-grad", " 1.0"        ,
       "--lr-warmup-fraction", ".01",
       "--activations-checkpoint-method", "uniform",
       "--log-interval", " 100"     ,
       "--save-interval", " 10000"  ,
       "--eval-interval", " 1000"   ,
       "--eval-iters", " 10"        ,
       "--no-bias-dropout-fusion" ,
       # --fp16
])



from functools import partial

import torch

from megatron import (
    get_args,
    get_timers,
    mpu,
    print_rank_0
)
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.model import GPTModel, ModelType
from megatron.training import pretrain, get_model
from megatron.utils import average_losses_across_data_parallel_group
from megatron.initialize import initialize_megatron
initialize_megatron(extra_args_provider=None, args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})


"""
Pipeline parallelism for T5
===========================

T5 is a model architecture with both encoder and decoder blocks.
Consequently, pipeline parallelism is implemented slightly differently
compared to architectures like GPT and BERT.

In particular, when pipeline_model_parallel_world_size > 1, each stage
either executes an encoder block or a decoder block. The
--pipeline-model-parallel-split-rank argument controls the rank at which
the split happens: all ranks lower than this argument execute the
encoder block, and all ranks equal to or higher than this argument value
execute the decoder block.

In the encoder section of the model, only one tensor is sent downstream:
the intermediate encoder_hidden_state. In the decoder section of the
model, two tensors are sent downstream in the forward pass: the fully
computed encoder_hidden_state, and the intermediate decoder_hidden_state.

In particular, these are the shapes of the tensors sent between
different workers:
    If rank is in decoder section:
        intermediate decoder_hidden_state (pre-transpose),
        complete encoder_hidden_state (post-transpose).
    If rank is at boundary between encoder and decoder sections:
        complete encoder_hidden_state (post-transpose).
    If rank is in encoder section:
        intermediate encoder_hidden_state (pre-transpose).

Additionally, we have code in the backward_step function in schedules.py
to accumulate the encoder_hidden_state gradient across skip connections
(encoder_hidden_state fed in as input to each layer in the decoder).
"""



def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds

def get_gpt_model():
    return get_model(model_provider_func=model_provider)[0].module.eval()

if __name__ == '__main__':
    print(get_gpt_model())
    from utils import get_sample
    get_gpt_model()(*get_sample(mode='torch'))
