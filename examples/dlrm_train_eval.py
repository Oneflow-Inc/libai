"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import argparse
import os
import sys
import glob
import time
import numpy as np
import psutil
import warnings
import oneflow as flow
import oneflow.nn as nn

from libai.models import DLRMModel
from libai.models.utils import CTRGraph
from libai.data import build_criteo_dataloader
from libai.scheduler import WarmupDecayLR


warnings.filterwarnings("ignore", category=FutureWarning)
from petastorm.reader import make_batch_reader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def get_args(print_args=True):
    def int_list(x):
        return list(map(int, x.split(",")))

    def str_list(x):
        return list(map(str, x.split(",")))

    parser = argparse.ArgumentParser()

    parser.add_argument("--disable_fusedmlp", action="store_true", help="disable fused MLP or not")
    parser.add_argument("--embedding_vec_size", type=int, default=128)
    parser.add_argument("--bottom_mlp", type=int_list, default="512,256,128")
    parser.add_argument("--top_mlp", type=int_list, default="1024,1024,512,256")
    parser.add_argument(
        "--disable_interaction_padding",
        action="store_true",
        help="disable interaction padding or not",
    )
    parser.add_argument(
        "--interaction_itself", action="store_true", help="interaction itself or not"
    )
    parser.add_argument("--model_load_dir", type=str, default=None)
    parser.add_argument("--model_save_dir", type=str, default=None)
    parser.add_argument(
        "--save_initial_model", action="store_true", help="save initial model parameters or not.",
    )
    parser.add_argument(
        "--save_model_after_each_eval", action="store_true", help="save model after each eval.",
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--eval_batches", type=int, default=1612, help="number of eval batches")
    parser.add_argument("--eval_batch_size", type=int, default=55296)
    parser.add_argument("--eval_interval", type=int, default=10000)
    parser.add_argument("--train_batch_size", type=int, default=55296)
    parser.add_argument("--learning_rate", type=float, default=24)
    parser.add_argument("--warmup_batches", type=int, default=2750)
    parser.add_argument("--decay_batches", type=int, default=27772)
    parser.add_argument("--decay_start", type=int, default=49315)
    parser.add_argument("--train_batches", type=int, default=75000)
    parser.add_argument("--loss_print_interval", type=int, default=1000)
    parser.add_argument(
        "--one_embedding_key_type",
        type=str,
        default="int64",
        help="OneEmbedding key type: int32, int64",
    )
    parser.add_argument(
        "--table_size_array",
        type=int_list,
        default="39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295,39664984,585935,12972,108,36",
        help="Embedding table size array for sparse fields",
    )
    parser.add_argument(
        "--persistent_path", type=str, required=True, help="path for persistent kv store",
    )
    parser.add_argument("--store_type", type=str, default="cached_host_mem")
    parser.add_argument("--cache_memory_budget_mb", type=int, default=8192)
    parser.add_argument("--amp", action="store_true", help="Run model with amp")
    parser.add_argument("--loss_scale_policy", type=str, default="static", help="static or dynamic")

    args = parser.parse_args()

    if print_args and flow.env.get_rank() == 0:
        _print_args(args)
    return args


def _print_args(args):
    """Print arguments."""
    print("------------------------ arguments ------------------------", flush=True)
    str_list = []
    for arg in vars(args):
        dots = "." * (48 - len(arg))
        str_list.append("  {} {} {}".format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print("-------------------- end of arguments ---------------------", flush=True)


def prefetch_eval_batches(data_dir, batch_size, num_batches):
    cached_eval_batches = []
    labels = []
    with build_criteo_dataloader(data_dir, batch_size, shuffle=False) as loader:
        for batch_dict in loader:
            labels.append(batch_dict.pop("label")) 
            cached_eval_batches.append(batch_to_global(batch_dict))

    labels = (
        np_to_global(np.concatenate(labels, axis=0)).to_global(sbp=flow.sbp.broadcast()).to_local()
    )
    return labels, cached_eval_batches


def train(args):
    rank = flow.env.get_rank()

    model = DLRMModel(
        embedding_vec_size=args.embedding_vec_size,
        bottom_mlp=args.bottom_mlp,
        top_mlp=args.top_mlp,
        use_fusedmlp=not args.disable_fusedmlp,
        persistent_path=args.persistent_path,
        table_size_array=args.table_size_array,
        one_embedding_key_type=args.one_embedding_key_type,
        one_embedding_store_type=args.store_type,
        cache_memory_budget_mb=args.cache_memory_budget_mb,
        interaction_itself=args.interaction_itself,
        interaction_padding=not args.disable_interaction_padding,
    )
    model.to_global(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)

    if args.model_load_dir:
        print(f"Loading model from {args.model_load_dir}")
        state_dict = flow.load(args.model_load_dir, global_src_rank=0)
        model.load_state_dict(state_dict, strict=False)

    def save_model(subdir):
        if not args.model_save_dir:
            return
        save_path = os.path.join(args.model_save_dir, subdir)
        if rank == 0:
            print(f"Saving model to {save_path}")
        state_dict = model.state_dict()
        flow.save(state_dict, save_path, global_dst_rank=0)

    if args.save_initial_model:
        save_model("initial_checkpoint")

    opt = flow.optim.SGD(model.parameters(), lr=args.learning_rate)
    lr_scheduler = WarmupDecayLR(opt, args.warmup_batches, args.decay_start, args.decay_batches)
    loss = flow.nn.BCEWithLogitsLoss(reduction="none").to("cuda")

    if args.loss_scale_policy == "static":
        grad_scaler = flow.amp.StaticGradScaler(1024)
    else:
        grad_scaler = flow.amp.GradScaler(
            init_scale=1073741824, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
        )

    eval_graph = CTRGraph(model, fp16=args.amp, is_train=False)
    train_graph = CTRGraph(model, loss, opt, lr_scheduler, grad_scaler, args.amp)

    if args.eval_interval > 0:
        cached_eval_labels, cached_eval_batches = prefetch_eval_batches(
            f"{args.data_dir}/test", args.eval_batch_size, args.eval_batches
        )

    model.train()
    with build_criteo_dataloader(f"{args.data_dir}/train", args.train_batch_size) as loader:
        step, last_step, last_time = -1, 0, time.time()
        #for step in range(1, args.train_batches + 1):
        for step, batch_dict in enumerate(loader):
            loss = train_graph(batch_to_global(batch_dict))
            if step % args.loss_print_interval == 0:
                loss = loss["bce_loss"].numpy().mean()
                if rank == 0:
                    latency = 0#(time.time() - last_time) / (step - last_step)
                    throughput = 0#args.train_batch_size / latency
                    last_step, last_time = step, time.time()
                    strtime = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"Rank[{rank}], Step {step}, Loss {loss:0.4f}, Latency "
                        + f"{(latency * 1000):0.3f} ms, Throughput {throughput:0.1f}, {strtime}"
                    )

            if args.eval_interval > 0 and step % args.eval_interval == 0:
                auc = eval(cached_eval_labels, cached_eval_batches, eval_graph, step)
                if args.save_model_after_each_eval:
                    save_model(f"step_{step}_val_auc_{auc:0.5f}")
                model.train()
                last_time = time.time()

    if args.eval_interval > 0 and step % args.eval_interval != 0:
        auc = eval(cached_eval_labels, cached_eval_batches, eval_graph, step)
        if args.save_model_after_each_eval:
            save_model(f"step_{step}_val_auc_{auc:0.5f}")


def np_to_global(np):
    t = flow.from_numpy(np)
    return t.to_global(placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0))


def batch_to_global(batch_dict):
    for k, v in batch_dict.items():
        batch_dict[k] = np_to_global(v)
    return batch_dict


def eval(labels, cached_eval_batches, eval_graph, cur_step=0):
    num_eval_batches = len(cached_eval_batches)
    if num_eval_batches <= 0:
        return
    eval_graph.module.eval()
    preds = []
    eval_start_time = time.time()
    for batch_dict in cached_eval_batches:
        pred = eval_graph(batch_dict)
        preds.append(pred.to_local())

    preds = (
        flow.cat(preds, dim=0)
        .to_global(placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0))
        .to_global(sbp=flow.sbp.broadcast())
        .to_local()
    )
    flow.comm.barrier()
    eval_time = time.time() - eval_start_time

    rank = flow.env.get_rank()
    auc = 0
    if rank == 0:
        auc_start_time = time.time()
        auc = flow.roc_auc_score(labels, preds).numpy()[0]
        auc_time = time.time() - auc_start_time
        host_mem_mb = psutil.Process().memory_info().rss // (1024 * 1024)
        stream = os.popen("nvidia-smi --query-gpu=memory.used --format=csv")
        device_mem_str = stream.read().split("\n")[rank + 1]

        strtime = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"Rank[{rank}], Step {cur_step}, AUC {auc:0.5f}, Eval_time {eval_time:0.2f} s, "
            + f"AUC_time {auc_time:0.2f} s, Eval_samples {labels.shape[0]}, "
            + f"GPU_Memory {device_mem_str}, Host_Memory {host_mem_mb} MiB, {strtime}"
        )

    return auc


if __name__ == "__main__":
    os.system(sys.executable + " -m oneflow --doctor")
    flow.boxing.nccl.enable_all_to_all(True)
    args = get_args()

    train(args)
