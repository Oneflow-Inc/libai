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

from libai.config import LazyConfig, instantiate, try_get_key
from libai.utils.logger import setup_logger
#from libai.models.utils import CTRGraph


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="config.yaml", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="output", metavar="FOLDER", help="path to output")
    args = parser.parse_args()
    return args


def prefetch_eval_batches(loader):
    cached_eval_batches = []
    labels = []
    for batch_dict in loader:
        labels.append(batch_dict.pop("label")) 
        cached_eval_batches.append(batch_to_global(batch_dict))

    labels = (
        np_to_global(np.concatenate(labels, axis=0)).to_global(sbp=flow.sbp.broadcast()).to_local()
    )
    return labels, cached_eval_batches


class CTRGraph(nn.Graph):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module = None,
        optimizer: flow.optim.Optimizer = None,
        lr_scheduler: flow.optim.lr_scheduler = None,
        grad_scaler = None,
        fp16=False,
        is_train=True,
    ):
        super().__init__()

        self.model = model
        self.is_train = is_train

        if is_train:
            self.loss = loss
            self.add_optimizer(optimizer, lr_sch=lr_scheduler)
            if fp16:
                self.config.enable_amp(True)
                self.set_grad_scaler(grad_scaler)

        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_cast_scale(True)

    def build(self, batch_dict):
        logits = self.model(batch_dict)
        if self.is_train:
            loss = self.loss(logits, batch_dict['label'].to("cuda"))
            reduce_loss = flow.mean(loss)
            reduce_loss.backward()
            return reduce_loss
        else:
            return logits.sigmoid()


def train(cfg):
    rank = flow.env.get_rank()
    logger = setup_logger(args.output_dir, distributed_rank=rank)
    logger.info("Rank of current process: {}. World size: {}".format(rank, flow.env.get_world_size()))
    logger.info("Command line arguments: " + str(cfg))

    # build dataloaders 
    assert try_get_key(cfg, "dataloader") is not None, "cfg must contain `dataloader` namespace"
    assert try_get_key(cfg.dataloader, "train") is not None, "dataloader must contain `train` namespace"
    assert try_get_key(cfg.dataloader, "validation") is not None, "dataloader must contain `validation` namespace"
    dataloaders = {k:instantiate(v) for k, v in cfg.dataloader.items()}

    # build model
    assert try_get_key(cfg, "model") is not None, "cfg must contain `model` namespace"
    model = instantiate(cfg.model)
    model.to_global(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)

    # build optimizer
    assert try_get_key(cfg, "optim") is not None, "cfg must contain `optim` namespace"
    cfg.optim.params = model.parameters()
    opt = instantiate(cfg.optim)

    # build learning rate scheduler
    assert try_get_key(cfg, "scheduler") is not None, "cfg must contain `scheduler` namespace"
    cfg.scheduler.optimizer = opt
    lr_scheduler = instantiate(cfg.scheduler)

    # build loss function
    if try_get_key(cfg, "loss") is not None:
        loss_fn = instantiate(cfg.loss)
    else:
        loss_fn = flow.nn.BCEWithLogitsLoss(reduction="none")
    loss_fn.to("cuda")

    # build loss function
    if try_get_key(cfg, "loss_scale") is not None:
        grad_scaler = instantiate(cfg.loss_scale)
    else:
        grad_scaler = flow.amp.StaticGradScaler(1024)

    eval_graph = CTRGraph(model, fp16=cfg.train.amp, is_train=False)
    train_graph = CTRGraph(model, loss_fn, opt, lr_scheduler, grad_scaler, cfg.train.amp)

    if cfg.train.eval_period > 0:
        cached_eval_labels, cached_eval_batches = prefetch_eval_batches(
            dataloaders["validation"]
        )

    model.train()
    last_step, last_time = 0, time.time()
    for step, batch_dict in enumerate(dataloaders["train"], start=1):
        loss = train_graph(batch_to_global(batch_dict))
        if step % cfg.train.log_period == 0:
            loss = loss.numpy()
            if rank == 0:
                latency = (time.time() - last_time) / (step - last_step)
                throughput = cfg.dataloader.train.batch_size / latency
                last_step, last_time = step, time.time()
                strtime = time.strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"Rank[{rank}], Step {step}, Loss {loss:0.4f}, Latency "
                    + f"{(latency * 1000):0.3f} ms, Throughput {throughput:0.1f}, {strtime}"
                )

        if cfg.train.eval_period > 0 and step % cfg.train.eval_period == 0:
            auc = eval(cached_eval_labels, cached_eval_batches, eval_graph, step)
            model.train()
            last_time = time.time()

    if cfg.train.eval_period > 0 and step % cfg.train.eval_period != 0:
        auc = eval(cached_eval_labels, cached_eval_batches, eval_graph, step)


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
    cfg = LazyConfig.load(args.config_file)
    train(cfg)

