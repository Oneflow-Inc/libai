# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
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

import os
import oneflow as flow
from .global_vars import get_args
from core import distribute as dist
from core.utils import print_rank_0, makedirs_ranks
from core.data import build_dataset
from core.models import build_model
from core.criterion import build_criterion
from core.optimizer import build_optimizer, build_lr_scheduler, build_grad_scaler
from core.meters import Logger, SimpleMeter, AverageMeter, SumMeter, TimeMeter, StopWatchMeter
from core.modules import ParallelLogits
from core.models.gpt_model import GPTEmbedding, TransformerLayer, ActivationCheckpointing, Transformer


class Trainer(object):
    def __init__(self):
        self.args = get_args()
        self.rank = flow.env.get_rank()
        self.world_size = flow.env.get_world_size()
        dist.init_distribute(self.args)

        self.model = build_model(self.args)
        self.train_data_loader = build_dataset(self.args, subset='train')
        self.eval_data_loader = build_dataset(self.args, subset='valid')
        self.criterion = build_criterion(self.args)
        self.optimizer = build_optimizer(self.args, self.model)
        self.lr_scheduler = build_lr_scheduler(self.args, self.optimizer)
        self.grad_scaler = build_grad_scaler(self.args)

        makedirs_ranks(self.args.save_path, exist_ok=True, consistent_dst_rank=0)

        if self.args.graph:
            flow.boxing.nccl.enable_use_compute_stream(True)

            self.train_graph = TrainGraph(
                self.args, 
                self.model, 
                self.train_data_loader, 
                self.criterion, 
                self.optimizer, 
                self.lr_scheduler, 
                self.grad_scaler
            )
            # self.eval_graph = EvalGraph(
            #     self.args, 
            #     self.model, 
            #     self.eval_data_loader, 
            #     self.criterion
            # )

        self.logger = Logger(self.rank)
        self.logger.register_metric("iters", SimpleMeter())
        self.logger.register_metric("samples", SumMeter())
        self.logger.register_metric("loss", AverageMeter(), "loss: {:.5f}", True)
        self.logger.register_metric("valid loss", AverageMeter(), "valid loss: {:.5f}", True)
        self.logger.register_metric("lr", SimpleMeter(), "lr: {:.5f}", False)
        self.logger.register_metric("throughput", TimeMeter(), "throughput: {:.2f}", True)

    def __call__(self):
        iteration = 0
        while iteration < self.args.train_iters:
            if self.args.graph:
                loss = self.train_graph()
            else:
                raise NotImplementedError
                # loss = self.train_eager()

            iteration += 1

            self.logger.meter("iters", iteration)
            self.logger.meter("samples", self.args.global_batch_size)
            self.logger.meter("loss", loss)
            self.logger.meter("lr", self.optimizer.get_lr())
            self.logger.meter("throughput", self.args.global_batch_size)
            # if iter % self.args.valid_interval == 0:
            #     pass
            if iteration % self.args.log_interval == 0:
                self.logger.print_metrics([self.world_size - 1])
            

    # def train_step(self):
    #     if self.args.graph:
    #         return self.train_graph()
    #     else:
    #         return self.train_eager_step()
    
    # def eval_step(self):
    #     if self.args.graph:
    #         return self.eval_graph()
    #     else:
    #         return self.eval_eager_step()

    def train_eager_step(self):
        self.model.train()
        data, label = self.data_loader()
        logits = self.model(data)
        loss = self.criterion(logits, label)
        loss.backward()
        self.optimizer.zero_grad()
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss
    
    def eval_eager_step(self):
        self.model.eval()
        data, label = self.data_loader()
        with flow.no_grad():
            outputs = logits = self.model(data)
            if self.criterion is not None:
                loss = self.criterion(logits)
                outputs = (logits, loss)
        return outputs

    def save(self, subdir):
        if self.args.save_path is None:
            return

        save_path = os.path.join(self.args.save_path, subdir)
        print_rank_0(f"Saving model to {save_path}")
        state_dict = self.model.state_dict()

        flow.save(state_dict, save_path, consistent_dst_rank=0)
    
    def load(self, subdir):
        assert self.args.save_path is None
        
        save_path = os.path.join(self.args.save_path, subdir)
        print_rank_0(f"Loading model from {save_path}")

        state_dict = flow.load(save_path, consistent_src_rank=0)
        self.model.load_state_dict(state_dict)

 
class TrainGraph(flow.nn.Graph):
    def __init__(self, args, model, data_loader, criterion=None, optimizer=None, lr_scheduler=None, grad_scaler=None):
        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        if optimizer is not None:
            self.add_optimizer(optimizer, lr_sch=lr_scheduler)
            if grad_scaler is not None:
                self.set_grad_scaler(grad_scaler)

        self.set_activation_checkpointing()
        self.set_pipeline_stage_id()
        self.config.set_gradient_accumulation_steps(args.num_accumulation_steps)

        if args.fp16:
            self.config.enable_amp(True)

        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_cast_scale(True)


    def set_activation_checkpointing(self):
        for module_block in self.model.modules():
            if isinstance(module_block.origin, TransformerLayer):
                module_block.config.activation_checkpointing = True

    def set_pipeline_stage_id(self):
        dist_util = dist.get_dist_util()

        self.data_loader.config.stage_id = dist_util.get_layer_stage_id(0)
        self.data_loader.data_decoder.config.stage_id = dist_util.get_layer_stage_id(0)

        for module_block in self.model.modules():
            if isinstance(module_block.origin, GPTEmbedding):
                module_block.config.stage_id = dist_util.get_layer_stage_id(0)
            elif isinstance(
                module_block.origin, (TransformerLayer, ActivationCheckpointing)
            ):
                module_block.config.stage_id = dist_util.get_layer_stage_id(
                    module_block.origin.layer_idx
                )
            elif isinstance(module_block.origin, (Transformer, ParallelLogits)):
                module_block.config.stage_id = dist_util.get_layer_stage_id(-1)
            else:
                pass

        self.data_loader.label_decoder.config.stage_id = dist_util.get_layer_stage_id(-1)
        self.criterion.config.stage_id = dist_util.get_layer_stage_id(-1)

    def build(self):
        data, label = self.data_loader()
        logits = self.model(data, None, False)
        loss = self.criterion(logits, label)
        loss.backward()
        return loss


class EvalGraph(flow.nn.Graph):
    def __init__(self, args, model, data_loader, criterion=None):
        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        
        self.set_activation_checkpointing()
        self.set_pipeline_stage_id()

        if args.fp16:
            self.config.enable_amp(True)

        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_cast_scale(True)
        self.config.set_gradient_accumulation_steps(args.num_accumulation_steps)

    def set_activation_checkpointing(self):
        for module_block in self.model.modules():
            if isinstance(module_block.origin, TransformerLayer):
                module_block.config.activation_checkpointing = True

    def set_pipeline_stage_id(self):
        dist_util = dist.get_dist_util()

        self.data_loader.config.stage_id = dist_util.get_layer_stage_id(0)
        self.data_loader.data_decoder.config.stage_id = dist_util.get_layer_stage_id(0)

        for module_block in self.model.modules():
            if isinstance(module_block.origin, GPTEmbedding):
                module_block.config.stage_id = dist_util.get_layer_stage_id(0)
            elif isinstance(
                module_block.origin, (TransformerLayer, ActivationCheckpointing)
            ):
                module_block.config.stage_id = dist_util.get_layer_stage_id(
                    module_block.origin.layer_idx
                )
            elif isinstance(module_block.origin, (Transformer, ParallelLogits)):
                module_block.config.stage_id = dist_util.get_layer_stage_id(-1)
            else:
                pass

        self.data_loader.label_decoder.config.stage_id = dist_util.get_layer_stage_id(-1)
        self.criterion.config.stage_id = dist_util.get_layer_stage_id(-1)

    def build(self):
        data, label = self.data_loader()
        with flow.no_grad():
            outputs = logits = self.model(data)
            if self.criterion is not None:
                loss = self.criterion(logits)
                outputs = (logits, loss)
        return outputs