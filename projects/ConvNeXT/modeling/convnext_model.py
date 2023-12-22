# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
from oneflow import nn

from libai.config import configurable
from libai.layers import LayerNorm, Linear
from libai.utils import distributed as dist
from projects.ConvNeXT.modeling.convnext_layers import ConvNextEncoder, ConvNextStage
from projects.ConvNeXT.modeling.embedding import ConvNextEmbeddings


class ConvNextModel(nn.Module):
    @configurable
    def __init__(
        self,
        num_channels,
        patch_size,
        num_stages,
        hidden_sizes,
        depths,
        layer_norm_eps=1e-12,
        drop_path_rate=0.0,
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.embeddings = ConvNextEmbeddings(num_channels, hidden_sizes, patch_size, layer_idx=0)
        self.encoder = ConvNextEncoder(hidden_sizes, depths, num_stages, drop_path_rate)
        self.layernorm = LayerNorm(hidden_sizes[-1], eps=layer_norm_eps, layer_idx=-1)

        # weight init
        if os.getenv("ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT", "0") != "1":
            self.apply(self._init_weight)

    def forward(self, x):
        embedding_output = self.embeddings(x)
        encoder_outputs = self.encoder(embedding_output)
        last_hidden_state = encoder_outputs
        pooled_output = self.layernorm(last_hidden_state.mean([-2, -1]))
        return {"last_hidden_state": last_hidden_state, "pooled_output": pooled_output}

    def _init_weight(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_channels": cfg.num_channels,
            "patch_size": cfg.patch_size,
            "num_stages": cfg.num_stages,
            "hidden_sizes": cfg.hidden_sizes,
            "depths": cfg.depths,
            "layer_norm_eps": cfg.layer_norm_eps,
            "drop_path_rate": cfg.drop_path_rate,
            "cfg": cfg,
        }


class ConvNextForImageClassification(nn.Module):
    @configurable
    def __init__(
        self,
        num_channels,
        patch_size,
        num_stages,
        hidden_sizes,
        depths,
        layer_norm_eps=1e-12,
        drop_path_rate=0.0,
        num_labels=None,
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_labels = num_labels
        self.convnext = ConvNextModel(
            num_channels=num_channels,
            patch_size=patch_size,
            num_stages=num_stages,
            hidden_sizes=hidden_sizes,
            depths=depths,
            layer_norm_eps=layer_norm_eps,
            drop_path_rate=drop_path_rate,
            cfg=self.cfg,
        )

        # Classifier head
        self.classifier = (
            Linear(hidden_sizes[-1], self.num_labels, layer_idx=-1)
            if num_labels > 0
            else nn.Identity()
        )

        # weight init
        if os.getenv("ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT", "0") != "1":
            self.apply(self._init_weight)

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_channels": cfg.num_channels,
            "patch_size": cfg.patch_size,
            "num_stages": cfg.num_stages,
            "hidden_sizes": cfg.hidden_sizes,
            "depths": cfg.depths,
            "layer_norm_eps": cfg.layer_norm_eps,
            "drop_path_rate": cfg.drop_path_rate,
            "num_labels": cfg.num_labels,
            "cfg": cfg,
        }

    def _init_weight(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, images, labels):
        outputs = self.convnext(images)
        pooled_output = outputs["pooled_output"]
        logits = self.classifier(pooled_output)

        if labels is not None:
            if self.cfg.problem_type is None:
                if self.num_labels == 1:
                    self.cfg.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == flow.long or labels.dtype == flow.int
                ):
                    self.cfg.problem_type = "single_label_classification"
                else:
                    self.cfg.problem_type = "multi_label_classification"

            if self.cfg.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.cfg.problem_type == "single_label_classification":
                loss = flow._C.sparse_softmax_cross_entropy(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.cfg.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            return {"losses": loss}
        else:
            return {"prediction_scores": logits}

    @staticmethod
    def set_activation_checkpoint(model):
        for module_block in model.convnext.encoder.modules():
            if hasattr(module_block, "origin"):
                # Old API in OneFlow 0.8
                if isinstance(module_block.origin, ConvNextStage):
                    module_block.config.activation_checkpointing = True
            else:
                if isinstance(module_block.to(nn.Module), ConvNextStage):
                    module_block.to(flow.nn.graph.GraphModule).activation_checkpointing = True

    @staticmethod
    def set_pipeline_stage_id(model):
        dist_utils = dist.get_dist_util()

        # Set pipeline parallelism stage_id
        if hasattr(model.convnext.embeddings, "config"):
            # Old API in OneFlow 0.8
            model.convnext.embeddings.config.set_stage(
                dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
            )
            for module_block in model.modules():
                if isinstance(module_block.origin, ConvNextStage):
                    module_block.config.set_stage(
                        dist_utils.get_layer_stage_id(module_block.layer_idx),
                        dist.get_layer_placement(module_block.layer_idx),
                    )
            model.convnext.layernorm.config.set_stage(
                dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
            )
            model.classifier.config.set_stage(
                dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
            )
        else:
            model.convnext.embeddings.to(flow.nn.graph.GraphModule).set_stage(
                dist_utils.get_layer_stage_id(0), dist.get_layer_placement(0)
            )
            for module_block in model.modules():
                if isinstance(module_block.to(nn.Module), ConvNextStage):
                    module_block.to(flow.nn.graph.GraphModule).set_stage(
                        dist_utils.get_layer_stage_id(module_block.layer_idx),
                        dist.get_layer_placement(module_block.layer_idx),
                    )
            model.convnext.layernorm.to(flow.nn.graph.GraphModule).set_stage(
                dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
            )
            model.classifier.to(flow.nn.graph.GraphModule).set_stage(
                dist_utils.get_layer_stage_id(-1), dist.get_layer_placement(-1)
            )
