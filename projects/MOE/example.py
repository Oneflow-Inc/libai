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


# --------------------------------------------------------
# MoE Example
# References:
# mixture-of-experts: https://github.com/davidmrau/mixture-of-experts/blob/master/example.py
# --------------------------------------------------------

import pdb

import oneflow as flow
from oneflow import nn
from oneflow.optim import Adam

from libai.moe.moe import MoE



def train(x,y, model, loss_fn, optim):
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    y_hat, aux_loss = model(x.float())
    # calculate prediction loss
    loss = loss_fn(y_hat, y)
    # combine losses
    total_loss = loss + aux_loss
    optim.zero_grad()
    total_loss.backward()
    optim.step()

    print("Training Results - loss: {:.2f}, aux_loss: {:.3f}".format(loss.item(), aux_loss.item()))
    return model

def eval(x, y, model, loss_fn):
    model.eval()
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    y_hat, aux_loss = model(x.float())
    loss = loss_fn(y_hat, y)
    total_loss = loss + aux_loss
    print("Evaluation Results - loss: {:.2f}, aux_loss: {:.3f}".format(loss.item(), aux_loss.item()))



def dummy_data(batch_size, input_size, num_classes):
    # dummy input
    x = flow.rand(batch_size, input_size)

    # dummy target
    y = flow.randint(0,num_classes, (batch_size, 1)).squeeze(1)
    print("x: ",x.shape)
    print("y: ",y.shape)
    return x,y


# arguments
input_size = 1000
num_classes = 20
num_experts = 10
hidden_size = 64
batch_size = 5
k = 4

# determine device
if flow.cuda.is_available():
	device = flow.device('cuda')
else:
	device = flow.device('cpu')
device = flow.device('cpu')


# instantiate the MoE layer
model = MoE(input_size, num_classes, num_experts,hidden_size, k=k, noisy_gating=True, device=device)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optim = Adam(model.parameters())

x, y = dummy_data(batch_size, input_size, num_classes)

# train
train(x.to(device), y.to(device), model, loss_fn, optim)
# evaluate
x, y = dummy_data(batch_size, input_size, num_classes)
eval(x.to(device), y.to(device), model, loss_fn)
