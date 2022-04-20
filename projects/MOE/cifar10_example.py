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

# This file is based on the `https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html`.

import argparse

import flowvision as vision
from libai.moe.moe import MoE
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
import oneflow.optim as optim

from .mlp import MLP

def _parse_args():
    parser = argparse.ArgumentParser("flags for train cifar10 example for moe layer")

    parser.add_argument(
        "--data_root", type=str, default='./data', help="load checkpoint"
    )

    # training hyper-parameters
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument("--mom", type=float, default=0.9, help="momentum")
    parser.add_argument("--epochs", type=int, default=10, help="training epochs")
    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="train batch size"
    )
    parser.add_argument("--val_batch_size", type=int, default=64, help="val batch size")

    return parser.parse_args()


def main(args):

    transform = vision.transforms.Compose(
        [vision.transforms.ToTensor(),
         vision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = vision.datasets.CIFAR10(root=args.data_root, train=True,
                                            download=True, transform=transform)
    trainloader = flow.utils.data.DataLoader(trainset, batch_size=args.train_batch_size,
                                             shuffle=True, num_workers=1)

    testset = vision.datasets.CIFAR10(root=args.data_root, train=False,
                                           download=True, transform=transform)
    testloader = flow.utils.data.DataLoader(testset, batch_size=args.val_batch_size,
                                            shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    device = flow.device('cuda')
    expert_network = MLP(input_size=3072, output_size=10, hidden_size=256)
    net = MoE(expert_network, 3072, 10, num_experts=10, noisy_gating=True, k=4)
    net.to(device)


    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.mom)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    for epoch in range(args.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs = inputs.view(inputs.shape[0], -1)
            outputs, aux_loss = net(inputs)
            loss = criterion(outputs, labels)
            total_loss = loss + aux_loss
            total_loss.backward()
            optimizer.step()       

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    with flow.no_grad():
        for i,data in enumerate(testloader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)


            outputs, _ = net(images.view(images.shape[0], -1))
            _, predicted = flow.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

if __name__ == "__main__":
    args = _parse_args()
    main(args)