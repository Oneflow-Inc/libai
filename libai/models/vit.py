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

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F

from .build import MODEL_ARCH_REGISTRY

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PositionEmbs(nn.Module):
    """Position Embedding
    """
    def __init__(self, num_patches, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(flow.randn(1, num_patches+1, hidden_dim, dtype=flow.float32, requires_grad=True))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x):
        out = x + self.pos_embedding

        if self.dropout:
            out = self.dropout(out)
        
        return out


class MlpBlock(nn.Module):
    """ Transformer Feed-Forward Block """
    def __init__(self, in_dim, mlp_dim, out_dim, act_layer=nn.GELU, dropout=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = act_layer()
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.dropout2(out)

        return out


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** 0.5

        self.query = nn.Linear(hidden_dim, self.num_heads * self.head_dim)
        self.key = nn.Linear(hidden_dim, self.num_heads * self.head_dim)
        self.value = nn.Linear(hidden_dim, self.num_heads * self.head_dim)
        self.out = nn.Linear(hidden_dim, self.num_heads * self.head_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def transpose_for_scores(self, x):
        b, num_patches, _ = x.shape
        x = x.view(b, num_patches, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, x):
        b, n, _ = x.shape
        # linear projection to get q,k,v
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # transpose matrix for attention computing
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # calculate attention map
        attn_map = flow.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_map = attn_map.softmax(dim=-1)

        # linear projection for merging heads information
        output = flow.matmul(attn_map, v)
        output = output.permute(0, 2, 1, 3)
        output_shape = tuple(output.size()[: -2]) + (self.num_heads * self.head_dim,)
        output = output.view(*output_shape)
        output = self.out(output)

        return self.dropout(output)


class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, num_heads, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, num_heads, dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MlpBlock(hidden_dim, mlp_dim, hidden_dim, dropout=dropout)
    
    def forward(self, x):
        # self-attention
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        out += residual

        # feed-forward
        residual = out
        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out
    

class Encoder(nn.Module):
    def __init__(self, num_patches, hidden_dim, mlp_dim, num_layers=12, num_heads=12, attn_dropout=0.0, dropout=0.1):
        super().__init__()

        # position embedding
        self.pos_embedding = PositionEmbs(num_patches, hidden_dim, dropout)

        # encoder blocks
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(
                hidden_dim, mlp_dim, num_heads, attn_dropout, dropout,
            )
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        out = self.pos_embedding(x)

        for layer in self.encoder_layers:
            out = layer(out)
        
        out = self.norm(out)
        return out


@MODEL_ARCH_REGISTRY.register()
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, hidden_dim=768, mlp_dim=3072, num_heads=12, num_layers=12, num_classes=1000, attn_dropout=0.0, dropout=0.1):
        super().__init__()
        image_h, image_w = pair(img_size)
        patch_h, patch_w = pair(patch_size)
        assert image_h % patch_h == 0 and image_w % patch_w == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_h // patch_h) * (image_w // patch_w)
        self.embedding = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)

        # cls token
        self.cls_token = nn.Parameter(flow.zeros(1, 1, hidden_dim))

        # transformer encoder
        self.transformer = Encoder(
            num_patches=num_patches,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            dropout=dropout,
        )

        # classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.permute(0, 2, 3, 1)
        b, h, w, c = emb.shape
        emb = emb.view(b, h*w, c)

        # prepend cls token
        cls_token = self.cls_token.repeat(b, 1, 1)
        emb = flow.cat([cls_token, emb], dim=1)

        # transformer
        feat = self.transformer(emb)

        # classifier
        logits = self.classifier(feat[:, 0])
        return logits