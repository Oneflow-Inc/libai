import oneflow as flow
import oneflow.nn as nn
from flowvision.layers import trunc_normal_

from libai.layers import Linear, BatchNorm2d
from libai.utils import distributed as dist

class DecodeHead(nn.Module):
    """Class for DecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """
    def __init__(self,
                 in_channels,
                 *,
                 num_classes,
                 feature_strides=[4, 8, 16, 32],
                 dropout_ratio=0.1,
                 in_index=[0, 1, 2, 3],
                 input_transform='multiple_select',
                 embedding_dim=256,
                 align_corners=False,
                 layer_idx=0,
                 ):
        super(DecodeHead, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.dropout_ratio = dropout_ratio
        self.in_index = in_index
        self.align_corners = align_corners
        self.feature_strides = feature_strides
        self.layer_idx = layer_idx
        
        if dropout_ratio > 0:
            # self.dropout = nn.Dropout2d(dropout_ratio)
            self.dropout = nn.Dropout(dropout_ratio)
        else:
            self.dropout = None
        
        mlps = []
        for i in range(4):
            mlp = MLP(input_dim=self.in_channels[i], embed_dim=self.embedding_dim, layer_idx=layer_idx)
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)
            
        self.linear_fuse = nn.Conv2d(
            in_channels=self.embedding_dim*4,
            out_channels=self.embedding_dim,
            kernel_size=1,
            bias=False,
        ).to_global(
            placement=dist.get_layer_placement(layer_idx),
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        )
        self.batch_norm = BatchNorm2d(self.embedding_dim, layer_idx=layer_idx)
        self.activation = nn.ReLU()
        
        self.linear_pred = nn.Conv2d(self.embedding_dim, self.num_classes, kernel_size=1).to_global(
            placement=dist.get_layer_placement(layer_idx),
            sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
        )
            
        
    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    
    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            raise Exception(print("transform not implement error"))
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs
        
    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        x = [i.to_global(placement=dist.get_layer_placement(self.layer_idx)) for i in x]
        batch_size = x[0].shape[0]
        
        ############## MLP decoder on C1-C4 ###########
        all_hidden_states = []
        for x_, mlp in zip(x, self.linear_c):
            height, weight = x_.shape[-2:]
            
            encoder_hidden_state = mlp(x_).permute(0,2,1).reshape(batch_size, -1, height, weight)
            encoder_hidden_state = nn.functional.interpolate(encoder_hidden_state, size=x[0].size()[2:], mode='bilinear',align_corners=False)
            all_hidden_states.append(encoder_hidden_state)
      
        hidden_states = self.linear_fuse(flow.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        logits = self.linear_pred(hidden_states)

        return logits


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768, layer_idx=0):
        super().__init__()
        self.proj = Linear(input_dim, embed_dim, layer_idx=layer_idx)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
