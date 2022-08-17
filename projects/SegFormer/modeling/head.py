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
        
        if dropout_ratio > 0:
            # self.dropout = nn.Dropout2d(dropout_ratio)    modified
            self.dropout = nn.Dropout(dropout_ratio)
        else:
            self.dropout = None
        
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim, layer_idx=layer_idx)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim, layer_idx=layer_idx)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim, layer_idx=layer_idx)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim, layer_idx=layer_idx)
            
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
            
    # def init_weights(self):
    #     """Initialize weights of classification layer."""
    #     normal_init(self.conv_seg, mean=0, std=0.01)
    
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
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = nn.functional.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = nn.functional.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = nn.functional.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(flow.cat([_c4, _c3, _c2, _c1], dim=1))
        _c = self.batch_norm(_c)
        _c = self.activation(_c)

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


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
