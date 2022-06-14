import oneflow as flow

from libai.utils import distributed as dist
    
    
def convert_to_distributed_default_setting(module):
    """
    Helper function to convert all eager local tensor in :attr:`nn.Module` in the model to
    global tensor with data parallelism as default.
    """
    for _, v in module.state_dict().items():
        if not v.is_global:
            module.to_global(
                sbp=dist.get_nd_sbp([flow.sbp.broadcast, flow.sbp.broadcast]),
                placement=dist.get_layer_placement(0),
            )
            return