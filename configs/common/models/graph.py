from omegaconf import DictConfig
from libai.config import LazyCall
from libai.models.utils import GraphBase

graph = dict(
    # options for graph or eager mode
    enabled=True,
    debug=-1,  # debug mode for graph
    auto_parallel=dict(
        enabled=False,
        enable_auto_parallel_ignore_user_sbp_config=False,  # ignore all .to_global() in graph
        trunk_algo=True,  # consider overlapping calculate time and transfer time
        sbp_collector=False,  # use proxy node when one node transfer to many nodes
    ),
    train_graph=LazyCall(GraphBase)(
        is_train=True,
    ),
    global_mode=dict(
        enabled=False,
    ),
    eval_graph=LazyCall(GraphBase)(is_train=False),
)

graph = DictConfig(graph)
