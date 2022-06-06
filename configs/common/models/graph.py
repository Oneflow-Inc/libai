from libai.config import LazyCall

from libai.models.utils import GraphBase

graph = dict(
    # options for graph or eager mode
    enabled=True,
    debug=-1,  # debug mode for graph
    auto_parallel=dict(
        enabled=False,
        prune_parallel_cast_ops=False,
        mainstem_algo=True,
        sbp_collector=False,
    ),
    train_graph=LazyCall(GraphBase)(
        is_train=True,
    ),
    eval_graph=LazyCall(GraphBase)(is_train=False),
)
