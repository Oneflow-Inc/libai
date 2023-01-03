from omegaconf import DictConfig
from libai.config import LazyCall
from libai.models.utils import CTRGraph


graph = dict(
    # options for graph or eager mode
    enabled=True,
    debug=-1,  # debug mode for graph
    train_graph=LazyCall(CTRGraph)(
        is_train=True,
    ),
    eval_graph=LazyCall(CTRGraph)(is_train=False),
)

graph = DictConfig(graph)
