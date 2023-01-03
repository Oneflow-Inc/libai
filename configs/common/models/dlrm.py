from omegaconf import DictConfig
from libai.config import LazyCall
from libai.models import DLRMModel


cfg = dict(
    embedding_vec_size = 128,
    bottom_mlp = [512, 256, 128],
    top_mlp = [1024, 1024, 512, 256],
    num_dense_fields = 13,
    num_sparse_fields = 26,
    use_fusedmlp = True,
    persistent_path = "persistent",
    table_size_array = [39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295,39664984,585935,12972,108,36],
    store_type = "device_mem", #"cached_host_mem", "cached_ssd"
    cache_memory_budget_mb = 8192,
    interaction_itself = True,
    interaction_padding = True,
)

cfg = DictConfig(cfg)

dlrm_model = LazyCall(DLRMModel)(cfg=cfg)

