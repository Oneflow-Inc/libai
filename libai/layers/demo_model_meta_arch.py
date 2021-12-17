import oneflow as flow
from oneflow import nn


class demo_model(nn.Module):
    def __init__(self, input_dim=512, out_dim=64):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        loss = self.get_loss(x)
        return loss
        
    def get_loss(self, x):
        return x.sum()
    

def build_graph(cfg, model, optimizer, lr_scheduler, fp16=False):
    class GraphModel(nn.Graph):
        def __init__(self, cfg, model, optimizer, lr_scheduler, fp16=False):
            super().__init__()
            self.model = model
            self.add_optimizer(optimizer, lr_sch=lr_scheduler)
            self.config.allow_fuse_add_to_output(True)
            self.config.allow_fuse_model_update_ops(True)
            if fp16:
                self.config.enable_amp(True)
                grad_scaler = flow.amp.GradScaler(
                    init_scale=2 ** 30,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=2000,
                )
                self.set_grad_scaler(grad_scaler)

        def build(self, x):
            loss = self.model(x)
            loss.backward()
            return loss
    graph_model = GraphModel(cfg, model, optimizer, lr_scheduler, fp16=False)
    return graph_model, None