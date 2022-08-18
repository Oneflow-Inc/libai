import oneflow as flow
import oneflow.nn as nn
from libai.layers import Linear

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(5,5)
        

    def forward(self, x):
        x = self.linear(x)
        x = nn.functional.interpolate(x, size=[10, 10], mode='bilinear')
        
        return x


model = NeuralNetwork()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = flow.optim.SGD(model.parameters(), 0.0001)
    


x_ = [flow.rand(1, 2, 5, 5) for _ in range(8)]
y_ = [flow.ones(1, 10, 10, dtype=flow.long) for _ in range(8)]
# print(net(x))

PLACEMENT = flow.placement("cuda", [[0, 1], [2, 3]])
BROADCAST = (flow.sbp.broadcast, flow.sbp.broadcast)
BS0 = (flow.sbp.broadcast, flow.sbp.split(0))

model = model.to_global(placement=PLACEMENT, sbp=BROADCAST)

for _ in range(5):
    for x, y in zip(x_, y_):
        global_x = x.to_global(placement=PLACEMENT, sbp=BS0)
        global_y = y.to_global(placement=PLACEMENT, sbp=BS0)
        pred = model(global_x)
        loss = criterion(pred, global_y)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
# python3 -m oneflow.distributed.launch --nproc_per_node=4 2d_sbp.py
        
 