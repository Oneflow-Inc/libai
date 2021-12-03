
import oneflow as flow

placement = flow.placement("cuda", {0: [0, 1]})

a = flow.empty((20, 10), dtype=flow.float32, placement=placement, sbp=flow.sbp.split(0))

a[11] = 1
print(a)

# b, c = flow.chunk(a, chunks=2, dim=-1)
# print(b, c)
