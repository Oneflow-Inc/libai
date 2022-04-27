import oneflow as flow
import pdb

def collate_fn(examples):
    pdb.set_trace()
    labels = []
    inputs = []
    for e in examples:
        inputs.append(e[0])
        labels.append(e[1])
    inputs = flow.FloatTensor(inputs)
    inputs = inputs.view(inputs.shape[0],-1)
    labels = flow.LongTensor(inputs)
    return (inputs,labels)

