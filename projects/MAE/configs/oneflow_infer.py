from libai.utils.checkpoint import Checkpointer
import oneflow as flow
from projects.MAE.modeling.vit import VisionTransformer
#from flowvision.models.vision_transformer import VisionTransformer
import os
from flowvision import datasets, transforms
from metrics import AverageMeter, accuracy,reduce_tensor
import PIL

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def build_dataset(data_path):
    transform = build_transform()
    root = os.path.join(data_path, 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset


def build_transform():
    input_size = 224
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    # eval transform
    t = []
    crop_pct = 224 / 256
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def evaluate(model, data_loader):
    model.eval()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    for idx, (images, target) in enumerate(data_loader):

        # compute output
        images = images.cuda()
        target = target.cuda()

        output = model(images)["prediction_scores"]
#        output = model(images)

        # measure accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)

        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        print(f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
              f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t')
    print(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

model = VisionTransformer(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    drop_path_rate=0.1,
    global_pool=True,
)
model=model.cuda()
Checkpointer(model).load('/work/libai/output/vit_base_82.658')
model.to_local()
model = flow.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False)

#state_dict = flow.load('/work/libai/vit_base_patch16_224')
#model.load_state_dict(state_dict,strict=False)

dataset_val = build_dataset('/imagenet')
#sampler_val = flow.utils.data.SequentialSampler(dataset_val)
sampler_val = flow.utils.data.distributed.DistributedSampler(dataset_val)
data_loader_val = flow.utils.data.DataLoader(
    dataset_val, sampler=sampler_val,
    batch_size=64,
    num_workers=8,
    drop_last=True
)
import time 
start_time=time.time()
evaluate(model, data_loader_val)
print(f'eval time:{time.time()-start_time}')
