# CLIP
Contributor{Xingyu.Liao: sherlockliao01@gmail.com}

> NOTE: We only support inference right now. Stay tuned for training part.

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.

## Approach

![CLIP](CLIP.png)



## Usage

```python
import clip
import oneflow as flow
from PIL import Image

device = "cuda" if flow.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = (
    preprocess(Image.open("CLIP.png"))
    .unsqueeze(0)
    .to_global(sbp=flow.sbp.split(0), placement=flow.placement("cuda", ranks=[0]))
)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to_global(
    sbp=flow.sbp.split(0), placement=flow.placement("cuda", ranks=[0])
)

with flow.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
```
