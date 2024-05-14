import os.path as osp
import sys
import argparse
from PIL import Image

import oneflow as flow
from libai.config import LazyConfig
from libai.engine.default import DefaultTrainer
from libai.utils.checkpoint import Checkpointer
from libai.data.structures import DistTensorData
from flowvision.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from flowvision.transforms import Compose, ToTensor, Normalize

# add "projects/MAE" to PYTHONPATH, to find model normally
sys.path.insert(0, osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'MAE'))


class InferenceEngine:
    def __init__(self, config_file, checkpoint_file, class_name_file):
        cfg = LazyConfig.load(config_file)
        self.model = DefaultTrainer.build_model(cfg).eval()
        Checkpointer(self.model).load(checkpoint_file)

        self.transforms = Compose([ToTensor(),
                                   Normalize(mean=IMAGENET_DEFAULT_MEAN,
                                             std=IMAGENET_DEFAULT_STD)])

        with open(class_name_file) as f:
            self.id2name = [line.strip() for line in f]
    
    def _preprocess_image(self, file_path):
        with open(file_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        
        h, w = img.size
        scale = 224 / w if w >= h else 224 / h
        img = img.resize((int(h * scale), int(w * scale)), Image.Resampling.LANCZOS)
        processed_img = Image.new('RGB', (224, 224), (0, 0, 0))
        processed_img.paste(img)
        img = self.transforms(processed_img)

        return img.unsqueeze(0)

    def infer(self, file_path):
        x = self._preprocess_image(file_path)
        dtd = DistTensorData(x)
        dtd.to_global()
        pred_scores = self.model(dtd.tensor)['prediction_scores']

        return self.id2name[flow.argmax(pred_scores).item()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help="The file path of config")
    parser.add_argument('--model_file', type=str, required=True, help="The file path of model")
    parser.add_argument('--image_file', type=str, required=True, help="The file path of input image")
    parser.add_argument('--class_name_file', type=str, default= './StanfordCars-Class-Names.txt',
                        required=False, help="The file path of class name mapping")
    args = parser.parse_args()

    engine = InferenceEngine(args.config_file, args.model_file, args.class_name_file)
    result = engine.infer(args.image_file)
    print(result)
