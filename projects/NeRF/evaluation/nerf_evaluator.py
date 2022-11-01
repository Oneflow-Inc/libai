import copy
import math
import os
from collections import OrderedDict
from datetime import datetime

import cv2
import flowvision.transforms as T
import imageio
import numpy as np
import oneflow as flow
from PIL import Image

from libai.evaluation.cls_evaluator import ClsEvaluator
from libai.utils import distributed as dist


class NerfEvaluator(ClsEvaluator):
    def __init__(self, img_wh, image_save_path=None):
        """
        Args:
            img_wh (tuple(int)): the width and height of the images in the validation set
            image_save_path (str): location of image storage
        """
        super().__init__(topk=(1, 5))
        self.img_wh = img_wh
        self.image_save_path = (
            str(os.path.dirname(os.path.realpath(__file__))) + "/../images"
            if image_save_path is None
            else image_save_path
        )
        if not os.path.exists(self.image_save_path):
            os.makedirs(self.image_save_path)
        self.toimage = T.ToPILImage()

    def current_time(self):
        currentDateAndTime = datetime.now()
        currentTime = currentDateAndTime.strftime("%H_%M_%S")
        return currentTime

    def process(self, inputs, outputs):
        """
        Inputs:
            inputs (dict): Inputs to NeRF System
            outputs (dict): Outputs to NeRF System

        Outputs:
            None
        """
        losses, rgbs = (
            outputs["losses"],
            outputs["rgbs"].squeeze(0),
        )
        typ = list(outputs.keys())[1]
        outputs.pop(typ)
        outputs.pop("losses")
        outputs.pop("rgbs")
        results = {k: v.squeeze(0) for k, v in outputs.items()}
        if len(self._predictions) == 0:
            W, H = self.img_wh
            img = results[f"rgb_{typ}"].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1)  # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            depth = self.visualize_depth(results[f"depth_{typ}"].view(H, W))  # (3, H, W)
            img = self.toimage(img)
            img_gt = self.toimage(img_gt)
            depth = self.toimage(depth)
            img.save(
                os.path.join(self.image_save_path, f"img_{self.current_time()}.png"), quality=100
            )
            img_gt.save(
                os.path.join(self.image_save_path, f"img_gt_{self.current_time()}.png"), quality=100
            )
            depth.save(
                os.path.join(self.image_save_path, f"depth_{self.current_time()}.png"), quality=100
            )
        psnr = self.psnr(results[f"rgb_{typ}"], rgbs)
        self._predictions.append({"losses": losses.item(), "psnr": psnr.item()})

    def evaluate(self):
        if not dist.is_main_process():
            return {}
        else:
            predictions = self._predictions

        total_correct_num = OrderedDict()
        total_correct_num["losses"] = 0
        total_correct_num["psnr"] = 0
        total_samples = 0
        for prediction in predictions:
            losses = prediction["losses"]
            psnr = prediction["psnr"]
            total_correct_num["losses"] += losses
            total_correct_num["psnr"] += psnr
            total_samples += 1

        self._results = OrderedDict()
        for key, value in total_correct_num.items():
            self._results[key] = value / total_samples

        return copy.deepcopy(self._results)

    def visualize_depth(self, depth, cmap=cv2.COLORMAP_JET):
        x = depth.cpu().numpy()
        x = np.nan_to_num(x)  # change nan to 0
        mi = np.min(x)  # get minimum depth
        ma = np.max(x)
        x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
        x = (255 * x).astype(np.uint8)
        x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
        x_ = T.ToTensor()(x_)  # (3, H, W)
        return x_

    def mse(self, image_pred, image_gt, valid_mask=None, reduction="mean"):
        value = (image_pred - image_gt) ** 2
        if valid_mask is not None:
            value = value[valid_mask]
        if reduction == "mean":
            return flow.mean(value)
        return value

    def psnr(self, image_pred, image_gt, valid_mask=None, reduction="mean"):
        return -10 * flow.log(self.mse(image_pred, image_gt, valid_mask, reduction)) / math.log(10)


class NerfVisEvaluator(NerfEvaluator):
    def __init__(self, img_wh, pose_dir_len, name):
        """
        Args:
            img_wh (tuple(int)): the width and height of the images in the validation set
        """
        super().__init__(img_wh=img_wh)
        self.image_list = []
        self.pose_dir_len = pose_dir_len
        self.name = name
        self.mp4_save_path = self.image_save_path

    def to8b(self, x):
        return (255 * np.clip(x, 0, 1)).astype(np.uint8)

    def process(self, inputs, outputs):
        """
        Inputs:
            inputs (dict): Inputs to NeRF System
            outputs (dict): Outputs to NeRF System

        Outputs:
            None
        """
        typ = list(outputs.keys())[0]
        outputs.pop(typ)
        results = {k: v.squeeze(0) for k, v in outputs.items()}
        W, H = self.img_wh
        img = results[f"rgb_{typ}"].view(H, W, 3).cpu().numpy()
        self.image_list.append(img)
        self._predictions.append({"losses": 0.0, "psnr": 0.0})
        if len(self._predictions) == self.pose_dir_len:
            mp4_save_path = os.path.join(self.mp4_save_path, f"{self.name}.mp4")
            imageio.mimwrite(
                mp4_save_path, self.to8b(np.stack(self.image_list, 0)), fps=30, quality=8
            )
            print("successfully save mp4 file!")
