# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np
import oneflow as flow
from flowvision import datasets
from flowvision import transforms as T
from oneflow.utils.data import Dataset
from PIL import Image

from libai.config import LazyCall, instantiate
from libai.data.structures import DistTensorData, Instance


# TODO: Somw tools about the fpm storage, which is not necessarily used
def read_pfm(filename):
    file = open(filename, "rb")

    header = file.readline().decode("utf-8").rstrip()
    if header == "PF":
        color = True
    elif header == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    image = np.flipud(image)

    if image.dtype.name != "float32":
        raise Exception("Image dtype must be float32.")

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

    file.write("PF\n".encode("utf-8") if color else "Pf\n".encode("utf-8"))
    file.write("{} {}\n".format(image.shape[1], image.shape[0]).encode("utf-8"))

    endian = image.dtype.byteorder

    if endian == "<" or endian == "=" and sys.byteorder == "little":
        scale = -scale

    file.write(("%f\n" % scale).encode("utf-8"))

    image.tofile(file)
    file.close()


# TODO: Preparatory conversion tools for 3D rendering
from kornia import create_meshgrid


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T  # (H, W, 3)
    rays_d = rays_d / flow.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    i = flow.tensor(i.numpy())
    j = flow.tensor(j.numpy())
    directions = flow.stack(
        [(i - W / 2) / focal, -(j - H / 2) / focal, -flow.ones_like(i)], -1
    )  # compute about tanx (H, W, 3)

    return directions


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf
    https://pengfeixc.com/blogs/computer-graphics/3D-matrix-transformation-part-three

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1.0 / (W / (2.0 * focal)) * ox_oz
    o1 = -1.0 / (H / (2.0 * focal)) * oy_oz
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = -1.0 / (W / (2.0 * focal)) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1.0 / (H / (2.0 * focal)) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2
    rays_o = flow.stack([o0, o1, o2], -1)  # (B, 3)
    rays_d = flow.stack([d0, d1, d2], -1)  # (B, 3)

    return rays_o, rays_d


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo)


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4 * np.pi, n_poses + 1)[:-1]:  # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))

        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0])  # (3)
        x = normalize(np.cross(y_, z))  # (3)
        y = np.cross(z, x)  # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)]  # (3, 4)

    return np.stack(poses_spiral, 0)  # (n_poses, 3, 4)


def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, -0.9 * t],
                [0, 0, 1, t],
                [0, 0, 0, 1],
            ]
        )

        rot_phi = lambda phi: np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(phi), -np.sin(phi), 0],
                [0, np.sin(phi), np.cos(phi), 0],
                [0, 0, 0, 1],
            ]
        )

        rot_theta = lambda th: np.array(
            [
                [np.cos(th), 0, -np.sin(th), 0],
                [0, 1, 0, 0],
                [np.sin(th), 0, np.cos(th), 0],
                [0, 0, 0, 1],
            ]
        )

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi / 5, radius)]  # 36 degree view downwards
    return np.stack(spheric_poses, 0)


# TODO: Blender and LLFF Datasets


def trun_dict_to_instance(dict):
    return Instance(**{key: DistTensorData(flow.tensor(value)) for key, value in dict.items()})


class NerfBaseDataset(Dataset):
    def __init__(self, root_dir, split, img_wh):
        super(NerfBaseDataset, self).__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.transform = T.Compose([T.ToTensor()])

    def load_meta(self):
        pass


class BlenderDataset(NerfBaseDataset):
    def __init__(self, root_dir, split="train", img_wh=(800, 800), **kwargs):
        """
        Args:
            root_dir: str,
            split: str,
            img_wh: tuple,
        """
        super(BlenderDataset, self).__init__(root_dir, split, img_wh)
        self.white_back = True
        self.load_meta()

    def load_meta(self):
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), "r") as f:
            self.meta = json.load(f)
        w, h = self.img_wh
        camera_angle_x = float(self.meta["camera_angle_x"])
        self.focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        self.directions = get_ray_directions(h, w, self.focal)  # (h, w, 3)

        if self.split == "train":  # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for frame in self.meta["frames"]:
                pose = np.array(frame["transform_matrix"])[:3, :4]
                self.poses += [pose]
                c2w = flow.Tensor(pose)
                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (4, h, w)
                img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
                self.all_rgbs += [img]
                rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
                self.all_rays += [
                    flow.cat(
                        [
                            rays_o,
                            rays_d,
                            self.near * flow.ones_like(rays_o[:, :1]),
                            self.far * flow.ones_like(rays_o[:, :1]),
                        ],
                        1,
                    )
                ]  # (h*w, 8)

            self.all_rays = flow.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = flow.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)

    def __len__(self):
        if self.split == "train":
            return len(self.all_rays)
        if self.split == "val":
            return 8  # only validate 8 images (to support <=8 gpus)
        return len(self.meta["frames"])

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            sample = OrderedDict(rays=self.all_rays[idx], rgbs=self.all_rgbs[idx])

        else:  # create data for each image separately
            frame = self.meta["frames"][idx]
            c2w = flow.Tensor(frame["transform_matrix"])[:3, :4]

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, H, W)
            valid_mask = (img[-1] > 0).flatten()  # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0)  # (H*W, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = flow.concat(
                [
                    rays_o,
                    rays_d,
                    self.near * flow.ones_like(rays_o[:, :1]),
                    self.far * flow.ones_like(rays_o[:, :1]),
                ],
                1,
            )  # (H*W, 8)
            sample = OrderedDict(rays=rays, rgbs=img, c2w=c2w, valid_mask=valid_mask)
        return trun_dict_to_instance(sample)


class LLFFDataset(NerfBaseDataset):
    def __init__(self, root_dir, split="train", img_wh=(504, 378), spheric_poses=False, val_num=1):
        """
        Args:
            root_dir: str,
            split: str,
            img_wh: tuple,
            spheric_poses: bool, whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
            val_num: int, number of val images (used for multigpu training, validate same image
                    for all gpus)
        """
        super(LLFFDataset, self).__init__(root_dir, split, img_wh)
        self.spheric_poses = spheric_poses
        self.val_num = max(1, val_num)  # at least 1
        self.load_meta()
        self.white_back = False

    def load_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))  # (N_images, 17)
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, "images/*")))
        if self.split in ["train", "val"]:
            assert len(poses_bounds) == len(
                self.image_paths
            ), "Mismatch between number of images and number of poses! Please rerun COLMAP!"
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.bounds = poses_bounds[:, -2:]  # (N_images, 2)
        H, W, self.focal = poses[0, :, -1]  # original intrinsics, same for all images
        H, W, self.focal = H.item(), W.item(), self.focal.item()
        assert (
            H * self.img_wh[0] == W * self.img_wh[1]
        ), f"You must set @img_wh to have the same aspect ratio as ({W}, {H}) !"
        self.focal *= self.img_wh[0] / W
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses)
        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        val_idx = np.argmin(distances_from_center)  # choose val image as the closest to
        near_original = self.bounds.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor
        self.directions = get_ray_directions(
            self.img_wh[1], self.img_wh[0], self.focal
        )  # (H, W, 3)

        if self.split == "train":  # create buffer of all rays and rgb data
            # use first N_images-1 to train, the LAST is val
            self.all_rays = []
            self.all_rgbs = []
            for i, image_path in enumerate(self.image_paths):
                if i == val_idx:  # exclude the val image
                    continue
                c2w = flow.Tensor(self.poses[i])
                img = Image.open(image_path).convert("RGB")
                assert (
                    img.size[1] * self.img_wh[0] == img.size[0] * self.img_wh[1]
                ), f"""{image_path} has different aspect ratio than img_wh, 
                        please check your data!"""
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (3, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
                self.all_rgbs += [img]

                rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
                if not self.spheric_poses:
                    near, far = 0, 1
                    rays_o, rays_d = get_ndc_rays(
                        self.img_wh[1], self.img_wh[0], self.focal, 1.0, rays_o, rays_d
                    )
                else:
                    near = self.bounds.min()
                    far = min(8 * near, self.bounds.max())  # focus on central object only

                self.all_rays += [
                    flow.concat(
                        [
                            rays_o,
                            rays_d,
                            near * flow.ones_like(rays_o[:, :1]),
                            far * flow.ones_like(rays_o[:, :1]),
                        ],
                        1,
                    )
                ]  # (h*w, 8)

            self.all_rays = flow.cat(self.all_rays, 0)  # ((N_images-1)*h*w, 8)
            self.all_rgbs = flow.cat(self.all_rgbs, 0)  # ((N_images-1)*h*w, 3)

        elif self.split == "val":
            self.c2w_val = self.poses[val_idx]
            self.image_path_val = self.image_paths[val_idx]

        else:  # for testing, create a parametric rendering path
            if self.split.endswith("train"):  # test on training set
                self.poses_test = self.poses
            elif not self.spheric_poses:
                focus_depth = 3.5  # hardcoded, this is numerically close to the formula
                # given in the original repo. Mathematically if near=1
                # and far=infinity, then this number will converge to 4
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
                self.poses_test = create_spiral_poses(radii, focus_depth)
            else:
                radius = 1.1 * self.bounds.min()
                self.poses_test = create_spheric_poses(radius)

    def __len__(self):
        if self.split == "train":
            return len(self.all_rays)
        if self.split == "val":
            return self.val_num
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            sample = OrderedDict(rays=self.all_rays[idx], rgbs=self.all_rgbs[idx])

        else:
            if self.split == "val":
                c2w = flow.Tensor(self.c2w_val)
            else:
                c2w = flow.Tensor(self.poses_test[idx])

            rays_o, rays_d = get_rays(self.directions, c2w)
            if not self.spheric_poses:
                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(
                    self.img_wh[1], self.img_wh[0], self.focal, 1.0, rays_o, rays_d
                )
            else:
                near = self.bounds.min()
                far = min(8 * near, self.bounds.max())

            rays = flow.cat(
                [
                    rays_o,
                    rays_d,
                    near * flow.ones_like(rays_o[:, :1]),
                    far * flow.ones_like(rays_o[:, :1]),
                ],
                1,
            )  # (h*w, 8)

            sample = OrderedDict(rays=rays, rgbs=None, c2w=c2w)
            if self.split == "val":
                img = Image.open(self.image_path_val).convert("RGB")
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (3, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 3)
                sample["rgbs"] = img

        return trun_dict_to_instance(sample)


#
# if __name__=="__main__":
#     # dataset=BlenderDataset(root_dir="/home/Bigdata/Nerf/nerf_synthetic/chair",split='train')
#     # for img in dataset:
#     #     print(img['rgbs'].shape)
#
#     dataset=LLFFDataset(root_dir="/home/Bigdata/Nerf/nerf_llff_data/fern",split="val")
#     dataset=instantiate(dataset)
#     for img in dataset:
#         img=img.get_fields()
#         print(img)
