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
import collections
from collections import defaultdict

import oneflow as flow
import oneflow.nn as nn

from libai.config.config import configurable
from projects.NeRF.modeling.NeRF import Embedding, NeRF


class NerfSystem(nn.Module):
    @configurable
    def __init__(
        self,
        D=8,
        W=256,
        in_channels_xyz=63,
        in_channels_dir=27,
        skips=[4],
        N_samples=64,
        use_disp=False,
        perturb=1.0,
        noise_std=1.0,
        N_importance=128,
        chunk=32 * 1204,
        dataset_type="blender",
        loss_func=None,
    ):
        """
        Args:
            D (int): number of layers for density (sigma) encoder
            W (int): number of hidden units in each layer
            in_channels_xyz (int): number of input channels for xyz (3+3*10*2=63 by default)
            in_channels_dir (int): number of input channels for direction (3+3*4*2=27 by default)
            skips (list(int)): add skip connection in the Dth layer
            N_samples (int): number of coarse samples
            use_disp (bool): use disparity depth sampling
            perturb (float): factor to perturb depth sampling points
            noise_std (float): std dev of noise added to regularize sigma
            N_importance (int): number of additional fine samples
            chunk (int): chunk size to split the input to avoid OOM
            dataset_type (str): the dataset applied for training and evaluating
            loss_func (callable): type of loss function
        """
        super(NerfSystem, self).__init__()
        self.N_samples = N_samples
        self.use_disp = use_disp
        self.perturb = perturb
        self.noise_std = noise_std
        self.N_importance = N_importance
        self.chunk = chunk
        self.white_back = True if dataset_type == "blender" else False
        self.loss_func = nn.MSELoss() if loss_func == None else loss_func
        self.embedding_xyz = Embedding(3, 10)  # 10 is the default number
        self.embedding_dir = Embedding(3, 4)  # 4 is the default number
        self.nerf_coarse = NeRF(
            D=D,
            W=W,
            in_channels_xyz=in_channels_xyz,
            in_channels_dir=in_channels_dir,
            skips=skips,
        )
        self.models = [self.nerf_coarse]
        if N_importance > 0:
            self.nerf_fine = NeRF(
                D=D,
                W=W,
                in_channels_xyz=in_channels_xyz,
                in_channels_dir=in_channels_dir,
                skips=skips,
            )
            self.models += [self.nerf_fine]
        self.models = nn.ModuleList(self.models)

    @classmethod
    def from_config(cls, cfg):
        return {
            "D": cfg.D,
            "W": cfg.W,
            "in_channels_xyz": cfg.in_channels_xyz,
            "in_channels_dir": cfg.in_channels_dir,
            "skips": cfg.skips,
            "N_samples": cfg.N_samples,
            "use_disp": cfg.use_disp,
            "perturb": cfg.perturb,
            "noise_std": cfg.noise_std,
            "N_importance": cfg.N_importance,
            "chunk": cfg.chunk,
            "dataset_type": cfg.dataset_type,
            "loss_func": cfg.loss_func,
        }

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.

        Inputs:
            bins (tensor): (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights (tensor): (N_rays, N_samples_)
            N_importance (int): the number of samples to draw from the distribution
            det (bool): deterministic or not
            eps (float): a small number to prevent division by zero

        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps  # prevent division by zero (don't do inplace op!)
        pdf = weights / flow.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
        cdf = flow.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
        cdf = flow.cat([flow.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)

        # padded to 0~1 inclusive
        if det:
            u = flow.linspace(0, 1, N_importance).to_global(placement=bins.placement, sbp=bins.sbp)
            u = u.expand(N_rays, N_importance)
        else:
            u = flow.rand(N_rays, N_importance).to_global(placement=bins.placement, sbp=bins.sbp)
        u = u.contiguous()

        inds = flow.searchsorted(cdf, u, right=True)
        below = flow.clamp(inds - 1, 0, 1e6)
        above = flow.clamp(inds, -1e6, N_samples_)

        inds_sampled = flow.stack([below, above], -1).view(N_rays, 2 * N_importance)
        cdf_g = flow.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = flow.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[..., 1] - cdf_g[..., 0]

        # denom equals 0 means a bin has weight 0, in which case it will not be sampled
        # anyway, therefore any value for it is fine (set to 1 here)
        denom = flow.masked_fill(denom, denom < eps, 1)
        samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
        return samples

    def inference(
        self,
        N_rays,
        model,
        embedding_xyz,
        xyz_,
        dir_,
        dir_embedded,
        z_vals,
        noise_std=1,
        chunk=1024 * 32,
        white_back=False,
        weights_only=False,
    ):
        """
        Helper function that performs model inference.

        Inputs:
            N_rays (tensor): rays (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
            model (nn.Module): NeRF model (coarse or fine)
            embedding_xyz (nn.Module): embedding module for xyz
            xyz_ (tensor): (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            dir_ (tensor): (N_rays, 3) ray directions
            dir_embedded (tensor): (N_rays, embed_dir_channels) embedded directions
            z_vals (tensor): (N_rays, N_samples_) depths of the sampled positions
            weights_only (tensor): do inference on sigma only or not

        Outputs:
            if weights_only:
                weights (tensor): (N_rays, N_samples_) weights of each sample
            else:
                rgb_final (tensor): (N_rays, 3) the final rgb image
                depth_final (tensor): (N_rays) depth map
                weights (tensor): (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz_.shape[1]
        # Embed directions
        xyz_ = xyz_.view(-1, 3)  # (N_rays*N_samples_, 3)
        if not weights_only:
            dir_embedded = flow.repeat_interleave(
                dir_embedded.to_local(), repeats=N_samples_, dim=0
            ).to_global(
                placement=xyz_.placement, sbp=xyz_.sbp
            )  # (N_rays*N_samples_, embed_dir_channels)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk):
            # Embed positions by chunk
            xyz_embedded = embedding_xyz(xyz_[i : i + chunk])
            if not weights_only:
                xyzdir_embedded = flow.cat([xyz_embedded, dir_embedded[i : i + chunk]], 1)
            else:
                xyzdir_embedded = xyz_embedded
            xyzdir_embedded = xyzdir_embedded
            out_chunk = model(xyzdir_embedded)
            out_chunks = out_chunks + [out_chunk]

        out = flow.cat(out_chunks, 0)
        if weights_only:
            sigmas = out.view(N_rays, N_samples_)
        else:
            rgbsigma = out.view(N_rays, N_samples_, 4)
            rgbs = rgbsigma[..., :3]  # (N_rays, N_samples_, 3)
            sigmas = rgbsigma[..., 3]  # (N_rays, N_samples_)

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:].clone() - z_vals[:, :-1].clone()  # (N_rays, N_samples_-1)
        delta_inf = 1e10 * flow.ones_like(deltas[:, :1]).to_global(
            sbp=deltas.sbp, placement=deltas.placement
        )  # (N_rays, 1) the last delta is infinity
        deltas = flow.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        deltas = deltas * flow.norm(dir_.unsqueeze(1), dim=-1)
        sigmas = sigmas  # TODO: error in it
        noise = (
            flow.randn(sigmas.shape).to_global(placement=sigmas.placement, sbp=sigmas.sbp)
            * noise_std
        )

        # compute alpha by the formula (3)
        alphas = 1 - flow.exp(-deltas * flow.relu(sigmas + noise))  # (N_rays, N_samples_)
        ne_alphas = 1 - alphas + 1e-10
        alphas_shifted = flow.cat(
            [
                flow.ones_like(alphas[:, :1]).to_global(sbp=alphas.sbp, placement=alphas.placement),
                ne_alphas,
            ],
            -1,
        )  # [1, a1, a2, ...]
        alphas = alphas
        rgbs = rgbs
        weights = alphas * flow.cumprod(alphas_shifted, -1)[:, :-1]  # (N_rays, N_samples_)
        # weights = alphas * alphas_shifted[:, :-1]  # (N_rays, N_samples_)
        weights_sum = weights.sum(1)  # (N_rays), the accumulated opacity along the rays
        # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
        if weights_only:
            return weights

        # compute final weighted outputs
        rgb_final = flow.sum(weights.unsqueeze(-1) * rgbs, -2)  # (N_rays, 3)
        depth_final = flow.sum(weights * z_vals, -1)  # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights

    def render_rays(
        self,
        models,
        embeddings,
        rays,
        N_samples=64,
        use_disp=False,
        perturb=0.0,
        N_importance=0.0,
        test_time=False,
        noise_std=1.0,
        chunk=1024 * 32,
        white_back=False,
    ):

        # Extract models from lists
        model_coarse = models[0]
        embedding_xyz = embeddings[0]
        embedding_dir = embeddings[1]

        # Decompose the inputs
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
        near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)

        # Embed direction
        dir_embedded = embedding_dir(rays_d)  # (N_rays, embed_dir_channels)

        # Sample depth points
        z_steps = flow.linspace(0, 1, N_samples).to_global(
            sbp=rays.sbp, placement=rays.placement
        )  # (N_samples)
        if not use_disp:  # use linear sampling in depth space
            z_vals = near * (1 - z_steps) + far * z_steps
        else:  # use linear sampling in disparity space
            z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

        z_vals = z_vals.expand(N_rays, N_samples)

        if perturb > 0:  # perturb sampling depths (z_vals)
            z_vals_mid = 0.5 * (
                z_vals[:, :-1] + z_vals[:, 1:]
            )  # (N_rays, N_samples-1) interval mid points
            # get intervals between samples
            upper = flow.cat([z_vals_mid, z_vals[:, -1:]], -1)
            lower = flow.cat([z_vals[:, :1], z_vals_mid], -1)

            perturb_rand = perturb * flow.rand(z_vals.shape).to_global(
                sbp=rays.sbp, placement=rays.placement
            )
            z_vals = lower + (upper - lower) * perturb_rand

        xyz_coarse_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(
            2
        )  # (N_rays, N_samples, 3)

        if test_time:
            weights_coarse = self.inference(
                rays.shape[0],
                model_coarse,
                embedding_xyz,
                xyz_coarse_sampled,
                rays_d,
                dir_embedded,
                z_vals,
                noise_std,
                chunk,
                white_back,
                weights_only=True,
            )
            result = {"opacity_coarse": weights_coarse.sum(1)}
        else:
            rgb_coarse, depth_coarse, weights_coarse = self.inference(
                rays.shape[0],
                model_coarse,
                embedding_xyz,
                xyz_coarse_sampled,
                rays_d,
                dir_embedded,
                z_vals,
                noise_std,
                chunk,
                white_back,
                weights_only=False,
            )
            result = {
                "rgb_coarse": rgb_coarse,
                "depth_coarse": depth_coarse,
                "opacity_coarse": weights_coarse.sum(1),
            }

        if N_importance > 0:  # sample points for fine model
            z_vals_mid = 0.5 * (
                z_vals[:, :-1] + z_vals[:, 1:]
            )  # (N_rays, N_samples-1) interval mid points
            z_vals_ = self.sample_pdf(
                z_vals_mid, weights_coarse[:, 1:-1], N_importance, det=(perturb == 0)
            )
            # detach so that grad doesn't propogate to weights_coarse from here

            z_vals, _ = flow.sort(flow.cat([z_vals, z_vals_], -1), -1)

            xyz_fine_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
            # (N_rays, N_samples+N_importance, 3)

            model_fine = models[1]
            rgb_fine, depth_fine, weights_fine = self.inference(
                rays.shape[0],
                model_fine,
                embedding_xyz,
                xyz_fine_sampled,
                rays_d,
                dir_embedded,
                z_vals,
                noise_std,
                chunk,
                white_back,
                weights_only=False,
            )
            result["rgb_fine"] = rgb_fine
            result["depth_fine"] = depth_fine
            result["opacity_fine"] = weights_fine.sum(1)

        return result

    def forward_features(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.chunk):
            rendered_ray_chunks = self.render_rays(
                self.models,
                [self.embedding_xyz, self.embedding_dir],
                rays[i : i + self.chunk],
                self.N_samples,
                self.use_disp,
                self.perturb,
                self.N_importance,
                False,
                self.noise_std,
                self.chunk,  # chunk size is effective in val mode
                self.white_back,
            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]
        for k, v in results.items():
            results[k] = flow.cat(v, 0)
        return results

    def forward(self, rays, rgbs, c2w=None, valid_mask=None):
        """
        Inputs:
            rays (tensor): (batchsize, 3+3+2) the set of input rays samples
            rgbs (tensor): (batchsize, 3) the set of input rgbs samples
            c2w (tensor): (3, 4) transformation matrix from camera coordinate to world coordinate
            valid_mask (tensor): (H W) valid color area

        Outputs:
            re (dict): regarding the series of outputs such as rgbs and loss obtained from the model predictions.
        """
        if c2w == None:
            results = self.forward_features(rays)
            losses = self.loss_func(results["rgb_coarse"], rgbs)
            if "rgb_fine" in results:
                losses += self.loss_func(results["rgb_fine"], rgbs)
            return {"losses": losses}
        else:
            rays = rays.squeeze()  # (H*W, 3)
            rgbs = rgbs.squeeze()  # (H*W, 3)
            results = self.forward_features(rays)
            losses = self.loss_func(results["rgb_coarse"], rgbs)
            if "rgb_fine" in results:
                losses += self.loss_func(results["rgb_fine"], rgbs)
            typ = "fine" if "rgb_fine" in results else "coarse"
            re = collections.OrderedDict()
            re["losses"] = losses
            re[typ] = flow.Tensor([0.0]).to_global(sbp=losses.sbp, placement=losses.placement)
            for key, value in results.items():
                re[key] = value.unsqueeze(0)
            re["rgbs"] = rgbs.unsqueeze(0)
            return re
