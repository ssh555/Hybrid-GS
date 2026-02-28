#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import math
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
from collections import defaultdict
from utils.loss_utils import  ssim
from utils.image_utils import psnr
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import json
import time


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        # self.alphamaps = []
        self.rgbmaps = []
        # self.normals = []
        # self.depth_normals = []
        self.viewpoint_stack = []
        self.flowmaps=[]
        self.dynamicmaps=[]
        self.staticmaps=[]

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack, model_path , stage = "validation"):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        metrics = defaultdict(list)
        
        lpips = LearnedPerceptualImagePatchSimilarity(
                    net_type="alex", normalize=True, 
                    ).to("cuda")
        
        fps_list = []
        
        
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            if stage == "validation":
                start_time = time.time()
                render_pkg = self.render(viewpoint_cam[1].cuda(), self.gaussians)
                end_time = time.time()
                fps = 1/(end_time-start_time)
                fps_list.append(fps)
                gt_image = viewpoint_cam[0].cuda()
            else:
                render_pkg = self.render(viewpoint_cam.cuda(), self.gaussians)
                
            rgb = render_pkg['render']
            #normalize rgb to 0-1 
            rgb = torch.clamp(rgb, 0, 1)
            alpha = render_pkg['alpha']
            flow = render_pkg['flow']
            flow = torch.cat([flow, torch.zeros_like(flow[:1])], dim=0)
            depth = render_pkg['depth']
            dynamic = render_pkg['render_4d']
            static = render_pkg['render_3d']

            #image clamp
            rgb = torch.clamp(rgb, 0, 1)
            dynamic = torch.clamp(dynamic, 0, 1)
            static = torch.clamp(static, 0, 1)
                
            self.rgbmaps.append(rgb.cpu())
            # self.depthmaps.append(depth.cpu())
            # self.flowmaps.append(flow.cpu())
            self.dynamicmaps.append(dynamic.cpu())
            self.staticmaps.append(static.cpu())

            
            if stage == "validation":
                metrics["psnr"].append(psnr(gt_image, rgb))
                metrics["ssim"].append(ssim(gt_image, rgb))
                metrics["lpips"].append(lpips(gt_image.unsqueeze(0), rgb.unsqueeze(0)))
        if stage == "validation":
            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "num_GS": self.gaussians.get_xyz.shape[0],
                    "static": self.gaussians.get_static_xyz.shape[0],
                }
            )
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.4f}",
                f"Number of GS: {stats['num_GS']}, Number of static: {stats['static']}",
                f"FPS: {np.mean(fps_list):.2f}, Max FPS: {np.max(fps_list):.2f}",
            )
            # save stats as json
            stats_dir = os.path.join(model_path, "stats")
            os.makedirs(stats_dir, exist_ok=True)
            with open(f"{stats_dir}/{stage}.json", "w") as f:
                json.dump(stats, f)

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")



    @torch.no_grad()
    def export_image(self, path,mode="validation"):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        # vis_path = os.path.join(path, "vis")
        dynamic_path = os.path.join(path, "dynamic")
        static_path = os.path.join(path, "static")
        # flow_path=os.path.join(path,"flow")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(dynamic_path, exist_ok=True)
        os.makedirs(static_path, exist_ok=True)
        # os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        # os.makedirs(flow_path,exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            if mode == "validation" and viewpoint_cam[0] is not None:
                gt = viewpoint_cam[0][0:3, :, :]
                save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.dynamicmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(dynamic_path, 'dynamic_{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.staticmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(static_path, 'static_{0:05d}'.format(idx) + ".png"))
            # save_img_u8(self.flowmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(flow_path, '{0:05d}'.format(idx) + ".png"))
            # save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))