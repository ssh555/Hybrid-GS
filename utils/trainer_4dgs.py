# 文件：utils/trainer_4dgs.py
import os
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import knn
from utils.trainer_base import BaseTrainer

class Trainer4DGS(BaseTrainer):
    def __init__(self, dataset, opt, pipe, testing_iterations, saving_iterations, args):
        super().__init__(dataset, opt, pipe, testing_iterations, saving_iterations, args)
        
        # 1. 完全按照 3D4DGS 的参数初始化 GaussianModel
        self.gaussians = GaussianModel(
            dataset.sh_degree, 
            gaussian_dim=args.gaussian_dim, 
            time_duration=args.time_duration, 
            rot_4d=args.rot_4d, 
            force_sh_3d=args.force_sh_3d, 
            sh_degree_t=2 if pipe.eval_shfs_4d else 0
        )
        
        # 2. 初始化 Scene (携带 3D4DGS 特有参数)
        os.makedirs(self.args.model_path, exist_ok=True)
        print("\n[Trainer4DGS] 正在通过 3D4DGS Scene 加载数据...")
        self.scene = Scene(
            self.dataset, 
            self.gaussians, 
            num_pts=args.num_pts, 
            num_pts_ratio=args.num_pts_ratio, 
            time_duration=args.time_duration
        )
        self.gaussians.training_setup(opt)

        # 3. 恢复权重
        self.first_iter = 0
        if self.args.start_checkpoint and os.path.exists(self.args.start_checkpoint):
            print(f"[Trainer4DGS] 恢复权重: {self.args.start_checkpoint}")
            (model_params, self.first_iter) = torch.load(self.args.start_checkpoint)
            self.gaussians.restore(model_params, opt)
            
        # 4. 背景与环境光贴图
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if pipe.env_map_res:
            self.env_map = nn.Parameter(torch.zeros((3, pipe.env_map_res, pipe.env_map_res), dtype=torch.float, device="cuda").requires_grad_(True))
            self.env_map_optimizer = torch.optim.Adam([self.env_map], lr=opt.feature_lr, eps=1e-15)
        else:
            self.env_map = None
            self.env_map_optimizer = None
        self.gaussians.env_map = self.env_map

    @torch.no_grad()
    def evaluate(self, iteration):
        print(f"\n[评估] 正在执行 Iteration {iteration} 的测试集评估...")
        test_cameras = self.scene.getTestCameras()
        if not test_cameras: return
            
        total_psnr, total_ssim, total_fps = 0.0, 0.0, 0.0
        for idx, batch_data in enumerate(tqdm(test_cameras, desc="Testing")):
            gt_image, viewpoint_cam = batch_data
            gt_image = gt_image.cuda()
            
            render_pkg, fps = self.metrics_tracker.measure_fps(
                render, viewpoint_cam, self.gaussians, self.pipe, self.background
            )
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            
            total_psnr += psnr(image, gt_image).mean().item()
            total_ssim += ssim(image, gt_image).mean().item()
            total_fps += fps
            
        avg_psnr = total_psnr / len(test_cameras)
        avg_ssim = total_ssim / len(test_cameras)
        avg_fps = total_fps / len(test_cameras)
        
        self.metrics_tracker.record_eval_metrics(iteration, avg_psnr, avg_ssim, avg_fps)

    def train(self):
        print(f"\n[Trainer4DGS] 开始标准 3D4DGS 训练，总迭代次数: {self.opt.iterations}")
        self.metrics_tracker.start_timer()
        
        training_dataset = self.scene.getTrainCameras()
        # 3D4DGS 官方使用 DataLoader 封装 Dataset
        training_dataloader = DataLoader(training_dataset, batch_size=self.args.batch_size, shuffle=True, 
                                         num_workers=12 if self.dataset.dataloader else 0, collate_fn=lambda x: x, drop_last=True)
        
        progress_bar = tqdm(range(self.first_iter + 1, self.opt.iterations + 1), desc="3D4DGS Training")
        iteration = self.first_iter
        
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        
        while iteration < self.opt.iterations:
            for batch_data in training_dataloader:
                iteration += 1
                if iteration > self.opt.iterations: break
                
                iter_start.record()
                self.gaussians.update_learning_rate(iteration)
                
                if iteration % self.opt.sh_increase_interval == 0:
                    self.gaussians.oneupSHdegree()
                if (iteration - 1) == self.args.debug_from:
                    self.pipe.debug = True

                # ================= 3D4DGS 核心前向与损失 =================
                batch_size = self.args.batch_size
                batch_point_grad, batch_visibility_filter, batch_radii = [], [], []
                batch_point_grad_static, batch_visibility_filter_static, batch_radii_static = [], [], []
                
                loss = 0
                for batch_idx in range(batch_size):
                    # 3D4DGS 的 dataset 返回的是 (gt_image, cam)
                    gt_image, viewpoint_cam = batch_data[batch_idx]
                    gt_image, viewpoint_cam = gt_image.cuda(), viewpoint_cam.cuda()

                    render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background)
                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    alpha = render_pkg["alpha"]
                    
                    viewspace_point_tensor_static = render_pkg.get("viewspace_points_static", [])
                    visibility_filter_static = render_pkg.get("visibility_filter_static", [])
                    radii_static = render_pkg.get("radii_static", [])

                    # 基础损失
                    Ll1 = l1_loss(image, gt_image)
                    Lssim = 1.0 - ssim(image, gt_image)
                    current_loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * Lssim
                    
                    # Opa Mask Loss
                    if self.opt.lambda_opa_mask > 0 and hasattr(viewpoint_cam, 'gt_alpha_mask') and viewpoint_cam.gt_alpha_mask is not None:
                        o = alpha.clamp(1e-6, 1-1e-6)
                        sky = 1 - viewpoint_cam.gt_alpha_mask
                        current_loss = current_loss + self.opt.lambda_opa_mask * (- sky * torch.log(1 - o)).mean()
                        
                    # Rigid Loss
                    if self.opt.lambda_rigid > 0:
                        k = 20
                        xyz_cur = self.gaussians.get_xyz
                        idx, dist = knn(xyz_cur[None].contiguous().detach(), xyz_cur[None].contiguous().detach(), k)
                        _, velocity = self.gaussians.get_current_covariance_and_mean_offset(1.0, self.gaussians.get_t + 0.1)
                        weight = torch.exp(-100 * dist)
                        vel_dist = torch.norm(velocity[idx] - velocity[None, :, None], p=2, dim=-1)
                        current_loss = current_loss + self.opt.lambda_rigid * ((weight * vel_dist).sum() / k / xyz_cur.shape[0])
                        
                    # Motion Loss
                    if self.opt.lambda_motion > 0:
                        _, velocity = self.gaussians.get_current_covariance_and_mean_offset(1.0, self.gaussians.get_t + 0.1)
                        current_loss = current_loss + self.opt.lambda_motion * velocity.norm(p=2, dim=1).mean()

                    current_loss = current_loss / batch_size
                    current_loss.backward()
                    loss += current_loss.item()
                    
                    # 收集梯度 (动静分离)
                    batch_point_grad.append(torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1))
                    batch_radii.append(radii)
                    batch_visibility_filter.append(visibility_filter)

                    static = len(viewspace_point_tensor_static) > 0
                    if static:
                        batch_point_grad_static.append(torch.norm(viewspace_point_tensor_static.grad[:,:2], dim=-1))
                        batch_radii_static.append(radii_static)
                        batch_visibility_filter_static.append(visibility_filter_static)

                # ================= 3D4DGS 批量梯度缩放 =================
                if batch_size > 1:
                    visibility_count = torch.stack(batch_visibility_filter,1).sum(1)
                    visibility_filter = visibility_count > 0
                    radii = torch.stack(batch_radii,1).max(1)[0]
                    
                    batch_viewspace_point_grad = torch.stack(batch_point_grad,1).sum(1)
                    batch_viewspace_point_grad[visibility_filter] = batch_viewspace_point_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                    batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)

                    if static:
                        visibility_count_static = torch.stack(batch_visibility_filter_static,1).sum(1)
                        visibility_filter_static = visibility_count_static > 0
                        radii_static = torch.stack(batch_radii_static,1).max(1)[0]
                        batch_viewspace_point_grad_static = torch.stack(batch_point_grad_static,1).sum(1)
                        batch_viewspace_point_grad_static[visibility_filter_static] = batch_viewspace_point_grad_static[visibility_filter_static] * batch_size / visibility_count_static[visibility_filter_static]
                        batch_viewspace_point_grad_static = batch_viewspace_point_grad_static.unsqueeze(1)
                    
                    if self.gaussians.gaussian_dim == 4:
                        batch_t_grad = self.gaussians._t.grad.clone()[:,0].detach()
                        batch_t_grad[visibility_filter] = batch_t_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                        batch_t_grad = batch_t_grad.unsqueeze(1)
                else:
                    if self.gaussians.gaussian_dim == 4:
                        batch_t_grad = self.gaussians._t.grad.clone().detach()
                        
                iter_end.record()

                # ================= 3D4DGS 致密化与修剪 =================
                with torch.no_grad():
                    if iteration % 100 == 0:
                        num_4d = self.gaussians.get_xyz.shape[0]
                        num_3d = self.gaussians.get_static_xyz.shape[0] if static else 0
                        self.metrics_tracker.record_training_stats(iteration, num_3d, num_4d)

                    if iteration < self.opt.densify_until_iter and (self.opt.densify_until_num_points < 0 or self.gaussians.get_xyz.shape[0] < self.opt.densify_until_num_points):
                        self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        if static:
                            self.gaussians.static_max_radii2D[visibility_filter_static] = torch.max(self.gaussians.static_max_radii2D[visibility_filter_static], radii_static[visibility_filter_static])
                        
                        if batch_size == 1:
                            self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, batch_t_grad if self.gaussians.gaussian_dim == 4 else None)
                        else:
                            self.gaussians.add_densification_stats_grad(batch_viewspace_point_grad, visibility_filter, batch_t_grad if self.gaussians.gaussian_dim == 4 else None)
                            if static:
                                self.gaussians.add_densification_stats_grad_static(batch_viewspace_point_grad_static, visibility_filter_static)

                        if iteration > self.opt.densify_from_iter: 
                            size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                            if iteration % self.opt.densification_interval == 0: 
                                self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, self.opt.thresh_opa_prune, self.scene.cameras_extent, size_threshold, self.opt.densify_grad_t_threshold)
                                # 3D4DGS 特有逻辑：动态点转静态点
                                if hasattr(self.gaussians, 'dynamic2static'):
                                    self.gaussians.dynamic2static(self.opt.scale_t_threshold)
                                    
                        if iteration % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                            self.gaussians.reset_opacity()
                            
                    # ================= 优化器步进 =================
                    if iteration < self.opt.iterations:
                        self.gaussians.optimizer.step()
                        self.gaussians.optimizer.zero_grad(set_to_none = True)
                        if self.pipe.env_map_res and iteration < self.pipe.env_optimize_until:
                            self.env_map_optimizer.step()
                            self.env_map_optimizer.zero_grad(set_to_none = True)

                    # 进度条与评估/保存
                    if iteration % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{loss:.4f}", "Pts(4D)": self.gaussians.get_xyz.shape[0], "Pts(3D)": self.gaussians.get_static_xyz.shape[0] if static else 0})
                        progress_bar.update(10)
                    
                    if iteration == self.opt.iterations:
                        self.evaluate(iteration)
                        torch.save((self.gaussians.capture(), iteration), os.path.join(self.args.model_path, f"chkpnt_{iteration}.pth"))
                        self.gaussians.save_ply(os.path.join(self.args.model_path, f"point_cloud_{iteration}.ply"))
                        
        progress_bar.close()
        self.metrics_tracker.save_log(os.path.join(self.args.model_path, "baseline_metrics.json"))