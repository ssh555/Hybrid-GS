# 文件：utils/trainer_swings.py
import os
import torch
from random import randint
from tqdm import tqdm
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from gaussian_renderer import render
from utils.general_utils import knn
from utils.trainer_4dgs import Trainer4DGS

class TrainerSWinGS(Trainer4DGS):
    def __init__(self, dataset, opt, pipe, testing_iterations, saving_iterations, args):
        super().__init__(dataset, opt, pipe, testing_iterations, saving_iterations, args)
        
        self.swin_size = getattr(args, 'swin_size', 50)
        self.total_frames = self.dataset.total_frames
        self.window_start = 0
        self.window_end = min(self.swin_size - 1, self.total_frames - 1)
        
        slide_steps = max(1, self.total_frames - self.swin_size)
        self.slide_interval = max(1, self.opt.iterations // slide_steps)

    def _update_sliding_window(self, iteration):
        """控制时间窗口滑动与高斯寿命延长"""
        if iteration > 0 and iteration % self.slide_interval == 0:
            if self.window_end < self.total_frames - 1:
                self.window_start += 1
                self.window_end += 1
                
                with torch.no_grad():
                    if hasattr(self.gaussians, '_expire_frame') and self.gaussians._expire_frame.numel() > 0:
                        alive_idx = self.gaussians._expire_frame == (self.window_end - 1)
                        self.gaussians._expire_frame[alive_idx] = self.window_end

    def _get_active_dynamic_mask(self, frame_id):
        """
        [SWinGS / HybridGS 核心] 获取当前帧存活的高斯掩码。
        过滤掉那些在当前时间戳尚未出生，或已经死亡的冗余高斯。
        """
        if not hasattr(self.gaussians, '_start_frame') or self.gaussians._start_frame.numel() == 0:
            return None
            
        # 存活条件：出生时间 <= 当前帧 AND 过期时间 >= 当前帧
        active_mask = (self.gaussians._start_frame <= frame_id) & (self.gaussians._expire_frame >= frame_id)
        
        # 融合 HybridGS 的动静硬约束
        if hasattr(self.gaussians, '_mask_dynamic'):
            # 1 是被硬约束冻结的静态背景 (永远可见)，其他则是动态点 (受生命周期限制)
            final_mask = (self.gaussians._mask_dynamic == 1) | ((self.gaussians._mask_dynamic != 1) & active_mask)
            return final_mask
            
        return active_mask

    @torch.no_grad()
    def evaluate(self, iteration):
        """重写测试集评估：引入时间窗口掩码过滤残影"""
        print(f"\n[评估 SWinGS/HybridGS] 正在执行 Iteration {iteration} 的测试集评估...")
        
        test_cameras = self.scene.getTestCameras()
        if not test_cameras: return
            
        total_psnr, total_ssim, total_fps = 0.0, 0.0, 0.0
        
        for idx, batch_data in enumerate(tqdm(test_cameras, desc="Testing")):
            gt_image, viewpoint_cam = batch_data
            gt_image = gt_image.cuda()
            
            # 优先找 fid，找不到就直接用当前测试图片的顺序索引 idx
            frame_id = getattr(viewpoint_cam, 'fid', idx)
            
            # 获取特定帧的干净掩码，过滤掉不在当前时间出生的点
            active_mask = self._get_active_dynamic_mask(frame_id)
            
            # 3. 必须把 active_dynamic_mask 传给渲染器！否则会满屏残影！
            # (如果在 renderer 内部没有处理 kwargs，可以直接作为位置参数传入，取决于你的 render 函数签名)
            render_pkg, fps = self.metrics_tracker.measure_fps(
                render, 
                viewpoint_cam, 
                self.gaussians, 
                self.pipe, 
                self.background, 
                active_dynamic_mask=active_mask  # 核心改动点！
            )
            
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            
            # 4. 计算指标 (复用原版公式)
            total_psnr += psnr(image, gt_image).mean().item()
            total_ssim += ssim(image, gt_image).mean().item()
            total_fps += fps
            
        avg_psnr = total_psnr / len(test_cameras)
        avg_ssim = total_ssim / len(test_cameras)
        avg_fps = total_fps / len(test_cameras)
        
        self.metrics_tracker.record_eval_metrics(iteration, avg_psnr, avg_ssim, avg_fps)
        print(f"[评估结果] PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | FPS: {avg_fps:.2f}")


    def train(self):
        print(f"\n[TrainerSWinGS] 开始长序列滑动窗口训练，窗口大小: {self.swin_size}")
        self.metrics_tracker.start_timer()
        
        # 3D4DGS 的 Dataset 会返回 (gt_image, cam)
        training_dataset = self.scene.getTrainCameras()
        
        # SWinGS 特有：初始化生命周期张量
        if not hasattr(self.gaussians, '_start_frame') or self.gaussians._start_frame.numel() == 0:
            num_pts = self.gaussians.get_xyz.shape[0]
            self.gaussians._start_frame = torch.zeros(num_pts, dtype=torch.int32, device="cuda")
            self.gaussians._expire_frame = torch.full((num_pts,), self.window_end, dtype=torch.int32, device="cuda")
            self.gaussians._mask_dynamic = torch.zeros(num_pts, dtype=torch.int8, device="cuda")

        progress_bar = tqdm(range(self.first_iter + 1, self.opt.iterations + 1), desc="SWinGS Training")
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        
        for iteration in range(self.first_iter + 1, self.opt.iterations + 1):
            iter_start.record()
            self._update_sliding_window(iteration)
            self.gaussians.update_learning_rate(iteration)
            
            if iteration % self.opt.sh_increase_interval == 0:
                self.gaussians.oneupSHdegree()
            if (iteration - 1) == self.args.debug_from:
                self.pipe.debug = True

            batch_size = self.args.batch_size
            batch_point_grad, batch_visibility_filter, batch_radii = [], [], []
            batch_point_grad_static, batch_visibility_filter_static, batch_radii_static = [], [], []
            
            loss = 0
            # =============== 批处理循环 (仅在窗口内采样) ===============
            for batch_idx in range(batch_size):
                # 【核心】：在当前滑动窗口中随机抽帧
                frame_id = randint(self.window_start, self.window_end)
                gt_image, viewpoint_cam = training_dataset[frame_id % len(training_dataset)]
                gt_image, viewpoint_cam = gt_image.cuda(), viewpoint_cam.cuda()

                render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                alpha = render_pkg["alpha"]
                
                viewspace_point_tensor_static = render_pkg.get("viewspace_points_static", [])
                visibility_filter_static = render_pkg.get("visibility_filter_static", [])
                radii_static = render_pkg.get("radii_static", [])

                # 计算基础损失
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
                
                batch_point_grad.append(torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1))
                batch_radii.append(radii)
                batch_visibility_filter.append(visibility_filter)

                static = len(viewspace_point_tensor_static) > 0
                if static:
                    batch_point_grad_static.append(torch.norm(viewspace_point_tensor_static.grad[:,:2], dim=-1))
                    batch_radii_static.append(radii_static)
                    batch_visibility_filter_static.append(visibility_filter_static)

            # =============== 梯度累加 ===============
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

            # =============== 致密化与优化器 ===============
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
                            if hasattr(self.gaussians, 'dynamic2static'):
                                self.gaussians.dynamic2static(self.opt.scale_t_threshold)
                                
                    if iteration % self.opt.opacity_reset_interval == 0 or (hasattr(self.dataset, 'white_background') and self.dataset.white_background and iteration == self.opt.densify_from_iter):
                        self.gaussians.reset_opacity()
                        
                # =============== [核心机制] SWinGS 生命周期梯度衰减 ===============
                if iteration < self.opt.iterations:
                    if hasattr(self.gaussians, '_start_frame') and self.gaussians._start_frame.numel() > 0:
                        age = (self.window_end - self.gaussians._start_frame).clamp(min=1)
                        decay_factor = 1.0 / age.float()
                        if self.gaussians._xyz.grad is not None:
                            self.gaussians._xyz.grad *= decay_factor.unsqueeze(-1)
                            
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    if self.pipe.env_map_res and iteration < self.pipe.env_optimize_until:
                        self.env_map_optimizer.step()
                        self.env_map_optimizer.zero_grad(set_to_none=True)

                # =============== 进度条与评估/保存 ===============
                if iteration % 10 == 0:
                    postfix = {"Loss": f"{loss:.4f}", "Win": f"[{self.window_start}-{self.window_end}]", "Pts(4D)": self.gaussians.get_xyz.shape[0]}
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)

                if iteration == self.opt.iterations:
                    self.evaluate(iteration)
                    os.makedirs(self.args.model_path, exist_ok=True)
                    torch.save((self.gaussians.capture(), iteration), os.path.join(self.args.model_path, f"chkpnt_{iteration}.pth"))
                    self.gaussians.save_ply(os.path.join(self.args.model_path, f"point_cloud_{iteration}.ply"))
                    
        progress_bar.close()
        self.metrics_tracker.save_log(os.path.join(self.args.model_path, "swings_metrics.json"))