# 文件：utils/trainer_hybrid.py
import os
import torch
import random
from tqdm import tqdm
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from gaussian_renderer import render
from utils.trainer_swings import TrainerSWinGS

class TrainerHybrid(TrainerSWinGS):
    def __init__(self, dataset, opt, pipe, testing_iterations, saving_iterations, args, debug_params):
        super().__init__(dataset, opt, pipe, testing_iterations, saving_iterations, args, debug_params)
        
        self.tau_avg = getattr(args, 'tau_avg', 0.01)
        self.tau_max = getattr(args, 'tau_max', 0.05)
        self.lambda_d = getattr(args, 'lambda_d', 0.1)
        self.use_soft = getattr(args, 'use_soft_constraint', True)
        self.use_hard = getattr(args, 'use_hard_constraint', True)

    def robust_hard_constraint_classifier(self, start_frame, end_frame):
        """保留你的硬约束分类器"""
        with torch.no_grad():
            if hasattr(self.gaussians, '_start_frame') and self.gaussians._start_frame.numel() > 0:
                alive_mask = (self.gaussians._start_frame <= end_frame) & (self.gaussians._expire_frame >= start_frame)
            else:
                alive_mask = torch.ones_like(self.gaussians._mask_dynamic, dtype=torch.bool)
                
            dynamic_mask = (self.gaussians._mask_dynamic != 1) & alive_mask
            if not dynamic_mask.any(): return None
            
            num_steps = min(8, end_frame - start_frame + 1)
            if num_steps < 2: return None
            
            frame_samples = torch.linspace(start_frame, end_frame, steps=num_steps, device="cuda").round().long()
            t_samples = frame_samples.float() / 30.0

            displacements = []
            for t in t_samples:
                _, d = self.gaussians.get_current_covariance_and_mean_offset(1.0, t, mask=dynamic_mask)
                displacements.append(d)
            displacements = torch.stack(displacements, dim=0)

            mean_d = displacements.mean(dim=0) 
            R_avg = torch.norm(displacements - mean_d.unsqueeze(0), p=2, dim=-1).mean(dim=0)
            step_diffs = torch.norm(displacements[1:] - displacements[:-1], p=2, dim=-1)
            R_max = step_diffs.max(dim=0)[0] if step_diffs.shape[0] > 0 else torch.zeros_like(R_avg)
            
            extent = self.scene.cameras_extent  
            dynamic_tau_avg = self.tau_avg * extent
            dynamic_tau_max = self.tau_max * extent

            is_static = (R_avg < dynamic_tau_avg) & (R_max < dynamic_tau_max)

            if is_static.any():
                global_static_mask = torch.zeros_like(self.gaussians._mask_dynamic, dtype=torch.bool)
                global_static_indices = torch.nonzero(dynamic_mask, as_tuple=True)[0][is_static]
                global_static_mask[global_static_indices] = True
                
                if hasattr(self.gaussians, 'kinematic_dynamic2static'):
                    self.gaussians.kinematic_dynamic2static(global_static_mask)
                return global_static_mask
            return None

    def train_phase1_window(self, win_idx, start_frame, end_frame):
        """重写第一阶段：融入 Hybrid 软硬约束，并完美保留绘图与 YAML 配置"""
        print(f"\n🚀 开始 HybridGS 阶段 1: 独立训练窗口 {win_idx} [{start_frame}-{end_frame}]")
        
        if not hasattr(self.gaussians, '_start_frame') or self.gaussians._start_frame.numel() == 0:
            num_pts = self.gaussians.get_xyz.shape[0]
            self.gaussians._start_frame = torch.zeros(num_pts, dtype=torch.int32, device="cuda")
            self.gaussians._expire_frame = torch.zeros(num_pts, dtype=torch.int32, device="cuda")
            self.gaussians._mask_dynamic = torch.zeros(num_pts, dtype=torch.int8, device="cuda")

        self.gaussians._start_frame[:] = start_frame
        self.gaussians._expire_frame[:] = end_frame

        training_dataset = self.scene.getTrainCameras()
        total_iters = self.opt.iterations
        warmup_iters = self.opt.warmup_iterations
        
        progress_bar = tqdm(range(1, total_iters + 1), desc=f"Win {win_idx} Phase 1 (Hybrid)")
        
        for iteration in range(1, total_iters + 1):
            self.global_iter += 1
            
            is_warmup = iteration <= warmup_iters
            if is_warmup and hasattr(self.gaussians, 'set_mlp_requires_grad'):
                self.gaussians.set_mlp_requires_grad(False)
            elif iteration == warmup_iters + 1 and hasattr(self.gaussians, 'set_mlp_requires_grad'):
                print("\n🔥 Warm-up 结束，解冻 MLP 并激活软约束！")
                self.gaussians.set_mlp_requires_grad(True)

            self.gaussians.update_learning_rate(iteration)
            if iteration % self.opt.sh_increase_interval == 0: self.gaussians.oneupSHdegree()

            batch_size = self.args.batch_size
            batch_point_grad, batch_visibility_filter, batch_radii = [], [], []
            batch_point_grad_static, batch_visibility_filter_static, batch_radii_static = [], [], []
            loss = 0
            
            for batch_idx in range(batch_size):
                t_id = random.randint(start_frame, end_frame)
                dataset_idx = random.choice(self.frames_dict[t_id])
                gt_image, viewpoint_cam = training_dataset[dataset_idx]
                gt_image, viewpoint_cam = gt_image.cuda(), viewpoint_cam.cuda()

                active_mask = self._get_active_dynamic_mask(t_id)
                render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background, active_dynamic_mask=active_mask)
                
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                alpha = render_pkg["alpha"]
                
                viewspace_point_tensor_static = render_pkg.get("viewspace_points_static", [])
                visibility_filter_static = render_pkg.get("visibility_filter_static", [])
                radii_static = render_pkg.get("radii_static", [])

                Ll1 = l1_loss(image, gt_image)
                Lssim = 1.0 - ssim(image, gt_image)
                current_loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * Lssim

                # --- 软约束 (解冻后生效) ---
                total_reg_loss = 0.0  
                if self.use_soft and not is_warmup:
                    warmup_start = int(total_iters * self.opt.warmup_start) 
                    warmup_end = int(total_iters * self.opt.warmup_end)
                    if iteration < warmup_start: current_lambda_d = 0.0
                    elif iteration > warmup_end: current_lambda_d = self.lambda_d
                    else: current_lambda_d = self.lambda_d * ((iteration - warmup_start) / (warmup_end - warmup_start))

                    if current_lambda_d > 0:
                        alive_mask = (self.gaussians._start_frame <= end_frame) & (self.gaussians._expire_frame >= start_frame)
                        active_dynamic_mask = (self.gaussians._mask_dynamic != 1) & alive_mask
                        active_indices = torch.nonzero(active_dynamic_mask, as_tuple=True)[0]
                        
                        if active_indices.numel() > 0:
                            if active_indices.numel() > 30000:
                                perm = torch.randperm(active_indices.numel(), device=active_indices.device)[:30000]
                                active_indices = active_indices[perm]
                                sampled_mask = torch.zeros_like(active_dynamic_mask)
                                sampled_mask[active_indices] = True
                                active_dynamic_mask = sampled_mask
                                
                            t_curr = t_id / 30.0
                            t_prev = max(0, t_id - 1) / 30.0
                            _, d_prev = self.gaussians.get_current_covariance_and_mean_offset(1.0, t_prev, mask=active_dynamic_mask)
                            _, d_curr = self.gaussians.get_current_covariance_and_mean_offset(1.0, t_curr, mask=active_dynamic_mask)
                            dt = max(t_curr - t_prev, 1e-6)
                            vel = (d_curr - d_prev) / dt
                            total_reg_loss += current_lambda_d * vel.norm(p=2, dim=1).mean()

                current_loss = current_loss + total_reg_loss
                
                if self.opt.lambda_opa_mask > 0 and hasattr(viewpoint_cam, 'gt_alpha_mask') and viewpoint_cam.gt_alpha_mask is not None:
                    o = alpha.clamp(1e-6, 1-1e-6)
                    sky = 1 - viewpoint_cam.gt_alpha_mask
                    current_loss = current_loss + self.opt.lambda_opa_mask * (- sky * torch.log(1 - o)).mean()

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

            if batch_size > 1:
                visibility_count = torch.stack(batch_visibility_filter,1).sum(1)
                visibility_filter = visibility_count > 0
                radii = torch.stack(batch_radii,1).max(1)[0]
                batch_viewspace_point_grad = torch.stack(batch_point_grad,1).sum(1)
                batch_viewspace_point_grad[visibility_filter] = batch_viewspace_point_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)
                
                has_static_in_batch = len(batch_point_grad_static) > 0
                if has_static_in_batch:
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

            with torch.no_grad():
                if iteration < self.opt.densify_until_iter:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    if batch_size == 1:
                        self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, batch_t_grad if self.gaussians.gaussian_dim == 4 else None)
                    else:
                        self.gaussians.add_densification_stats_grad(batch_viewspace_point_grad, visibility_filter, batch_t_grad if self.gaussians.gaussian_dim == 4 else None)

                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0: 
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        active_grad_threshold = self.opt.densify_grad_threshold
                        
                        try: num_static = self.gaussians.get_static_xyz.shape[0] if (hasattr(self.gaussians, 'get_static_xyz') and self.gaussians.get_static_xyz is not None) else 0
                        except: num_static = 0
                        
                        if (self.gaussians.get_xyz.shape[0] + num_static) >= getattr(self.opt, 'densify_until_num_points', 4000000):
                            active_grad_threshold = 99999.0 
                            
                        self.gaussians.densify_and_prune(active_grad_threshold, self.opt.thresh_opa_prune, self.scene.cameras_extent, size_threshold)
                                
                if iteration % self.opt.opacity_reset_interval == 0:
                    self.gaussians.reset_opacity()

                # --- 硬约束判定 ---
                freeze_start_iter = int(total_iters * self.opt.freeze_start)
                freeze_end_iter = int(total_iters * self.opt.freeze_end)
                if self.use_hard and iteration >= freeze_start_iter and iteration <= freeze_end_iter and iteration % self.opt.freeze_internal == 0:
                    global_static_mask = self.robust_hard_constraint_classifier(start_frame, end_frame)
                    # 【核心保护修改】：一旦转为背景静态点，强制它活到视频最后一帧！
                    # 否则漫游渲染时走到后面的帧，背景会全部消失
                    if global_static_mask is not None:
                        self.gaussians._expire_frame[global_static_mask] = self.total_frames - 1
                # 防止静态点更新
                static_mask = (self.gaussians._mask_dynamic == 1)
                if static_mask.any() and self.gaussians._t.grad is not None:
                    self.gaussians._t.grad[static_mask] = 0.0

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration % 10 == 0:
                try: num_3d = self.gaussians.get_static_xyz.shape[0] if hasattr(self.gaussians, 'get_static_xyz') else 0
                except: num_3d = 0
                
                progress_bar.set_postfix({"Loss": f"{loss:.4f}", "P_4D": self.gaussians.get_xyz.shape[0], "P_3D": num_3d})
                progress_bar.update(10)
                
                self.loss_iterations.append(self.global_iter)
                self.loss_history.append(loss)
                self.pts_4d_history.append(self.gaussians.get_xyz.shape[0])
                self.pts_3d_history.append(num_3d)

            if iteration == self.opt.iterations:
                self.evaluate(self.global_iter, tag=f"Phase1_Win{win_idx}")
                os.makedirs(self.args.model_path, exist_ok=True)
                torch.save(self.gaussians.capture(), os.path.join(self.args.model_path, f"hybrid_phase1_win{win_idx}.pth"))
                try: num_3d = self.gaussians.get_static_xyz.shape[0] if hasattr(self.gaussians, 'get_static_xyz') else 0
                except: num_3d = 0
                self.metrics_tracker.record_training_stats(self.global_iter, num_3d, self.gaussians.get_xyz.shape[0])
                if self.debug_params.save_ply_interval > 0:
                    self.gaussians.save_ply(os.path.join(self.args.model_path, f"debug_point_cloud_win{win_idx}.ply"))

        progress_bar.close()

    def _draw_metrics_chart(self):
        """重写绘图，加入静态3D点云曲线"""
        try:
            import matplotlib.pyplot as plt
            fig, ax1 = plt.subplots(figsize=(12, 6))
            color_loss = '#1f77b4'
            ax1.set_xlabel("Global Iteration", fontsize=12)
            ax1.set_ylabel("Total Loss", color=color_loss, fontsize=12, fontweight='bold')
            line_loss, = ax1.plot(self.loss_iterations, self.loss_history, label="Training Loss", color=color_loss, linewidth=1.5, alpha=0.9)
            ax1.tick_params(axis='y', labelcolor=color_loss)
            ax1.grid(True, linestyle='--', alpha=0.6)
            
            if len(self.loss_history) > 100:
                ax1.set_ylim(0, max(self.loss_history[len(self.loss_history)//10:]) * 1.5)
            
            ax2 = ax1.twinx() 
            color_4d = '#ff7f0e'  
            color_3d = '#2ca02c'  
            ax2.set_ylabel("Number of Gaussian Points", color='#333333', fontsize=12, fontweight='bold')
            line_4d, = ax2.plot(self.loss_iterations, self.pts_4d_history, label="4D Dynamic Points", color=color_4d, linewidth=2.0, alpha=0.85)
            lines = [line_loss, line_4d]
            if sum(self.pts_3d_history) > 0:
                line_3d, = ax2.plot(self.loss_iterations, self.pts_3d_history, label="3D Static Points", color=color_3d, linewidth=2.0, linestyle='--', alpha=0.85)
                lines.append(line_3d)
                
            ax2.tick_params(axis='y', labelcolor='#333333')
            ax1.legend(lines, [l.get_label() for l in lines], loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=11)
            plt.title("HybridGS Loss and Densification over Global Iterations", fontsize=14, fontweight='bold', pad=30)
            loss_plot_path = os.path.join(self.args.model_path, "hybrid_loss_and_points_curve.png")
            plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 [指标可视化] 联动图已保存至: {loss_plot_path}")
        except Exception as e:
            print(f"⚠️ 绘图失败: {e}")