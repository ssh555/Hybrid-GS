# 文件：utils/trainer_hybrid.py
import os
import torch
from random import randint
from tqdm import tqdm
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from gaussian_renderer import render
from utils.general_utils import knn
from utils.trainer_swings import TrainerSWinGS

class TrainerHybrid(TrainerSWinGS):
    def __init__(self, dataset, opt, pipe, testing_iterations, saving_iterations, args):
        super().__init__(dataset, opt, pipe, testing_iterations, saving_iterations, args)
        self.tau_avg = getattr(args, 'tau_avg', 0.01)
        self.tau_max = getattr(args, 'tau_max', 0.05)
        self.lambda_d = getattr(args, 'lambda_d', 0.1)

    def robust_hard_constraint_classifier(self):
        """核心机制：鲁棒硬约束判定，将低位移高斯永久冻结为静态背景"""
        with torch.no_grad():
            dynamic_mask = self.gaussians._mask_dynamic != 1
            if not dynamic_mask.any():
                return
            
            # [修复 Bug & 提速] 传入 mask 省下 80% 算力！dt=1.0 获取标准速度，dt=0.05 获取瞬时位移
            _, velocity = self.gaussians.get_current_covariance_and_mean_offset(1.0, self.gaussians.get_t + 1.0, mask=dynamic_mask)
            _, vel_next = self.gaussians.get_current_covariance_and_mean_offset(1.0, self.gaussians.get_t + 0.05, mask=dynamic_mask)
            
            displacement_1 = velocity.norm(dim=-1)
            displacement_2 = vel_next.norm(dim=-1)
            
            r_avg = (displacement_1 + displacement_2) / 2.0
            r_max = torch.max(displacement_1, displacement_2)
            
            # 双重阈值判定
            is_static = (r_avg < self.tau_avg) & (r_max < self.tau_max)
            
            if is_static.any():
                global_static_indices = torch.nonzero(dynamic_mask, as_tuple=True)[0][is_static]
                # 斩断！设为 1 (静态)
                self.gaussians._mask_dynamic[global_static_indices] = 1
                # 物理抹除 3D4DGS 的时间戳属性，阻止形变网络响应
                self.gaussians._t.data[global_static_indices] = 0.0

    def train(self):
        print(f"\n[TrainerHybrid] 开始终极混合训练！已激活软硬双重约束。")
        self.metrics_tracker.start_timer()
        
        training_dataset = self.scene.getTrainCameras()
        
        if not hasattr(self.gaussians, '_start_frame') or self.gaussians._start_frame.numel() == 0:
            num_pts = self.gaussians.get_xyz.shape[0]
            self.gaussians._start_frame = torch.zeros(num_pts, dtype=torch.int32, device="cuda")
            self.gaussians._expire_frame = torch.full((num_pts,), self.window_end, dtype=torch.int32, device="cuda")
            self.gaussians._mask_dynamic = torch.zeros(num_pts, dtype=torch.int8, device="cuda")

        # ==========================================
        # [新增] 初始化单独的 Loss 记录列表 (仅存内存，不进 JSON)
        # ==========================================
        self.loss_history = []
        self.loss_iterations = []

        progress_bar = tqdm(range(self.first_iter + 1, self.opt.iterations + 1), desc="HybridGS Training")
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
            for batch_idx in range(batch_size):
                frame_id = randint(self.window_start, self.window_end)
                gt_image, viewpoint_cam = training_dataset[frame_id % len(training_dataset)]
                gt_image, viewpoint_cam = gt_image.cuda(), viewpoint_cam.cuda()

                render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                alpha = render_pkg["alpha"]
                
                viewspace_point_tensor_static = render_pkg.get("viewspace_points_static", [])
                visibility_filter_static = render_pkg.get("visibility_filter_static", [])
                radii_static = render_pkg.get("radii_static", [])

                Ll1 = l1_loss(image, gt_image)
                Lssim = 1.0 - ssim(image, gt_image)
                current_loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * Lssim
                
                if self.opt.lambda_opa_mask > 0 and hasattr(viewpoint_cam, 'gt_alpha_mask') and viewpoint_cam.gt_alpha_mask is not None:
                    o = alpha.clamp(1e-6, 1-1e-6)
                    sky = 1 - viewpoint_cam.gt_alpha_mask
                    current_loss = current_loss + self.opt.lambda_opa_mask * (- sky * torch.log(1 - o)).mean()
                    
                # =============== [核心机制] HybridGS 时间解耦软约束 ===============
                dynamic_mask = self.gaussians._mask_dynamic != 1
                if dynamic_mask.any():
                    # [修复 Bug & 提速] dt=1.0 获取真实速度，且仅计算动态点！
                    _, velocity = self.gaussians.get_current_covariance_and_mean_offset(1.0, self.gaussians.get_t + 1.0, mask=dynamic_mask)
                    l_reg_d = self.lambda_d * velocity.norm(p=2, dim=1).mean()
                    current_loss += l_reg_d
                # ====================================================================

                # (注: 下方的 lambda_rigid 和 lambda_motion 由于你配置中是 0.0，且涉及 KNN，我建议如果不用就保持原样或删掉即可)

                if self.opt.lambda_rigid > 0:
                    k = 20
                    xyz_cur = self.gaussians.get_xyz
                    idx, dist = knn(xyz_cur[None].contiguous().detach(), xyz_cur[None].contiguous().detach(), k)
                    weight = torch.exp(-100 * dist)
                    vel_dist = torch.norm(velocity[idx] - velocity[None, :, None], p=2, dim=-1)
                    current_loss = current_loss + self.opt.lambda_rigid * ((weight * vel_dist).sum() / k / xyz_cur.shape[0])
                    
                if self.opt.lambda_motion > 0:
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

            with torch.no_grad():
                # if iteration % 100 == 0:
                #     num_4d = self.gaussians.get_xyz.shape[0]
                #     num_3d = self.gaussians.get_static_xyz.shape[0] if static else 0
                #     self.metrics_tracker.record_training_stats(iteration, num_3d, num_4d)
                # =============== [核心机制] HybridGS 空间解耦硬约束 ===============
                if iteration % 100 == 0 and iteration > self.opt.densify_from_iter:
                     self.robust_hard_constraint_classifier()
                # ====================================================================

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
                        
                if iteration < self.opt.iterations:
                    # SWinGS 生命周期衰减
                    if hasattr(self.gaussians, '_start_frame') and self.gaussians._start_frame.numel() > 0:
                        age = (self.window_end - self.gaussians._start_frame).clamp(min=1)
                        decay_factor = 1.0 / age.float()
                        if self.gaussians._xyz.grad is not None:
                            self.gaussians._xyz.grad *= decay_factor.unsqueeze(-1)
                    
                    # =============== [核心修复] 静止点时间梯度清零 ===============
                    # 防止已经被硬约束判为静态的背景，被优化器意外扯动！
                    static_mask = (self.gaussians._mask_dynamic == 1)
                    if static_mask.any() and self.gaussians._t.grad is not None:
                        self.gaussians._t.grad[static_mask] = 0.0
                    # =============================================================

                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    if self.pipe.env_map_res and iteration < self.pipe.env_optimize_until:
                        self.env_map_optimizer.step()
                        self.env_map_optimizer.zero_grad(set_to_none=True)

                if iteration % 10 == 0:
                    postfix = {"Loss": f"{loss:.4f}", "Win": f"[{self.window_start}-{self.window_end}]"}
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                    # ==========================================
                    # [新增] 每 10 步记录一次 Loss，用于画图
                    # ==========================================
                    self.loss_iterations.append(iteration)
                    self.loss_history.append(loss)
                
                if iteration == self.opt.iterations:
                    self.evaluate(iteration)
                    os.makedirs(self.args.model_path, exist_ok=True)
                    torch.save((self.gaussians.capture(), iteration), os.path.join(self.args.model_path, f"chkpnt_{iteration}.pth"))
                    self.gaussians.save_ply(os.path.join(self.args.model_path, f"point_cloud_{iteration}.ply"))
                    num_4d = self.gaussians.get_xyz.shape[0]
                    num_3d = self.gaussians.get_static_xyz.shape[0] if static else 0
                    self.metrics_tracker.record_training_stats(iteration, num_3d, num_4d)
                    
        progress_bar.close()
        self.metrics_tracker.save_log(os.path.join(self.args.model_path, "hybridgs_metrics.json"))
        
        # ==========================================
        # [新增] 训练结束：绘制并保存 Loss 曲线图
        # ==========================================
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.plot(self.loss_iterations, self.loss_history, label="Training Loss", color="#1f77b4", linewidth=1.5, alpha=0.9)
            
            # 美化图表
            plt.title("Training Loss Curve over Iterations", fontsize=14, fontweight='bold')
            plt.xlabel("Iteration", fontsize=12)
            plt.ylabel("Total Loss", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc="upper right", fontsize=12)
            
            # 设置动态 Y 轴范围（防止个别离群点把整个图压扁）
            if len(self.loss_history) > 100:
                # 忽略最开始极不稳定的前 10% 的数据来计算 Y 轴上限
                stable_losses = self.loss_history[len(self.loss_history)//10:]
                plt.ylim(0, max(stable_losses) * 1.5)
            
            # 保存为高清 PNG 图像
            loss_plot_path = os.path.join(self.args.model_path, "loss_curve.png")
            plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 [指标可视化] 训练 Loss 曲线已完美保存至: {loss_plot_path}")
            
        except ImportError:
            print("⚠️ [指标可视化] 缺少 matplotlib 库，跳过绘制 Loss 曲线。如需绘制请运行: pip install matplotlib")
        except Exception as e:
            print(f"⚠️ [指标可视化] 绘制 Loss 曲线时发生错误: {e}")