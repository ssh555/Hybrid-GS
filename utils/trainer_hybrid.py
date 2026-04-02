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

        # 读取消融开关，默认全开（满血 HybridGS）
        self.use_soft = getattr(args, 'use_soft_constraint', True)
        self.use_hard = getattr(args, 'use_hard_constraint', True)

    def robust_hard_constraint_classifier(self):
        """核心机制：基于物理学单帧绝对位移的精准冻结"""
        with torch.no_grad():
            if hasattr(self.gaussians, '_start_frame') and self.gaussians._start_frame.numel() > 0:
                alive_mask = (self.gaussians._start_frame <= self.window_end) & (self.gaussians._expire_frame >= self.window_start)
            else:
                alive_mask = torch.ones_like(self.gaussians._mask_dynamic, dtype=torch.bool)
                
            dynamic_mask = (self.gaussians._mask_dynamic != 1) & alive_mask
            if not dynamic_mask.any():
                return None
            
            t_plus_1 = self.gaussians.get_t + 1.0
            _, physical_velocity = self.gaussians.get_current_covariance_and_mean_offset(1.0, t_plus_1, mask=dynamic_mask)
            duration = self.gaussians.time_duration[1] - self.gaussians.time_duration[0]
            frame_time = duration / self.total_frames if hasattr(self, 'total_frames') and self.total_frames > 0 else 0.0333
            print(f"[INFO] Duration: {duration:.2f}s, Frame Time: {frame_time:.4f}s, Checking {dynamic_mask.sum().item()} dynamic points for hard constraint...")
            frame_displacement = physical_velocity.norm(dim=-1) * frame_time
            is_static = (frame_displacement < self.tau_avg) & (frame_displacement < self.tau_max)
            
            if is_static.any():
                global_static_mask = torch.zeros_like(self.gaussians._mask_dynamic, dtype=torch.bool)
                global_static_indices = torch.nonzero(dynamic_mask, as_tuple=True)[0][is_static]
                global_static_mask[global_static_indices] = True
                if global_static_mask is not None and global_static_mask.any():
                    if hasattr(self.gaussians, 'kinematic_dynamic2static'):
                        # 🚀 调用自定义的物理转移函数
                        self.gaussians.kinematic_dynamic2static(global_static_mask)
                    else:
                        print("⚠️ 架构缺失：请在 gaussian_model.py 中实现 kinematic_dynamic2static(mask)！")
                return global_static_mask
            
            return None
        
    def train(self):
        print(f"\n[TrainerHybrid] 开始终极混合训练！已激活软硬双重约束。")
        self.metrics_tracker.start_timer()
        
        training_dataset = self.scene.getTrainCameras()
        
        # ==========================================
        # 🌟 核心修复：建立 [帧号 -> 相机索引] 的全映射字典
        # 严格基于 camxx_xxxx 命名规范，完美支持多相机与乱序
        # ==========================================
        import random
        self.frames_dict = {}
        for idx, cam in enumerate(training_dataset):
            try:
                # 提取 cam.image_name (如 "cam00_0150") 的后半段作为帧号
                frame_id = int(cam.image_name.split('_')[-1])
            except:
                # 极端情况 fallback
                frame_id = getattr(cam, 'fid', idx % self.total_frames)
                
            if frame_id not in self.frames_dict:
                self.frames_dict[frame_id] = []
            self.frames_dict[frame_id].append(idx)

        # ==========================================
        # 🚀 植入动态滑动缓存池：永远只占用当前窗口的内存！
        # ==========================================
        self.window_cache = {}

        if not hasattr(self.gaussians, '_start_frame') or self.gaussians._start_frame.numel() == 0:
            num_pts = self.gaussians.get_xyz.shape[0]
            self.gaussians._start_frame = torch.zeros(num_pts, dtype=torch.int32, device="cuda")
            self.gaussians._expire_frame = torch.full((num_pts,), self.window_end, dtype=torch.int32, device="cuda")
            self.gaussians._mask_dynamic = torch.zeros(num_pts, dtype=torch.int8, device="cuda")

        # ==========================================
        # [修改] 初始化记录列表，增加点云数量监控
        # ==========================================
        self.loss_history = []
        self.loss_iterations = []
        self.pts_4d_history = []  # 记录 4D 动态高斯点数量
        self.pts_3d_history = []  # 记录 3D 静态高斯点数量

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
                # 1. 窗口采样与数据读取
                t_id = random.randint(self.window_start, self.window_end)
                dataset_idx = random.choice(self.frames_dict[t_id])
                frame_id = t_id

                if dataset_idx not in self.window_cache:
                    self.window_cache[dataset_idx] = training_dataset[dataset_idx]
                gt_image, viewpoint_cam = self.window_cache[dataset_idx]
                gt_image, viewpoint_cam = gt_image.cuda(), viewpoint_cam.cuda()

                active_mask = self._get_active_dynamic_mask(frame_id)
                render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background, active_dynamic_mask=active_mask)

                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                alpha = render_pkg["alpha"]
                
                viewspace_point_tensor_static = render_pkg.get("viewspace_points_static", [])
                visibility_filter_static = render_pkg.get("visibility_filter_static", [])
                radii_static = render_pkg.get("radii_static", [])

                # 2. 基础图像 Loss
                Ll1 = l1_loss(image, gt_image)
                Lssim = 1.0 - ssim(image, gt_image)
                current_loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * Lssim
                
                if self.opt.lambda_opa_mask > 0 and hasattr(viewpoint_cam, 'gt_alpha_mask') and viewpoint_cam.gt_alpha_mask is not None:
                    o = alpha.clamp(1e-6, 1-1e-6)
                    sky = 1 - viewpoint_cam.gt_alpha_mask
                    current_loss = current_loss + self.opt.lambda_opa_mask * (- sky * torch.log(1 - o)).mean()

                # =========================================================
                # 3. 终极修复：统一 HybridGS 软硬双重约束 (完全消融解耦，防爆，防时间穿梭)
                # =========================================================
                total_reg_loss = 0.0  
                
                # 🌟 修复：严格使用当前真实帧号的归一化时间
                current_t = frame_id / self.total_frames if hasattr(self, 'total_frames') else self.gaussians.get_t

                # --- 软约束模块 (控制动态点的平滑度与 KNN 刚性) ---
                if self.use_soft:
                    warmup_start = int(self.opt.iterations * self.opt.warmup_start) 
                    warmup_end = int(self.opt.iterations * self.opt.warmup_end)
                    if iteration < warmup_start: current_lambda_d = 0.0
                    elif iteration > warmup_end: current_lambda_d = self.lambda_d
                    else: current_lambda_d = self.lambda_d * ((iteration - warmup_start) / (warmup_end - warmup_start))

                    if current_lambda_d > 0 or self.opt.lambda_rigid > 0:
                        if hasattr(self.gaussians, '_start_frame') and self.gaussians._start_frame.numel() > 0:
                            alive_mask = (self.gaussians._start_frame <= self.window_end) & (self.gaussians._expire_frame >= self.window_start)
                        else:
                            alive_mask = torch.ones_like(self.gaussians._mask_dynamic, dtype=torch.bool)
                        
                        active_dynamic_mask = (self.gaussians._mask_dynamic != 1) & alive_mask
                        active_indices = torch.nonzero(active_dynamic_mask, as_tuple=False).squeeze()
                        
                        if active_indices.numel() > 0:
                            # 绝对防爆：强制最多只抽 30,000 点
                            if active_indices.numel() > 30000:
                                perm = torch.randperm(active_indices.numel(), device=active_indices.device)[:30000]
                                active_indices = active_indices[perm]
                                sampled_mask = torch.zeros_like(active_dynamic_mask)
                                sampled_mask[active_indices] = True
                                active_dynamic_mask = sampled_mask

                            # 正确使用 current_t 算速度！
                            t_plus_1 = self.gaussians.get_t + 1.0
                            _, physical_velocity = self.gaussians.get_current_covariance_and_mean_offset(1.0, t_plus_1, mask=active_dynamic_mask)
                            
                            if current_lambda_d > 0:
                                total_reg_loss += current_lambda_d * physical_velocity.norm(p=2, dim=1).mean()

                            # KNN 刚性约束
                            if self.opt.lambda_rigid > 0 and active_indices.numel() > 10:
                                k_neighbors = 10
                                xyz_dynamic = self.gaussians.get_xyz[active_dynamic_mask].contiguous()
                                idx, dist = knn(xyz_dynamic[None].detach(), xyz_dynamic[None].detach(), k_neighbors)
                                weight = torch.exp(-100 * dist)
                                vel_dist = torch.norm(physical_velocity[idx.squeeze(0)] - physical_velocity.unsqueeze(1), p=2, dim=-1)
                                coherence_loss = (weight * vel_dist).sum() / k_neighbors / xyz_dynamic.shape[0]
                                total_reg_loss += self.opt.lambda_rigid * 5.0 * coherence_loss

                # --- 硬约束模块 ---
                if self.use_hard:
                    static_mask = (self.gaussians._mask_dynamic == 1)
                    if static_mask.any():
                        static_indices = torch.nonzero(static_mask, as_tuple=False).squeeze()
                        
                        # 备用防爆针：防止几百万个静态点把显存撑爆
                        if static_indices.numel() > 30000:
                            perm = torch.randperm(static_indices.numel(), device=static_indices.device)[:30000]
                            static_indices = static_indices[perm]
                            sampled_static_mask = torch.zeros_like(static_mask)
                            sampled_static_mask[static_indices] = True
                        else:
                            sampled_static_mask = static_mask

                        t_plus_1_static = self.gaussians.get_t + 1.0
                        _, static_velocity = self.gaussians.get_current_covariance_and_mean_offset(1.0, t_plus_1_static, mask=sampled_static_mask)
                        total_reg_loss += 10.0 * static_velocity.norm(p=2, dim=1).mean()


                # =========================================================
                # 4. 统一反向传播 (剔除性能毒瘤，全场唯一的 Backward!)
                # =========================================================
                current_loss = current_loss + total_reg_loss
                current_loss = current_loss / batch_size
                current_loss.backward()  # <--- 唯一下达梯度指令的地方
                loss += current_loss.item()
                
                # 5. 梯度收集 (维持你写好的静动态点分离逻辑)
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
                    
            iter_end.record()

            with torch.no_grad():
                if iteration < self.opt.densify_until_iter:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    
                    if batch_size == 1:
                        self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, batch_t_grad if self.gaussians.gaussian_dim == 4 else None)
                    else:
                        self.gaussians.add_densification_stats_grad(batch_viewspace_point_grad, visibility_filter, batch_t_grad if self.gaussians.gaussian_dim == 4 else None)

                    if iteration > self.opt.densify_from_iter: 
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        if iteration % self.opt.densification_interval == 0: 
                            
                            # 🚀 动态阈值法：让系统“只排泄，不进食”
                            active_grad_threshold = self.opt.densify_grad_threshold
                            active_grad_t_threshold = getattr(self.opt, 'densify_grad_t_threshold', 0.00005)
                            
                            # 触顶防爆：如果点数超过上限，把生点门槛拉到极其巨大 (99999.0)
                            # 这样它绝对生不出新点，但底层的 prune (修剪) 依然会完美执行，清理显存！
                            try:
                                num_static = self.gaussians.get_static_xyz.shape[0] if (hasattr(self.gaussians, 'get_static_xyz') and self.gaussians.get_static_xyz is not None) else 0
                            except:
                                num_static = 0
                            current_pts = self.gaussians.get_xyz.shape[0] + num_static
                            max_points = getattr(self.opt, 'densify_until_num_points', 4000000)
                            if max_points > 0 and current_pts >= max_points:
                                active_grad_threshold = 99999.0 
                                active_grad_t_threshold = 99999.0 

                            self.gaussians.densify_and_prune(active_grad_threshold, self.opt.thresh_opa_prune, self.scene.cameras_extent, size_threshold, active_grad_t_threshold)
                            
                            # if hasattr(self.gaussians, 'dynamic2static'):
                            #     self.gaussians.dynamic2static(self.opt.scale_t_threshold)
                            # 硬约束冻结 代替 原3D4DGS冻结
                                
                # 大扫除独立出来
                if iteration % self.opt.opacity_reset_interval == 0 or (hasattr(self.dataset, 'white_background') and self.dataset.white_background and iteration == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

                # 硬约束冻结 代替 原3D4DGS冻结
                # =============== [核心机制] HybridGS 空间解耦硬约束 ===============
                freeze_start_iter = int(self.opt.iterations * self.opt.freeze_start)
                freeze_end_iter = int(self.opt.iterations * self.opt.freeze_end)
                
                # 在窗口滑动时，触发严格的物理降维
                if self.use_hard and iteration > freeze_start_iter and iteration < freeze_end_iter and iteration % self.slide_interval == 0:
                    self.robust_hard_constraint_classifier()
                # ====================================================================

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
                    postfix = {"Loss": f"{loss:.4f}", "Win": f"[{self.window_start}-{self.window_end}]", "Pts4" : f"{self.gaussians.get_xyz.shape[0]}", "Pts3" : f"{self.gaussians.get_static_xyz.shape[0] if static else 0}"}
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                    
                    # ==========================================
                    # [修改] 每 10 步记录 Loss 和 高斯点数量
                    # ==========================================
                    self.loss_iterations.append(iteration)
                    self.loss_history.append(loss)
                    
                    # 安全获取 4D 动态点数量
                    num_4d = self.gaussians.get_xyz.shape[0] if self.gaussians.get_xyz is not None else 0
                    self.pts_4d_history.append(num_4d)
                    
                    # 安全获取 3D 静态点数量
                    try:
                        num_3d = self.gaussians.get_static_xyz.shape[0] if (hasattr(self.gaussians, 'get_static_xyz') and self.gaussians.get_static_xyz is not None) else 0
                    except:
                        num_3d = 0
                    self.pts_3d_history.append(num_3d)
                
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
        # [修改] 训练结束：绘制并保存双 Y 轴指标图 (Loss + Points)
        # ==========================================
        try:
            import matplotlib.pyplot as plt
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # --- 1. 左侧 Y 轴：绘制 Loss 曲线 ---
            color_loss = '#1f77b4'
            ax1.set_xlabel("Iteration", fontsize=12)
            ax1.set_ylabel("Total Loss", color=color_loss, fontsize=12, fontweight='bold')
            line_loss, = ax1.plot(self.loss_iterations, self.loss_history, label="Training Loss", color=color_loss, linewidth=1.5, alpha=0.9)
            ax1.tick_params(axis='y', labelcolor=color_loss)
            ax1.grid(True, linestyle='--', alpha=0.6)
            
            # 设置动态 Loss Y 轴范围（防止离群点把图压扁）
            if len(self.loss_history) > 100:
                stable_losses = self.loss_history[len(self.loss_history)//10:]
                ax1.set_ylim(0, max(stable_losses) * 1.5)
            
            # --- 2. 右侧 Y 轴：绘制点云数量曲线 ---
            ax2 = ax1.twinx()  # 创建共享 X 轴的第二个 Y 轴
            color_4d = '#ff7f0e'  # 橙色代表 4D 动态点
            color_3d = '#2ca02c'  # 绿色代表 3D 静态点
            
            ax2.set_ylabel("Number of Gaussian Points", color='#333333', fontsize=12, fontweight='bold')
            line_4d, = ax2.plot(self.loss_iterations, self.pts_4d_history, label="4D Dynamic Points", color=color_4d, linewidth=2.0, alpha=0.85)
            
            lines = [line_loss, line_4d]
            
            # 如果存在 3D 静态点，也画上去
            if sum(self.pts_3d_history) > 0:
                line_3d, = ax2.plot(self.loss_iterations, self.pts_3d_history, label="3D Static Points", color=color_3d, linewidth=2.0, linestyle='--', alpha=0.85)
                lines.append(line_3d)
                
            ax2.tick_params(axis='y', labelcolor='#333333')
            
            # --- 3. 图例与保存 ---
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=11)
            
            plt.title("Training Loss and Densification over Iterations", fontsize=14, fontweight='bold', pad=30)
            
            # 保存为高清 PNG 图像
            loss_plot_path = os.path.join(self.args.model_path, "loss_and_points_curve.png")
            plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 [指标可视化] 训练 Loss 与高斯点数量联动图已保存至: {loss_plot_path}")
            
        except ImportError:
            print("⚠️ [指标可视化] 缺少 matplotlib 库，跳过绘制图表。如需绘制请运行: pip install matplotlib")
        except Exception as e:
            print(f"⚠️ [指标可视化] 绘制联动图表时发生错误: {e}")