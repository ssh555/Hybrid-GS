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
    def __init__(self, dataset, opt, pipe, testing_iterations, saving_iterations, args, debug_params):
        super().__init__(dataset, opt, pipe, testing_iterations, saving_iterations, args, debug_params)
        self.tau_avg = getattr(args, 'tau_avg', 0.01)
        self.tau_max = getattr(args, 'tau_max', 0.05)
        self.lambda_d = getattr(args, 'lambda_d', 0.1)

        # 读取消融开关，默认全开（满血 HybridGS）
        self.use_soft = getattr(args, 'use_soft_constraint', True)
        self.use_hard = getattr(args, 'use_hard_constraint', True)
        t = self.gaussians.get_t.detach().squeeze()

        self.t_min = torch.quantile(t, 0.02)
        self.t_max = torch.quantile(t, 0.98)
    
    def _update_sliding_window(self, iteration):
        super()._update_sliding_window(iteration)
        t = self.gaussians.get_t.detach().squeeze()

        self.t_min = torch.quantile(t, 0.02)
        self.t_max = torch.quantile(t, 0.98)

    def get_effective_model_time(self, frame_idx):
        frame_idx = torch.as_tensor(
            frame_idx,
            dtype=torch.float32,
            device=self.t_min.device
        )
        alpha = frame_idx / max(self.total_frames - 1, 1)
        return alpha * (self.t_max - self.t_min) + self.t_min


    def robust_hard_constraint_classifier(self):
        """
        🌟 核心机制修正：基于滑动窗口内 R_avg 和 R_max 的双重联合判定
        彻底剔除假静态（间歇性动态），只将真正的永久背景冻结为 3D
        """
        with torch.no_grad():
            if hasattr(self.gaussians, '_start_frame') and self.gaussians._start_frame.numel() > 0:
                alive_mask = (self.gaussians._start_frame <= self.window_end) & (self.gaussians._expire_frame >= self.window_start)
            else:
                alive_mask = torch.ones_like(self.gaussians._mask_dynamic, dtype=torch.bool)
                
            dynamic_mask = (self.gaussians._mask_dynamic != 1) & alive_mask
            if not dynamic_mask.any():
                return None
            
            # 1. 在当前时间窗口内均匀采样 8 个时间戳，用于评估一段时间内的运动潜力
            num_steps = 8
            t_samples = torch.linspace(self.window_start, self.window_end, steps=num_steps, device="cuda")
            if hasattr(self, 'total_frames') and self.total_frames > 0:
                t_samples = self.get_effective_model_time(t_samples)  # 将帧号映射到模型时间域

            # 2. 收集窗口内各个时间戳的位移 d
            displacements = []
            for t in t_samples:
                # 3D-4DGS 的底层：获取给定时间 t 的位移向量
                _, d = self.gaussians.get_current_covariance_and_mean_offset(1.0, t, mask=dynamic_mask)
                displacements.append(d)
                
            displacements = torch.stack(displacements, dim=0) # Shape: [num_steps, N_dynamic, 3]

            # =====================================================================
            # 3. 计算 R_avg (平均位移)：衡量宏观活跃度
            #    将各个时间戳的位移减去该时间段内的平均位移，求范数均值
            # =====================================================================
            mean_d = displacements.mean(dim=0) # [N_dynamic, 3]
            R_avg = torch.norm(displacements - mean_d.unsqueeze(0), p=2, dim=-1).mean(dim=0) # [N_dynamic]

            # =====================================================================
            # 4. 计算 R_max (最大瞬时位移)：衡量突发性动作潜力
            #    计算相邻时间步之间的位移差最大值
            # =====================================================================
            step_diffs = torch.norm(displacements[1:] - displacements[:-1], p=2, dim=-1) # [num_steps-1, N_dynamic]
            R_max = step_diffs.max(dim=0)[0] # [N_dynamic]
            # print(f"[INFO] {R_avg} {R_max}")
            # 5. 联合判定：必须同时满足平均极小 AND 没有突发潜力，才是死物背景！
            is_static = (R_avg < self.tau_avg) & (R_max < self.tau_max)

            if is_static.any():
                global_static_mask = torch.zeros_like(self.gaussians._mask_dynamic, dtype=torch.bool)
                global_static_indices = torch.nonzero(dynamic_mask, as_tuple=True)[0][is_static]
                global_static_mask[global_static_indices] = True
                
                # 触发 4D 到 3D 的永久转换 (切断 MLP)
                if hasattr(self.gaussians, 'kinematic_dynamic2static'):
                    self.gaussians.kinematic_dynamic2static(global_static_mask)
                else:
                    print("⚠️ 架构缺失：请在 gaussian_model.py 中实现 kinematic_dynamic2static(mask)！")
                
                # print(f"❄️ [硬约束触发] 成功将 {is_static.sum().item()} 个背景高斯永久降维为 3D！")
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
                # 🌟 核心修正：仅针对前景的软约束 (位移收敛正则化 L_reg_d)
                # 硬约束点已被彻底剥离，这里绝不去算它们的 Loss！
                # =========================================================
                total_reg_loss = 0.0  
                # current_t = self.get_effective_model_time(frame_id) if hasattr(self, 'total_frames') else self.gaussians.get_t
                t_prev = self.get_effective_model_time(max(frame_id - 1, 0))
                t_curr = self.get_effective_model_time(frame_id)



                if self.use_soft:
                    warmup_start = int(self.opt.iterations * self.opt.warmup_start) 
                    warmup_end = int(self.opt.iterations * self.opt.warmup_end)
                    if iteration < warmup_start: current_lambda_d = 0.0
                    elif iteration > warmup_end: current_lambda_d = self.lambda_d
                    else: current_lambda_d = self.lambda_d * ((iteration - warmup_start) / (warmup_end - warmup_start))

                    if current_lambda_d > 0 or self.opt.lambda_rigid > 0:
                        alive_mask = (self.gaussians._start_frame <= self.window_end) & (self.gaussians._expire_frame >= self.window_start)
                        active_dynamic_mask = (self.gaussians._mask_dynamic != 1) & alive_mask
                        active_indices = torch.nonzero(active_dynamic_mask, as_tuple=True)[0]
                        
                        if active_indices.numel() > 0:
                            if active_indices.numel() > 30000:
                                perm = torch.randperm(active_indices.numel(), device=active_indices.device)[:30000]
                                active_indices = active_indices[perm]
                                sampled_mask = torch.zeros_like(active_dynamic_mask)
                                sampled_mask[active_indices] = True
                                active_dynamic_mask = sampled_mask

                            # 获取当前的位移向量 d (公式中的 \mathbf{d})
                            _, d_prev = self.gaussians.get_current_covariance_and_mean_offset(1.0, t_prev, mask=active_dynamic_mask)
                            _, d_curr = self.gaussians.get_current_covariance_and_mean_offset(1.0, t_curr, mask=active_dynamic_mask)

                            dt = (t_curr - t_prev).clamp(min=1e-6)  # 防止除零
                            vel = (d_curr - d_prev) / dt  # 近似速度

                            # 位移收敛正则化：L_reg_d = \sum ||d||_2
                            # 作用：静止时使其归 0，充当时间平滑损失 L_time
                            if current_lambda_d > 0:
                                total_reg_loss += current_lambda_d * vel.norm(p=2, dim=1).mean()


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
                if self.use_hard and iteration >= freeze_start_iter and iteration <= freeze_end_iter and iteration % self.opt.freeze_internal == 0:
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
                
                if self.debug_params.save_ply_interval > 0 and iteration % self.debug_params.save_ply_interval == 0:
                    os.makedirs(self.args.model_path, exist_ok=True)
                    self.gaussians.save_ply(os.path.join(self.args.model_path, f"debug_point_cloud_{iteration}.ply"))
                    
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