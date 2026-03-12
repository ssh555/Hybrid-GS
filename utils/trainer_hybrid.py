# My HybridGS Trainer
# 文件：utils/trainer_hybrid.py
import os
import torch
from tqdm import tqdm
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from utils.trainer_swings import TrainerSWinGS
from utils.general_utils import knn

class TrainerHybrid(TrainerSWinGS):
    """
    核心创新模型：HybridGS
    结合空间解耦（硬约束）与时间解耦（软约束）的混合 3D-4D 高斯表示。
    """
    def __init__(self, dataset, opt, pipe, testing_iterations, saving_iterations, args):
        super().__init__(dataset, opt, pipe, testing_iterations, saving_iterations, args)
        
        # 接收并初始化外部传进来的超参数 (需确保 args 中有这些参数)
        self.tau_avg = getattr(args, 'tau_avg', 0.01)       # 硬约束：平均位移阈值
        self.tau_max = getattr(args, 'tau_max', 0.05)       # 硬约束：最大瞬时位移阈值
        self.lambda_d = getattr(args, 'lambda_d', 0.1)      # 软约束：位移收敛惩罚系数

    def robust_hard_constraint_classifier(self):
        """
        核心机制 1：鲁棒硬约束判定 (空间解耦)
        将绝对静止的舞台背景永久冻结放入全局背景显存池。
        执行双重阈值联合判定逻辑 (R_avg < tau_avg AND R_max < tau_max)。
        """
        with torch.no_grad():
            # 1. 提取当前仍然是动态的高斯 (mask_dynamic == 0 未分类, 或 2 动态)
            dynamic_mask = self.gaussians._mask_dynamic != 1
            if not dynamic_mask.any():
                return
            
            # 2. 获取动态高斯在当前时刻 t 的速度 (位移特征)
            _, velocity = self.gaussians.get_current_covariance_and_mean_offset(1.0, self.gaussians.get_t)
            
            # 3. 在极小的时间偏移后采样，模拟瞬时位移变化
            _, vel_next = self.gaussians.get_current_covariance_and_mean_offset(1.0, self.gaussians.get_t + 0.05)
            
            # 计算位移的模长 (Norm)
            displacement_1 = velocity[dynamic_mask].norm(dim=-1)
            displacement_2 = vel_next[dynamic_mask].norm(dim=-1)
            
            # 4. 计算 R_avg (平均位移) 和 R_max (最大瞬时位移)
            r_avg = (displacement_1 + displacement_2) / 2.0
            r_max = torch.max(displacement_1, displacement_2)
            
            # 5. 双重阈值联合判定
            is_static = (r_avg < self.tau_avg) & (r_max < self.tau_max)
            
            if is_static.any():
                # 找到满足静止条件的全局索引
                global_static_indices = torch.nonzero(dynamic_mask, as_tuple=True)[0][is_static]
                
                # [核心斩断逻辑]：永久冻结为 3D 静态背景 (mask = 1)
                self.gaussians._mask_dynamic[global_static_indices] = 1
                
                # [新增：物理归零压缩法] 抹除时间属性数据，不仅保证物理静止，更为 ZIP 无损压缩提供极大冗余！
                self.gaussians._t.data[global_static_indices] = 0.0
                if hasattr(self.gaussians, '_scaling_t'):
                    self.gaussians._scaling_t.data[global_static_indices] = 0.0
                # 顺手将时间维度的旋转特征也彻底归零（如果有的话）
                if hasattr(self.gaussians, '_rot_r'):
                    self.gaussians._rot_r.data[global_static_indices] = 0.0

    def train(self):
        """重写训练大循环，全面注入软硬约束"""
        print(f"\n[TrainerHybrid] 开始训练！核心机制已激活：")
        print(f" --> 软约束 L_reg_d (lambda_d={self.lambda_d})")
        print(f" --> 硬约束双判定 (tau_avg={self.tau_avg}, tau_max={self.tau_max})")
        self.metrics_tracker.start_timer()

        # 初始化生命周期张量 (继承自 SWinGS 的逻辑)
        if self.gaussians._start_frame.numel() == 0:
            num_pts = self.gaussians.get_xyz.shape[0]
            self.gaussians._start_frame = torch.zeros(num_pts, dtype=torch.int32, device="cuda")
            self.gaussians._expire_frame = torch.full((num_pts,), self.window_end, dtype=torch.int32, device="cuda")
            self.gaussians._mask_dynamic = torch.zeros(num_pts, dtype=torch.int8, device="cuda")

        # 预加载初始时间窗口的图片
        self.dataset.prefetch_window(self.window_start, self.window_end)
        
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        progress_bar = tqdm(range(self.first_iter + 1, self.opt.iterations + 1), desc="HybridGS Training")
        
        for iteration in range(self.first_iter + 1, self.opt.iterations + 1):
            iter_start.record()
            
            # 1. 更新滑动窗口与学习率
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
            # =============== 批处理循环 ===============
            for batch_idx in range(batch_size):
                # 从当前滑动窗口中随机抽帧
                frame_id = randint(self.window_start, self.window_end)
                # ====== [新增/修改] 兼容标准 3DGS 场景读取器的 4D 相机获取逻辑 ======
                train_cameras = self.scene.getTrainCameras()
                # 确保索引不越界，安全获取当前帧对应的相机
                viewpoint_cam = train_cameras[frame_id % len(train_cameras)]
                
                # 极其关键：4DGS 的高斯球形变网络强依赖时间戳，而原版 COLMAP 相机没有。
                # 我们在这里动态为相机注入 fid (帧序号) 和 time (0.0~1.0 归一化时间)
                if not hasattr(viewpoint_cam, 'fid'):
                    viewpoint_cam.fid = frame_id
                if not hasattr(viewpoint_cam, 'time'):
                    viewpoint_cam.time = frame_id / max(1, len(train_cameras) - 1)
                # ======================================================================
                gt_image = viewpoint_cam.original_image.cuda()
                
                # 获取混合掩码 (包含 3D静态背景 + 当前存活的4D高斯)
                active_mask = self._get_active_dynamic_mask(frame_id)
                
                # 前向渲染
                render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background, active_dynamic_mask=active_mask)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                alpha = render_pkg["alpha"]
                
                viewspace_point_tensor_static = render_pkg.get("viewspace_points_static", [])
                visibility_filter_static = render_pkg.get("visibility_filter_static", [])
                radii_static = render_pkg.get("radii_static", [])

                # --- 计算基础光度损失 ---
                Ll1 = l1_loss(image, gt_image)
                Lssim = 1.0 - ssim(image, gt_image)
                current_loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * Lssim
                
                if hasattr(self.opt, 'lambda_opa_mask') and self.opt.lambda_opa_mask > 0 and hasattr(viewpoint_cam, 'gt_alpha_mask') and viewpoint_cam.gt_alpha_mask is not None:
                    o = alpha.clamp(1e-6, 1-1e-6)
                    sky = 1 - viewpoint_cam.gt_alpha_mask
                    current_loss += self.opt.lambda_opa_mask * (- sky * torch.log(1 - o)).mean()
                    
                # ==========================================================
                # 核心机制 2：软约束 L_reg_d (时间位移正则化，解决闪烁)
                # ==========================================================
                _, velocity = self.gaussians.get_current_covariance_and_mean_offset(1.0, self.gaussians.get_t)
                # 仅对当前出场的高斯施加软约束惩罚
                active_velocity = velocity[active_mask] if active_mask is not None else velocity
                
                # [核心修复：增加非空判定，防止 NaN]
                if active_velocity.shape[0] > 0:
                    # 植入位移收敛正则化项
                    l_reg_d = self.lambda_d * active_velocity.norm(p=2, dim=1).mean()
                    current_loss += l_reg_d
                
                # --- 保留原版的刚性约束 (Rigid Loss) ---
                if hasattr(self.opt, 'lambda_rigid') and self.opt.lambda_rigid > 0:
                    num_active = active_mask.sum().item() if active_mask is not None else self.gaussians.get_xyz.shape[0]
                    k = min(20, num_active - 1)
                    if k > 0:
                        xyz_cur = self.gaussians.get_xyz[active_mask] if active_mask is not None else self.gaussians.get_xyz
                        idx, dist = knn(xyz_cur[None].contiguous().detach(), xyz_cur[None].contiguous().detach(), k)
                        weight = torch.exp(-100 * dist)
                        vel_dist = torch.norm(active_velocity[idx] - active_velocity[None, :, None], p=2, dim=-1)
                        current_loss += self.opt.lambda_rigid * ((weight * vel_dist).sum() / k / num_active)

                current_loss = current_loss / batch_size
                current_loss.backward()
                loss += current_loss.item()
                
                # 收集梯度数据供致密化使用
                batch_point_grad.append(torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1))
                batch_radii.append(radii)
                batch_visibility_filter.append(visibility_filter)

                static = len(viewspace_point_tensor_static) > 0
                if static:
                    batch_point_grad_static.append(torch.norm(viewspace_point_tensor_static.grad[:,:2], dim=-1))
                    batch_radii_static.append(radii_static)
                    batch_visibility_filter_static.append(visibility_filter_static)

            # =============== 梯度累加逻辑 ===============
            if batch_size > 1:
                visibility_count = torch.stack(batch_visibility_filter, 1).sum(1)
                visibility_filter = visibility_count > 0
                radii = torch.stack(batch_radii, 1).max(1)[0]
                batch_viewspace_point_grad = torch.stack(batch_point_grad, 1).sum(1)
                batch_viewspace_point_grad[visibility_filter] = batch_viewspace_point_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)
                
                if static:
                    visibility_count_static = torch.stack(batch_visibility_filter_static, 1).sum(1)
                    visibility_filter_static = visibility_count_static > 0
                    radii_static = torch.stack(batch_radii_static, 1).max(1)[0]
                    batch_viewspace_point_grad_static = torch.stack(batch_point_grad_static, 1).sum(1)
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

            # =============== 致密化、硬约束与指标记录 ===============
            with torch.no_grad():
                # [核心机制触发] 定期执行硬约束判定，淘汰背景空间冗余
                # 放在 densify_from_iter 之后，让高斯先发育一段时间再判定
                if iteration % 100 == 0 and iteration > self.opt.densify_from_iter:
                    self.robust_hard_constraint_classifier()
            
                if iteration % 100 == 0:
                    # 分别计算记录硬约束生效后的纯3D点，和依然存活的4D点
                    num_3d = (self.gaussians._mask_dynamic == 1).sum().item()
                    num_4d = (self.gaussians._mask_dynamic != 1).sum().item()
                    self.metrics_tracker.record_training_stats(iteration, num_3d, num_4d)
                    
                # 原版的克隆、分裂与剪枝逻辑
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
                            extent = getattr(self.dataset, 'cameras_extent', 1.0)
                            self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, self.opt.thresh_opa_prune, extent, size_threshold, self.opt.densify_grad_t_threshold)
                            
                    if iteration % self.opt.opacity_reset_interval == 0 or (hasattr(self.dataset, 'white_background') and self.dataset.white_background and iteration == self.opt.densify_from_iter):
                        self.gaussians.reset_opacity()

                # =============== 优化器步进 ===============
                if iteration < self.opt.iterations:
                    # 继承自 SWinGS 的自适应梯度缩放机制
                    if hasattr(self.gaussians, '_start_frame') and self.gaussians._start_frame.numel() > 0:
                        age = (self.window_end - self.gaussians._start_frame).clamp(min=1)
                        decay_factor = 1.0 / age.float()
                        if self.gaussians._xyz.grad is not None:
                            self.gaussians._xyz.grad *= decay_factor.unsqueeze(-1)

                    # [新增：核心修复，安全抹除静态背景的时间梯度]
                    static_mask = (self.gaussians._mask_dynamic == 1)
                    if static_mask.any():
                        if self.gaussians._t.grad is not None:
                            self.gaussians._t.grad[static_mask] = 0.0
                        if hasattr(self.gaussians, '_scaling_t') and self.gaussians._scaling_t.grad is not None:
                            self.gaussians._scaling_t.grad[static_mask] = 0.0

                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    if self.env_map_optimizer and iteration < self.pipe.env_optimize_until:
                        self.env_map_optimizer.step()
                        self.env_map_optimizer.zero_grad(set_to_none=True)

                # =============== 进度条与日志 ===============
                if iteration % 10 == 0:
                    postfix = {"Loss": f"{loss:.4f}", "Win": f"[{self.window_start}-{self.window_end}]"}
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                
                if iteration in self.testing_iterations:
                    self.evaluate(iteration)
                    
                if iteration in self.saving_iterations:
                    os.makedirs(self.args.model_path, exist_ok=True)
                    torch.save((self.gaussians.capture(), iteration), os.path.join(self.args.model_path, f"chkpnt_{iteration}.pth"))

        progress_bar.close()
        
        # 记录专属的 HybridGS JSON 日志
        log_path = os.path.join(self.args.model_path, "hybridgs_metrics.json")
        self.metrics_tracker.save_log(log_path)
        print(f"[TrainerHybrid] 训练完成！软硬约束指标与模型已保存至 {log_path}。")