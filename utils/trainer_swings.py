# SwinGS模型封装类
# 文件：utils/trainer_swings.py
import os
import torch
from random import randint
from tqdm import tqdm
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from utils.trainer_4dgs import Trainer4DGS
from utils.general_utils import knn
from utils.image_utils import psnr # 确保导入了 psnr

class TrainerSWinGS(Trainer4DGS):
    """
    引入滑动窗口和显式生命周期管理的长序列基线 (SWinGS)。
    继承自 Trainer4DGS，重写训练大循环以支持局部时间窗口优化。
    """
    def __init__(self, dataset, opt, pipe, testing_iterations, saving_iterations, args):
        super().__init__(dataset, opt, pipe, testing_iterations, saving_iterations, args)
        
        # SWinGS 核心参数
        self.swin_size = args.swin_size  # 滑动窗口大小 (如: 50帧)
        self.total_frames = self.dataset.total_frames
        
        # 初始化第一个窗口
        self.window_start = 0
        self.window_end = min(self.swin_size - 1, self.total_frames - 1)
        
        # 计算滑动频率: 总迭代次数 / 需要滑动的总步数
        slide_steps = max(1, self.total_frames - self.swin_size)
        self.slide_interval = max(1, self.opt.iterations // slide_steps)

    def _update_sliding_window(self, iteration):
        """
        核心机制 1：控制时间窗口滑动与显存按需加载
        """
        if iteration > 0 and iteration % self.slide_interval == 0:
            if self.window_end < self.total_frames - 1:
                self.window_start += 1
                self.window_end += 1
                
                # 触发 Lazy DataLoader 的显存释放与新帧预加载
                self.dataset.release_window(self.window_start)
                self.dataset.prefetch_window(self.window_start, self.window_end)
                # print(f"\n[SWinGS] 窗口已滑动: [{self.window_start} -> {self.window_end}]")
                # [新增算法逻辑] 延长当前存活高斯的寿命，或者让太老的自然死亡
                # SWinGS 的核心就是让离开窗口的动态点过期
                with torch.no_grad():
                    if self.gaussians._expire_frame.numel() > 0:
                        # 找到在旧窗口末尾还活着的高斯，给它们续命到新窗口末尾
                        # (具体的生命周期策略可根据你的 HybridGS 需求调整)
                        alive_idx = self.gaussians._expire_frame == (self.window_end - 1)
                        self.gaussians._expire_frame[alive_idx] = self.window_end

    def _get_active_dynamic_mask(self, frame_id):
        """
        核心机制 2：获取当前帧存活的高斯掩码。
        这里预留了高斯生命周期判定的接口。
        """
        # 如果模型内部尚未初始化生命周期数组，则返回 None（全量渲染）
        if self.gaussians._start_frame.numel() == 0:
            return None
            
        # 提取生命周期掩码: 高斯的出生时间 <= 当前帧 AND 高斯的过期时间 >= 当前帧
        active_mask = (self.gaussians._start_frame <= frame_id) & (self.gaussians._expire_frame >= frame_id)
        
        # 加上硬约束判定：如果是已被标记为绝对静止(mask==1)的，或者是动态存活的
        if hasattr(self.gaussians, '_mask_dynamic'):
            # 1 是静态背景，始终可见；2 是动态前景，受生命周期限制
            final_mask = (self.gaussians._mask_dynamic == 1) | ((self.gaussians._mask_dynamic == 2) & active_mask)
            return final_mask
            
        return active_mask

    @torch.no_grad()
    def evaluate(self, iteration):
        """重写测试集评估：必须引入时间窗口掩码过滤残影"""
        print(f"\n[评估 SWinGS] 正在执行 Iteration {iteration} 的测试集评估...")
        
        test_cameras = self.dataset.getTestCameras() if hasattr(self.dataset, 'getTestCameras') else []
        if not test_cameras:
            return
            
        total_psnr = 0.0
        total_ssim = 0.0
        total_fps = 0.0
        
        for viewpoint_cam in tqdm(test_cameras, desc="Testing"):
            gt_image = viewpoint_cam.original_image.cuda()
            
            # [核心修复] 获取测试相机的真实帧号/时间戳
            # 假设你的 Camera 类有 uid 或 timestamp 可以映射回 frame_id
            frame_id = getattr(viewpoint_cam, 'uid', 0) 
            
            # 获取特定帧的干净掩码，过滤掉不在当前时间出生的点
            active_mask = self._get_active_dynamic_mask(frame_id)
            
            render_pkg, fps = self.metrics_tracker.measure_fps(
                render, 
                viewpoint_camera=viewpoint_cam, 
                pc=self.gaussians, 
                pipe=self.pipe, 
                bg_color=self.background, 
                active_dynamic_mask=active_mask # 关键：必须传掩码！
            )
            
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            total_psnr += psnr(image, gt_image).mean().item()
            total_ssim += ssim(image, gt_image).mean().item()
            total_fps += fps
            
        avg_psnr = total_psnr / len(test_cameras)
        avg_ssim = total_ssim / len(test_cameras)
        avg_fps = total_fps / len(test_cameras)
        
        self.metrics_tracker.record_eval_metrics(iteration, avg_psnr, avg_ssim, avg_fps)
        print(f"[评估结果] PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | FPS: {avg_fps:.2f}")
        # 记录 TEST JSON 日志
        log_path = os.path.join(self.args.model_path, "test_metrics.json")
        self.metrics_tracker.save_log(log_path)
        print(f"测试完成！软硬约束指标与模型已保存至 {log_path}。")

    def train(self):
        """SWinGS 滑动窗口训练大循环"""
        print(f"\n[TrainerSWinGS] 开始训练，窗口大小: {self.swin_size}，总帧数: {self.total_frames}")
        self.metrics_tracker.start_timer()

        # [新增算法逻辑] 初始化高斯的生命周期
        if self.gaussians._start_frame.numel() == 0:
            num_pts = self.gaussians.get_xyz.shape[0]
            # 初始高斯的生命周期设为 [0, window_end]
            self.gaussians._start_frame = torch.zeros(num_pts, dtype=torch.int32, device="cuda")
            self.gaussians._expire_frame = torch.full((num_pts,), self.window_end, dtype=torch.int32, device="cuda")
            # 初始化动静掩码 (默认 0：未分类)
            self.gaussians._mask_dynamic = torch.zeros(num_pts, dtype=torch.int8, device="cuda")

        # 预加载初始窗口
        self.dataset.prefetch_window(self.window_start, self.window_end)
        
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        progress_bar = tqdm(range(self.first_iter + 1, self.opt.iterations + 1), desc="SWinGS Training")
        
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
                # 【重要修改】：只在当前活跃的滑动窗口内随机抽帧！
                frame_id = randint(self.window_start, self.window_end)
                viewpoint_cam = self.dataset.get_camera_data(frame_id)
                gt_image = viewpoint_cam.original_image.cuda()
                
                # 获取当前帧活跃的动态掩码
                active_mask = self._get_active_dynamic_mask(frame_id)
                
                # 前向渲染 (传入 active_dynamic_mask 执行时间切片过滤)
                render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background, active_dynamic_mask=active_mask)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                alpha = render_pkg["alpha"]
                
                viewspace_point_tensor_static = render_pkg.get("viewspace_points_static", [])
                visibility_filter_static = render_pkg.get("visibility_filter_static", [])
                radii_static = render_pkg.get("radii_static", [])

# =============== 计算基础与正则损失 ===============
                Ll1 = l1_loss(image, gt_image)
                Lssim = 1.0 - ssim(image, gt_image)
                current_loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * Lssim
                
                if hasattr(self.opt, 'lambda_opa_mask') and self.opt.lambda_opa_mask > 0 and hasattr(viewpoint_cam, 'gt_alpha_mask') and viewpoint_cam.gt_alpha_mask is not None:
                    o = alpha.clamp(1e-6, 1-1e-6)
                    sky = 1 - viewpoint_cam.gt_alpha_mask
                    current_loss += self.opt.lambda_opa_mask * (- sky * torch.log(1 - o)).mean()
                    
                # [核心算法修复]：只针对当前窗口存活的动态高斯计算刚性约束
                if hasattr(self.opt, 'lambda_rigid') and self.opt.lambda_rigid > 0:
                    # 获取当前存活的高斯数量
                    num_active = active_mask.sum().item() if active_mask is not None else self.gaussians.get_xyz.shape[0]
                    k = min(20, num_active - 1) # 防止存活点不足 20 个导致 KNN 报错
                    
                    if k > 0:
                        xyz_cur = self.gaussians.get_xyz[active_mask] if active_mask is not None else self.gaussians.get_xyz
                        idx, dist = knn(xyz_cur[None].contiguous().detach(), xyz_cur[None].contiguous().detach(), k)
                        
                        _, velocity = self.gaussians.get_current_covariance_and_mean_offset(1.0, self.gaussians.get_t + 0.1)
                        active_velocity = velocity[active_mask] if active_mask is not None else velocity
                        
                        weight = torch.exp(-100 * dist)
                        vel_dist = torch.norm(active_velocity[idx] - active_velocity[None, :, None], p=2, dim=-1)
                        current_loss += self.opt.lambda_rigid * ((weight * vel_dist).sum() / k / num_active)

                # [核心算法修复]：只针对当前存活的高斯计算运动正则
                if hasattr(self.opt, 'lambda_motion') and self.opt.lambda_motion > 0:
                    _, velocity = self.gaussians.get_current_covariance_and_mean_offset(1.0, self.gaussians.get_t + 0.1)
                    active_velocity = velocity[active_mask] if active_mask is not None else velocity
                    # [核心修复：增加非空判定，防止 NaN]
                    if active_velocity.shape[0] > 0:
                        current_loss += self.opt.lambda_motion * active_velocity.norm(p=2, dim=1).mean()

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

            # =============== 梯度累加逻辑 (同基线) ===============
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

            # =============== 致密化与指标记录 ===============
            with torch.no_grad():
                # 记录指标
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
                            extent = getattr(self.dataset, 'cameras_extent', 1.0)
                            self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, self.opt.thresh_opa_prune, extent, size_threshold, self.opt.densify_grad_t_threshold)
                            if hasattr(self.gaussians, 'dynamic2static'):
                                self.gaussians.dynamic2static(self.opt.scale_t_threshold)
                                
                    if iteration % self.opt.opacity_reset_interval == 0 or (hasattr(self.dataset, 'white_background') and self.dataset.white_background and iteration == self.opt.densify_from_iter):
                        self.gaussians.reset_opacity()

                # 优化器步进
                if iteration < self.opt.iterations:
                    # [新增] SWinGS 自适应梯度缩放机制
                    if hasattr(self.gaussians, '_start_frame') and self.gaussians._start_frame.numel() > 0:
                        # 估算每个高斯存活的窗口数量 (当前帧号 - 出生帧号)
                        # 为防止除零，最小存活单位设为 1
                        age = (self.window_end - self.gaussians._start_frame).clamp(min=1)
                        decay_factor = 1.0 / age.float()
                        
                        # 仅对动态点应用空间梯度衰减，防止老点被过度拉扯
                        if self.gaussians._xyz.grad is not None:
                            self.gaussians._xyz.grad *= decay_factor.unsqueeze(-1)
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    if self.env_map_optimizer and iteration < self.pipe.env_optimize_until:
                        self.env_map_optimizer.step()
                        self.env_map_optimizer.zero_grad(set_to_none=True)

                # 进度与评估
                if iteration % 10 == 0:
                    postfix = {"Loss": f"{loss:.4f}", "Win": f"[{self.window_start}-{self.window_end}]"}
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                
                if iteration in self.testing_iterations:
                    self.evaluate(iteration)
                    
                if iteration in self.saving_iterations:
                    os.makedirs(self.args.model_path, exist_ok=True)
                    torch.save((self.gaussians.capture(), iteration), os.path.join(self.args.model_path, f"chkpnt_{iteration}.pth"))
                    # [新增代码] 保存标准的 .ply (防报错，且方便后续用第三方软件可视化查看)
                    ply_path = os.path.join(self.args.model_path, f"point_cloud/iteration_{iteration}")
                    os.makedirs(ply_path, exist_ok=True)
                    self.gaussians.save_ply(os.path.join(ply_path, "point_cloud.ply"))

        progress_bar.close()
        
        log_path = os.path.join(self.args.model_path, "swings_metrics.json")
        self.metrics_tracker.save_log(log_path)
        print(f"[TrainerSWinGS] 训练完成！")