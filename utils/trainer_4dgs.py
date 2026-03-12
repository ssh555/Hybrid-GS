# 3D4DGS模型封装类
# 文件：utils/trainer_4dgs.py
import os
import torch
import random
from torch import nn
from random import randint
from tqdm import tqdm
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr # 确保导入了 psnr
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.general_utils import knn
from utils.trainer_base import BaseTrainer

class Trainer4DGS(BaseTrainer):
    """
    原生 3D-4DGS 完整版封装 (Baseline)
    包含环境贴图优化、多重正则化损失(Rigid/Motion)、梯度累加机制，
    并完整接入了 Lazy DataLoader 与全方位评估追踪。
    """
    def __init__(self, dataset, opt, pipe, testing_iterations, saving_iterations, args):
        super().__init__(dataset, opt, pipe, testing_iterations, saving_iterations, args)
        
        # 1. 修正时间维度缩放
        if hasattr(self.dataset, 'frame_ratio') and self.dataset.frame_ratio > 1:
            self.args.time_duration = [self.args.time_duration[0] / self.dataset.frame_ratio, 
                                       self.args.time_duration[1] / self.dataset.frame_ratio]
            
        # 2. 初始化 GaussianModel
        self.gaussians = GaussianModel(
            dataset.sh_degree if hasattr(dataset, 'sh_degree') else 3, 
            gaussian_dim=self.args.gaussian_dim, 
            time_duration=self.args.time_duration, 
            rot_4d=self.args.rot_4d, 
            force_sh_3d=self.args.force_sh_3d, 
            sh_degree_t=2 if self.pipe.eval_shfs_4d else 0
        )
        self.gaussians.training_setup(self.opt)

        # =========================================================
        # [新增] 强行挂载官方 Scene！
        # 它的作用：读取 COLMAP 相机、加载 points3d.ply 稀疏点云并种下高斯种子！
        # =========================================================
        from scene import Scene
        print("\n[Trainer] 正在通过 Scene 加载 COLMAP 数据与点云...")

        # 【新增修复代码】提前建好输出文件夹！防止 Scene 拷贝点云备份时找不到路！
        os.makedirs(self.args.model_path, exist_ok=True)
        print(f"[Trainer] 已确保模型输出路径存在: {self.args.model_path}")
        self.scene = Scene(self.dataset, self.gaussians)
        
        # 顺手把真实的帧数更新一下，覆盖之前 trainer_base 里的估算值
        self.dataset.total_frames = len(self.scene.getTrainCameras())
        print(f"[Trainer] Scene 加载完毕，总帧数锁定为: {self.dataset.total_frames}\n")
        # =========================================================

        # 3. 恢复 Checkpoint
        self.first_iter = 0
        if self.args.start_checkpoint:
            # 只有当文件确实存在时，才进行加载
            if os.path.exists(self.args.start_checkpoint):
                print(f"[加载] 发现 Checkpoint: {self.args.start_checkpoint}，正在恢复模型状态...")
                (model_params, self.first_iter) = torch.load(self.args.start_checkpoint)
                self.gaussians.restore(model_params, self.opt)
            else:
                # 如果文件不存在，仅打印警告而不崩溃
                # 这样你训练时就不需要去 YAML 里注释掉这一行了
                print(f"[提示] 未找到 Checkpoint 文件: {self.args.start_checkpoint}")
                print("[提示] 将作为全新任务从第 0 代开始训练。")

        # 4. 背景颜色
        bg_color = [1, 1, 1] if hasattr(self.dataset, 'white_background') and self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # 5. 环境光贴图优化
        if self.pipe.env_map_res:
            self.env_map = nn.Parameter(torch.zeros((3, self.pipe.env_map_res, self.pipe.env_map_res), dtype=torch.float, device="cuda").requires_grad_(True))
            self.env_map_optimizer = torch.optim.Adam([self.env_map], lr=self.opt.feature_lr, eps=1e-15)
        else:
            self.env_map = None
            self.env_map_optimizer = None
        self.gaussians.env_map = self.env_map

    @torch.no_grad()
    def evaluate(self, iteration):
        """
        在特定的 iteration 执行测试集评估，并将 PSNR、SSIM 和 FPS 记入 Tracker。
        使用 @torch.no_grad() 是为了在评估时不计算梯度，节省显存并提速。
        """
        print(f"\n[评估] 正在执行 Iteration {iteration} 的测试集评估...")
        
        test_cameras = self.dataset.getTestCameras() if hasattr(self.dataset, 'getTestCameras') else []
        if not test_cameras:
            print("[警告] 未找到测试集相机，跳过评估。")
            return
            
        total_psnr = 0.0
        total_ssim = 0.0
        total_fps = 0.0
        
        for viewpoint_cam in tqdm(test_cameras, desc="Testing"):
            gt_image = viewpoint_cam.original_image.cuda()
            
            # 使用 MetricsTracker 包装以获取精准渲染耗时
            render_pkg, fps = self.metrics_tracker.measure_fps(
                render, 
                viewpoint_camera=viewpoint_cam, 
                pc=self.gaussians, 
                pipe=self.pipe, 
                bg_color=self.background, 
                active_dynamic_mask=None
            )
            
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            
            # 累计指标
            total_psnr += psnr(image, gt_image).mean().item()
            total_ssim += ssim(image, gt_image).mean().item()
            total_fps += fps
            
        avg_psnr = total_psnr / len(test_cameras)
        avg_ssim = total_ssim / len(test_cameras)
        avg_fps = total_fps / len(test_cameras)
        
        # 将测试结果记录到 MetricsTracker 的字典中
        self.metrics_tracker.record_eval_metrics(iteration, avg_psnr, avg_ssim, avg_fps)
        
        print(f"[评估结果] PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | FPS: {avg_fps:.2f}")

    def train(self):
        """完整复刻原版 3D-4DGS 的训练大循环，接入 Lazy DataLoader 与全面评估"""
        print(f"\n[Trainer4DGS] 开始训练 Baseline，总迭代次数: {self.opt.iterations}")
        self.metrics_tracker.start_timer()
        
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        
        progress_bar = tqdm(range(self.first_iter + 1, self.opt.iterations + 1), desc="Training progress")
        total_frames = self.dataset.total_frames
        print('[INFO] 数据集总帧数 (total_frames):', total_frames)
        for iteration in range(self.first_iter + 1, self.opt.iterations + 1):
            iter_start.record()
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
                # Lazy DataLoader：随机选取一帧
                frame_id = randint(0, total_frames - 1)
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
                
                # 前向渲染
                render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background, active_dynamic_mask=None)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                alpha = render_pkg["alpha"]
                
                viewspace_point_tensor_static = render_pkg.get("viewspace_points_static", [])
                visibility_filter_static = render_pkg.get("visibility_filter_static", [])
                radii_static = render_pkg.get("radii_static", [])

                # 基础光度损失
                Ll1 = l1_loss(image, gt_image)
                Lssim = 1.0 - ssim(image, gt_image)
                current_loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * Lssim
                
                # Opa Mask Loss
                if hasattr(self.opt, 'lambda_opa_mask') and self.opt.lambda_opa_mask > 0 and hasattr(viewpoint_cam, 'gt_alpha_mask') and viewpoint_cam.gt_alpha_mask is not None:
                    o = alpha.clamp(1e-6, 1-1e-6)
                    sky = 1 - viewpoint_cam.gt_alpha_mask
                    Lopa_mask = (- sky * torch.log(1 - o)).mean()
                    current_loss += self.opt.lambda_opa_mask * Lopa_mask
                    
                # Rigid Loss (刚性正则化)
                if hasattr(self.opt, 'lambda_rigid') and self.opt.lambda_rigid > 0:
                    k = 20
                    xyz_cur = self.gaussians.get_xyz
                    idx, dist = knn(xyz_cur[None].contiguous().detach(), xyz_cur[None].contiguous().detach(), k)
                    _, velocity = self.gaussians.get_current_covariance_and_mean_offset(1.0, self.gaussians.get_t + 0.1)
                    weight = torch.exp(-100 * dist)
                    vel_dist = torch.norm(velocity[idx] - velocity[None, :, None], p=2, dim=-1)
                    Lrigid = (weight * vel_dist).sum() / k / xyz_cur.shape[0]
                    current_loss += self.opt.lambda_rigid * Lrigid
                    
                # Motion Loss (运动正则化)
                if hasattr(self.opt, 'lambda_motion') and self.opt.lambda_motion > 0:
                    _, velocity = self.gaussians.get_current_covariance_and_mean_offset(1.0, self.gaussians.get_t + 0.1)
                    Lmotion = velocity.norm(p=2, dim=1).mean()
                    current_loss += self.opt.lambda_motion * Lmotion

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

            # =============== 致密化与剪枝 ===============
            with torch.no_grad():
                # [核心指标追踪]：显存、耗时、3D/4D高斯数量
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

                # =============== 优化器步进 ===============
                if iteration < self.opt.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    if self.env_map_optimizer and iteration < self.pipe.env_optimize_until:
                        self.env_map_optimizer.step()
                        self.env_map_optimizer.zero_grad(set_to_none=True)

                # =============== 进度与评估 ===============
                if iteration % 10 == 0:
                    postfix = {"Loss": f"{loss:.4f}", "Pts": self.gaussians.get_xyz.shape[0]}
                    if static: postfix["Static"] = self.gaussians.get_static_xyz.shape[0]
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                
                # [触发测试集评估]
                if iteration in self.testing_iterations:
                    self.evaluate(iteration)
                    
                if iteration in self.saving_iterations:
                    print(f"\n[Trainer4DGS] 正在保存模型至 Iteration {iteration}")
                    os.makedirs(self.args.model_path, exist_ok=True)
                    torch.save((self.gaussians.capture(), iteration), os.path.join(self.args.model_path, f"chkpnt_{iteration}.pth"))

        progress_bar.close()
        
        # 保存最终追踪日志
        log_path = os.path.join(self.args.model_path, "baseline_metrics.json")
        self.metrics_tracker.save_log(log_path)
        print(f"[Trainer4DGS] 训练完成！指标已保存至 {log_path}")