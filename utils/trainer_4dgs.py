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
    def __init__(self, dataset, opt, pipe, testing_iterations, saving_iterations, args, debug_params):
        super().__init__(dataset, opt, pipe, testing_iterations, saving_iterations, args, debug_params)
        
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
        print(f"\n[{self.__class__.__name__}] 正在通过 3D4DGS Scene 加载数据...")
        self.scene = Scene(
            self.dataset, 
            self.gaussians, 
            num_pts=args.num_pts, 
            num_pts_ratio=args.num_pts_ratio, 
            time_duration=args.time_duration
        )
        self.gaussians.training_setup(opt)
        self.gaussians.bind_current_window(0, self.dataset.total_frames - 1)

        if self.debug_params.del_ply_on_start:
            # 删除self.args.model_path路径下的chkpnt_*.pth和point_cloud_*.ply文件，避免与当前训练产生混淆
            for filename in os.listdir(self.args.model_path):
                if filename.startswith("chkpnt_") and filename.endswith(".pth"):
                    os.remove(os.path.join(self.args.model_path, filename))
                elif filename.startswith("point_cloud_") and filename.endswith(".ply"):
                    os.remove(os.path.join(self.args.model_path, filename))

        # 3. 恢复权重
        self.first_iter = 0
        if self.args.start_checkpoint and os.path.exists(self.args.start_checkpoint):
            print(f"[Trainer4DGS] 恢复权重: {self.args.start_checkpoint}")
            (model_params, self.first_iter) = torch.load(self.args.start_checkpoint, weights_only=False)
            self.gaussians.restore(model_params, opt)

        if self.debug_params.is_load_ply and os.path.exists(self.debug_params.load_ply_path):
            print(f"[Debug Trainer4DGS] 从 PLY 文件加载点云: {self.debug_params.load_ply_path}")
            self.gaussians.load_from_ply(self.debug_params.load_ply_path)
            
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
            
        total_psnr, total_ssim, total_lpips, total_fps = 0.0, 0.0, 0.0, 0.0
        for idx, batch_data in enumerate(tqdm(test_cameras, desc="Testing")):
            gt_image, viewpoint_cam = batch_data
            gt_image = gt_image.cuda()
            
            render_pkg, fps = self.metrics_tracker.measure_fps(
                render, viewpoint_cam, self.gaussians, self.pipe, self.background
            )
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            
            psnr_val, ssim_val, lpips_val = self.metrics_tracker.calculate_image_metrics(
                gt_image,
                image
            )

            total_psnr += psnr_val
            total_ssim += ssim_val
            total_lpips += lpips_val
            total_fps += fps
            
        avg_psnr = total_psnr / len(test_cameras)
        avg_ssim = total_ssim / len(test_cameras)
        avg_lpips = total_lpips / len(test_cameras)
        avg_fps = total_fps / len(test_cameras)
        
        self.metrics_tracker.record_eval_metrics(iteration, avg_psnr, avg_ssim, avg_lpips, avg_fps)
        print(f"[评估结果 ({len(test_cameras)} 帧)] PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f} | FPS: {avg_fps:.2f}")


    def train(self):
        print(f"\n[Trainer4DGS] 开始标准 3D4DGS 训练，总迭代次数: {self.opt.iterations}")
        self.metrics_tracker.start_timer()
        
        training_dataset = self.scene.getTrainCameras()
        # 3D4DGS 官方使用 DataLoader 封装 Dataset
        training_dataloader = DataLoader(training_dataset, batch_size=self.args.batch_size, shuffle=True, 
                                         num_workers=12 if self.dataset.dataloader else 0, collate_fn=lambda x: x, drop_last=True)
        # ==========================================
        # [新增] 初始化记录列表，增加点云数量监控
        # ==========================================
        self.loss_history = []
        self.loss_iterations = []
        self.pts_4d_history = []
        self.pts_3d_history = []
        
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
                    if iteration < self.opt.densify_until_iter and (self.opt.densify_until_num_points < 0 or (self.gaussians.get_xyz.shape[0] + (self.gaussians.get_static_xyz.shape[0] if static else 0)) < self.opt.densify_until_num_points):
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
                        postfix = {"Loss": f"{loss:.4f}", "Pts(4D)": self.gaussians.get_xyz.shape[0], "Pts(3D)": self.gaussians.get_static_xyz.shape[0] if static else 0}
                        progress_bar.set_postfix(postfix)
                        progress_bar.update(10)
                        
                        # ==========================================
                        # [修改] 每 10 步记录 Loss 和 高斯点数量
                        # ==========================================
                        self.loss_iterations.append(iteration)
                        self.loss_history.append(loss)
                        
                        num_4d = self.gaussians.get_xyz.shape[0] if self.gaussians.get_xyz is not None else 0
                        self.pts_4d_history.append(num_4d)
                        
                        try:
                            num_3d = self.gaussians.get_static_xyz.shape[0] if (hasattr(self.gaussians, 'get_static_xyz') and self.gaussians.get_static_xyz is not None) else 0
                        except:
                            num_3d = 0
                        self.pts_3d_history.append(num_3d)
                    
                    if iteration == self.opt.iterations:
                        self.evaluate(iteration)
                        torch.save((self.gaussians.capture(), iteration), os.path.join(self.args.model_path, f"chkpnt_{iteration}.pth"))
                        self.gaussians.save_ply(os.path.join(self.args.model_path, f"point_cloud_{iteration}.ply"))
                        num_4d = self.gaussians.get_xyz.shape[0]
                        num_3d = self.gaussians.get_static_xyz.shape[0] if static else 0
                        self.metrics_tracker.record_training_stats(iteration, num_3d, num_4d)
                        
        progress_bar.close()
        self.metrics_tracker.save_log(os.path.join(self.args.model_path, "baseline_metrics.json"))

        # ==========================================
        # [修改] 训练结束：绘制并保存双 Y 轴指标图
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
            
            # 设置动态 Loss Y 轴范围
            if len(self.loss_history) > 100:
                stable_losses = self.loss_history[len(self.loss_history)//10:]
                ax1.set_ylim(0, max(stable_losses) * 1.5)
            
            # --- 2. 右侧 Y 轴：绘制点云数量曲线 ---
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
            
            # --- 3. 图例与保存 ---
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=11)
            
            plt.title("Training Loss and Densification over Iterations", fontsize=14, fontweight='bold', pad=30)
            
            loss_plot_path = os.path.join(self.args.model_path, "loss_and_points_curve.png")
            plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 [指标可视化] 训练 Loss 与高斯点数量联动图已保存至: {loss_plot_path}")
            
        except ImportError:
            print("⚠️ [指标可视化] 缺少 matplotlib 库，跳过绘制图表。如需绘制请运行: pip install matplotlib")
        except Exception as e:
            print(f"⚠️ [指标可视化] 绘制联动图表时发生错误: {e}")