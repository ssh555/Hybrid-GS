# 文件：utils/trainer_swings.py
import os
import torch
import random
import gc
from tqdm import tqdm
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from gaussian_renderer import render
from utils.general_utils import knn
from utils.trainer_4dgs import Trainer4DGS

class TrainerSWinGS(Trainer4DGS):
    def __init__(self, dataset, opt, pipe, testing_iterations, saving_iterations, args, debug_params):
        super().__init__(dataset, opt, pipe, testing_iterations, saving_iterations, args, debug_params)
        
        self.swin_size = getattr(args, 'swin_size', 50)
        self.total_frames = self.dataset.total_frames
        
        # [SWinGS 严格算法] 划分带 1 帧重叠的块状窗口 (Block Windows)
        self.window_blocks = []
        curr_start = 0
        while curr_start < self.total_frames - 1:
            curr_end = min(curr_start + self.swin_size - 1, self.total_frames - 1)
            self.window_blocks.append((curr_start, curr_end))
            if curr_end == self.total_frames - 1:
                break
            curr_start = curr_end # 下一窗口的起点是本窗口的终点 (1帧重叠)
            
        print(f"[TrainerSWinGS] 单卡严格串行模式初始化！总帧数: {self.total_frames}")
        print(f"[TrainerSWinGS] 窗口规划: {self.window_blocks}")

        # 建立帧字典，完美支持多相机与乱序
        self.training_dataset = self.scene.getTrainCameras()
        self.frames_dict = {}
        for idx, cam in enumerate(self.training_dataset):
            try:
                frame_id = int(cam.image_name.split('_')[-1])
            except:
                frame_id = getattr(cam, 'fid', idx % self.total_frames)
            if frame_id not in self.frames_dict:
                self.frames_dict[frame_id] = []
            self.frames_dict[frame_id].append(idx)

        self.window_cache = {}
        
        # 记录全局绘图数据
        self.loss_history = []
        self.loss_iterations = []
        self.pts_4d_history = []
        self.pts_3d_history = []
        self.global_iter = 0  # 全局迭代步数，用于贯穿所有窗口

    def _get_active_dynamic_mask(self, frame_id):
        if not hasattr(self.gaussians, '_start_frame') or self.gaussians._start_frame.numel() == 0:
            return None
        active_mask = (self.gaussians._start_frame <= frame_id) & (self.gaussians._expire_frame >= frame_id)
        if hasattr(self.gaussians, '_mask_dynamic'):
            final_mask = (self.gaussians._mask_dynamic == 1) | ((self.gaussians._mask_dynamic != 1) & active_mask)
            return final_mask
        return active_mask

    @torch.no_grad()
    def evaluate(self, iteration, start_frame=0, end_frame=None, tag=""):
        print(f"\n[评估 {tag}] 正在执行 Iteration {iteration} 的测试集评估...")
        test_cameras = self.scene.getTestCameras()
        if not test_cameras: return
            
        total_psnr, total_ssim, total_fps = 0.0, 0.0, 0.0
        valid_frames_count = 0  # 记录当前窗口内有效测试帧的数量
        
        for idx, batch_data in enumerate(tqdm(test_cameras, desc="Testing")):
            gt_image, viewpoint_cam = batch_data
            try: frame_id = int(viewpoint_cam.image_name.split('_')[-1])
            except: frame_id = getattr(viewpoint_cam, 'fid', idx % self.total_frames)
            
            # 直接跳过不属于当前窗口的测试帧！防止纯黑画面拉低平均分！
            if end_frame is not None:
                if frame_id < start_frame or frame_id > end_frame:
                    continue
                    
            if gt_image is not None: gt_image = gt_image.cuda()
            
            active_mask = self._get_active_dynamic_mask(frame_id)
            if active_mask is not None and not active_mask.any():
                image = self.background.clone().view(3, 1, 1).expand(3, viewpoint_cam.image_height, viewpoint_cam.image_width)
                fps = 1000.0
            else:
                render_pkg, fps = self.metrics_tracker.measure_fps(
                    render, viewpoint_cam, self.gaussians, self.pipe, self.background, active_dynamic_mask=active_mask
                )
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            
            if gt_image is not None:
                total_psnr += psnr(image, gt_image).mean().item()
                total_ssim += ssim(image, gt_image).mean().item()
                valid_frames_count += 1
            total_fps += fps
            
        if valid_frames_count == 0:
            print("[评估警告] 当前窗口没有分配到测试帧。")
            return
            
        avg_psnr = total_psnr / valid_frames_count
        avg_ssim = total_ssim / valid_frames_count
        avg_fps = total_fps / valid_frames_count
        
        self.metrics_tracker.record_eval_metrics(iteration, avg_psnr, avg_ssim, avg_fps)
        print(f"[评估结果 ({valid_frames_count} 帧)] PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | FPS: {avg_fps:.2f}")

    def train_phase1_window(self, win_idx, start_frame, end_frame):
        """阶段一：独立训练当前窗口（使用 YAML 中的 self.opt.iterations）"""
        print(f"\n🚀 开始 SWinGS 阶段 1: 独立训练窗口 {win_idx} [{start_frame}-{end_frame}]")
        
        if not hasattr(self.gaussians, '_start_frame') or self.gaussians._start_frame.numel() == 0:
            num_pts = self.gaussians.get_xyz.shape[0]
            self.gaussians._start_frame = torch.zeros(num_pts, dtype=torch.int32, device="cuda")
            self.gaussians._expire_frame = torch.zeros(num_pts, dtype=torch.int32, device="cuda")
            self.gaussians._mask_dynamic = torch.zeros(num_pts, dtype=torch.int8, device="cuda")

        # 框定生命周期在当前窗口
        self.gaussians._start_frame[:] = start_frame
        self.gaussians._expire_frame[:] = end_frame

        # 使用 YAML 配置作为单窗口的迭代总数
        total_iters = self.opt.iterations
        warmup_iters = self.opt.warmup_iterations
        
        progress_bar = tqdm(range(1, total_iters + 1), desc=f"Win {win_idx} Phase 1")
        
        for iteration in range(1, total_iters + 1):
            self.global_iter += 1
            
            # [核心] SWinGS Warm-up
            is_warmup = iteration <= warmup_iters
            if is_warmup and hasattr(self.gaussians, 'set_mlp_requires_grad'):
                self.gaussians.set_mlp_requires_grad(False)
            elif iteration == warmup_iters + 1 and hasattr(self.gaussians, 'set_mlp_requires_grad'):
                print("\n🔥 Warm-up 结束，解冻 MLP 变形网络！")
                self.gaussians.set_mlp_requires_grad(True)

            self.gaussians.update_learning_rate(iteration)
            if iteration % self.opt.sh_increase_interval == 0:
                self.gaussians.oneupSHdegree()

            batch_size = self.args.batch_size
            batch_point_grad, batch_visibility_filter, batch_radii = [], [], []
            loss = 0
            
            for batch_idx in range(batch_size):
                t_id = random.randint(start_frame, end_frame)
                dataset_idx = random.choice(self.frames_dict[t_id])
                gt_image, viewpoint_cam = self.window_cache[dataset_idx]
                gt_image, viewpoint_cam = gt_image.cuda(), viewpoint_cam.cuda()

                active_mask = self._get_active_dynamic_mask(t_id)
                render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background, active_dynamic_mask=active_mask)

                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                alpha = render_pkg["alpha"]

                Ll1 = l1_loss(image, gt_image)
                Lssim = 1.0 - ssim(image, gt_image)
                current_loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * Lssim

                # 保留你原代码的 Opa Mask Loss
                if self.opt.lambda_opa_mask > 0 and hasattr(viewpoint_cam, 'gt_alpha_mask') and viewpoint_cam.gt_alpha_mask is not None:
                    o = alpha.clamp(1e-6, 1-1e-6)
                    sky = 1 - viewpoint_cam.gt_alpha_mask
                    current_loss = current_loss + self.opt.lambda_opa_mask * (- sky * torch.log(1 - o)).mean()

                # 保留你原代码的 Motion & Rigid Loss (仅解冻后生效)
                if not is_warmup and ((self.opt.lambda_motion > 0) or (self.opt.lambda_rigid > 0)):
                    current_t = t_id / self.total_frames
                    _, active_velocity = self.gaussians.get_current_covariance_and_mean_offset(1.0, current_t, mask=active_mask)
                    
                    if self.opt.lambda_rigid > 0:
                        k_neighbors = 10
                        xyz_active = self.gaussians.get_xyz[active_mask].contiguous()
                        if xyz_active.shape[0] > 30000:
                            perm = torch.randperm(xyz_active.shape[0], device="cuda")[:30000]
                            xyz_cur = xyz_active[perm].contiguous()
                            velocity_cur = active_velocity[perm]
                        else:
                            xyz_cur = xyz_active
                            velocity_cur = active_velocity

                        if xyz_cur.shape[0] > k_neighbors:
                            idx, dist = knn(xyz_cur[None].detach(), xyz_cur[None].detach(), k_neighbors)
                            weight = torch.exp(-100 * dist)
                            vel_dist = torch.norm(velocity_cur[idx.squeeze(0)] - velocity_cur.unsqueeze(1), p=2, dim=-1)
                            coherence_loss = (weight * vel_dist).sum() / k_neighbors / xyz_cur.shape[0]
                            current_loss = current_loss + (self.opt.lambda_rigid * 5.0) * coherence_loss

                    if self.opt.lambda_motion > 0:
                        current_loss = current_loss + (self.opt.lambda_motion * 2.0) * active_velocity.norm(p=2, dim=1).mean()

                current_loss = current_loss / batch_size
                current_loss.backward()
                loss += current_loss.item()
                
                batch_point_grad.append(torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1))
                batch_radii.append(radii)
                batch_visibility_filter.append(visibility_filter)

            # 梯度累加与优化器 (保留你所有原版逻辑)
            if batch_size > 1:
                visibility_count = torch.stack(batch_visibility_filter,1).sum(1)
                visibility_filter = visibility_count > 0
                radii = torch.stack(batch_radii,1).max(1)[0]
                batch_viewspace_point_grad = torch.stack(batch_point_grad,1).sum(1)
                batch_viewspace_point_grad[visibility_filter] = batch_viewspace_point_grad[visibility_filter] * batch_size / visibility_count[visibility_filter]
                batch_viewspace_point_grad = batch_viewspace_point_grad.unsqueeze(1)
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

                    if iteration > self.opt.densify_from_iter: 
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        if iteration % self.opt.densification_interval == 0: 
                            active_grad_threshold = self.opt.densify_grad_threshold
                            active_grad_t_threshold = getattr(self.opt, 'densify_grad_t_threshold', 0.00005)
                            current_pts = self.gaussians.get_xyz.shape[0]
                            max_points = getattr(self.opt, 'densify_until_num_points', 4000000)
                            if max_points > 0 and current_pts >= max_points:
                                active_grad_threshold, active_grad_t_threshold = 99999.0, 99999.0 
                            self.gaussians.densify_and_prune(active_grad_threshold, self.opt.thresh_opa_prune, self.scene.cameras_extent, size_threshold, active_grad_t_threshold)
                                
                if iteration % self.opt.opacity_reset_interval == 0:
                    self.gaussians.reset_opacity()
                        
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

                if iteration % 10 == 0:
                    postfix = {"Loss": f"{loss:.4f}", "Pts": self.gaussians.get_xyz.shape[0]}
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(10)
                    
                    # 保存绘图数据
                    self.loss_iterations.append(self.global_iter)
                    self.loss_history.append(loss)
                    self.pts_4d_history.append(self.gaussians.get_xyz.shape[0])
                    self.pts_3d_history.append(0)

                # 每个窗口结束前保留测试和保存
                if iteration == self.opt.iterations:
                    self.evaluate(self.global_iter, start_frame=start_frame, end_frame=end_frame, tag=f"Phase1_Win{win_idx}")
                    os.makedirs(self.args.model_path, exist_ok=True)
                    self.metrics_tracker.record_training_stats(self.global_iter, 0, self.gaussians.get_xyz.shape[0])

        progress_bar.close()

    def train_phase2_finetune(self, win_idx, start_frame, end_frame, overlap_image_cache):
        """阶段二：时序一致性微调（使用 finetune_iterations 配置）"""
        print(f"\n🔄 SWinGS 阶段 2: 时序微调窗口 {win_idx} [{start_frame}-{end_frame}]")
        
        # 冻结 MLP
        if hasattr(self.gaussians, 'set_mlp_requires_grad'):
            self.gaussians.set_mlp_requires_grad(False)

        finetune_iters = self.opt.finetune_iterations  # 细调迭代数 = 配置比例 * 每窗口总迭代数
        
        progress_bar = tqdm(range(1, finetune_iters + 1), desc=f"Win {win_idx} Phase 2")
        
        for iteration in range(1, finetune_iters + 1):
            self.global_iter += 1
            self.gaussians.update_learning_rate(iteration)

            batch_size = self.args.batch_size
            loss = 0
            
            for batch_idx in range(batch_size):
                # 75% 概率执行时序一致性约束，25% 正常训练 (MLP已被冻结，只微调 Canonical)
                is_consistency_step = random.random() < 0.75

                if is_consistency_step and overlap_image_cache is not None:
                    t_id = start_frame
                    dataset_idx = random.choice(self.frames_dict[t_id])
                    gt_image, viewpoint_cam = self.window_cache[dataset_idx] # 【修改】从缓存读取
                    gt_image, viewpoint_cam = gt_image.cuda(), viewpoint_cam.cuda()
                    
                    active_mask = self._get_active_dynamic_mask(t_id)
                    render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background, active_dynamic_mask=active_mask)
                    image = render_pkg["render"]
                    
                    # 一致性损失 L_consistency
                    current_loss = l1_loss(image, overlap_image_cache)
                    lambda_time = getattr(self.opt, 'lambda_time', 1.0)
                    current_loss = lambda_time * current_loss
                else:
                    t_id = random.randint(start_frame, end_frame)
                    dataset_idx = random.choice(self.frames_dict[t_id])
                    gt_image, viewpoint_cam = self.window_cache[dataset_idx]
                    gt_image, viewpoint_cam = gt_image.cuda(), viewpoint_cam.cuda()
                    
                    active_mask = self._get_active_dynamic_mask(t_id)
                    render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background, active_dynamic_mask=active_mask)
                    image = render_pkg["render"]

                    Ll1 = l1_loss(image, gt_image)
                    Lssim = 1.0 - ssim(image, gt_image)
                    current_loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * Lssim

                current_loss = current_loss / batch_size
                current_loss.backward()
                loss += current_loss.item()

            with torch.no_grad():
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{loss:.4f}"})
                    progress_bar.update(10)
                    self.loss_iterations.append(self.global_iter)
                    self.loss_history.append(loss)
                    self.pts_4d_history.append(self.gaussians.get_xyz.shape[0])
                    self.pts_3d_history.append(0)

                if iteration == finetune_iters:
                    self.evaluate(self.global_iter, start_frame=start_frame, end_frame=end_frame, tag=f"Phase2_Win{win_idx}")
                    os.makedirs(self.args.model_path, exist_ok=True)
                    # self.gaussians.save_ply(os.path.join(self.args.model_path, f"final_point_cloud_win{win_idx}.ply"))
        self.metrics_tracker.record_training_stats(self.global_iter, 0, self.gaussians.get_xyz.shape[0])

        progress_bar.close()

    def train(self):
        self.metrics_tracker.start_timer()
        
        # ==========================================
        # 大循环阶段一：按顺序独立训练所有窗口
        # ==========================================
        for win_idx, (start, end) in enumerate(self.window_blocks):
            torch.cuda.empty_cache()
            gc.collect()

            # 【提前把本窗口需要的图像全部读入 CPU 内存缓存】
            self.window_cache = {}
            for frame_id in range(start, end + 1):
                if frame_id in self.frames_dict:
                    for d_idx in self.frames_dict[frame_id]:
                        if d_idx not in self.window_cache:
                            self.window_cache[d_idx] = self.training_dataset[d_idx]

            # 【修复1：生命周期平滑继承】防止漫游时点云断裂消失
            with torch.no_grad():
                if win_idx == 0:
                    self.gaussians._start_frame[:] = start
                    self.gaussians._expire_frame[:] = end
                else:
                    # 把依然存活的点的寿命延长到本窗口末尾
                    alive_mask = (self.gaussians._start_frame <= start) & (self.gaussians._expire_frame >= start)
                    self.gaussians._expire_frame[alive_mask] = end

            self.train_phase1_window(win_idx, start, end)
            self.window_cache.clear()  # 释放当前窗口的图像缓存，准备下一个窗口

        # ==========================================
        # 大循环阶段二：时序一致性微调串联
        # ==========================================
        for win_idx in range(1, len(self.window_blocks)):
            start, end = self.window_blocks[win_idx]
            # 【提前把本窗口需要的图像全部读入 CPU 内存缓存】
            self.window_cache = {}
            for frame_id in range(start, end + 1):
                if frame_id in self.frames_dict:
                    for d_idx in self.frames_dict[frame_id]:
                        if d_idx not in self.window_cache:
                            self.window_cache[d_idx] = self.training_dataset[d_idx]
            # 1. 载入前一窗口 (w-1) 模型生成基准帧缓存
            # 注: 如果代码库没有好的 restore 方法，建议先手动实现
            # self.gaussians.restore(torch.load(os.path.join(self.args.model_path, f"phase1_win{win_idx-1}.pth")), self.opt)
            overlap_frame_id = start
            dataset_idx = random.choice(self.frames_dict[overlap_frame_id])
            _, viewpoint_cam = self.training_dataset[dataset_idx]
            viewpoint_cam = viewpoint_cam.cuda()
            
            with torch.no_grad():
                # 注意这里要切回 w-1 对应的激活掩码
                self.gaussians._start_frame[:] = self.window_blocks[win_idx-1][0]
                self.gaussians._expire_frame[:] = self.window_blocks[win_idx-1][1]
                active_mask = self._get_active_dynamic_mask(overlap_frame_id)
                render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background, active_dynamic_mask=active_mask)
                overlap_image_cache = render_pkg["render"].detach().clone()
            
            # 2. 载入当前窗口 (w) 模型准备微调
            # self.gaussians.restore(torch.load(os.path.join(self.args.model_path, f"phase1_win{win_idx}.pth")), self.opt)
            self.gaussians._start_frame[:] = start
            self.gaussians._expire_frame[:] = end
            
            self.train_phase2_finetune(win_idx, start, end, overlap_image_cache)
            torch.cuda.empty_cache()
            gc.collect()
            self.window_cache.clear()  # 释放当前窗口的图像缓存，准备下一个窗口
        
        # 【修复2：训练完毕后保存全局唯一的大模型】
        print(f"\n🎉 训练完毕！正在生成全序列最终标准大模型: chkpnt_{self.opt.iterations}.pth")
        self._save_checkpoint(str(self.opt.iterations))
        # 最终评估和记录
        self.metrics_tracker.record_training_stats(self.global_iter, 0, self.gaussians.get_xyz.shape[0])
        self.evaluate(iteration=self.global_iter, start_frame=0, end_frame=self.total_frames - 1, tag="FINAL_GLOBAL")

        
        print("\n🎉 SWinGS 两阶段严格训练完成！正在生成图表...")
        
        self.metrics_tracker.save_log(os.path.join(self.args.model_path, "swings_metrics.json"))
        self._draw_metrics_chart()

    def _draw_metrics_chart(self):
        """完全保留你原版代码的双Y轴 Matplotlib 绘图"""
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
            ax2.set_ylabel("Number of Gaussian Points", color='#333333', fontsize=12, fontweight='bold')
            line_4d, = ax2.plot(self.loss_iterations, self.pts_4d_history, label="4D Dynamic Points", color=color_4d, linewidth=2.0, alpha=0.85)
            
            lines = [line_loss, line_4d]
            ax1.legend(lines, [l.get_label() for l in lines], loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=11)
            plt.title("SWinGS Loss and Densification over Global Iterations", fontsize=14, fontweight='bold', pad=30)
            
            loss_plot_path = os.path.join(self.args.model_path, "swings_loss_and_points_curve.png")
            plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 [指标可视化] 联动图已保存至: {loss_plot_path}")
        except Exception as e:
            print(f"⚠️ [指标可视化] 绘制图表时发生错误: {e}")

    def _save_checkpoint(self, name_suffix):
        """【核心修复】标准 3DGS 漫游渲染器专用的保存格式"""
        os.makedirs(self.args.model_path, exist_ok=True)
        # 1. 保存包含 (模型参数, 迭代次数) 的元组，符合渲染器读取标准
        save_path = os.path.join(self.args.model_path, f"chkpnt_{name_suffix}.pth")
        torch.save((self.gaussians.capture(), self.global_iter), save_path)
        
        # 2. 生成标准的 point_cloud 目录结构
        point_cloud_path = os.path.join(self.args.model_path, f"point_cloud/iteration_{name_suffix}/point_cloud.ply")
        os.makedirs(os.path.dirname(point_cloud_path), exist_ok=True)
        self.gaussians.save_ply(point_cloud_path)


