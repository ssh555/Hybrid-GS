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
                # 记录即将被淘汰的旧起始帧
                old_start = self.window_start
                self.window_start += 1
                self.window_end += 1

                # ===================================================
                # 🧹 动态内存垃圾回收：窗口一走，立刻把旧照片踢出内存！
                # ===================================================
                if hasattr(self, 'frames_dict') and hasattr(self, 'window_cache'):
                    if old_start in self.frames_dict:
                        for d_idx in self.frames_dict[old_start]:
                            self.window_cache.pop(d_idx, None)  # 物理释放内存

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
            
            # 优先从文件名中精准提取真实帧号
            try:
                frame_id = int(viewpoint_cam.image_name.split('_')[-1])
            except:
                frame_id = getattr(viewpoint_cam, 'fid', idx % self.total_frames)
            
            # 获取特定帧的干净掩码，过滤掉不在当前时间出生的点
            active_mask = self._get_active_dynamic_mask(frame_id)
            
            # 3. 必须把 active_dynamic_mask 传给渲染器！否则会满屏残影！
            render_pkg, fps = self.metrics_tracker.measure_fps(
                render, 
                viewpoint_cam, 
                self.gaussians, 
                self.pipe, 
                self.background, 
                active_dynamic_mask=active_mask
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

        # SWinGS 特有：初始化生命周期张量
        if not hasattr(self.gaussians, '_start_frame') or self.gaussians._start_frame.numel() == 0:
            num_pts = self.gaussians.get_xyz.shape[0]
            self.gaussians._start_frame = torch.zeros(num_pts, dtype=torch.int32, device="cuda")
            self.gaussians._expire_frame = torch.full((num_pts,), self.window_end, dtype=torch.int32, device="cuda")
            self.gaussians._mask_dynamic = torch.zeros(num_pts, dtype=torch.int8, device="cuda")

        # ==========================================
        # [新增] 初始化记录列表，增加点云数量监控
        # ==========================================
        self.loss_history = []
        self.loss_iterations = []
        self.pts_4d_history = []
        self.pts_3d_history = []

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
            
            loss = 0
            # =============== 批处理循环 (仅在窗口内采样) ===============
            for batch_idx in range(batch_size):
                # 1. SWinGS 官方算法：在当前活跃的滑动窗口内进行均匀随机抽样 (SGD核心)
                t_id = random.randint(self.window_start, self.window_end)
                
                # 2. 从预处理的字典中，随机抽取该帧对应的一个相机视角
                # 这个做法 100% 避免了之前的数组越界和单视角 Bug
                dataset_idx = random.choice(self.frames_dict[t_id])
                
                # 3. 提取真实图像和相机位姿
                frame_id = t_id
                # ===================================================
                # 🚀 动态缓存读取机制
                # ===================================================
                if dataset_idx not in self.window_cache:
                    # 第一次从硬盘/内存读取
                    raw_img, raw_cam = training_dataset[dataset_idx]
                    
                    # 🚀 极致优化：直接将张量转移到 GPU 并缓存！
                    # 使用 non_blocking=True 开启异步传输，不阻塞主线程
                    gpu_img = raw_img.cuda(non_blocking=True)
                    
                    # 注意：3DGS 的 Camera 类可能没有完整的 .cuda() 方法，
                    # 但我们通常只需要它的 image_name, FoVx 等属性，或者原作者已经处理好了
                    try:
                        gpu_cam = raw_cam.cuda()
                    except AttributeError:
                        gpu_cam = raw_cam  # 如果不支持 cuda() 就不强求，主要是图片耗时
                        
                    self.window_cache[dataset_idx] = (gpu_img, gpu_cam)

                # 🚀 光速读取：直接从显存字典里拿出来用，没有任何搬运开销！
                gt_image, viewpoint_cam = self.window_cache[dataset_idx]

                # render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background)
                # 🚀 替换为带有防御掩码的终极版：
                active_mask = self._get_active_dynamic_mask(frame_id)
                render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background, active_dynamic_mask=active_mask)

                image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                alpha = render_pkg["alpha"]
                

                # 计算基础损失
                Ll1 = l1_loss(image, gt_image)
                Lssim = 1.0 - ssim(image, gt_image)
                current_loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * Lssim
                
                # Opa Mask Loss
                if self.opt.lambda_opa_mask > 0 and hasattr(viewpoint_cam, 'gt_alpha_mask') and viewpoint_cam.gt_alpha_mask is not None:
                    o = alpha.clamp(1e-6, 1-1e-6)
                    sky = 1 - viewpoint_cam.gt_alpha_mask
                    current_loss = current_loss + self.opt.lambda_opa_mask * (- sky * torch.log(1 - o)).mean()
                    

                    
                need_velocity = (self.opt.lambda_motion > 0) or (self.opt.lambda_rigid > 0)
                
                if need_velocity:
                    current_t = frame_id / self.total_frames if hasattr(self, 'total_frames') else self.gaussians.get_t
                    
                    # ⚠️ 核心提速：只给当前活着的点算速度 (MLP 唯一一次前向传播)
                    _, active_velocity = self.gaussians.get_current_covariance_and_mean_offset(1.0, current_t, mask=active_mask)
                    
                    # --- 1. Rigid Loss (KNN 刚性防炸裂约束) ---
                    if self.opt.lambda_rigid > 0:
                        k_neighbors = 10
                        xyz_active = self.gaussians.get_xyz[active_mask].contiguous()
                        
                        # 绝对防爆机制：强制最多只随机抽 30,000 个点算 KNN！
                        if xyz_active.shape[0] > 30000:
                            perm = torch.randperm(xyz_active.shape[0], device="cuda")[:30000]
                            xyz_cur = xyz_active[perm].contiguous()
                            velocity_cur = active_velocity[perm]
                        else:
                            xyz_cur = xyz_active
                            velocity_cur = active_velocity

                        # 防止点数太少导致 KNN 报错
                        if xyz_cur.shape[0] > k_neighbors:
                            idx, dist = knn(xyz_cur[None].detach(), xyz_cur[None].detach(), k_neighbors)
                            weight = torch.exp(-100 * dist)
                            vel_dist = torch.norm(velocity_cur[idx.squeeze(0)] - velocity_cur.unsqueeze(1), p=2, dim=-1)
                            coherence_loss = (weight * vel_dist).sum() / k_neighbors / xyz_cur.shape[0]
                            current_loss = current_loss + (self.opt.lambda_rigid * 5.0) * coherence_loss
                    
                if self.opt.lambda_motion > 0:
                    current_t = frame_id / self.total_frames if hasattr(self, 'total_frames') else self.gaussians.get_t
                    
                    # 💡 极限提速核心：无论当前窗口有多少活着的点，最多只抽 10,000 个算速度！
                    active_indices = torch.nonzero(active_mask, as_tuple=False).squeeze()
                    
                    if active_indices.numel() > 10000:
                        perm = torch.randperm(active_indices.numel(), device="cuda")[:10000]
                        sampled_indices = active_indices[perm]
                        sampled_mask = torch.zeros_like(active_mask)
                        sampled_mask[sampled_indices] = True
                    else:
                        sampled_mask = active_mask

                    # MLP 唯一一次前向传播，且最多只算 1 万个点！耗时极其微小！
                    _, sampled_velocity = self.gaussians.get_current_covariance_and_mean_offset(1.0, current_t, mask=sampled_mask)
                    
                    # 惩罚拉扯，防碎纸屑
                    current_loss = current_loss + (self.opt.lambda_motion * 2.0) * sampled_velocity.norm(p=2, dim=1).mean()

                # =============== 统一回传 =================
                current_loss = current_loss / batch_size
                current_loss.backward()
                loss += current_loss.item()
                
                batch_point_grad.append(torch.norm(viewspace_point_tensor.grad[:,:2], dim=-1))
                batch_radii.append(radii)
                batch_visibility_filter.append(visibility_filter)

            # =============== 梯度累加 ===============
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
                    
            iter_end.record()

            # =============== 致密化与优化器 ===============
            with torch.no_grad():
                # 1. 移除了最外层的点数限制，只要在 densify_until_iter 期限内，就永远进得来！
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
                            current_pts = self.gaussians.get_xyz.shape[0]
                            max_points = getattr(self.opt, 'densify_until_num_points', 4000000)
                            if max_points > 0 and current_pts >= max_points:
                                active_grad_threshold = 99999.0 
                                active_grad_t_threshold = 99999.0 

                            self.gaussians.densify_and_prune(active_grad_threshold, self.opt.thresh_opa_prune, self.scene.cameras_extent, size_threshold, active_grad_t_threshold)
                            
                            if hasattr(self.gaussians, 'dynamic2static'):
                                self.gaussians.dynamic2static(self.opt.scale_t_threshold)
                                
                # 大扫除独立出来
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
                    
                    # ==========================================
                    # [修改] 每 10 步记录 Loss 和 高斯点数量
                    # ==========================================
                    self.loss_iterations.append(iteration)
                    self.loss_history.append(loss)
                    
                    num_4d = self.gaussians.get_xyz.shape[0] if self.gaussians.get_xyz is not None else 0
                    self.pts_4d_history.append(num_4d)
                    
                    num_3d = 0
                    self.pts_3d_history.append(num_3d)

                if iteration == self.opt.iterations:
                    self.evaluate(iteration)
                    os.makedirs(self.args.model_path, exist_ok=True)
                    torch.save((self.gaussians.capture(), iteration), os.path.join(self.args.model_path, f"chkpnt_{iteration}.pth"))
                    self.gaussians.save_ply(os.path.join(self.args.model_path, f"point_cloud_{iteration}.ply"))
                    num_4d = self.gaussians.get_xyz.shape[0]
                    num_3d = 0
                    self.metrics_tracker.record_training_stats(iteration, num_3d, num_4d)


                    
        progress_bar.close()
        self.metrics_tracker.save_log(os.path.join(self.args.model_path, "swings_metrics.json"))

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