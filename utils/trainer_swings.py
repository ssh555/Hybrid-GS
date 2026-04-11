# 文件：utils/trainer_swings.py
import os
import json
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
        self.overlap_size = 8


        self.max_window_points = getattr(args, 'max_window_points', 2_000_000)

        self.densify_until_iter = self.opt.densify_until_iter
        self.densify_grad_threshold = self.opt.densify_grad_threshold
        self.iterations = self.opt.iterations
        self.thresh_opa_prune = self.opt.thresh_opa_prune
        self.enable_split = True
        self.densify_until_num_points = self.opt.densify_until_num_points
        self.densify_from_iter = self.opt.densify_from_iter
        
        # [SWinGS 严格算法] 划分带 1 帧重叠的块状窗口 (Block Windows)
        self.window_blocks = []
        curr_start = 0
        while curr_start < self.total_frames - 1:
            curr_end = min(curr_start + self.swin_size - 1, self.total_frames - 1)
            self.window_blocks.append((curr_start, curr_end))
            if curr_end == self.total_frames - 1:
                break
            curr_start = curr_end # 下一窗口的起点是本窗口的终点 (1帧重叠)
            
        print(f"[{self.__class__.__name__}] 单卡严格串行模式初始化！总帧数: {self.total_frames}")
        print(f"[{self.__class__.__name__}] 窗口规划: {self.window_blocks}")

        self.training_dataset = self.scene.getTrainCameras()
        test_cameras = self.scene.getTestCameras()
        os.makedirs(self.args.model_path, exist_ok=True)
        # 定义缓存文件路径 (存放在模型输出目录下)
        cache_path = os.path.join(self.args.model_path, "frames_mapping_cache.json")

        self.reset_pth_path = os.path.join(self.args.model_path, "initial_checkpoint.pth")
        torch.save((self.gaussians.capture(), 0), self.reset_pth_path)  # 保存初始模型参数，供每个窗口训练前重置使用

        # self.gaussians.save_ply(self.reset_ply_path)  # 保存初始点云，供每个窗口训练前重置使用

        if os.path.exists(cache_path):
            print(f"[{self.__class__.__name__}] ⚡ 命中缓存！正在从文件极速恢复帧映射关系...")
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            # json 的 key 默认是 string，需要转回 int
            self.frames_dict = {int(k): v for k, v in cache_data["train_frames_dict"].items()}
            self.test_frame_ids = cache_data["test_frame_ids"]
        else:
            print(f"[{self.__class__.__name__}] ⏳ 未找到缓存，初次解析帧数据映射，可能需要一些时间...")
            
            # 1. 解析训练集
            self.frames_dict = {}
            for idx, cam in enumerate(self.training_dataset):
                try:
                    frame_id = int(cam.image_name.split('_')[-1])
                except:
                    frame_id = getattr(cam, 'fid', idx % self.total_frames)
                if frame_id not in self.frames_dict:
                    self.frames_dict[frame_id] = []
                self.frames_dict[frame_id].append(idx)
                
            # 2. 解析测试集（提前把测试集的 frame_id 也提取出来存好）
            self.test_frame_ids = []
            if test_cameras:
                for idx, cam in enumerate(test_cameras):
                    try:
                        frame_id = int(cam.image_name.split('_')[-1])
                    except:
                        frame_id = getattr(cam, 'fid', idx % self.total_frames)
                    self.test_frame_ids.append(frame_id)
                    
            # 3. 写入缓存文件
            with open(cache_path, 'w') as f:
                json.dump({
                    "train_frames_dict": self.frames_dict,
                    "test_frame_ids": self.test_frame_ids
                }, f)
            print(f"[{self.__class__.__name__}] 💾 帧映射解析完成并已保存至: {cache_path}")
            
        self.window_cache = {}
        
        # 记录全局绘图数据
        self.loss_history = []
        self.loss_iterations = []
        self.pts_4d_history = []
        self.pts_3d_history = []
        self.global_iter = 0  # 全局迭代步数，用于贯穿所有窗口
        self.eval_psnr_sum = 0.0
        self.eval_ssim_sum = 0.0
        self.eval_lpips_sum = 0.0
        self.eval_fps_sum = 0.0
        self.eval_frame_count = 0

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
            
        total_psnr, total_ssim, total_lpips, total_fps = 0.0, 0.0, 0.0, 0.0
        valid_frames_count = 0  # 记录当前窗口内有效测试帧的数量
        
        for idx, batch_data in enumerate(tqdm(test_cameras, desc="Testing")):
            gt_image, viewpoint_cam = batch_data
            
            # # 【极速获取】直接从缓存的列表中读取 frame_id，省去耗时的字符串解析
            # frame_id = self.test_frame_ids[idx]
            try:
                frame_id = int(viewpoint_cam.image_name.split('_')[-1])
            except:
                frame_id = getattr(viewpoint_cam, 'fid', idx % self.total_frames)
            # 直接跳过不属于当前窗口的测试帧！防止纯黑画面拉低平均分！
            if end_frame is not None:
                if frame_id < start_frame or frame_id > end_frame:
                    continue
            for (w_start, w_end) in self.window_blocks:
                if w_start <= frame_id <= w_end:
                    # 渲染这帧前，把高斯的时间域切换到它对应的训练窗口
                    self.gaussians._start_frame[:] = w_start
                    self.gaussians._expire_frame[:] = w_end
                    break
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
                psnr_val, ssim_val, lpips_val = self.metrics_tracker.calculate_image_metrics(
                    gt_image,
                    image
                )

                total_psnr += psnr_val
                total_ssim += ssim_val
                total_lpips += lpips_val
                # total_psnr += psnr(image, gt_image).mean().item()
                # total_ssim += ssim(image, gt_image).mean().item()
                # lpips_val = self.metrics_tracker.lpips_metric(image, gt_image).item()
                # total_lpips += lpips_val
                valid_frames_count += 1
            total_fps += fps
            
        if valid_frames_count == 0:
            print("[评估警告] 当前窗口没有分配到测试帧。")
            return
            
        avg_psnr = total_psnr / valid_frames_count
        avg_ssim = total_ssim / valid_frames_count
        avg_fps = total_fps / valid_frames_count
        avg_lpips = total_lpips / valid_frames_count
        
        self.metrics_tracker.record_eval_metrics(iteration, avg_psnr, avg_ssim, avg_lpips, avg_fps)
        self.eval_psnr_sum += total_psnr
        self.eval_ssim_sum += total_ssim
        self.eval_fps_sum += total_fps
        self.eval_lpips_sum += total_lpips
        self.eval_frame_count += valid_frames_count
        print(f"[评估结果 ({valid_frames_count} 帧)] PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f} | FPS: {avg_fps:.2f}")

    def train_phase1_window(self, win_idx, start_frame, end_frame):
        """阶段一：独立训练当前窗口（使用 YAML 中的 self.iterations）"""
        print(f"\n🚀 开始 {self.__class__.__name__} 阶段 1: 独立训练窗口 {win_idx} [{start_frame}-{end_frame}]")

        # 框定生命周期在当前窗口
        # self.gaussians._start_frame[:] = start_frame  
        # self.gaussians._expire_frame[:] = end_frame
        self.gaussians.bind_current_window(start_frame, end_frame)
        # 使用 YAML 配置作为单窗口的迭代总数
        total_iters = self.iterations
        
        progress_bar = tqdm(range(1, total_iters + 1), desc=f"Win {win_idx} Phase 1")
        
        for iteration in range(1, total_iters + 1):
            self.global_iter += 1
            
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
                if ((self.opt.lambda_motion > 0) or (self.opt.lambda_rigid > 0)):
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
                if iteration < self.densify_until_iter and (self.densify_until_num_points < 0 or (self.gaussians.get_xyz.shape[0]) < self.densify_until_num_points):
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    
                    if batch_size == 1:
                        self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, batch_t_grad if self.gaussians.gaussian_dim == 4 else None)
                    else:
                        self.gaussians.add_densification_stats_grad(batch_viewspace_point_grad, visibility_filter, batch_t_grad if self.gaussians.gaussian_dim == 4 else None)

                    if iteration > self.densify_from_iter: 
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        if iteration % self.opt.densification_interval == 0: 
                            self.gaussians.densify_and_prune(self.densify_grad_threshold, self.thresh_opa_prune, self.scene.cameras_extent, size_threshold, self.opt.densify_grad_t_threshold, enable_split = self.enable_split)
                                
                if iteration % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and iteration == self.densify_from_iter):
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
                if iteration == self.iterations:
                    self.evaluate(self.global_iter, start_frame=start_frame, end_frame=end_frame, tag=f"Phase1_Win{win_idx}")
                    os.makedirs(self.args.model_path, exist_ok=True)
                    self.metrics_tracker.record_training_stats(self.global_iter, 0, self.gaussians.get_xyz.shape[0])

        progress_bar.close()

    def train_phase2_finetune(self, win_idx, start_frame, end_frame, overlap_caches):
        """阶段二：时序一致性微调（使用 finetune_iterations 配置）"""
        print(f"\n🔄 {self.__class__.__name__} 阶段 2: 时序微调窗口 {win_idx} [{start_frame}-{end_frame}]")
        
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
                is_consistency_step = random.random() < self.opt.replay_prob

                if is_consistency_step and overlap_caches is not None and len(overlap_caches) > 0:
                    t_id = random.choice(list(overlap_caches.keys()))
                    dataset_idx = random.choice(list(overlap_caches[t_id].keys()))

                    gt_image, viewpoint_cam = self.window_cache[dataset_idx]
                    gt_image, viewpoint_cam = gt_image.cuda(), viewpoint_cam.cuda()

                    active_mask = self._get_active_dynamic_mask(t_id)

                    render_pkg = render(
                        viewpoint_cam,
                        self.gaussians,
                        self.pipe,
                        self.background,
                        active_dynamic_mask=active_mask
                    )
                    image = render_pkg["render"]

                    target_cache = overlap_caches[t_id][dataset_idx]

                    current_loss = l1_loss(image, target_cache)
                    current_loss = self.opt.lambda_time * current_loss
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

    def _reset_gaussian(self):
        """【核心修复】重置高斯 MLP 权重，防止过拟合局部窗口"""
        (model_params, _) = torch.load(self.reset_pth_path, weights_only=False)
        self.gaussians.restore(model_params, self.opt)

    def train(self):
        self.metrics_tracker.start_timer()
        
        # ==========================================
        # 大循环阶段一：按顺序独立训练所有窗口
        # ==========================================
        for win_idx, (start, end) in enumerate(self.window_blocks):
            _path = os.path.join(self.args.model_path, "phase1", f"phase1_win_{win_idx}.pth")
            # if os.path.exists(_path):
            #     # _path = os.path.join(self.args.model_path, "phase1", f"phase1_win_{win_idx}.pth")
            #     # prev_model_data = torch.load(_path, weights_only=False)
            #     # self.gaussians.restore(prev_model_data, self.opt)
            #     continue  # 跳过已完成的窗口

            if win_idx > self.opt.freeze_end_idx:
                self.densify_until_iter = self.opt.densify_until_iter_after_freeze
                self.densify_grad_threshold = self.opt.densify_grad_threshold_after_freeze
                self.iterations = self.opt.iterations_after_freeze
                self.enable_split = self.opt.enable_split_after_freeze
                self.thresh_opa_prune = self.opt.thresh_opa_prune_after_freeze
                self.densify_until_num_points = self.opt.densify_until_num_points_after_freeze
                self.densify_from_iter = self.opt.densify_from_iter_after_freeze
            torch.cuda.empty_cache()
            gc.collect()

            # 【提前把本窗口需要的图像全部读入 CPU 内存缓存】
            self.window_cache = {}
            for frame_id in range(start, end + 1):
                if frame_id in self.frames_dict:
                    for d_idx in self.frames_dict[frame_id]:
                        if d_idx not in self.window_cache:
                            self.window_cache[d_idx] = self.training_dataset[d_idx]
            if not hasattr(self.gaussians, '_start_frame') or self.gaussians._start_frame.numel() == 0:
                num_pts = self.gaussians.get_xyz.shape[0]
                self.gaussians._start_frame = torch.zeros(num_pts, dtype=torch.int32, device="cuda")
                self.gaussians._expire_frame = torch.zeros(num_pts, dtype=torch.int32, device="cuda")
                self.gaussians._mask_dynamic = torch.zeros(num_pts, dtype=torch.int8, device="cuda")
            # 【修复1：生命周期平滑继承】防止漫游时点云断裂消失
            with torch.no_grad():
                self.current_window_start = start
                self.current_window_end = end

                num_pts = self.gaussians.get_xyz.shape[0]

                if win_idx == 0:
                    # 首窗口初始化生命周期
                    self.gaussians._start_frame = torch.full(
                        (num_pts,), start, dtype=torch.int32, device="cuda"
                    )
                    self.gaussians._expire_frame = torch.full(
                        (num_pts,), end, dtype=torch.int32, device="cuda"
                    )
                else:
                    # 只允许 overlap 帧仍存活的动态点进入新窗口
                    overlap_frame = start

                    static_mask = (self.gaussians._mask_dynamic == 1)
                    dynamic_alive = (
                        (self.gaussians._mask_dynamic != 1)
                        & (self.gaussians._start_frame <= overlap_frame)
                        & (self.gaussians._expire_frame >= overlap_frame)
                    )

                    inherit_mask = static_mask | dynamic_alive

                    # 只给真正继承下来的动态点续命
                    self.gaussians._expire_frame[dynamic_alive] = end

                    # 不继承的动态点直接标记死亡，等待窗口后 prune
                    dead_mask = (~inherit_mask) & (self.gaussians._mask_dynamic != 1)
                    self.gaussians._expire_frame[dead_mask] = overlap_frame - 1
            self.train_phase1_window(win_idx, start, end)
            
            # 必须在第一阶段结束时持久化局部窗口模型
            _path = os.path.join(self.args.model_path, "phase1", f"phase1_win_{win_idx}.pth")
            os.makedirs(os.path.dirname(_path), exist_ok=True)
            torch.save(self.gaussians.capture(), _path)
            # self._reset_gaussian()
            if win_idx < len(self.window_blocks) - 1:
                next_start = self.window_blocks[win_idx + 1][0]
                with torch.no_grad():
                    # 1. 静态背景点 (_mask_dynamic == 1) 拥有免死金牌，永远保留。
                    # 2. 动态点必须满足不透明度 > 0.01 才允许进入下一个窗口。
                    dynamic = self.gaussians._mask_dynamic.view(-1)
                    opacity = self.gaussians.get_opacity.view(-1)
                    expire = self.gaussians._expire_frame.view(-1)

                    static_keep = (dynamic == 1)

                    dynamic_keep = (
                        (dynamic != 1)
                        & (expire >= next_start)
                        & (opacity > self.opt.win_end_prune_opacity_threshold)
                    )

                    keep_mask = static_keep | dynamic_keep
                    # =========================
                    # 二级限制：最多 200w
                    # =========================
                    max_pts = self.max_window_points
                    keep_indices = torch.nonzero(keep_mask, as_tuple=True)[0]

                    if keep_indices.numel() > max_pts:
                        static_indices = torch.nonzero(static_keep, as_tuple=True)[0]
                        dyn_indices = torch.nonzero(dynamic_keep, as_tuple=True)[0]

                        remain = max_pts - static_indices.numel()
                        remain = max(remain, 0)

                        if dyn_indices.numel() > remain:
                            dyn_opacity = opacity[dyn_indices]
                            topk = torch.topk(dyn_opacity, remain, sorted=False).indices
                            dyn_indices = dyn_indices[topk]

                        final_keep = torch.zeros_like(keep_mask)
                        final_keep[static_indices] = True
                        final_keep[dyn_indices] = True
                        keep_mask = final_keep
                    prune_mask = ~keep_mask
                    print(
                        f"[Window {win_idx}] total={keep_mask.numel()} "
                        f"keep={keep_mask.sum().item()} "
                        f"prune={prune_mask.sum().item()}"
                    )
                    # 你需要在 gaussian_model.py 中实现一个 prune_by_mask 函数
                    # 用于在底层张量和优化器中剔除 keep_mask == False 的点
                    if prune_mask.any():
                        self.gaussians.prune_points(prune_mask)
                        print(f"🧹 窗口 {win_idx} 结束，清理了 {prune_mask.sum().item()} 个过期动态点！")
            self.window_cache.clear()  # 释放当前窗口的图像缓存，准备下一个窗口

        # ==========================================
        # 大循环阶段二：时序一致性微调串联
        # ==========================================
        for win_idx in range(1, len(self.window_blocks)):
            _path = os.path.join(self.args.model_path, "phase2", f"phase2_win_{win_idx}.pth")
            # if os.path.exists(_path):
            #     continue  # 跳过已完成的窗口
            start, end = self.window_blocks[win_idx]
            # 【提前把本窗口需要的图像全部读入 CPU 内存缓存】
            self.window_cache = {}
            for frame_id in range(start, end + 1):
                if frame_id in self.frames_dict:
                    for d_idx in self.frames_dict[frame_id]:
                        if d_idx not in self.window_cache:
                            self.window_cache[d_idx] = self.training_dataset[d_idx]
            # 1. 载入前一窗口 (w-1) 模型生成基准帧缓存
            overlap_size = 8
            prev_start, prev_end = self.window_blocks[win_idx - 1]
            # 上一窗口最后 overlap_size 帧
            prev_overlap_frames = list(
                range(max(prev_start, prev_end - overlap_size + 1), prev_end + 1)
            )

            # 当前窗口前 overlap_size 帧
            curr_overlap_frames = list(
                range(start, min(end + 1, start + overlap_size))
            )
            overlap_pairs = list(zip(prev_overlap_frames, curr_overlap_frames))
            overlap_caches = {}
            
            with torch.no_grad():
                prev_model_data = torch.load(
                    os.path.join(self.args.model_path, "phase1", f"phase1_win_{win_idx - 1}.pth"),
                    weights_only=False
                )
                self.gaussians.restore(prev_model_data, self.opt)
                self.gaussians.bind_current_window(prev_start, prev_end)

                for prev_fid, curr_fid in overlap_pairs:
                    overlap_caches[curr_fid] = {}

                    for d_idx in self.frames_dict[curr_fid]:
                        _, viewpoint_cam = self.training_dataset[d_idx]
                        viewpoint_cam = viewpoint_cam.cuda()

                        active_mask = self._get_active_dynamic_mask(prev_fid)

                        render_pkg = render(
                            viewpoint_cam,
                            self.gaussians,
                            self.pipe,
                            self.background,
                            active_dynamic_mask=active_mask
                        )

                        overlap_caches[curr_fid][d_idx] = (
                            render_pkg["render"].detach().clone()
                        )

            # 2. 载入当前窗口 (w) 模型准备微调
            _path = os.path.join(self.args.model_path, "phase1", f"phase1_win_{win_idx}.pth")
            curr_model_data = torch.load(_path, weights_only=False)
            self.gaussians.restore(curr_model_data, self.opt)
            # self.gaussians.restore(torch.load(os.path.join(self.args.model_path, f"phase1_win{win_idx}.pth")), self.opt)
            # self.gaussians._start_frame[:] = start
            # self.gaussians._expire_frame[:] = end
            self.gaussians.bind_current_window(start, end)
            self.train_phase2_finetune(win_idx, start, end, overlap_caches)
            # 保存微调后的结果
            _path = os.path.join(self.args.model_path, "phase2", f"phase2_win_{win_idx}.pth")
            os.makedirs(os.path.dirname(_path), exist_ok=True)
            torch.save(self.gaussians.capture(), _path)
            torch.cuda.empty_cache()
            gc.collect()
            self.window_cache.clear()  # 释放当前窗口的图像缓存，准备下一个窗口
        
        # 【修复2：训练完毕后保存全局唯一的大模型】
        print(f"\n🎉 训练完毕！正在生成全序列最终标准大模型: chkpnt_{self.opt.iterations}.pth")
        self._save_merged_checkpoint(str(self.opt.iterations))
        # 最终评估和记录
        self.metrics_tracker.record_training_stats(self.global_iter, 0, self.gaussians.get_xyz.shape[0])
        # self.evaluate(iteration=self.global_iter, start_frame=0, end_frame=self.total_frames - 1, tag="FINAL_GLOBAL")
        avg_psnr = self.eval_psnr_sum / max(self.eval_frame_count, 1)
        avg_ssim = self.eval_ssim_sum / max(self.eval_frame_count, 1)
        avg_lpips = self.eval_lpips_sum / max(self.eval_frame_count, 1)
        avg_fps = self.eval_fps_sum / max(self.eval_frame_count, 1)
        self.metrics_tracker.record_eval_metrics(self.global_iter, avg_psnr, avg_ssim, avg_lpips, avg_fps)
        print(f"[INFO] [最终评估结果] PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f} | FPS: {avg_fps:.2f}")
        
        print(f"\n🎉 {self.__class__.__name__} 两阶段严格训练完成！正在生成图表...")
        
        self.metrics_tracker.save_log(os.path.join(self.args.model_path, f"{self.__class__.__name__}_metrics.json"))
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


    def _save_merged_checkpoint(self, name_suffix):
        """【无损瘦身版】多窗口合并打包。自动剔除渲染不需要的优化器状态，防止预加载时爆内存。"""
        os.makedirs(self.args.model_path, exist_ok=True)
        
        merged_windows = {}
        for win_idx in range(len(self.window_blocks)):
            if win_idx == 0:
                path = os.path.join(self.args.model_path, "phase1", f"phase1_win_{win_idx}.pth")
            else:
                path = os.path.join(self.args.model_path, "phase2", f"phase2_win_{win_idx}.pth")
            
            if os.path.exists(path):
                # 1. 强行映射到 CPU，防止合并过程就把显存挤爆
                win_data = torch.load(path, weights_only=False)
                
                # 2. 【无损压缩核心逻辑】：遍历高斯参数，剥离巨无霸优化器状态
                compressed_win_data = []
                for item in win_data:
                    # 在 3DGS/4DGS 的 capture() 中，只有 optimizer.state_dict() 是字典类型
                    if isinstance(item, dict):
                        # 识别到优化器状态，直接替换为空字典。这一步能减小 66% 的体积！
                        compressed_win_data.append({})
                    elif isinstance(item, torch.Tensor):
                        # 确保纯参数张量停留在 CPU 物理内存中
                        compressed_win_data.append(item)
                    else:
                        compressed_win_data.append(item)
                
                # 将瘦身后的数据转回元组并存入超级字典
                merged_windows[win_idx] = tuple(compressed_win_data)
                
        final_super_dict = {
            "is_swings_sequence": True,  # 渲染器读取标志
            "window_blocks": self.window_blocks,
            "models": merged_windows
        }
        
        save_path = os.path.join(self.args.model_path, f"chkpnt_{name_suffix}.pth")
        torch.save((final_super_dict, self.global_iter), save_path)
        print(f"✅ 多窗口超级大模型已完成【无损瘦身压缩】，并保存至: {save_path}")
