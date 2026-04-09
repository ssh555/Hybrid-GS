# 文件：utils/metrics_tracker.py
import torch
import time
import json
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class MetricsTracker:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsTracker, cls).__new__(cls)
            cls._instance.metrics_log = {
                # --- 独立双时间轴 ---
                "train_iterations": [],    # 对应高频记录的训练指标 (如每100步)
                "test_iterations": [],     # 对应低频记录的测试指标 (如7000, 30000步)
                
                # --- 空间与时间质量指标 (挂载于 test_iterations) ---
                "psnr": [],
                "ssim": [],
                "lpips": [],
                "fps": [],                 # 渲染评估时的平均 FPS
                "temporal_psnr": [],       # 记录时间一致性/抗闪烁度 (T-PSNR)
                
                # --- 效率与状态指标 (挂载于 train_iterations) ---
                "vram_peak_mb": [],        # 捕捉VRAM消耗
                "training_time_s": [],     # 记录训练耗时，证明 HybridGS 更快
                "num_3d_gaussians": [],    # 记录被硬约束冻结的静态高斯数量
                "num_4d_gaussians": [],    # 记录活跃的动态前景高斯数量
            }
            cls._instance.start_time = None
            cls._instance.lpips_metric = LearnedPerceptualImagePatchSimilarity(
                net_type='alex',
                normalize=True
            ).cuda()
        return cls._instance

    def start_timer(self):
        """记录训练开始的绝对时间"""
        self.start_time = time.time()
        
    def record_training_stats(self, iteration, num_3d, num_4d):
        """
        统一记录所有高频训练状态指标，确保数组长度绝对一致！
        供论文中证明：背景转化为3D高斯，4D高斯数量维持低位，且时间/显存下降。
        """
        self.metrics_log["train_iterations"].append(iteration)
        
        # 记录时间
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.metrics_log["training_time_s"].append(elapsed)
        
        # 记录高斯数量
        self.metrics_log["num_3d_gaussians"].append(num_3d)
        self.metrics_log["num_4d_gaussians"].append(num_4d)
        
        # 记录显存
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        self.metrics_log["vram_peak_mb"].append(vram_mb)

    def record_eval_metrics(self, iteration, avg_psnr, avg_ssim, avg_lpips, avg_fps):
        self.metrics_log["test_iterations"].append(iteration)
        self.metrics_log["psnr"].append(avg_psnr)
        self.metrics_log["ssim"].append(avg_ssim)
        self.metrics_log["lpips"].append(avg_lpips)
        self.metrics_log["fps"].append(avg_fps)
        
    def measure_fps(self, render_func, *args, **kwargs):
        """利用CUDA底层事件同步机制包裹渲染核心函数，获取极致精准的渲染耗时"""
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        out = render_func(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time_ms = start_event.elapsed_time(end_event)
        fps = 1000.0 / elapsed_time_ms if elapsed_time_ms > 0 else 0
        
        # 注意：这里不再单独 append fps，直接返回交给 evaluate 函数计算平均值
        return out, fps
        
    def calculate_image_metrics(self, gt_image, rendered_image):
        """
        自动计算空间域指标：PSNR、SSIM与LPIPS
        输入:
            gt_image: [3,H,W] or [1,3,H,W], range [0,1]
            rendered_image: same as gt
        返回:
            psnr, ssim, lpips
        """
        from utils.image_utils import psnr
        from utils.loss_utils import ssim

        # 保证 batch 维度
        if gt_image.dim() == 3:
            gt_image = gt_image.unsqueeze(0)
        if rendered_image.dim() == 3:
            rendered_image = rendered_image.unsqueeze(0)

        gt_image = gt_image.clamp(0, 1)
        rendered_image = rendered_image.clamp(0, 1)

        with torch.no_grad():
            psnr_val = psnr(rendered_image, gt_image).mean().item()
            ssim_val = ssim(rendered_image, gt_image).mean().item()
            lpips_val = self.lpips_metric(rendered_image, gt_image).item()

        return psnr_val, ssim_val, lpips_val

    def calculate_temporal_metrics(self, rendered_seq, gt_seq):
        """计算时间域指标：Temporal PSNR / Flickering Metric"""
        pass
        
    def save_log(self, filepath):
        """将追踪数据实时序列化至JSON格式日志文件"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics_log, f, indent=4)