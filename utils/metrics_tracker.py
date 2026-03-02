import torch
import time
import json

class MetricsTracker:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsTracker, cls).__new__(cls)
            cls._instance.metrics_log = {
                "iteration": [],
                "psnr": [],
                "ssim": [],
                "lpips": [],
                "vram_peak_mb": [],
                "fps": [],
                # ================= 新增核心指标 =================
                "training_time_s": [],     # 记录训练耗时，证明 HybridGS 更快
                "num_3d_gaussians": [],    # 记录被硬约束冻结的静态高斯数量
                "num_4d_gaussians": [],    # 记录活跃的动态前景高斯数量
                "temporal_psnr": []        # 记录时间一致性/抗闪烁度 (T-PSNR)
            }
            cls._instance.start_time = None
        return cls._instance

    def start_timer(self):
        """记录训练开始的绝对时间"""
        self.start_time = time.time()
        
    def record_training_time(self, iteration):
        """记录从开始到当前迭代的累计训练时间"""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.metrics_log["training_time_s"].append(elapsed)
            # 保证列表长度一致（如果不独立记录的话，建议和其他指标一起append）

    def record_gaussian_stats(self, num_3d, num_4d):
        """
        记录3D和4D高斯的数量。
        用于在论文中证明：随着时间推移，背景被转化为3D高斯，4D高斯数量保持在一个低水平。
        """
        self.metrics_log["num_3d_gaussians"].append(num_3d)
        self.metrics_log["num_4d_gaussians"].append(num_4d)

    def record_vram(self, iteration):
        """利用底层的接口捕捉VRAM消耗"""
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        if iteration not in self.metrics_log["iteration"]:
            self.metrics_log["iteration"].append(iteration)
        self.metrics_log["vram_peak_mb"].append(vram_mb)
        
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
        self.metrics_log["fps"].append(fps)
        return out, fps
        
    def calculate_image_metrics(self, gt_image, rendered_image):
        """自动计算空间域指标：PSNR、SSIM与LPIPS"""
        # 具体指标计算代码
        pass

    def calculate_temporal_metrics(self, rendered_seq, gt_seq):
        """
        计算时间域指标：Temporal PSNR / Flickering Metric
        输入必须是连续的几帧图像，评估帧间连贯性。
        用于证明 SWinGS 和 L_reg_d 软约束抑制了闪烁。
        """
        pass
        
    def save_log(self, filepath):
        """将追踪数据实时序列化至JSON格式日志文件"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics_log, f, indent=4)