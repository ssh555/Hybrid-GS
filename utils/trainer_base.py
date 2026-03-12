import torch
from utils.metrics_tracker import MetricsTracker

class BaseTrainer:
    def __init__(self, dataset, opt, pipe, testing_iterations, saving_iterations, args):
        self.dataset = dataset
        
        # =========================================================
        # [新增] 动态填补 total_frames 属性，完美兼容标准 COLMAP 数据集
        # =========================================================
        if not hasattr(self.dataset, 'total_frames'):
            import os, glob
            # 自动去 images 文件夹里数一下到底有多少张图片（即多少帧）
            img_path = os.path.join(self.dataset.source_path, self.dataset.images)
            num_frames = len(glob.glob(os.path.join(img_path, "*")))
            self.dataset.total_frames = num_frames
            print(f"[数据注入] 成功为当前数据集绑定 total_frames: {num_frames}")
        # =========================================================

        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations
        self.args = args
        
        # 实例化统一的单例监控记录仪
        self.metrics_tracker = MetricsTracker()
        
        # 初始化高斯模型（稍后在 gaussian_model.py 中重构）
        # self.gaussians = GaussianModel(...)

    def train(self):
        """定义训练循环骨架，子类必须实现特定步骤"""
        raise NotImplementedError("子类必须实现具体的训练循环")
        
    def evaluate(self, iteration):
        """统一的评估管线"""
        # 调用 MetricsTracker 计算 PSNR, SSIM, LPIPS 并捕捉 VRAM 消耗
        pass
        
    def save_model(self, iteration):
        """统一的模型序列化保存管线"""
        pass