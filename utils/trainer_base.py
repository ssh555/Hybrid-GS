# 文件：utils/trainer_base.py
import os
import glob
import torch
from utils.metrics_tracker import MetricsTracker

class BaseTrainer:
    def __init__(self, dataset, opt, pipe, testing_iterations, saving_iterations, args):
        self.dataset = dataset
        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations
        self.args = args
        
        # 兼容 3D4DGS 的时间缩放逻辑
        if hasattr(self.dataset, 'frame_ratio') and self.dataset.frame_ratio > 1:
            self.args.time_duration = [self.args.time_duration[0] / self.dataset.frame_ratio, 
                                       self.args.time_duration[1] / self.dataset.frame_ratio]

        # 动态填补 total_frames 属性，兼容标准 COLMAP 数据集
        if not hasattr(self.dataset, 'total_frames'):
            img_path = os.path.join(self.dataset.source_path, self.dataset.images)
            num_frames = len(glob.glob(os.path.join(img_path, "*")))
            self.dataset.total_frames = num_frames
            print(f"[数据注入] 成功为当前数据集绑定 total_frames: {num_frames}")

        # 实例化统一的单例监控记录仪
        self.metrics_tracker = MetricsTracker()

    def train(self):
        raise NotImplementedError("子类必须实现具体的训练循环")
        
    def evaluate(self, iteration):
        pass