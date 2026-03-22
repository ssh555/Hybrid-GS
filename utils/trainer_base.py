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

        # 动态填补 total_frames 属性，严格解析 camxx_xxxx 命名规范
        if not hasattr(self.dataset, 'total_frames'):
            img_path = os.path.join(self.dataset.source_path, self.dataset.images)
            img_files = glob.glob(os.path.join(img_path, "*"))
            max_frame = 0
            for f in img_files:
                basename = os.path.basename(f)
                name_without_ext = os.path.splitext(basename)[0]
                try:
                    # 解析 camxx_xxxx 中的第二部分 xxxx
                    parts = name_without_ext.split('_')
                    if len(parts) >= 2:
                        frame_idx = int(parts[-1])
                        if frame_idx > max_frame:
                            max_frame = frame_idx
                except ValueError:
                    continue
            self.dataset.total_frames = max_frame + 1
            print(f"\n[数据注入] 🎯 成功解析 camxx_xxxx 规范，绑定真实总帧数: {self.dataset.total_frames}")

        # 实例化统一的单例监控记录仪
        self.metrics_tracker = MetricsTracker()

    def train(self):
        raise NotImplementedError("子类必须实现具体的训练循环")
        
    def evaluate(self, iteration):
        pass