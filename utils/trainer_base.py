# 文件：utils/trainer_base.py
import os
import glob
import torch
from utils.metrics_tracker import MetricsTracker

class BaseTrainer:
    def __init__(self, dataset, opt, pipe, testing_iterations, saving_iterations, args, debug_params):
        self.dataset = dataset
        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations
        self.args = args
        self.debug_params = debug_params
        
        # 兼容 3D4DGS 的时间缩放逻辑
        if hasattr(self.dataset, 'frame_ratio') and self.dataset.frame_ratio > 1:
            self.args.time_duration = [self.args.time_duration[0] / self.dataset.frame_ratio, 
                                       self.args.time_duration[1] / self.dataset.frame_ratio]

        # 动态填补 total_frames 属性，严格解析 camxx_xxxx 命名规范
        if not hasattr(self.dataset, 'total_frames'):
            cam00_path = os.path.join(self.dataset.source_path, self.dataset.images, 'cam00')
            # 直接使用 scandir 统计文件总数
            try:
                # 只统计文件(is_file)，排除子文件夹（如果有的话）
                with os.scandir(cam00_path) as entries:
                    file_count = sum(1 for entry in entries if entry.is_file())
                
                self.dataset.total_frames = file_count
                print(f"\n[数据注入] ⚡ 通过 os.scandir 快速统计 cam00，判定总帧数: {self.dataset.total_frames}")
                
            except FileNotFoundError:
                self.dataset.total_frames = 1
                print(f"\n[数据注入] ⚠️ 警告：找不到目录 {cam00_path}，设为默认值 1")

        # 实例化统一的单例监控记录仪
        self.metrics_tracker = MetricsTracker()

    def train(self):
        raise NotImplementedError("子类必须实现具体的训练循环")
        
    def evaluate(self, iteration):
        pass