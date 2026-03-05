# 文件：ui_main.py
import sys
import os
import time
import json
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QFileDialog, QProgressBar, QGroupBox, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

# 导入 3DGS 核心组件
from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.camera_trajectory import generate_smooth_trajectory
from render import get_inference_active_mask

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class RenderWorker(QThread):
    """
    后台渲染线程：负责加载模型、生成轨迹、调用 CUDA 渲染并传回图像
    """
    # 定义信号：传回渲染好的 QImage, 当前帧号, 总帧数, 实时 FPS
    frame_ready = pyqtSignal(QImage, int, int, float)
    render_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path
        self.is_running = True
        self.is_paused = False

    def run(self):
        try:
            # 1. 解析配置
            from argparse import ArgumentParser, Namespace
            parser = ArgumentParser()
            lp = ModelParams(parser)
            pp = PipelineParams(parser)
            args = parser.parse_args([])
            
            cfg = OmegaConf.load(self.config_path)
            def recursive_merge(key, host):
                if isinstance(host[key], DictConfig):
                    for key1 in host[key].keys():
                        recursive_merge(key1, host[key])
                else:
                    setattr(args, key, host[key])
            for k in cfg.keys():
                recursive_merge(k, cfg)
                
            dataset = lp.extract(args)
            pipe = pp.extract(args)
            
            # 2. 初始化模型
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            
            gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=getattr(args, 'gaussian_dim', 4), 
                                      time_duration=getattr(args, 'time_duration', [-0.5, 0.5]), 
                                      rot_4d=getattr(args, 'rot_4d', True), 
                                      force_sh_3d=getattr(args, 'force_sh_3d', True))
            
            checkpoint = os.path.join(dataset.model_path, "chkpnt_30000.pth")
            if not os.path.exists(checkpoint):
                self.error_occurred.emit(f"找不到权重文件: {checkpoint}")
                return

            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, None)
            
            scene = Scene(dataset, gaussians, load_iteration=first_iter, shuffle=False)
            train_cameras = scene.getTrainCameras()
            
            # 3. 规划运镜轨迹 (300帧)
            num_cams = len(train_cameras)
            keyframe_indices = [0, num_cams//4, num_cams//2, int(num_cams*0.75), num_cams-1]
            keyframes = [train_cameras[i] for i in keyframe_indices]
            num_frames = 300
            trajectory = generate_smooth_trajectory(keyframes, num_frames=num_frames)
            
            # 4. 实时渲染循环
            for i, cam in enumerate(trajectory):
                if not self.is_running:
                    break
                while self.is_paused:
                    time.sleep(0.1)
                    
                start_time = time.time()
                
                # 获取掩码并渲染
                active_mask = get_inference_active_mask(gaussians, getattr(cam, 'uid', 0))
                render_pkg = render(cam, gaussians, pipe, background, active_dynamic_mask=active_mask)
                
                # 图像后处理
                rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                img_np = (rendered_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                
                # 转换为 PyQt QImage
                h, w, ch = img_np.shape
                bytes_per_line = ch * w
                qimg = QImage(img_np.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                fps = 1.0 / (time.time() - start_time)
                
                # 发送信号更新 UI
                self.frame_ready.emit(qimg, i + 1, num_frames, fps)
                
            self.render_finished.emit()

        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self.is_running = False
    
    def pause_resume(self):
        self.is_paused = not self.is_paused


class MetricsCanvas(FigureCanvas):
    """绘制训练指标折线图的画板"""
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot_metrics(self, json_path):
        if not os.path.exists(json_path):
            return
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            iters = [int(k) for k in data.keys()]
            vram = [v.get('VRAM_GB', 0) for v in data.values()]
            
            self.axes.clear()
            self.axes.plot(iters, vram, 'b-', label='VRAM (GB)')
            self.axes.set_title('Training VRAM Usage')
            self.axes.set_xlabel('Iteration')
            self.axes.set_ylabel('GB')
            self.axes.legend()
            self.draw()
        except Exception as e:
            print("解析 JSON 失败:", e)


class HybridGSViewer(QMainWindow):
    """
    主视窗 UI
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HybridGS 可视化漫游系统 - 毕业设计演示")
        self.resize(1280, 720)
        self.worker = None
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ================= 左侧控制面板 =================
        left_panel = QWidget()
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)

        # 1. 配置加载区
        config_group = QGroupBox("1. 模型配置加载")
        config_layout = QVBoxLayout()
        self.lbl_config = QLabel("当前配置: configs/n3v/default.yaml")
        btn_load_config = QPushButton("选择 YAML 配置文件")
        btn_load_config.clicked.connect(self.select_config)
        config_layout.addWidget(self.lbl_config)
        config_layout.addWidget(btn_load_config)
        config_group.setLayout(config_layout)

        # 2. 播放控制区
        play_group = QGroupBox("2. 漫游渲染控制")
        play_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶ 开始渲染/播放")
        self.btn_pause = QPushButton("⏸ 暂停/继续")
        self.btn_stop = QPushButton("⏹ 停止")
        
        self.btn_play.clicked.connect(self.start_rendering)
        self.btn_pause.clicked.connect(self.pause_rendering)
        self.btn_stop.clicked.connect(self.stop_rendering)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        
        btn_layout.addWidget(self.btn_play)
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_stop)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        self.lbl_status = QLabel("状态: 待机中")
        self.lbl_fps = QLabel("实时 FPS: 0.0")
        self.lbl_fps.setFont(QFont("Arial", 12, QFont.Bold))
        self.lbl_fps.setStyleSheet("color: #2E86C1;")

        play_layout.addLayout(btn_layout)
        play_layout.addWidget(self.progress_bar)
        play_layout.addWidget(self.lbl_status)
        play_layout.addWidget(self.lbl_fps)
        play_group.setLayout(play_layout)

        # 3. 数据监控面板 (Matplotlib)
        monitor_group = QGroupBox("3. 训练指标分析 (VRAM / Loss)")
        monitor_layout = QVBoxLayout()
        self.metrics_canvas = MetricsCanvas(self)
        btn_plot = QPushButton("加载训练日志 (metrics.json)")
        btn_plot.clicked.connect(self.load_metrics)
        monitor_layout.addWidget(self.metrics_canvas)
        monitor_layout.addWidget(btn_plot)
        monitor_group.setLayout(monitor_layout)

        left_layout.addWidget(config_group)
        left_layout.addWidget(play_group)
        left_layout.addWidget(monitor_group)
        left_layout.addStretch()

        # ================= 右侧渲染视窗 =================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        viewer_group = QGroupBox("实时漫游渲染视窗 (HybridGS Rendering)")
        viewer_layout = QVBoxLayout()
        self.lbl_video = QLabel("点击左侧【开始渲染】加载模型并预览漫游画面")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("background-color: black; color: white; font-size: 16px;")
        self.lbl_video.setMinimumSize(800, 600)
        
        viewer_layout.addWidget(self.lbl_video)
        viewer_group.setLayout(viewer_layout)
        right_layout.addWidget(viewer_group)

        # 使用分割器调整比例
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)
        
        self.current_config_path = "configs/n3v/default.yaml"

    def select_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 YAML 配置文件", "configs", "YAML Files (*.yaml)")
        if path:
            self.current_config_path = path
            self.lbl_config.setText(f"当前配置: {os.path.basename(path)}")

    def load_metrics(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 metrics.json", "output", "JSON Files (*.json)")
        if path:
            self.metrics_canvas.plot_metrics(path)

    def start_rendering(self):
        if not os.path.exists(self.current_config_path):
            self.lbl_status.setText(f"错误: 找不到配置 {self.current_config_path}")
            return

        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("状态: 模型加载中，请稍候...")
        
        # 启动后台渲染线程
        self.worker = RenderWorker(self.current_config_path)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.render_finished.connect(self.rendering_done)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.start()

    def pause_rendering(self):
        if self.worker:
            self.worker.pause_resume()
            status = "暂停中" if self.worker.is_paused else "渲染中..."
            self.lbl_status.setText(f"状态: {status}")

    def stop_rendering(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        self.rendering_done()
        self.lbl_status.setText("状态: 已停止")

    def update_frame(self, qimg, current_frame, total_frames, fps):
        # 刷新画面
        pixmap = QPixmap.fromImage(qimg)
        # 保持比例自适应窗口缩放
        scaled_pixmap = pixmap.scaled(self.lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_video.setPixmap(scaled_pixmap)
        
        # 刷新进度和指标
        self.progress_bar.setMaximum(total_frames)
        self.progress_bar.setValue(current_frame)
        self.lbl_status.setText(f"状态: 正在渲染 ({current_frame}/{total_frames})")
        self.lbl_fps.setText(f"实时渲染速度: {fps:.1f} FPS")

    def rendering_done(self):
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("状态: 渲染播放完成")

    def show_error(self, error_msg):
        self.lbl_status.setText("发生错误，请查看控制台")
        print(f"[UI 错误] {error_msg}")
        self.rendering_done()

    def closeEvent(self, event):
        # 关闭窗口时安全结束线程
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置深色主题风格，更显高级感
    app.setStyle("Fusion")
    
    viewer = HybridGSViewer()
    viewer.show()
    sys.exit(app.exec_())