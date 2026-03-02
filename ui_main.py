# 预留的可视化集成入口

class HybridGSMainWindow:
    """整合训练配置、实时图表与3D漫游的图形用户界面"""
    
    def init_control_panel(self):
        """构建控制台：动态下发 swin_size, num_gs, tau_avg 等超参数"""
        pass

    def init_monitor_panel(self):
        """构建监控看板：通过 QThread 异步监听 MetricsTracker 推送的数据流并绘图"""
        pass

    def init_viewer_panel(self):
        """构建 OpenGL/WebGL 交互视窗：绑定鼠标键盘事件，实现自由视点漫游"""
        pass

    def start_training_thread(self, model_type: str):
        """拉起后台守护进程执行具体模型的训练逻辑"""
        pass