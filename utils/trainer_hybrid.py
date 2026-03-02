# 派生的 HybridGS 核心训练策略
# Ours
# 文件：utils/trainer_hybrid.py
from .trainer_base import TrainerSWinGS

class TrainerHybrid(TrainerSWinGS): # 继承SWinGS以获得滑动窗口能力
    """空间解耦(硬约束)与时间解耦(软约束)深度融合的混合模型"""
    
    def robust_hard_constraint_classifier(self):
        """
        鲁棒硬约束判定器 (解决空间冗余)：
        执行双重阈值联合判定逻辑 (R_i < tau_avg AND R_{i,max} < tau_max)。
        只将绝对静止的舞台背景永久冻结放入全局背景显存池。
        """
        pass

    def compute_loss(self, render_pkg, gt_image, d_displacement):
        """
        重写损失函数 (解决时间冗余与闪烁)：
        计算 L_color + L_reg_d，植入位移收敛正则化项。
        迫使动态高斯在静止片段物理坍缩为3D特征，运动时被光度梯度冲破束缚。
        """
        pass