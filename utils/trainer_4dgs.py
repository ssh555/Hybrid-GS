# 文件：utils/trainer_4dgs.py
from trainer_base import BaseTrainer

class Trainer4DGS(BaseTrainer):
    """原生 3D-4DGS 封装，使用全局平均时间尺度进行静态转化"""
    
    def adaptive_convert_4d_to_3d(self):
        """原生的定期探针：提取时间维度缩放参数判定静态区域并降维"""
        pass