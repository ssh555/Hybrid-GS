# 新增懒加载机制的数据集类
# 长序列按需懒加载机制（Lazy DataLoader）
# 2-3分钟的长序列表演视频 -> 一次性读取几千帧的高清图片到 GPU -> 显存爆炸
# 文件：scene/dataset.py
import os
import torch
import torchvision
from PIL import Image
import json
import numpy as np
from scene.cameras import Camera
from utils.graphics_utils import fov2focal, focal2fov

class LazyCameraDataloader:
    """
    长序列动态视频的按需懒加载器 (Lazy DataLoader)
    针对 SWinGS 和 HybridGS 设计，配合滑动窗口动态申请与释放显存，彻底避免 OOM
    """
    def __init__(self, dataset_path, args, resolution_scale=1.0):
        self.dataset_path = dataset_path
        self.args = args
        self.resolution_scale = resolution_scale
        
        # 1. 仅加载轻量级的元数据 (Metadata)，绝对不加载图像张量
        self.image_paths = []     # 存储所有帧的磁盘路径索引
        self.cameras_meta = {}    # 存储相机内参、外参和时间戳等轻量级对象
        self.loaded_images = {}   # 内存/显存中的活动图像字典 {frame_id: image_tensor}
        
        self.total_frames = 0
        self._load_metadata()

    def _load_metadata(self):
        """
        解析通过 MP4 -> PNG -> COLMAP 生成的 transforms_train.json 格式数据。
        提取图像路径、外参矩阵(C2W)、内参(FOV)以及时间戳(Time)，放入内存中。
        """
        # 1. 确定 json 文件路径 (优先加载 train，也可以根据传参加载)
        json_path = os.path.join(self.dataset_path, "transforms_train.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(self.dataset_path, "transforms.json") # 备用路径
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"在 {self.dataset_path} 目录下未找到 transforms_train.json 或 transforms.json！")
        
        print(f"[LazyLoader] 正在读取长序列元数据，路径: {json_path}")
        
        with open(json_path, 'r') as f:
            meta = json.load(f)
            
        # 2. 提取全局或基础内参 (如果有的话)
        # 多数脚本会在顶层写入 camera_angle_x
        global_camera_angle_x = meta.get("camera_angle_x", None)
        global_fl_x = meta.get("fl_x", None)
        global_fl_y = meta.get("fl_y", None)
        global_cx = meta.get("cx", None)
        global_cy = meta.get("cy", None)
        global_w = meta.get("w", None)
        global_h = meta.get("h", None)

        # 3. 提取所有帧并进行时间序列排序
        frames = meta["frames"]
        # 对于长序列表演视频，必须确保帧是按照时间严格顺序排列的
        # 如果有 'time' 字段按 time 排，否则按文件名字符串排序
        frames.sort(key=lambda x: x.get("time", x["file_path"]))
        
        for frame_id, frame in enumerate(frames):
            # --- A. 解析并拼接图像绝对路径 ---
            # json里的 file_path 通常是 "./images/00001" 或 "images/00001.png"
            image_name = frame["file_path"]
            
            # 去除前导的 "./" 保证路径拼接正确
            if image_name.startswith("./"):
                image_name = image_name[2:]
                
            # 补充扩展名 (如果脚本生成的 json 中没有后缀)
            if not (image_name.endswith(".png") or image_name.endswith(".jpg")):
                image_name += ".png" 
                
            full_image_path = os.path.join(self.dataset_path, image_name)
            self.image_paths.append(full_image_path)
            
            # --- B. 解析 4DGS 必需的时间戳 ---
            # 归一化时间戳，通常在 [0, 1] 或 [-0.5, 0.5] 之间
            timestamp = frame.get("time", 0.0) 
            
            # --- C. 解析外参矩阵 (C2W: Camera to World) ---
            # json 中的 transform_matrix 默认是 C2W 矩阵
            c2w = np.array(frame["transform_matrix"])
            
            # --- D. 解析逐帧的特定内参 (支持动态内参变焦相机) ---
            camera_angle_x = frame.get("camera_angle_x", global_camera_angle_x)
            fl_x = frame.get("fl_x", global_fl_x)
            fl_y = frame.get("fl_y", global_fl_y)
            cx = frame.get("cx", global_cx)
            cy = frame.get("cy", global_cy)
            w = frame.get("w", global_w)
            h = frame.get("h", global_h)
            
            # 4. 将提取的轻量级元数据存入字典
            self.cameras_meta[frame_id] = {
                "c2w": c2w,
                "timestamp": float(timestamp),
                "camera_angle_x": camera_angle_x,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "image_name": os.path.basename(image_name)
            }
            
        self.total_frames = len(self.image_paths)
        print(f"[LazyLoader] 元数据加载成功！共解析出 {self.total_frames} 帧动态序列。")

    def _read_image_to_tensor(self, frame_id):
        """从磁盘异步读取单张图像并转换为 GPU 张量"""
        if frame_id >= len(self.image_paths):
            return None
            
        img_path = self.image_paths[frame_id]
        image = Image.open(img_path).convert("RGB")
        
        # 降采样逻辑（如果开启了降采样）
        if self.resolution_scale != 1.0:
            new_size = (int(image.size[0] * self.resolution_scale), int(image.size[1] * self.resolution_scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
        # 转换为 C, H, W 张量并送入显存
        transform = torchvision.transforms.ToTensor()
        img_tensor = transform(image).cuda()
        return img_tensor

    def prefetch_window(self, start_frame: int, end_frame: int):
        """
        异步I/O预抓取：将当前滑动窗口 [start_frame, end_frame] 内的图像加载到显存
        """
        for frame_id in range(start_frame, min(end_frame + 1, self.total_frames)):
            if frame_id not in self.loaded_images:
                # 只有当图像不在显存时才执行高昂的磁盘 I/O 操作
                self.loaded_images[frame_id] = self._read_image_to_tensor(frame_id)

    def release_window(self, current_start_frame: int):
        """
        显存清理：释放已经滑出当前时间窗口的陈旧图像数据，死死压制显存峰值
        """
        frames_to_delete = []
        for frame_id in self.loaded_images.keys():
            if frame_id < current_start_frame:
                frames_to_delete.append(frame_id)
                
        for frame_id in frames_to_delete:
            # 从字典中删除引用
            del self.loaded_images[frame_id]
            
        if len(frames_to_delete) > 0:
            # 强制清空 CUDA 缓存，避免显存碎片化导致的中途崩溃
            torch.cuda.empty_cache()

    def get_camera_data(self, frame_id: int):
        """
        获取指定帧的相机对象与高清图像张量，供前向渲染与损失计算使用。
        在此处将 JSON 中的 C2W 矩阵转换为渲染器需要的 W2C 和投影矩阵。
        """
        # 1. 触发懒加载，获取 GPU 张量图像
        if frame_id not in self.loaded_images:
            self.loaded_images[frame_id] = self._read_image_to_tensor(frame_id)
            
        image_tensor = self.loaded_images[frame_id]
        meta = self.cameras_meta[frame_id]
        
        # 2. 从 meta 中恢复相机内参 (FOV)
        if meta["camera_angle_x"] is not None:
            fovx = meta["camera_angle_x"]
            fovy = focal2fov(fov2focal(fovx, image_tensor.shape[2]), image_tensor.shape[1])
        elif meta["fl_x"] is not None:
            fovx = focal2fov(meta["fl_x"], image_tensor.shape[2])
            fovy = focal2fov(meta["fl_y"], image_tensor.shape[1])
        else:
            raise ValueError("JSON 中既没有 camera_angle_x 也没有 focal length (fl_x/fl_y)！")
            
        # 3. 处理外参：原版 json 给的是 C2W (Camera to World)
        # 3DGS 渲染器需要的是 R, T (World to Camera)
        c2w = meta["c2w"]
        # 对 NeRF 的 OpenGL 相机坐标系(Y上, Z后)转 COLMAP 坐标系(Y下, Z前)
        c2w[:3, 1:3] *= -1 
        
        # 矩阵求逆得到 W2C
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # 3DGS 中的 R 是转置的 W2C 旋转矩阵
        T = w2c[:3, 3]                 # W2C 平移向量
        
        # 4. 组装并返回标准的 Camera 对象
        # 注意：这里的传参要和你原代码 `scene/cameras.py` 里的 Camera 类的 __init__ 对齐
        cam = Camera(
            colmap_id=frame_id, 
            R=R, 
            T=T, 
            FoVx=fovx, 
            FoVy=fovy, 
            image=image_tensor, 
            gt_alpha_mask=None,
            image_name=meta["image_name"], 
            uid=frame_id,
            data_device="cuda",
            timestamp=meta["timestamp"]  # <--- 4DGS 最核心的时间戳参数
        )
        
        return cam
    