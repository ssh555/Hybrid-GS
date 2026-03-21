# 文件名：interactive_viewer.py
# pip install viser
# 运行命令: python interactive_viewer.py --config ./configs/n3v/3D4DGS.yaml --start_checkpoint output/3d4dgs/miku_shaungxue_daxi/test_short/chkpnt_30000.pth
import os
import time
import math
import torch
import numpy as np
import viser
import viser.transforms as tf
from argparse import ArgumentParser
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class MiniCam:
    """一个轻量级的虚拟相机类，用来欺骗 3DGS 渲染器，让它以为这是一个真实相机"""
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = world_view_transform.inverse()[3, :3]
        # 4D 专属时间属性
        self.fid = 0
        self.time = 0.0
        self.timestamp = 0.0

@torch.no_grad()
def main(dataset: ModelParams, pipe: PipelineParams, args):
    print(f"[引擎] 正在初始化 4DGS 实时漫游引擎...")
    
    # 1. 初始化模型
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    sh_degree = dataset.sh_degree if hasattr(dataset, 'sh_degree') else 3
    gaussians = GaussianModel(sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, 
                              rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    
    checkpoint = args.start_checkpoint
    print(f"[引擎] 正在加载模型权重: {checkpoint}")
    (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
    gaussians.restore(model_params, None)
    
    # 2. 获取总帧数 (根据你之前说的，大约 200 帧左右)
    # 我们可以通过读取原相机序列来获取准确的总帧数
    scene = Scene(dataset, gaussians, shuffle=False)
    train_cameras = [c[1] if isinstance(c, tuple) else c for c in scene.getTrainCameras()]
    max_frames = max([getattr(c, 'fid', 0) for c in train_cameras]) 
    if max_frames == 0: max_frames = 200 # 兜底默认值
    
    # 3. 启动 Viser Web 服务器
    server = viser.ViserServer(port=8080)
    print("\n" + "="*50)
    print(f"🚀 漫游引擎启动成功！")
    print(f"👉 请在浏览器中打开: http://localhost:8080")
    print("="*50 + "\n")
    
    # 在网页 UI 上添加控制面板
    gui_frame = server.gui.add_slider("🎬 时间轴 (Frame)", min=0, max=max_frames, step=1, initial_value=0)
    gui_res_scale = server.gui.add_slider("🖥️ 渲染画质缩放", min=0.1, max=1.0, step=0.1, initial_value=0.5)
    gui_hint = server.gui.add_markdown("🕹️ **操作指南**：\n- **左键拖拽**: 旋转视角\n- **右键拖拽**: 平移\n- **滚轮/WASD**: 前进后退")

    # 4. 主渲染循环 (死循环，不断监听浏览器里相机的移动并实时渲染)
    while True:
        # 获取当前连接的用户（通常只有一个，就是你自己打开的浏览器）
        clients = server.get_clients()
        for client_id, client in clients.items():
            cam_state = client.camera
            
            # 计算当前分辨率 (拖动时降低分辨率可以大幅提升流畅度)
            W = int(cam_state.aspect * 1000 * gui_res_scale.value)
            H = int(1000 * gui_res_scale.value)
            
            # --- 核心数学：坐标系转换 (Viser -> 3DGS) ---
            # Viser (Web) 的相机是: +X向右, +Y向上, +Z向后
            # 3DGS (COLMAP) 的相机是: +X向右, +Y向下, +Z向前
            c2w = np.eye(4)
            c2w[:3, :3] = tf.SO3(cam_state.wxyz).as_matrix()
            c2w[:3, 3] = cam_state.position
            
            # 翻转 Y 和 Z 轴
            c2w[:3, 1:3] *= -1 
            
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3].T  # 3DGS 要求 R 是转置的
            T = w2c[:3, 3]
            
            # 计算 FOV
            fovy = cam_state.fov
            fovx = 2 * math.atan(math.tan(fovy / 2) * cam_state.aspect)
            
            # 构建 PyTorch 矩阵
            world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()
            projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            
            # 生成虚拟相机
            view_cam = MiniCam(W, H, fovy, fovx, 0.01, 100.0, world_view_transform, full_proj_transform)
            
            # 注入 4D 动态时间戳
            view_cam.fid = int(gui_frame.value)
            view_cam.time = float(gui_frame.value / max(1, max_frames))
            view_cam.timestamp = view_cam.time
            
            # --- 执行前向渲染 ---
            render_pkg = render(view_cam, gaussians, pipe, background)
            rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            
            # 转换成图片并发送给浏览器！
            img_np = (rendered_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            # client.set_background_image(img_np, format="jpeg")
            client.scene.set_background_image(img_np, format="jpeg")
        # 极短的休眠，防止死循环把 CPU 跑满
        time.sleep(0.01)

if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gaussian_dim", type=int, default=4)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--rot_4d", action="store_true", default=True)
    parser.add_argument("--force_sh_3d", action="store_true", default=True)
    parser.add_argument("--start_checkpoint", type=str, required=True)
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for k in host[key].keys(): recursive_merge(k, host[key])
        elif hasattr(args, key): setattr(args, key, host[key])
    for k in cfg.keys(): recursive_merge(k, cfg)
        
    main(lp.extract(args), pp.extract(args), args)