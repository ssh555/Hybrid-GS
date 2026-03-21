import os
import time
import math
import copy
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
from utils.graphics_utils import getProjectionMatrix

@torch.no_grad()
def main(dataset: ModelParams, pipe: PipelineParams, args):
    print(f"[引擎] 正在初始化 4DGS 实时漫游引擎...")
    
    # 1. 初始化模型 (深灰色背景防迷路)
    bg_color = [1, 1, 1] if dataset.white_background else [0.2, 0.2, 0.2]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    sh_degree = dataset.sh_degree if hasattr(dataset, 'sh_degree') else 3
    gaussians = GaussianModel(sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, 
                              rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    
    # 2. 先初始化 Scene (让它生成废点)
    scene = Scene(dataset, gaussians, shuffle=False)
    
    # 3. 强行加载你的 20000 步完美权重，覆盖废点！
    checkpoint = args.start_checkpoint
    print(f"[引擎] 正在加载模型权重: {checkpoint}")
    (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
    gaussians.restore(model_params, None)
    
    # 获取所有的真实训练相机
    train_cameras = [c[1] if isinstance(c, tuple) else c for c in scene.getTrainCameras()]
    max_frames = len(train_cameras) - 1
    
    # 4. 启动 Viser Web 服务器
    server = viser.ViserServer(port=8080)
    print("\n" + "="*50)
    print(f"🚀 漫游引擎启动成功！")
    print(f"👉 请在浏览器中打开: http://localhost:8080")
    print("="*50 + "\n")
    
    # ==========================================
    # 🌟 传送门：解决出生不在舞台 + 修正旋转圆心
    # ==========================================
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        print("👋 浏览器已连接，正在将视角传送到舞台中心...")
        c0 = train_cameras[0]
        # 把 3DGS 坐标转换给 Viser
        w2c = c0.world_view_transform.transpose(0, 1).cpu().numpy()
        c2w = np.linalg.inv(w2c)
        c2w[:3, 1:3] *= -1 # 翻转 Y 和 Z
        
        # 将用户传送到第一帧相机的物理位置
        client.camera.position = c2w[:3, 3]
        client.camera.wxyz = tf.SO3.from_matrix(c2w[:3, :3]).wxyz
        # 【关键】设置相机的聚焦点为舞台中心 (0,0,0)，这样旋转操作就不会反了！
        client.camera.look_at = (0.0, 0.0, 0.0)
        client.scene.add_grid("grid", visible=False)

    # 网页 UI 控制台
    gui_frame = server.gui.add_slider("🎬 4D 时间轴", min=0, max=max_frames, step=1, initial_value=0)
    gui_res_scale = server.gui.add_slider("🖥️ 画质缩放 (卡顿请调低)", min=0.1, max=1.0, step=0.1, initial_value=0.5)
    server.gui.add_markdown("🕹️ **操作指南**：\n- **左键拖拽**: 围绕舞台旋转\n- **右键拖拽**: 平移相机\n- **滚轮**: 拉近拉远")
    gui_pose_info = server.gui.add_markdown("📍 **当前位姿**: `等待获取...`")

    while True:
        clients = server.get_clients()
        for client_id, client in clients.items():
            cam_state = client.camera
            
            # 实时更新 UI 坐标
            pos, rot = cam_state.position, cam_state.wxyz
            gui_pose_info.content = f"📍 **坐标**:\n `{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}`"
            
            # ==========================================
            # 🌟 夺舍术：克隆真实相机，完美保留所有 4D 属性！
            # ==========================================
            frame_idx = int(gui_frame.value)
            frame_idx = min(frame_idx, len(train_cameras) - 1)
            # 克隆对应的真实相机（这就连带把 timestamp, fid 等全偷过来了，时间轴绝对有效！）
            view_cam = copy.copy(train_cameras[frame_idx])
            
            # 计算当前画质分辨率
            W = int(cam_state.aspect * 1000 * gui_res_scale.value)
            H = int(1000 * gui_res_scale.value)
            view_cam.image_width = W
            view_cam.image_height = H
            
            # --- Viser 相机移动 映射回 3DGS ---
            c2w = np.eye(4)
            c2w[:3, :3] = tf.SO3(cam_state.wxyz).as_matrix()
            c2w[:3, 3] = cam_state.position
            
            c2w[:3, 1:3] *= -1 # Viser to COLMAP
            w2c = np.linalg.inv(c2w)
            
            # 构建标准的 Torch 矩阵
            world_view_transform = torch.tensor(w2c, dtype=torch.float32).transpose(0, 1).cuda()
            fovy = cam_state.fov
            fovx = 2 * math.atan(math.tan(fovy / 2) * cam_state.aspect)
            
            projection_matrix = getProjectionMatrix(znear=0.01, zfar=1000.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            
            # 强行覆盖克隆相机的空间位置
            view_cam.FoVy = fovy
            view_cam.FoVx = fovx
            view_cam.world_view_transform = world_view_transform
            view_cam.projection_matrix = projection_matrix
            view_cam.full_proj_transform = full_proj_transform
            view_cam.camera_center = world_view_transform.inverse()[3, :3]
            
            # 渲染画面！
            render_pkg = render(view_cam, gaussians, pipe, background)
            rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            
            # 发送给网页
            img_np = (rendered_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            client.scene.set_background_image(img_np, format="jpeg")
            
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