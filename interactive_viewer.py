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
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

# 全局播放状态
class PlayState:
    is_playing = False

play_state = PlayState()

@torch.no_grad()
def main(dataset: ModelParams, pipe: PipelineParams, args):
    print(f"[引擎] 正在初始化 4DGS 终极自适应漫游引擎...")
    
    # 1. 初始化模型 (深灰色背景)
    bg_color = [1, 1, 1] if dataset.white_background else [0.2, 0.2, 0.2]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    sh_degree = dataset.sh_degree if hasattr(dataset, 'sh_degree') else 3
    gaussians = GaussianModel(sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, 
                              rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    
    # 2. 读取原始完美相机
    scene = Scene(dataset, gaussians, shuffle=False)
    train_cameras = [c[1] if isinstance(c, tuple) else c for c in scene.getTrainCameras()]
    max_frames = len(train_cameras) - 1
    
    checkpoint = args.start_checkpoint
    print(f"[引擎] 正在加载模型权重: {checkpoint}")
    (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
    gaussians.restore(model_params, None)
    
    server = viser.ViserServer(port=8080)
    print("\n" + "="*50)
    print(f"🚀 毕设专属 4D 引擎启动成功！请打开: http://localhost:8080")
    print("="*50 + "\n")
    
    # ==========================================
    # 🌟 核心修复 1：精准提取 COLMAP 真实环境的上方向
    # ==========================================
    c0 = train_cameras[0]
    w2c_colmap = c0.world_view_transform.transpose(0, 1).cpu().numpy()
    c2w_colmap = np.linalg.inv(w2c_colmap)
    c2w_opengl = c2w_colmap.copy()
    c2w_opengl[:, 1:3] *= -1 # 转换到 Viser 坐标系
    
    init_pos = c2w_opengl[:3, 3]
    init_wxyz = tf.SO3.from_matrix(c2w_opengl[:3, :3]).wxyz
    # 动态获取相机本地的 Y 轴作为 Up Vector，彻底杜绝翻转和倒立！！！
    real_up_direction = c2w_opengl[:3, 1]

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.scene.add_grid("grid", visible=False)
        client.camera.position = init_pos
        client.camera.wxyz = init_wxyz
        client.camera.up_direction = real_up_direction # 绝对吻合真实场景的重力方向！

    # ==========================================
    # 🎬 UI 控制台：自动播放器与画质控制
    # ==========================================
    with server.gui.add_folder("🎬 4D 动画播放器"):
        gui_time_idx = server.gui.add_slider("当前帧进度", min=0, max=max_frames, step=1, initial_value=0)
        with server.gui.add_folder("控制面板", expand_by_default=True):
            gui_btn_prev = server.gui.add_button("⏮️ 上一帧")
            gui_btn_play = server.gui.add_button("▶️ 自动播放")
            gui_btn_next = server.gui.add_button("⏭️ 下一帧")
            
    with server.gui.add_folder("🖥️ 渲染设置 (完全自适应)"):
        gui_res_h = server.gui.add_slider("基准清晰度 (纵向像素)", min=400, max=1200, step=100, initial_value=800)
        gui_reset_cam = server.gui.add_button("🔄 迷路一键回放")

    # --- 按钮绑定逻辑 ---
    @gui_btn_play.on_click
    def _(_):
        play_state.is_playing = not play_state.is_playing
        gui_btn_play.name = "⏸️ 暂停播放" if play_state.is_playing else "▶️ 自动播放"

    @gui_btn_prev.on_click
    def _(_):
        play_state.is_playing = False
        gui_btn_play.name = "▶️ 自动播放"
        gui_time_idx.value = max(0, gui_time_idx.value - 1)

    @gui_btn_next.on_click
    def _(_):
        play_state.is_playing = False
        gui_btn_play.name = "▶️ 自动播放"
        gui_time_idx.value = min(max_frames, gui_time_idx.value + 1)

    @gui_reset_cam.on_click
    def _(_):
        for client in server.get_clients().values():
            client.camera.position = init_pos
            client.camera.wxyz = init_wxyz

    while True:
        # --- 自动播放推进 ---
        if play_state.is_playing:
            next_frame = gui_time_idx.value + 1
            if next_frame > max_frames:
                next_frame = 0 # 循环播放
            gui_time_idx.value = next_frame

        clients = server.get_clients()
        for client_id, client in clients.items():
            cam_state = client.camera
            
            # ==========================================
            # 🌟 核心修复 2：自适应画幅与 FOV (拒绝拉伸与黑边)
            # ==========================================
            aspect = cam_state.aspect # 获取浏览器当前长宽比
            H = int(gui_res_h.value)
            W = int(H * aspect)       # 宽度动态适应浏览器，严丝合缝！
            
            # 维持原版纵向视野，横向视野根据浏览器动态展开
            fovy = c0.FoVy 
            fovx = 2 * math.atan(math.tan(fovy / 2) * aspect)
            
            # ==========================================
            # 🌟 核心修复 3：标准坐标系转换 (真实自由漫游)
            # ==========================================
            c2w_opengl = np.eye(4)
            c2w_opengl[:3, :3] = tf.SO3(cam_state.wxyz).as_matrix()
            c2w_opengl[:3, 3] = cam_state.position
            
            # Viser 转 3DGS 标准操作
            c2w_opencv = c2w_opengl.copy()
            c2w_opencv[:, 1:3] *= -1 
            w2c_opencv = np.linalg.inv(c2w_opencv)
            
            R = w2c_opencv[:3, :3].T 
            T = w2c_opencv[:3, 3]
            
            # 构建绝对安全的 Torch 矩阵
            world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0), dtype=torch.float32).transpose(0, 1).cuda()
            projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            
            # ==========================================
            # 🌟 核心修复 4：克隆夺舍 (保证离线级渲染画质)
            # ==========================================
            # 直接深拷贝原版相机，保留一切隐藏的 PyTorch 属性
            view_cam = copy.deepcopy(c0)
            view_cam.image_width = W
            view_cam.image_height = H
            view_cam.FoVy = fovy
            view_cam.FoVx = fovx
            view_cam.world_view_transform = world_view_transform
            view_cam.projection_matrix = projection_matrix
            view_cam.full_proj_transform = full_proj_transform
            view_cam.camera_center = world_view_transform.inverse()[3, :3]
            
            # 注入 4D 动画时间属性
            current_time_idx = int(gui_time_idx.value)
            view_cam.fid = current_time_idx
            view_cam.time = float(current_time_idx / max(1, max_frames))
            if hasattr(view_cam, 'timestamp'):
                view_cam.timestamp = view_cam.time
            
            # --- 渲染 ---
            render_pkg = render(view_cam, gaussians, pipe, background)
            rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            
            # 直接铺满整个网页背景！
            img_np = (rendered_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            client.scene.set_background_image(img_np, format="jpeg")
            
        time.sleep(0.02) # 控制在最高约 50fps

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