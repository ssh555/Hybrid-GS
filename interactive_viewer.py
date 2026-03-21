import os
import time
import math
import copy
import torch
import numpy as np
import viser
from argparse import ArgumentParser
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.graphics_utils import getProjectionMatrix

def apply_pan_and_zoom(cam, zoom_factor, pan_x, pan_y):
    """绝对安全的焦距缩放与平移算法（不破坏矩阵正交性）"""
    # 1. 安全缩放 (修改 FOV)
    fovx = cam.FoVx / zoom_factor
    fovy = cam.FoVy / zoom_factor
    
    # 2. 安全平移
    w2c_T = cam.world_view_transform.cpu().numpy()
    w2c = w2c_T.T
    R = w2c[:3, :3]
    T = w2c[:3, 3]
    
    # 获取相机的局部 Right(X) 和 Up(Y) 轴
    right = R[0, :]
    up = R[1, :]
    
    # 计算世界空间偏移量
    shift_world = (pan_x * right) + (pan_y * up)
    T_new = T - R @ shift_world
    
    # 3. 完美重建正交矩阵
    w2c[:3, 3] = T_new
    cam.world_view_transform = torch.tensor(w2c.T, dtype=torch.float32).cuda()
    cam.camera_center = cam.world_view_transform.inverse()[3, :3]
    cam.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
    
    return cam

# ==========================================
# 🌟 全局播放状态机
# ==========================================
class PlayState:
    is_playing = False

play_state = PlayState()

@torch.no_grad()
def main(dataset: ModelParams, pipe: PipelineParams, args):
    print(f"[引擎] 正在初始化 4DGS 播放器引擎...")
    
    # 初始化模型
    bg_color = [1, 1, 1] if dataset.white_background else [0.2, 0.2, 0.2]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    sh_degree = dataset.sh_degree if hasattr(dataset, 'sh_degree') else 3
    gaussians = GaussianModel(sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, 
                              rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    
    # 读取原始完美相机
    scene = Scene(dataset, gaussians, shuffle=False)
    train_cameras = [c[1] if isinstance(c, tuple) else c for c in scene.getTrainCameras()]
    max_frames = len(train_cameras) - 1
    
    checkpoint = args.start_checkpoint
    print(f"[引擎] 正在加载模型权重: {checkpoint}")
    (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
    gaussians.restore(model_params, None)
    
    server = viser.ViserServer(port=8080)
    print("\n" + "="*50)
    print(f"🚀 毕设专属 4D 播放器启动成功！请打开: http://localhost:8080")
    print("="*50 + "\n")
    
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.scene.add_grid("grid", visible=False)
        client.camera.up_direction = (0, 0, 1)

    # ==========================================
    # 🎬 核心 UI：多媒体播放控制台
    # ==========================================
    with server.gui.add_folder("🎬 4D 动画播放器"):
        gui_time_idx = server.gui.add_slider("当前帧进度", min=0, max=max_frames, step=1, initial_value=0)
        
        # 将按钮放在同一行
        with server.gui.add_folder("控制面板", expand_by_default=True):
            gui_btn_prev = server.gui.add_button("⏮️ 上一帧")
            gui_btn_play = server.gui.add_button("▶️ 自动播放")
            gui_btn_next = server.gui.add_button("⏭️ 下一帧")
            
    with server.gui.add_folder("🎥 镜头与画质控制"):
        # 修复畸变：改为整数步进切换真实机位！
        gui_cam_pos = server.gui.add_slider("🎥 切换原版机位", min=0, max=max_frames, step=1, initial_value=0)
        gui_zoom = server.gui.add_slider("🔍 镜头拉近/推远", min=0.5, max=3.0, step=0.05, initial_value=1.0)
        gui_pan_x = server.gui.add_slider("↔️ 镜头水平微调", min=-2.0, max=2.0, step=0.05, initial_value=0.0)
        gui_pan_y = server.gui.add_slider("↕️ 镜头垂直微调", min=-2.0, max=2.0, step=0.05, initial_value=0.0)
        gui_res_scale = server.gui.add_slider("🖥️ 渲染画质 (卡顿请降低)", min=0.1, max=1.0, step=0.1, initial_value=0.8)
        gui_reset_cam = server.gui.add_button("🔄 重置镜头")

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
        gui_zoom.value = 1.0
        gui_pan_x.value = 0.0
        gui_pan_y.value = 0.0

    while True:
        # --- 自动播放核心逻辑 ---
        if play_state.is_playing:
            next_frame = gui_time_idx.value + 1
            if next_frame > max_frames:
                next_frame = 0 # 播到结尾自动循环
            gui_time_idx.value = next_frame

        clients = server.get_clients()
        for client_id, client in clients.items():
            cam_state = client.camera
            
            # 1. 提取绝对原版安全机位 (拒绝插值畸变)
            cam_val = int(gui_cam_pos.value)
            view_cam = copy.deepcopy(train_cameras[cam_val])
            
            # 2. 注入安全的平移与缩放
            view_cam = apply_pan_and_zoom(view_cam, gui_zoom.value, gui_pan_x.value, gui_pan_y.value)
            
            # 3. 注入 4D 动画时间属性
            current_time_idx = int(gui_time_idx.value)
            view_cam.fid = current_time_idx
            view_cam.time = float(current_time_idx / max(1, max_frames))
            if hasattr(view_cam, 'timestamp'):
                view_cam.timestamp = view_cam.time
            
            # --- 渲染画质计算 ---
            render_w = int(view_cam.image_width * gui_res_scale.value)
            render_h = int(view_cam.image_height * gui_res_scale.value)
            view_cam.image_width = render_w
            view_cam.image_height = render_h
            
            # --- 执行渲染 ---
            render_pkg = render(view_cam, gaussians, pipe, background)
            rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            img_np = (rendered_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            
            # ==========================================
            # 🌟 动态自适应填充 (防拉伸)
            # ==========================================
            browser_aspect = cam_state.aspect 
            render_aspect = render_w / render_h
            
            if browser_aspect > render_aspect:
                canvas_H = render_h
                canvas_W = int(render_h * browser_aspect)
            else:
                canvas_W = render_w
                canvas_H = int(render_w / browser_aspect)
                
            canvas = np.full((canvas_H, canvas_W, 3), 50, dtype=np.uint8) 
            start_y = (canvas_H - render_h) // 2
            start_x = (canvas_W - render_w) // 2
            canvas[start_y:start_y+render_h, start_x:start_x+render_w] = img_np
            
            client.scene.set_background_image(canvas, format="jpeg")
            
        time.sleep(0.02) # 约 50 FPS 的播放帧率控制

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