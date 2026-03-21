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

# ==========================================
# 🌟 代理相机类 (保护原版 4D 属性不被污染)
# ==========================================
class ProxyCam:
    def __init__(self, base_cam):
        self.base = base_cam
        self.world_view_transform = base_cam.world_view_transform.clone()
        self.projection_matrix = base_cam.projection_matrix.clone()
        self.full_proj_transform = base_cam.full_proj_transform.clone()
        self.camera_center = base_cam.camera_center.clone()
        self.image_width = base_cam.image_width
        self.image_height = base_cam.image_height
        self.FoVx = base_cam.FoVx
        self.FoVy = base_cam.FoVy
        self.fid = getattr(base_cam, 'fid', 0)
        self.time = getattr(base_cam, 'time', 0.0)
        self.timestamp = getattr(base_cam, 'timestamp', 0.0)

# ==========================================
# 🌟 核心数学工具：安全矩阵转换与插值
# ==========================================
def get_c2w_from_cam(cam):
    """从 3DGS 相机安全提取 C2W (OpenCV 坐标系)"""
    w2c = cam.world_view_transform.transpose(0, 1).cpu().numpy()
    return np.linalg.inv(w2c)

def slerp_c2w(c2w_1, c2w_2, alpha):
    """SVD 奇异值分解插值：绝对平滑，绝对不炸刺！"""
    c2w = np.eye(4)
    # 1. 线性插值位置
    c2w[:3, 3] = (1 - alpha) * c2w_1[:3, 3] + alpha * c2w_2[:3, 3]
    # 2. 混合旋转矩阵
    R_mix = (1 - alpha) * c2w_1[:3, :3] + alpha * c2w_2[:3, :3]
    # 3. SVD 正交化（消除畸变）
    U, _, Vt = np.linalg.svd(R_mix)
    c2w[:3, :3] = U @ Vt
    return c2w

class PlayState:
    is_playing = False

play_state = PlayState()

@torch.no_grad()
def main(dataset: ModelParams, pipe: PipelineParams, args):
    print(f"[引擎] 正在初始化 4DGS 导演级引擎...")
    bg_color = [1, 1, 1] if dataset.white_background else [0.15, 0.15, 0.15]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    sh_degree = dataset.sh_degree if hasattr(dataset, 'sh_degree') else 3
    gaussians = GaussianModel(sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration,
                              rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)

    # 读取训练相机
    scene = Scene(dataset, gaussians, shuffle=False)
    train_cameras = [c[1] if isinstance(c, tuple) else c for c in scene.getTrainCameras()]
    max_frames = len(train_cameras) - 1
    native_W = train_cameras[0].image_width
    native_H = train_cameras[0].image_height

    checkpoint = args.start_checkpoint
    print(f"[引擎] 正在加载模型权重...")
    (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
    gaussians.restore(model_params, None)

    server = viser.ViserServer(port=8080)
    print("\n" + "="*50)
    print(f"🚀 毕设终极引擎启动成功！请打开: http://localhost:8080")
    print("="*50 + "\n")

    # ==========================================
    # 🌟 修复倒立：计算绝对准确的初始物理参数
    # ==========================================
    c2w_opencv_0 = get_c2w_from_cam(train_cameras[0])
    c2w_opengl_0 = c2w_opencv_0.copy()
    c2w_opengl_0[:, 1:3] *= -1 # OpenCV 转 OpenGL

    init_pos = c2w_opengl_0[:3, 3]
    init_rot = tf.SO3.from_matrix(c2w_opengl_0[:3, :3]).wxyz
    init_up = c2w_opengl_0[:3, 1] # 提取真实的 Y 轴作为上方向
    init_look = init_pos - c2w_opengl_0[:3, 2] * 2.0 # 向前看

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.scene.add_grid("grid", visible=False)
        client.camera.position = init_pos
        client.camera.wxyz = init_rot
        client.camera.up_direction = init_up
        client.camera.look_at = init_look

    # ==========================================
    # 🎬 UI 控制台
    # ==========================================
    with server.gui.add_folder("🎥 漫游模式控制"):
        gui_mode = server.gui.add_dropdown(
            "模式选择", 
            ("🎬 导播轨道 (推荐: 平滑且画质最高)", "🕹️ 自由漫游 (修复倒立版)"), 
            initial_value="🎬 导播轨道 (推荐: 平滑且画质最高)"
        )
        # 注意这是 float 滑块，支持平滑插值过渡！
        gui_cam_track = server.gui.add_slider("导播机位轨道", min=0.0, max=float(max_frames), step=0.01, initial_value=0.0)
        gui_sync_btn = server.gui.add_button("🔄 将自由漫游视角重置回导播轨")

    with server.gui.add_folder("🎬 4D 动画与画质"):
        gui_time_idx = server.gui.add_slider("4D 时间轴", min=0, max=max_frames, step=1, initial_value=0)
        with server.gui.add_folder("自动播放面板", expand_by_default=True):
            gui_play = server.gui.add_button("▶️ 自动播放 / 暂停")
        gui_res_scale = server.gui.add_slider("🖥️ 分辨率缩放 (1.0为原生画质)", min=0.2, max=1.0, step=0.1, initial_value=1.0)

    @gui_play.on_click
    def _(_):
        play_state.is_playing = not play_state.is_playing
        gui_play.name = "⏸️ 暂停" if play_state.is_playing else "▶️ 自动播放"

    @gui_sync_btn.on_click
    def _(_):
        for client in server.get_clients().values():
            client.camera.position = init_pos
            client.camera.wxyz = init_rot
            client.camera.up_direction = init_up
            client.camera.look_at = init_look

    while True:
        # 自动播放时间推进
        if play_state.is_playing:
            gui_time_idx.value = (gui_time_idx.value + 1) % (max_frames + 1)

        clients = server.get_clients()
        for client_id, client in clients.items():
            cam_state = client.camera
            
            # --- 分辨率计算 ---
            W_render = int(native_W * gui_res_scale.value)
            H_render = int(native_H * gui_res_scale.value)
            
            view_cam = ProxyCam(train_cameras[0])
            view_cam.image_width = W_render
            view_cam.image_height = H_render

            if gui_mode.value == "🎬 导播轨道 (推荐: 平滑且画质最高)":
                # ==========================================
                # 🌟 SVD 平滑插值模式 (画质绝杀，绝对不炸刺)
                # ==========================================
                t = gui_cam_track.value
                idx1 = int(math.floor(t))
                idx2 = min(int(math.ceil(t)), max_frames)
                alpha = t - idx1
                
                c2w_1 = get_c2w_from_cam(train_cameras[idx1])
                c2w_2 = get_c2w_from_cam(train_cameras[idx2])
                c2w_opencv = slerp_c2w(c2w_1, c2w_2, alpha)
                
                # 转换回 3DGS 矩阵
                w2c_opencv = np.linalg.inv(c2w_opencv)
                R = w2c_opencv[:3, :3].T
                T = w2c_opencv[:3, 3]
                
                # 强行使用训练集原版 FOV (不拉伸)
                fovy = train_cameras[idx1].FoVy
                fovx = train_cameras[idx1].FoVx
                
            else:
                # ==========================================
                # 🌟 修复版自由漫游模式 (操作符合直觉)
                # ==========================================
                c2w_opengl = np.eye(4)
                c2w_opengl[:3, :3] = tf.SO3(cam_state.wxyz).as_matrix()
                c2w_opengl[:3, 3] = cam_state.position
                
                # OpenGL 转 OpenCV
                c2w_opencv = c2w_opengl.copy()
                c2w_opencv[:, 1:3] *= -1 
                
                w2c_opencv = np.linalg.inv(c2w_opencv)
                R = w2c_opencv[:3, :3].T
                T = w2c_opencv[:3, 3]
                
                # 使用浏览器当前的 FOV，但锁定渲染比例
                fovy = cam_state.fov
                fovx = 2 * math.atan(math.tan(fovy / 2) * (W_render / float(H_render)))

            # 构建底层 Torch 矩阵
            wvt = torch.tensor(getWorld2View2(R, T, np.array([0.,0.,0.]), 1.0), dtype=torch.float32).transpose(0, 1).cuda()
            proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
            
            view_cam.FoVx = fovx
            view_cam.FoVy = fovy
            view_cam.world_view_transform = wvt
            view_cam.projection_matrix = proj
            view_cam.full_proj_transform = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
            view_cam.camera_center = wvt.inverse()[3, :3]
            
            # 注入时间
            current_time = int(gui_time_idx.value)
            view_cam.fid = current_time
            view_cam.time = float(current_time / max(1, max_frames))
            view_cam.timestamp = view_cam.time

            # 执行渲染！
            render_pkg = render(view_cam, gaussians, pipe, background)
            rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            img_np = (rendered_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            # ==========================================
            # 🌟 智能填缝算法 (拒绝拉伸，保持离线级画质)
            # ==========================================
            browser_aspect = cam_state.aspect
            render_aspect = W_render / float(H_render)
            
            if browser_aspect > render_aspect:
                canvas_H = H_render
                canvas_W = int(H_render * browser_aspect)
            else:
                canvas_W = W_render
                canvas_H = int(W_render / browser_aspect)
                
            canvas = np.full((canvas_H, canvas_W, 3), 40, dtype=np.uint8)
            start_y = (canvas_H - H_render) // 2
            start_x = (canvas_W - W_render) // 2
            canvas[start_y:start_y+H_render, start_x:start_x+W_render] = img_np
            
            client.scene.set_background_image(canvas, format="jpeg")

        time.sleep(0.02)

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