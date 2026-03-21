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

def interpolate_camera(cam1, cam2, t):
    """摄像机轨道插值算法：保证平滑过渡"""
    cam = copy.deepcopy(cam1)
    cam.camera_center = cam1.camera_center * (1 - t) + cam2.camera_center * t
    cam.world_view_transform = cam1.world_view_transform * (1 - t) + cam2.world_view_transform * t
    cam.projection_matrix = cam1.projection_matrix * (1 - t) + cam2.projection_matrix * t
    cam.full_proj_transform = cam1.full_proj_transform * (1 - t) + cam2.full_proj_transform * t
    return cam

def apply_pan_and_zoom(cam, zoom_factor, pan_x, pan_y):
    """焦距缩放与镜头平移算法（绝对安全，不改变世界坐标系尺度）"""
    # 1. 缩放 (改变 FOV 视野角)
    cam.FoVy = cam.FoVy / zoom_factor
    cam.FoVx = cam.FoVx / zoom_factor
    
    # 2. 平移 (在相机的局部坐标系内移动)
    # 获取 W2C 矩阵并转回 Numpy
    W2C_T = cam.world_view_transform.cpu().numpy() 
    W2C = W2C_T.T
    R = W2C[:3, :3]
    T = W2C[:3, 3]
    
    # 相机的局部 X(右) 和 Y(上) 轴
    local_right = R[0, :]
    local_up = R[1, :]
    
    # 在世界空间中计算偏移量 (这里可以调节灵敏度，0.5 是基准)
    shift_world = (pan_x * local_right * 0.5) + (pan_y * local_up * 0.5)
    cam.camera_center = cam.camera_center + torch.tensor(shift_world, dtype=torch.float32).cuda()
    
    # 更新平移矩阵
    T_new = T - (R @ shift_world)
    W2C[:3, 3] = T_new
    
    # 3. 重新计算所有投影矩阵并写回
    cam.world_view_transform = torch.tensor(W2C.T, dtype=torch.float32).cuda()
    cam.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=cam.FoVx, fovY=cam.FoVy).transpose(0, 1).cuda()
    cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
    
    return cam

@torch.no_grad()
def main(dataset: ModelParams, pipe: PipelineParams, args):
    print(f"[引擎] 正在初始化 4DGS 导播级漫游引擎...")
    
    # 初始化模型 (灰色背景填缝)
    bg_color = [1, 1, 1] if dataset.white_background else [0.2, 0.2, 0.2]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    sh_degree = dataset.sh_degree if hasattr(dataset, 'sh_degree') else 3
    gaussians = GaussianModel(sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, 
                              rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    
    # 读取原始完美相机轨迹
    scene = Scene(dataset, gaussians, shuffle=False)
    train_cameras = [c[1] if isinstance(c, tuple) else c for c in scene.getTrainCameras()]
    max_frames = len(train_cameras) - 1
    
    # 加载权重
    checkpoint = args.start_checkpoint
    print(f"[引擎] 正在加载模型权重: {checkpoint}")
    (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
    gaussians.restore(model_params, None)
    
    # 启动 Viser 界面
    server = viser.ViserServer(port=8080)
    print("\n" + "="*50)
    print(f"🚀 毕设专属导播台启动成功！请打开: http://localhost:8080")
    print("="*50 + "\n")
    
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.scene.add_grid("grid", visible=False)
        # 彻底锁定前端相机的输入，防止鼠标乱划导致灰屏！
        client.camera.up_direction = (0, 0, 1)

    # ==========================================
    # 🌟 导播台控制面板 (纯 UI 控制，绝对安全)
    # ==========================================
    with server.gui.add_folder("🎥 漫游与运镜控制"):
        # 注意：这里是 float 滑块！拖动它可以实现极其平滑的运镜插值
        gui_cam_pos = server.gui.add_slider("🎥 机位轨道 (拖动漫游)", min=0.0, max=float(max_frames), step=0.01, initial_value=0.0)
        gui_zoom = server.gui.add_slider("🔍 镜头拉近/推远", min=0.5, max=3.0, step=0.05, initial_value=1.0)
        gui_pan_x = server.gui.add_slider("↔️ 镜头水平微调", min=-2.0, max=2.0, step=0.05, initial_value=0.0)
        gui_pan_y = server.gui.add_slider("↕️ 镜头垂直微调", min=-2.0, max=2.0, step=0.05, initial_value=0.0)
        gui_reset_cam = server.gui.add_button("🔄 视角归位")

    with server.gui.add_folder("🎬 4D 动画与画质"):
        gui_time_idx = server.gui.add_slider("🎬 4D 时间轴", min=0, max=max_frames, step=1, initial_value=0)
        gui_res_scale = server.gui.add_slider("🖥️ 渲染画质 (卡顿请降低)", min=0.1, max=1.0, step=0.1, initial_value=0.8)
    
    server.gui.add_markdown("ℹ️ **提示**: 鼠标已被锁定防止飞出边界，请使用**上方滑块**进行完美的平滑运镜漫游！")

    @gui_reset_cam.on_click
    def _(_):
        gui_zoom.value = 1.0
        gui_pan_x.value = 0.0
        gui_pan_y.value = 0.0

    while True:
        clients = server.get_clients()
        for client_id, client in clients.items():
            cam_state = client.camera
            
            # 1. 核心突破：获取当前机位并进行轨道插值
            cam_val = gui_cam_pos.value
            idx1 = int(math.floor(cam_val))
            idx2 = min(int(math.ceil(cam_val)), max_frames)
            t = cam_val - idx1
            
            # 生成平滑过渡的虚拟相机！
            view_cam = interpolate_camera(train_cameras[idx1], train_cameras[idx2], t)
            
            # 2. 注入平移与缩放 (满足毕设微调需求)
            view_cam = apply_pan_and_zoom(view_cam, gui_zoom.value, gui_pan_x.value, gui_pan_y.value)
            
            # 3. 注入 4D 动画时间属性
            current_time_idx = int(gui_time_idx.value)
            view_cam.fid = current_time_idx
            view_cam.time = float(current_time_idx / max(1, max_frames))
            if hasattr(view_cam, 'timestamp'):
                view_cam.timestamp = view_cam.time
            
            # --- 渲染画质缩放 ---
            render_w = int(view_cam.image_width * gui_res_scale.value)
            render_h = int(view_cam.image_height * gui_res_scale.value)
            view_cam.image_width = render_w
            view_cam.image_height = render_h
            
            # --- 执行渲染 ---
            render_pkg = render(view_cam, gaussians, pipe, background)
            rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            img_np = (rendered_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            
            # ==========================================
            # 🌟 动态自适应填充 (保留原始比例，周围填灰色)
            # ==========================================
            browser_aspect = cam_state.aspect 
            render_aspect = render_w / render_h
            
            # 根据浏览器宽高比，动态计算黑色/灰色画布的大小
            if browser_aspect > render_aspect:
                canvas_H = render_h
                canvas_W = int(render_h * browser_aspect)
            else:
                canvas_W = render_w
                canvas_H = int(render_w / browser_aspect)
                
            # 创建填缝背景 (灰色)
            canvas = np.full((canvas_H, canvas_W, 3), 50, dtype=np.uint8) # 50 是深灰色
            
            # 把渲染图完美贴在正中央
            start_y = (canvas_H - render_h) // 2
            start_x = (canvas_W - render_w) // 2
            canvas[start_y:start_y+render_h, start_x:start_x+render_w] = img_np
            
            # 发送给网页
            client.scene.set_background_image(canvas, format="jpeg")
            
        time.sleep(0.02) # 50fps 刷新

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