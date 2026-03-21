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
    """动态自适应虚拟相机"""
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
        self.fid = 0
        self.time = 0.0
        self.timestamp = 0.0

@torch.no_grad()
def main(dataset: ModelParams, pipe: PipelineParams, args):
    print(f"[引擎] 正在初始化 4DGS 自适应漫游引擎...")
    
    # 1. 初始化模型 (深灰色背景)
    bg_color = [1, 1, 1] if dataset.white_background else [0.2, 0.2, 0.2]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    sh_degree = dataset.sh_degree if hasattr(dataset, 'sh_degree') else 3
    gaussians = GaussianModel(sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, 
                              rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    
    # 2. 读取场景并覆盖权重
    scene = Scene(dataset, gaussians, shuffle=False)
    train_cameras = [c[1] if isinstance(c, tuple) else c for c in scene.getTrainCameras()]
    max_frames = len(train_cameras) - 1
    
    checkpoint = args.start_checkpoint
    print(f"[引擎] 正在加载模型权重: {checkpoint}")
    (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
    gaussians.restore(model_params, None)
    
    # 3. 启动 Viser
    server = viser.ViserServer(port=8080)
    print("\n" + "="*50)
    print(f"🚀 毕设专用漫游引擎启动成功！")
    print(f"👉 请在浏览器打开: http://localhost:8080")
    print("="*50 + "\n")
    
    # 记录出生点，用于防迷路重置
    c0 = train_cameras[0]
    w2c_init = c0.world_view_transform.transpose(0, 1).cpu().numpy()
    c2w_init = np.linalg.inv(w2c_init)
    c2w_init[:3, 1:3] *= -1 
    init_pos = c2w_init[:3, 3]
    init_wxyz = tf.SO3.from_matrix(c2w_init[:3, :3]).wxyz

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = init_pos
        client.camera.wxyz = init_wxyz
        client.camera.look_at = (0.0, 0.0, 0.0) 
        client.scene.add_grid("grid", visible=False)

    # ==========================================
    # 🌟 毕设专属 UI 控制台
    # ==========================================
    gui_time_idx = server.gui.add_slider("🎬 4D 时间轴", min=0, max=max_frames, step=1, initial_value=0)
    gui_res_scale = server.gui.add_slider("🖥️ 渲染画质 (卡顿请降低)", min=0.1, max=1.0, step=0.1, initial_value=0.8)
    gui_reset_btn = server.gui.add_button("🔄 一键重置视角 (防迷路)")
    server.gui.add_markdown("🕹️ **操作**: 左键旋转舞台 | 右键平移 | 滚轮缩放")
    
    @gui_reset_btn.on_click
    def _(_):
        for client in server.get_clients().values():
            client.camera.position = init_pos
            client.camera.wxyz = init_wxyz
            client.camera.look_at = (0.0, 0.0, 0.0)

    # 基础分辨率基准 (可动态缩放)
    BASE_RES = 1000

    while True:
        clients = server.get_clients()
        for client_id, client in clients.items():
            cam_state = client.camera
            
            # 🌟 核心突破：完美自适应浏览器比例！
            aspect = cam_state.aspect
            scale = gui_res_scale.value
            if aspect > 1.0:
                W = int(BASE_RES * scale)
                H = int((BASE_RES / aspect) * scale)
            else:
                H = int(BASE_RES * scale)
                W = int((BASE_RES * aspect) * scale)
                
            # 动态接管光学 FOV
            fovy = cam_state.fov
            fovx = 2 * math.atan(math.tan(fovy / 2) * aspect)
            
            # 坐标转换
            c2w = np.eye(4)
            c2w[:3, :3] = tf.SO3(cam_state.wxyz).as_matrix()
            c2w[:3, 3] = cam_state.position
            c2w[:3, 1:3] *= -1 
            w2c = np.linalg.inv(c2w)
            
            R = w2c[:3, :3].T 
            T = w2c[:3, 3]
            
            world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0), dtype=torch.float32).transpose(0, 1).cuda()
            projection_matrix = getProjectionMatrix(znear=0.01, zfar=1000.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            
            view_cam = MiniCam(W, H, fovy, fovx, 0.01, 1000.0, world_view_transform, full_proj_transform)
            
            current_time_idx = int(gui_time_idx.value)
            view_cam.fid = current_time_idx
            view_cam.time = float(current_time_idx / max(1, max_frames))
            view_cam.timestamp = view_cam.time
            
            # 执行渲染
            render_pkg = render(view_cam, gaussians, pipe, background)
            rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            
            # 直接发送原生比例的图片，拒绝黑边，拒绝拉伸！
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