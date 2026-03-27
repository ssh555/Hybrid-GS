import math
import time
import torch
import viser
import argparse
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import sys
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import random
from utils.general_utils import safe_state, knn

# 🚀 降维打击：直接调用你本地已经编译好的、用来训练的 4DGS 渲染器！
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel

class MiniCam:
    """把网页端的 Viser 相机，完美伪装成 3DGS 训练时的视角相机"""
    def __init__(self, cam: viser.CameraHandle, W, H, frame_id):
        self.image_width = W
        self.image_height = H
        self.FoVy = cam.fov
        self.FoVx = 2 * math.atan(math.tan(self.FoVy / 2) * (W / H))
        self.znear = 0.01
        self.zfar = 100.0
        
        # 兼容 4DGS 的时间戳变量名 (防止你的底层代码找不着)
        self.fid = frame_id        
        self.timestamp = frame_id  
        
        # 提取相机外参 (Viser OpenGL -> 3DGS OpenCV)
        try:
            c2w = torch.tensor(cam.c2w, dtype=torch.float32, device="cuda")
        except AttributeError:
            w, x, y, z = cam.wxyz
            pos = cam.position
            R = torch.tensor([
                [1 - 2*(y**2 + z**2), 2*(x*y - w*z),   2*(x*z + w*y)],
                [2*(x*y + w*z),       1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
                [2*(x*z - w*y),       2*(y*z + w*x),   1 - 2*(x**2 + y**2)]
            ], dtype=torch.float32, device="cuda")
            T = torch.tensor(pos, dtype=torch.float32, device="cuda")
            c2w = torch.eye(4, dtype=torch.float32, device="cuda")
            c2w[:3, :3] = R
            c2w[:3, 3] = T
            
        c2w[:, 1:3] *= -1
        w2c = torch.linalg.inv(c2w)
        
        self.world_view_transform = w2c.transpose(0, 1).cuda()
        self.camera_center = c2w[:3, 3].cuda()
        
        # 计算投影矩阵
        P = torch.zeros((4, 4), device="cuda")
        P[0, 0] = 1.0 / math.tan(self.FoVx / 2.0)
        P[1, 1] = 1.0 / math.tan(self.FoVy / 2.0)
        P[2, 2] = self.zfar / (self.zfar - self.znear)
        P[2, 3] = -(self.zfar * self.znear) / (self.zfar - self.znear)
        P[3, 2] = 1.0
        self.projection_matrix = P.transpose(0, 1)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="混合3D与4D高斯场景重建统一训练框架")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument("--config", type=str)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[6_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[6_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = "无效参数，但是删除会影响其他地方的参数解析，暂时保留")
    
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument('--num_pts', type=int, default=100_000)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    parser.add_argument("--val", action="store_true", default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
        
    cfg = OmegaConf.load(args.config)
    
    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])
    
    for k in cfg.keys():
        recursive_merge(k, cfg)
        
    if args.exhaust_test:
        args.test_iterations = args.test_iterations + [i for i in range(0,args.iterations,500)]

    # 系统状态初始化
    setup_seed(args.seed)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    pipe = pp.extract(args)

    gaussians = GaussianModel(sh_degree=cfg.ModelParams.sh_degree, gaussian_dim=cfg.gaussian_dim)
    model_data, _ = torch.load(cfg.start_checkpoint, weights_only=False)
    gaussians.restore(model_data, op.extract(args))

    server = viser.ViserServer(port=args.port)
    
    with server.gui.add_folder("4D 控制面板 (Native)"):
        # ⚠️ SWinGS 必须用真实的帧号(Frame)驱动，不能用 0~1 的小数！
        gui_frame = server.gui.add_slider("播放进度 (帧)", min=0, max=300, step=1, initial_value=0)
        gui_playing = server.gui.add_checkbox("自动播放", initial_value=False)
        gui_res = server.gui.add_slider("分辨率高度", min=240, max=1080, step=120, initial_value=720)
        gui_bg = server.gui.add_rgb("背景颜色", initial_value=(0, 0, 0))
        gui_fps = server.gui.add_markdown("**FPS: --**")
        
    need_render = True
    @gui_frame.on_update
    def _(_): nonlocal need_render; need_render = True
    @gui_res.on_update
    def _(_): nonlocal need_render; need_render = True
    @gui_bg.on_update
    def _(_): nonlocal need_render; need_render = True
    
    @server.on_client_connect
    def on_connect(client: viser.ClientHandle):
        @client.camera.on_update
        def on_cam(cam: viser.CameraHandle):
            nonlocal need_render; need_render = True

    while True:
        if gui_playing.value:
            gui_frame.value = (gui_frame.value + 1) % 301
            need_render = True

        if need_render:
            clients = server.get_clients()
            if not clients:
                time.sleep(0.05)
                continue
                
            start_t = time.time()
            try:
                with torch.no_grad():
                    for client_id, client in clients.items():
                        cam = client.camera
                        try:
                            W, H = cam.resolution
                        except AttributeError:
                            H = gui_res.value
                            W = int(H * getattr(cam, 'aspect', 16/9))
                            
                        frame_id = gui_frame.value
                        viewpoint = MiniCam(cam, W, H, frame_id)
                        
                        # 兼容某些使用全局 _t 的 4DGS 底层
                        if hasattr(gaussians, '_t') and gaussians._t is not None:
                            gaussians._t.data.fill_(frame_id / 300.0)
                        
                        # ========================================================
                        # 完美复现你在 trainer_hybrid.py 里的残影过滤魔法！
                        # ========================================================
                        active_mask = None
                        if hasattr(gaussians, '_start_frame') and gaussians._start_frame.numel() > 0:
                            active_mask = (gaussians._start_frame <= frame_id) & (gaussians._expire_frame >= frame_id)
                            if hasattr(gaussians, '_mask_dynamic'):
                                active_mask = (gaussians._mask_dynamic == 1) | ((gaussians._mask_dynamic != 1) & active_mask)
                                
                        bg_color = torch.tensor(gui_bg.value, dtype=torch.float32, device="cuda") / 255.0
                        
                        # 🚀 直接调用原生渲染！你训练用的什么，这里就画出什么！
                        render_pkg = render(viewpoint, gaussians, pipe, bg_color, active_dynamic_mask=active_mask)
                        
                        image = render_pkg["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                        
                        try:
                            client.scene.set_background_image(image, format="jpeg")
                        except AttributeError:
                            client.camera.set_background_image(image, format="jpeg")
                            
                    end_t = time.time()
                    gui_fps.content = f"**FPS: {1.0 / (end_t - start_t + 1e-5):.1f}**"
            except Exception as e:
                print(f"渲染警告: {e}")
            need_render = False
        time.sleep(0.01)

if __name__ == "__main__":
    main()