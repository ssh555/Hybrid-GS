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

# ==========================================
# 🌟 代理相机 (保证不丢失任何原版 4D 属性)
# ==========================================
class ProxyCam:
    def __init__(self, base_cam):
        self.base = base_cam
        self.override_wvt = None
        self.override_proj = None
        self.override_full = None
        self.override_center = None
        self.override_w = None
        self.override_h = None
        self.override_fovx = None
        self.override_fovy = None

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def world_view_transform(self): return self.override_wvt if self.override_wvt is not None else self.base.world_view_transform
    @property
    def projection_matrix(self): return self.override_proj if self.override_proj is not None else self.base.projection_matrix
    @property
    def full_proj_transform(self): return self.override_full if self.override_full is not None else self.base.full_proj_transform
    @property
    def camera_center(self): return self.override_center if self.override_center is not None else self.base.camera_center
    @property
    def image_width(self): return self.override_w if self.override_w is not None else self.base.image_width
    @property
    def image_height(self): return self.override_h if self.override_h is not None else self.base.image_height
    @property
    def FoVx(self): return self.override_fovx if self.override_fovx is not None else self.base.FoVx
    @property
    def FoVy(self): return self.override_fovy if self.override_fovy is not None else self.base.FoVy

def get_c2w_from_colmap(cam):
    w2c = cam.world_view_transform.transpose(0, 1).cpu().numpy()
    c2w_opencv = np.linalg.inv(w2c)
    c2w_opengl = c2w_opencv.copy()
    c2w_opengl[:, 1:3] *= -1
    return c2w_opengl

@torch.no_grad()
def main(dataset: ModelParams, pipe: PipelineParams, args):
    print(f"[引擎] 正在初始化 4DGS 离线级播放引擎...")
    bg_color = [1, 1, 1] if dataset.white_background else [0.2, 0.2, 0.2]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    sh_degree = dataset.sh_degree if hasattr(dataset, 'sh_degree') else 3
    gaussians = GaussianModel(sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration,
                              rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)

    # 读取绝对完美的原版相机
    scene = Scene(dataset, gaussians, shuffle=False)
    train_cameras = [c[1] if isinstance(c, tuple) else c for c in scene.getTrainCameras()]
    max_frames = len(train_cameras) - 1

    checkpoint = args.start_checkpoint
    (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
    gaussians.restore(model_params, None)

    server = viser.ViserServer(port=8080)
    print("\n" + "="*50)
    print(f"🚀 毕设专属双模播放器启动成功！请打开: http://localhost:8080")
    print("="*50 + "\n")

    # ==========================================
    # 🎬 UI 控制台
    # ==========================================
    with server.gui.add_folder("🎬 播放控制台"):
        gui_mode = server.gui.add_dropdown(
            "漫游模式", 
            ("原画轨迹播放 (100%安全高清)", "自由漫游 (警告:靠太近会炸刺)"), 
            initial_value="原画轨迹播放 (100%安全高清)"
        )
        gui_play = server.gui.add_checkbox("▶️ 自动播放/暂停", initial_value=False)
        gui_frame = server.gui.add_slider("时间与机位进度", min=0, max=max_frames, step=1, initial_value=0)
        
    server.gui.add_markdown("ℹ️ **答辩建议**：使用【原画轨迹】展示完美画质；如需漫游，请切换模式，若出现乱码刺，请疯狂往后滚鼠标拉远距离！")

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.scene.add_grid("grid", visible=False)
        c2w = get_c2w_from_colmap(train_cameras[0])
        client.camera.position = c2w[:3, 3]
        client.camera.wxyz = tf.SO3.from_matrix(c2w[:3, :3]).wxyz

    while True:
        clients = server.get_clients()

        # 播放逻辑
        if gui_play.value:
            gui_frame.value = (gui_frame.value + 1) % (max_frames + 1)

        current_idx = int(gui_frame.value)
        base_cam = train_cameras[current_idx]

        for client_id, client in clients.items():
            if gui_mode.value == "原画轨迹播放 (100%安全高清)":
                # 🌟 核心真理：不做任何修改，直接把原版相机塞进渲染器！和离线渲染一模一样！
                view_cam = base_cam
                
                # 让网页的相机视角自动跟随原始轨迹移动
                c2w = get_c2w_from_colmap(base_cam)
                client.camera.position = c2w[:3, 3]
                client.camera.wxyz = tf.SO3.from_matrix(c2w[:3, :3]).wxyz
            
            else:
                # 🌟 自由漫游模式：允许你接管相机
                cam_state = client.camera
                aspect = cam_state.aspect
                fovy = cam_state.fov
                fovx = 2 * math.atan(math.tan(fovy / 2) * aspect)

                c2w_opengl = np.eye(4)
                c2w_opengl[:3, :3] = tf.SO3(cam_state.wxyz).as_matrix()
                c2w_opengl[:3, 3] = cam_state.position
                c2w_opencv = c2w_opengl.copy()
                c2w_opencv[:, 1:3] *= -1

                w2c = np.linalg.inv(c2w_opencv)
                R = w2c[:3, :3].T
                T = w2c[:3, 3]

                wvt = torch.tensor(getWorld2View2(R, T, np.array([0.,0.,0.]), 1.0), dtype=torch.float32).transpose(0, 1).cuda()
                proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
                full_proj = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
                cam_center = wvt.inverse()[3, :3]

                view_cam = ProxyCam(base_cam)
                view_cam.override_wvt = wvt
                view_cam.override_proj = proj
                view_cam.override_full = full_proj
                view_cam.override_center = cam_center
                view_cam.override_w = int(800 * aspect)
                view_cam.override_h = 800
                view_cam.override_fovx = fovx
                view_cam.override_fovy = fovy

            # 执行渲染！
            render_pkg = render(view_cam, gaussians, pipe, background)
            rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            img_np = (rendered_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            # 发送给网页
            client.scene.set_background_image(img_np, format="jpeg")

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