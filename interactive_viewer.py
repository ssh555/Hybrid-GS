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
from utils.graphics_utils import getProjectionMatrix

# ==============================
# ProxyCam
# ==============================
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
    def world_view_transform(self):
        return self.override_wvt if self.override_wvt is not None else self.base.world_view_transform

    @property
    def projection_matrix(self):
        return self.override_proj if self.override_proj is not None else self.base.projection_matrix

    @property
    def full_proj_transform(self):
        return self.override_full if self.override_full is not None else self.base.full_proj_transform

    @property
    def camera_center(self):
        return self.override_center if self.override_center is not None else self.base.camera_center

    @property
    def image_width(self):
        return self.override_w if self.override_w is not None else self.base.image_width

    @property
    def image_height(self):
        return self.override_h if self.override_h is not None else self.base.image_height

    @property
    def FoVx(self):
        return self.override_fovx if self.override_fovx is not None else self.base.FoVx

    @property
    def FoVy(self):
        return self.override_fovy if self.override_fovy is not None else self.base.FoVy


# ==============================
# Utils
# ==============================
def get_c2w(cam):
    w2c = cam.world_view_transform.transpose(0, 1).cpu().numpy()
    c2w = np.linalg.inv(w2c)
    c2w[:, 1:3] *= -1
    return c2w


# ==============================
# 主函数
# ==============================
@torch.no_grad()
def main(dataset: ModelParams, pipe: PipelineParams, args):

    background = torch.tensor(
        [1,1,1] if dataset.white_background else [0.2,0.2,0.2],
        dtype=torch.float32,
        device="cuda"
    )

    gaussians = GaussianModel(
        dataset.sh_degree,
        gaussian_dim=args.gaussian_dim,
        time_duration=args.time_duration,
        rot_4d=args.rot_4d,
        force_sh_3d=args.force_sh_3d
    )

    scene = Scene(dataset, gaussians, shuffle=False)
    train_cams = [c[1] if isinstance(c, tuple) else c for c in scene.getTrainCameras()]

    # ========= 解析多视角多帧 =========
    base_cam = train_cams[0]
    base_T = base_cam.T.cpu().numpy() if hasattr(base_cam.T, 'cpu') else base_cam.T
    base_R = base_cam.R.cpu().numpy() if hasattr(base_cam.R, 'cpu') else base_cam.R

    max_frames = 0
    for cam in train_cams:
        cam_T = cam.T.cpu().numpy() if hasattr(cam.T, 'cpu') else cam.T
        cam_R = cam.R.cpu().numpy() if hasattr(cam.R, 'cpu') else cam.R
        if np.allclose(base_T, cam_T, atol=1e-5) and np.allclose(base_R, cam_R, atol=1e-5):
            max_frames += 1
        else:
            break

    view_cams = []
    for i in range(0, len(train_cams), max_frames):
        view_cams.append(train_cams[i:i+max_frames])
    max_cams = len(view_cams)

    print(f"[渲染器] {max_cams} 个视角 × {max_frames} 帧")

    # ========= 加载模型 =========
    model_params, _ = torch.load(args.start_checkpoint, weights_only=False)
    gaussians.restore(model_params, None)

    server = viser.ViserServer(port=8080)

    # ================= UI =================
    with server.gui.add_folder("🎬 导播台"):
        gui_free_roam = server.gui.add_checkbox("自由漫游", False)
        gui_sync_cam = server.gui.add_button("重置视角")

        cam_id = server.gui.add_slider("机位", 0, max_cams-1, 1, 0)

        btn_play = server.gui.add_button("▶")
        btn_pause = server.gui.add_button("⏸")

        slider_frame = server.gui.add_slider("帧", 0, max_frames-1, 1, 0)
        slider_speed = server.gui.add_slider("速度", 0.25, 2.0, 0.05, 1.0)
        gui_res_scale = server.gui.add_slider("分辨率倍率", 0.5, 2.0, 0.1, 1.0)

    play_state = {"playing": False}

    @btn_play.on_click
    def _(_): play_state["playing"] = True

    @btn_pause.on_click
    def _(_): play_state["playing"] = False

    @gui_sync_cam.on_click
    def _(_): gui_free_roam.value = False

    last_time = time.time()
    FPS = 30.0

    # ================= 主循环 =================
    while True:
        now = time.time()
        dt = now - last_time
        last_time = now

        if play_state["playing"]:
            slider_frame.value += FPS * dt * slider_speed.value

        slider_frame.value %= max_frames

        frame_idx = int(slider_frame.value)
        selected_cam = view_cams[cam_id.value][frame_idx]

        for client in server.get_clients().values():

            scale = gui_res_scale.value
            render_w = int(selected_cam.image_width * scale)
            render_h = int(selected_cam.image_height * scale)

            view_cam = ProxyCam(selected_cam)
            view_cam.override_w = render_w
            view_cam.override_h = render_h

            # ================= 导播模式 =================
            if not gui_free_roam.value:
                c2w = get_c2w(selected_cam)
                client.camera.position = c2w[:3, 3]
                client.camera.wxyz = tf.SO3.from_matrix(c2w[:3, :3]).wxyz

            # ================= 自由漫游 =================
            else:
                cam = client.camera

                # ✅ float32
                c2w_gl = np.eye(4, dtype=np.float32)
                c2w_gl[:3, :3] = tf.SO3(cam.wxyz).as_matrix().astype(np.float32)
                c2w_gl[:3, 3] = np.array(cam.position, dtype=np.float32)

                # GL → CV（只一次）
                c2w_cv = c2w_gl.copy()
                c2w_cv[:, 1:3] *= -1

                w2c = np.linalg.inv(c2w_cv)

                wvt = torch.tensor(w2c, dtype=torch.float32, device="cuda").transpose(0,1)

                proj = getProjectionMatrix(
                    znear=0.01,
                    zfar=100.0,
                    fovX=selected_cam.FoVx,
                    fovY=selected_cam.FoVy
                ).transpose(0,1).cuda()

                view_cam.override_wvt = wvt
                view_cam.override_proj = proj
                view_cam.override_full = wvt @ proj
                view_cam.override_center = torch.tensor(c2w_cv[:3,3], dtype=torch.float32, device="cuda")
                view_cam.override_fovx = selected_cam.FoVx
                view_cam.override_fovy = selected_cam.FoVy

            # ================= 渲染 =================
            out = render(view_cam, gaussians, pipe, background)

            img = torch.clamp(out["render"], 0, 1)
            img = (img.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)

            # ================= 防拉伸 =================
            aspect_browser = client.camera.aspect
            aspect_render = render_w / render_h

            if aspect_browser > aspect_render:
                H = render_h
                W = int(H * aspect_browser)
            else:
                W = render_w
                H = int(W / aspect_browser)

            canvas = np.zeros((H, W, 3), dtype=np.uint8)

            y0 = (H - render_h)//2
            x0 = (W - render_w)//2

            canvas[y0:y0+render_h, x0:x0+render_w] = img

            client.scene.set_background_image(canvas, format="png")

        time.sleep(0.01)


# ================= 启动 =================
if __name__ == "__main__":
    parser = ArgumentParser()

    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--config", required=True)
    parser.add_argument("--start_checkpoint", type=str)

    parser.add_argument("--gaussian_dim", type=int, default=4)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5,0.5])
    parser.add_argument("--rot_4d", action="store_true", default=True)
    parser.add_argument("--force_sh_3d", action="store_true", default=True)

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    def merge(k, host):
        if isinstance(host[k], DictConfig):
            for kk in host[k]:
                merge(kk, host[k])
        elif hasattr(args, k):
            setattr(args, k, host[k])

    for k in cfg:
        merge(k, cfg)

    main(lp.extract(args), pp.extract(args), args)