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


# ==============================
# Proxy Camera
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
    def world_view_transform(self): return self.override_wvt or self.base.world_view_transform
    @property
    def projection_matrix(self): return self.override_proj or self.base.projection_matrix
    @property
    def full_proj_transform(self): return self.override_full or self.base.full_proj_transform
    @property
    def camera_center(self): return self.override_center or self.base.camera_center
    @property
    def image_width(self): return self.override_w or self.base.image_width
    @property
    def image_height(self): return self.override_h or self.base.image_height
    @property
    def FoVx(self): return self.override_fovx or self.base.FoVx
    @property
    def FoVy(self): return self.override_fovy or self.base.FoVy


# ==============================
# Utils
# ==============================
def get_c2w(cam):
    w2c = cam.world_view_transform.transpose(0, 1).cpu().numpy()
    c2w = np.linalg.inv(w2c)
    c2w[:, 1:3] *= -1
    return c2w


def lerp(a, b, t):
    return a * (1 - t) + b * t


# ==============================
# Main
# ==============================
@torch.no_grad()
def main(dataset: ModelParams, pipe: PipelineParams, args):

    bg_color = [1,1,1] if dataset.white_background else [0.2,0.2,0.2]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(
        dataset.sh_degree,
        gaussian_dim=args.gaussian_dim,
        time_duration=args.time_duration,
        rot_4d=args.rot_4d,
        force_sh_3d=args.force_sh_3d
    )

    scene = Scene(dataset, gaussians, shuffle=False)
    train_cams = [c[1] if isinstance(c, tuple) else c for c in scene.getTrainCameras()]
    max_frames = len(train_cams) - 1

    model_params, _ = torch.load(args.start_checkpoint, weights_only=False)
    gaussians.restore(model_params, None)

    # ===== 安全区域 =====
    centers = np.array([get_c2w(cam)[:3,3] for cam in train_cams])
    scene_center = centers.mean(axis=0)
    scene_radius = np.linalg.norm(centers - scene_center, axis=1).max()

    # ===== UI =====
    server = viser.ViserServer(port=8080)

    play_state = {"playing": False, "direction": 1}

    with server.gui.add_folder("🎬 播放器"):

        mode = server.gui.add_dropdown(
            "模式",
            ("原始轨迹", "插值播放", "自由漫游"),
            initial_value="原始轨迹"
        )

        btn_play = server.gui.add_button("▶️ 播放")
        btn_pause = server.gui.add_button("⏸ 暂停")

        btn_prev = server.gui.add_button("⏮ 上一帧")
        btn_next = server.gui.add_button("⏭ 下一帧")

        btn_backward = server.gui.add_button("⏪ 倒放")
        btn_forward = server.gui.add_button("⏩ 快进")

        slider_frame = server.gui.add_slider("进度", 0, max_frames, 0.01, 0)
        slider_speed = server.gui.add_slider("速度", 0.25, 2.0, 0.05, 1.0)

        loop_toggle = server.gui.add_checkbox("循环播放", True)
        txt = server.gui.add_text("帧信息", "0")

    # ===== 事件绑定 =====
    @btn_play.on_click
    def _(_):
        play_state["playing"] = True
        play_state["direction"] = 1

    @btn_pause.on_click
    def _(_):
        play_state["playing"] = False

    @btn_forward.on_click
    def _(_):
        play_state["playing"] = True
        play_state["direction"] = 1

    @btn_backward.on_click
    def _(_):
        play_state["playing"] = True
        play_state["direction"] = -1

    @btn_next.on_click
    def _(_):
        slider_frame.value = min(slider_frame.value + 1, max_frames)

    @btn_prev.on_click
    def _(_):
        slider_frame.value = max(slider_frame.value - 1, 0)

    # ===== 主循环 =====
    while True:

        # ===== 播放逻辑 =====
        if play_state["playing"]:
            slider_frame.value += play_state["direction"] * slider_speed.value

        if slider_frame.value > max_frames:
            if loop_toggle.value:
                slider_frame.value = 0
            else:
                slider_frame.value = max_frames
                play_state["playing"] = False

        if slider_frame.value < 0:
            if loop_toggle.value:
                slider_frame.value = max_frames
            else:
                slider_frame.value = 0
                play_state["playing"] = False

        txt.value = f"{int(slider_frame.value)} / {max_frames}"

        f_idx = int(slider_frame.value)
        t = slider_frame.value - f_idx

        cam_a = train_cams[f_idx]
        cam_b = train_cams[(f_idx+1) % len(train_cams)]

        for client in server.get_clients().values():

            # ========= 模式1 =========
            if mode.value == "原始轨迹":
                view_cam = cam_a
                c2w = get_c2w(cam_a)
                client.camera.position = c2w[:3,3]
                client.camera.wxyz = tf.SO3.from_matrix(c2w[:3,:3]).wxyz

            # ========= 模式2 =========
            elif mode.value == "插值播放":

                c2w_a = get_c2w(cam_a)
                c2w_b = get_c2w(cam_b)

                pos = lerp(c2w_a[:3,3], c2w_b[:3,3], t)

                R = tf.SO3.from_matrix(c2w_a[:3,:3]).slerp(
                    tf.SO3.from_matrix(c2w_b[:3,:3]), t)

                client.camera.position = pos
                client.camera.wxyz = R.wxyz

                view_cam = cam_a

            # ========= 模式3 =========
            else:
                cam_state = client.camera

                pos = cam_state.position
                offset = pos - scene_center
                dist = np.linalg.norm(offset)

                max_r = scene_radius * 1.2
                min_r = scene_radius * 0.3

                if dist > max_r:
                    pos = scene_center + offset/dist * max_r
                if dist < min_r:
                    pos = scene_center + offset/dist * min_r

                fovy = np.clip(cam_state.fov, np.deg2rad(30), np.deg2rad(75))
                aspect = cam_state.aspect
                fovx = 2 * math.atan(math.tan(fovy/2)*aspect)

                c2w = np.eye(4)
                c2w[:3,:3] = tf.SO3(cam_state.wxyz).as_matrix()
                c2w[:3,3] = pos
                c2w[:,1:3] *= -1

                w2c = np.linalg.inv(c2w)
                R = w2c[:3,:3].T
                T = w2c[:3,3]

                wvt = torch.tensor(
                    getWorld2View2(R, T, np.array([0,0,0]), 1.0),
                    dtype=torch.float32
                ).transpose(0,1).cuda()

                proj = getProjectionMatrix(
                    znear=0.1,
                    zfar=50.0,
                    fovX=fovx,
                    fovY=fovy
                ).transpose(0,1).cuda()

                full = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
                center = wvt.inverse()[3,:3]

                view_cam = ProxyCam(cam_a)
                view_cam.override_wvt = wvt
                view_cam.override_proj = proj
                view_cam.override_full = full
                view_cam.override_center = center
                view_cam.override_w = int(800*aspect)
                view_cam.override_h = 800
                view_cam.override_fovx = fovx
                view_cam.override_fovy = fovy

            # ===== 渲染 =====
            out = render(view_cam, gaussians, pipe, background)
            img = torch.clamp(out["render"], 0,1)
            img = (img.cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)

            client.scene.set_background_image(img, format="jpeg")

        time.sleep(0.02)


# ==============================
# Entry
# ==============================
if __name__ == "__main__":
    parser = ArgumentParser()

    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--config", required=True)
    parser.add_argument("--start_checkpoint", required=True)

    parser.add_argument("--gaussian_dim", type=int, default=4)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5,0.5])
    parser.add_argument("--rot_4d", action="store_true", default=True)
    parser.add_argument("--force_sh_3d", action="store_true", default=True)

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    def merge(k, host):
        if isinstance(host[k], DictConfig):
            for kk in host[k]: merge(kk, host[k])
        elif hasattr(args, k):
            setattr(args, k, host[k])

    for k in cfg: merge(k, cfg)

    main(lp.extract(args), pp.extract(args), args)