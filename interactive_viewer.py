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
# SLERP
# ==============================
def slerp(q1, q2, t):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.dot(q1, q2)

    if dot < 0:
        q2 = -q2
        dot = -dot

    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    q3 = q2 - q1 * dot
    q3 /= np.linalg.norm(q3)

    return q1 * np.cos(theta) + q3 * np.sin(theta)


# ==============================
# ProxyCam（修复版）
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


def letterbox(img, target_h, target_w):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = np.array(
        torch.nn.functional.interpolate(
            torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float(),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False
        )[0].permute(1,2,0)
    ).astype(np.uint8)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h)//2
    x0 = (target_w - new_w)//2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas


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
    # 我们以第 1 个相机的空间位置为基准 (View 0)
    base_cam = train_cams[0]
    base_T = base_cam.T.cpu().numpy() if hasattr(base_cam.T, 'cpu') else base_cam.T
    base_R = base_cam.R.cpu().numpy() if hasattr(base_cam.R, 'cpu') else base_cam.R # [新增] 获取基准相机的旋转矩阵
    
    max_frames = 0
    
    for cam in train_cams:
        cam_T = cam.T.cpu().numpy() if hasattr(cam.T, 'cpu') else cam.T
        cam_R = cam.R.cpu().numpy() if hasattr(cam.R, 'cpu') else cam.R # [新增] 获取当前相机的旋转矩阵
        
        # [核心修复] 必须位置(T)和旋转角度(R)都极其接近（误差小于1e-5），才被认定是绝对的同一个静态视角
        if np.allclose(base_T, cam_T, atol=1e-5) and np.allclose(base_R, cam_R, atol=1e-5):
            max_frames += 1
        else:
            break
    # 以max_frames作为时间维度的长度，拆分train_cams成多个视角的时间序列
    # 不同视角的连续帧会被分到不同的列表中，确保每个列表中的相机都是同一个视角的连续时间帧
    view_cams = []
    for i in range(0, len(train_cams), max_frames):
        view_cams.append(train_cams[i:i+max_frames])
    max_cams = len(view_cams)
    print(f"[渲染器] 数据集共有 {len(train_cams)} 图像，分为 {len(view_cams)} 个视角，每个视角最多 {max_cams} 帧")

    model_params, _ = torch.load(args.start_checkpoint, weights_only=False)
    gaussians.restore(model_params, None)

    TARGET_W, TARGET_H = train_cams[0].resolution[0], train_cams[0].resolution[1]

    server = viser.ViserServer(port=8080)

    play_state = {"playing": False, "direction": 1}

    with server.gui.add_folder("🎬 控制台"):

        cam_id = server.gui.add_slider("相机选择", 0, len(train_cams)-1, 1, 0)

        mode = server.gui.add_dropdown(
            "模式",
            ("训练相机", "自由漫游"),
            initial_value="训练相机"
        )

        btn_play = server.gui.add_button("▶️")
        btn_pause = server.gui.add_button("⏸")

        slider_frame = server.gui.add_slider("时间", 0, max_frames, 0.01, 0)
        slider_speed = server.gui.add_slider("速度", 0.25, 2.0, 0.05, 1.0)

    @btn_play.on_click
    def _(_): play_state["playing"] = True

    @btn_pause.on_click
    def _(_): play_state["playing"] = False

    while True:

        if play_state["playing"]:
            slider_frame.value += slider_speed.value

        if slider_frame.value > max_frames:
            slider_frame.value = 0

        frame_idx = int(slider_frame.value)

        selected_cam = view_cams[cam_id.value][frame_idx % len(view_cams[cam_id.value])]

        for client in server.get_clients().values():

            # ===== 模式1：训练相机 =====
            if mode.value == "训练相机":

                view_cam = selected_cam

                c2w = get_c2w(selected_cam)
                client.camera.position = c2w[:3,3]
                client.camera.wxyz = tf.SO3.from_matrix(c2w[:3,:3]).wxyz

            # ===== 模式2：自由漫游 =====
            else:
                cam_state = client.camera

                aspect = TARGET_W / TARGET_H
                fovy = np.clip(cam_state.fov, np.deg2rad(40), np.deg2rad(70))
                fovx = 2 * math.atan(math.tan(fovy/2)*aspect)

                c2w = np.eye(4)
                c2w[:3,:3] = tf.SO3(cam_state.wxyz).as_matrix()
                c2w[:3,3] = cam_state.position
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

                view_cam = ProxyCam(selected_cam)
                view_cam.override_wvt = wvt
                view_cam.override_proj = proj
                view_cam.override_full = full
                view_cam.override_center = center
                view_cam.override_w = TARGET_W
                view_cam.override_h = TARGET_H

            # ===== 渲染 =====
            out = render(view_cam, gaussians, pipe, background)
            img = torch.clamp(out["render"], 0,1)
            img = (img.cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)

            img = letterbox(img, TARGET_H, TARGET_W)

            client.scene.set_background_image(img, format="png")

        time.sleep(0.02)


# ==============================
# 启动
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