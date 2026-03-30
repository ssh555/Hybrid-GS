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
    print(f"[INFO] [渲染器] 数据集解析完成，共 {max_cams} 个视角，每个视角 {max_frames} 帧。")

    model_params, _ = torch.load(args.start_checkpoint, weights_only=False)
    gaussians.restore(model_params, None)

    server = viser.ViserServer(port=8080)

    # ==============================
    # 自动拟合训练相机轨道
    # ==============================
    cam_centers = []
    for cams in view_cams:
        c2w = get_c2w(cams[0])
        cam_centers.append(c2w[:3, 3])

    cam_centers = np.stack(cam_centers, axis=0)

    stage_center = cam_centers.mean(axis=0)
    stage_center[1] = np.percentile(cam_centers[:, 1], 30)

    relative = cam_centers - stage_center

    radius_all = np.linalg.norm(relative[:, [0, 2]], axis=1)
    yaw_all = np.arctan2(relative[:, 0], relative[:, 2])
    height_all = relative[:, 1]

    radius_mean = float(np.mean(radius_all))
    height_mean = float(np.mean(height_all))

    yaw_min = float(np.min(yaw_all))
    yaw_max = float(np.max(yaw_all))

    r_min = float(np.percentile(radius_all, 10))
    r_max = float(np.percentile(radius_all, 90))

    h_min = float(np.percentile(height_all, 10))
    h_max = float(np.percentile(height_all, 90))

    # ==============================
    # 安全收缩边界（避免贴训练边缘炸刺）
    # ==============================
    SAFE_MARGIN = 0.08

    yaw_span = yaw_max - yaw_min
    r_span = r_max - r_min
    h_span = h_max - h_min

    yaw_min += yaw_span * SAFE_MARGIN
    yaw_max -= yaw_span * SAFE_MARGIN

    r_min += r_span * SAFE_MARGIN
    r_max -= r_span * SAFE_MARGIN

    h_min += h_span * SAFE_MARGIN
    h_max -= h_span * SAFE_MARGIN

    roam_state = {
        "yaw": float((yaw_min + yaw_max) * 0.5),
        "radius": float(np.clip(radius_mean, r_min, r_max)),
        "height": float(np.clip(height_mean, h_min, h_max)),
    }

    print("[INFO] [自由漫游约束初始化完成]")
    print(f"[INFO] yaw: {np.degrees(yaw_min):.1f}° ~ {np.degrees(yaw_max):.1f}°")
    print(f"[INFO] radius: {r_min:.3f} ~ {r_max:.3f}")
    print(f"[INFO] height: {h_min:.3f} ~ {h_max:.3f}")

    warning_state = {
        "yaw": False,
        "radius": False,
        "height": False,
    }

    # ==========================================
    # 🎬 UI 控制台
    # ==========================================
    with server.gui.add_folder("🎬 导播台面板"):
        gui_free_roam = server.gui.add_checkbox("🕹️ 启用自由漫游", initial_value=False)
        gui_sync_cam = server.gui.add_button("🎯 视角归位 (一键对齐原轨迹)")
        
        cam_id = server.gui.add_slider("🎥 原版机位切换", 0, max_cams-1, 1, 0)

        with server.gui.add_folder("播放控制", expand_by_default=True):
            btn_play = server.gui.add_button("▶️ 播放")
            btn_pause = server.gui.add_button("⏸ 暂停")

        slider_frame = server.gui.add_slider("⏱️ 播放进度", 0, max_frames-1, 0.01, 0)
        slider_speed = server.gui.add_slider("⚡ 播放速度倍率", 0.25, 2.0, 0.05, 1.0)

        gui_res_scale = server.gui.add_slider("🖥️ 渲染质量倍率 (调高极清晰)", 0.5, 2.0, 0.1, 1.0)
    with server.gui.add_folder("🕹️ 扇形漫游控制", expand_by_default=True):
        btn_w = server.gui.add_button("W 前进")
        btn_s = server.gui.add_button("S 后退")
        btn_a = server.gui.add_button("A 左移")
        btn_d = server.gui.add_button("D 右移")
        btn_q = server.gui.add_button("Q 上升")
        btn_e = server.gui.add_button("E 下降")

    play_state = {"playing": False}

    @btn_play.on_click
    def _(_): play_state["playing"] = True

    @btn_pause.on_click
    def _(_): play_state["playing"] = False

    @gui_sync_cam.on_click
    def _(_):
        gui_free_roam.value = False

    MOVE_RADIUS = radius_mean * 0.03
    MOVE_YAW = np.radians(2.0)
    MOVE_HEIGHT = 0.05

    @btn_w.on_click
    def _(_):
        roam_state["radius"] -= MOVE_RADIUS

    @btn_s.on_click
    def _(_):
        roam_state["radius"] += MOVE_RADIUS

    @btn_a.on_click
    def _(_):
        roam_state["yaw"] -= MOVE_YAW

    @btn_d.on_click
    def _(_):
        roam_state["yaw"] += MOVE_YAW

    @btn_q.on_click
    def _(_):
        roam_state["height"] += MOVE_HEIGHT

    @btn_e.on_click
    def _(_):
        roam_state["height"] -= MOVE_HEIGHT

    last_update_time = time.time()
    TARGET_FPS = 30.0

    while True:
        current_time = time.time()
        dt = current_time - last_update_time
        last_update_time = current_time

        if play_state["playing"]:
            slider_frame.value += TARGET_FPS * dt * slider_speed.value

        if slider_frame.value >= max_frames:
            slider_frame.value %= max_frames

        frame_idx = int(slider_frame.value)
        selected_cam = view_cams[cam_id.value][frame_idx % len(view_cams[cam_id.value])]

        for client in server.get_clients().values():
            scale = gui_res_scale.value
            browser_aspect = client.camera.aspect

            if not gui_free_roam.value:
                # ==========================================================
                # 🎬 导播模式：利用黑边防畸变，完美呈现过拟合的 2D 纸片视角
                # ==========================================================
                render_w = int(selected_cam.image_width * scale)
                render_h = int(selected_cam.image_height * scale)
                
                view_cam = ProxyCam(selected_cam)
                view_cam.override_w = render_w
                view_cam.override_h = render_h
                
                c2w_gl = get_c2w(selected_cam)
                client.camera.position = c2w_gl[:3, 3]
                client.camera.wxyz = tf.SO3.from_matrix(c2w_gl[:3, :3]).wxyz

                out = render(view_cam, gaussians, pipe, background)
                img = torch.clamp(out["render"], 0, 1)
                img_np = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

                render_aspect = render_w / render_h
                if browser_aspect > render_aspect:
                    canvas_h = render_h
                    canvas_w = int(render_h * browser_aspect)
                else:
                    canvas_w = render_w
                    canvas_h = int(render_w / browser_aspect)

                canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                y0 = (canvas_h - render_h) // 2
                x0 = (canvas_w - render_w) // 2
                canvas[y0:y0+render_h, x0:x0+render_w] = img_np

                client.scene.set_background_image(canvas, format="png")

            else:
                # ==========================================
                # 🕹️ 扇形柱体 WASDQE 漫游（稳定版）
                # ==========================================
                render_h = int(selected_cam.image_height * scale)
                render_w = int(render_h * browser_aspect)

                # -----------------------------
                # clamp 到训练范围
                # -----------------------------
                roam_state["yaw"] = float(np.clip(roam_state["yaw"], yaw_min, yaw_max))
                roam_state["radius"] = float(np.clip(roam_state["radius"], r_min, r_max))
                roam_state["height"] = float(np.clip(roam_state["height"], h_min, h_max))

                yaw = roam_state["yaw"]
                radius = roam_state["radius"]
                height = roam_state["height"]

                # -----------------------------
                # 相机位置（扇形柱体）
                # -----------------------------
                cam_pos = np.array([
                    stage_center[0] + radius * np.sin(yaw),
                    stage_center[1] + height,
                    stage_center[2] + radius * np.cos(yaw),
                ], dtype=np.float32)

                # -----------------------------
                # 始终看向舞台中心
                # -----------------------------
                target = stage_center.copy()
                target[1] += height * 0.15

                forward = target - cam_pos
                forward /= (np.linalg.norm(forward) + 1e-8)

                world_up = np.array([0, 1, 0], dtype=np.float32)

                right = np.cross(world_up, forward)
                right /= (np.linalg.norm(right) + 1e-8)

                up = np.cross(forward, right)
                up /= (np.linalg.norm(up) + 1e-8)

                c2w_cv = np.eye(4, dtype=np.float32)
                c2w_cv[:3, 0] = right
                c2w_cv[:3, 1] = up
                c2w_cv[:3, 2] = forward
                c2w_cv[:3, 3] = cam_pos

                w2c_cv = np.linalg.inv(c2w_cv)

                wvt = torch.tensor(
                    w2c_cv,
                    dtype=torch.float32,
                    device="cuda"
                ).transpose(0, 1)

                fovy = selected_cam.FoVy
                fovx = 2.0 * math.atan(
                    math.tan(fovy / 2.0) * browser_aspect
                )

                proj = getProjectionMatrix(
                    znear=0.1,
                    zfar=100.0,
                    fovX=fovx,
                    fovY=fovy
                ).transpose(0, 1).cuda()

                view_cam = ProxyCam(selected_cam)
                view_cam.override_w = render_w
                view_cam.override_h = render_h
                view_cam.override_wvt = wvt
                view_cam.override_proj = proj
                view_cam.override_full = wvt @ proj
                view_cam.override_center = torch.tensor(
                    cam_pos,
                    dtype=torch.float32,
                    device="cuda"
                )
                view_cam.override_fovx = fovx
                view_cam.override_fovy = fovy

                out = render(view_cam, gaussians, pipe, background)
                img = torch.clamp(out["render"], 0, 1)
                img_np = (
                    img.cpu().numpy().transpose(1, 2, 0) * 255
                ).astype(np.uint8)

                # ==========================================
                # 非刷屏 Warning 日志
                # ==========================================
                yaw_warn = yaw <= yaw_min + yaw_span * 0.1 or yaw >= yaw_max - yaw_span * 0.1
                r_warn = radius <= r_min + r_span * 0.1 or radius >= r_max - r_span * 0.1
                h_warn = height <= h_min + h_span * 0.1 or height >= h_max - h_span * 0.1

                if yaw_warn and not warning_state["yaw"]:
                    print(f"[WARNING] yaw 接近训练边界: {np.degrees(yaw):.1f}°")
                if r_warn and not warning_state["radius"]:
                    print(f"[WARNING] 半径接近训练边界: {radius:.3f}")
                if h_warn and not warning_state["height"]:
                    print(f"[WARNING] 高度接近训练边界: {height:.3f}")

                warning_state["yaw"] = yaw_warn
                warning_state["radius"] = r_warn
                warning_state["height"] = h_warn

                client.scene.set_background_image(img_np, format="png")

        time.sleep(0.01)

if __name__ == "__main__":
    parser = ArgumentParser()

    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--config", required=True)
    parser.add_argument("--start_checkpoint", type=str, default = "无效参数，但是删除会影响其他地方的参数解析，暂时保留")

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