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
    base_R = base_cam.R.cpu().numpy() if hasattr(base_cam.R, 'cpu') else base_cam.R
    
    max_frames = 0
    
    for cam in train_cams:
        cam_T = cam.T.cpu().numpy() if hasattr(cam.T, 'cpu') else cam.T
        cam_R = cam.R.cpu().numpy() if hasattr(cam.R, 'cpu') else cam.R
        
        # 必须位置(T)和旋转角度(R)都极其接近（误差小于1e-5），才被认定是绝对的同一个静态视角
        if np.allclose(base_T, cam_T, atol=1e-5) and np.allclose(base_R, cam_R, atol=1e-5):
            max_frames += 1
        else:
            break
            
    # 以max_frames作为时间维度的长度，拆分train_cams成多个视角的时间序列
    view_cams = []
    for i in range(0, len(train_cams), max_frames):
        view_cams.append(train_cams[i:i+max_frames])
    max_cams = len(view_cams)
    print(f"[渲染器] 数据集共有 {len(train_cams)} 图像，分为 {len(view_cams)} 个视角，每个视角最多 {max_frames} 帧")

    model_params, _ = torch.load(args.start_checkpoint, weights_only=False)
    gaussians.restore(model_params, None)

    server = viser.ViserServer(port=8080)
    play_state = {"playing": False}

    # ==========================================
    # 🎬 强大的 UI 控制台
    # ==========================================
    with server.gui.add_folder("🎬 控制台"):
        # 保持整数步进的固定机位切换，确保 100% 贴合完美原机位
        cam_id = server.gui.add_slider("🎥 相机机位切换", 0, max_cams-1, 1, 0)

        with server.gui.add_folder("播放控制", expand_by_default=True):
            btn_play = server.gui.add_button("▶️ 播放")
            btn_pause = server.gui.add_button("⏸ 暂停")

        slider_frame = server.gui.add_slider("⏱️ 播放进度", 0, max_frames-1, 0.01, 0)
        slider_speed = server.gui.add_slider("⚡ 播放速度倍率", 0.25, 2.0, 0.05, 1.0)
        
        gui_res_scale = server.gui.add_slider("🖥️ 原生渲染质量倍率 (调高极清晰)", 0.5, 2.0, 0.1, 1.0)

    @btn_play.on_click
    def _(_): play_state["playing"] = True

    @btn_pause.on_click
    def _(_): play_state["playing"] = False

    # ==========================================
    # 🌟 真实时间戳初始化 (对齐离线视频的物理时长)
    # ==========================================
    last_update_time = time.time()
    TARGET_FPS = 30.0  # 假设你离线合成的视频是 30 FPS

    while True:
        # 计算距离上一次渲染过了多少秒 (Delta Time)
        current_time = time.time()
        dt = current_time - last_update_time
        last_update_time = current_time

        if play_state["playing"]:
            # 使用真实时间推进，保证无论渲染画质倍率多高（渲染多卡），物理播放总时长绝对恒定！
            slider_frame.value += TARGET_FPS * dt * slider_speed.value

        # 无缝循环播放
        if slider_frame.value >= max_frames:
            slider_frame.value %= max_frames

        frame_idx = int(slider_frame.value)
        selected_cam = view_cams[cam_id.value][frame_idx % len(view_cams[cam_id.value])]

        for client in server.get_clients().values():
            
            # 1. 设置视角的空间位置（让前端的视角跟着变）
            c2w = get_c2w(selected_cam)
            client.camera.position = c2w[:3,3]
            client.camera.wxyz = tf.SO3.from_matrix(c2w[:3,:3]).wxyz

            # 2. 获取原始分辨率，并应用画质倍率
            scale = gui_res_scale.value
            native_w = selected_cam.image_width
            native_h = selected_cam.image_height
            render_w = int(native_w * scale)
            render_h = int(native_h * scale)

            # 3. 使用代理相机，覆盖底层的分辨率（直接输出高清画面！）
            view_cam = ProxyCam(selected_cam)
            view_cam.override_w = render_w
            view_cam.override_h = render_h

            # ===== 底层高清渲染 =====
            out = render(view_cam, gaussians, pipe, background)
            img = torch.clamp(out["render"], 0, 1)
            img_np = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            # ==========================================
            # 🌟 智能自适应黑边填充逻辑 (绝对不再变形！)
            # ==========================================
            browser_aspect = client.camera.aspect
            render_aspect = render_w / render_h

            if browser_aspect > render_aspect:
                # 浏览器比渲染图更“宽” -> 左右留黑边
                canvas_h = render_h
                canvas_w = int(render_h * browser_aspect)
            else:
                # 浏览器比渲染图更“高” -> 上下留黑边
                canvas_w = render_w
                canvas_h = int(render_w / browser_aspect)

            # 创建一张纯黑幕布
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            # 计算居中坐标，把极其清晰的原生渲染图“贴”在正中间
            y0 = (canvas_h - render_h) // 2
            x0 = (canvas_w - render_w) // 2
            canvas[y0:y0+render_h, x0:x0+render_w] = img_np

            # 发送给前端 (使用了 jpeg, jpeg_quality=100 以兼顾最高画质与传输帧率)
            client.scene.set_background_image(canvas, format="jpeg", jpeg_quality=100)

        # 限制最高空转帧率，防止 CPU 占用过高
        time.sleep(0.01)


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