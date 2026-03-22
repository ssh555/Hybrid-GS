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
    print(f"[渲染器] 数据集解析完成，共 {max_cams} 个视角，每个视角 {max_frames} 帧。")

    model_params, _ = torch.load(args.start_checkpoint, weights_only=False)
    gaussians.restore(model_params, None)

    # ==========================================
    # 🌟 核心突破：计算场景绝对真实的物理重力方向！
    # ==========================================
    up_vectors = []
    for cam in train_cams:
        c2w = get_c2w(cam)
        # 在 OpenGL 坐标系下，相机的 +Y 轴代表它的上方
        up_vectors.append(c2w[:3, 1])
    
    # 取所有相机的平均上方，作为世界基准上方
    avg_up = np.mean(up_vectors, axis=0)
    avg_up /= np.linalg.norm(avg_up)
    print(f"[引擎] 成功计算场景真实重力基准向量: {avg_up}")

    server = viser.ViserServer(port=8080)

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        # 强制接管前端陀螺仪，杜绝轨道相机自动 180 度乱翻转！
        client.camera.up_direction = tuple(avg_up)

    # ==========================================
    # 🎬 UI 控制台
    # ==========================================
    with server.gui.add_folder("🎬 导播台面板"):
        gui_free_roam = server.gui.add_checkbox("🕹️ 启用自由漫游", initial_value=False)
        gui_invert_cam = server.gui.add_checkbox("🔄 画面倒立急救 (点我翻转)", initial_value=False)
        gui_sync_cam = server.gui.add_button("🎯 视角归位 (一键对齐原轨迹)")
        
        cam_id = server.gui.add_slider("🎥 原版机位切换", 0, max_cams-1, 1, 0)

        with server.gui.add_folder("播放控制", expand_by_default=True):
            btn_play = server.gui.add_button("▶️ 播放")
            btn_pause = server.gui.add_button("⏸ 暂停")

        slider_frame = server.gui.add_slider("⏱️ 播放进度", 0, max_frames-1, 0.01, 0)
        slider_speed = server.gui.add_slider("⚡ 播放速度倍率", 0.25, 2.0, 0.05, 1.0)
        gui_res_scale = server.gui.add_slider("🖥️ 渲染质量倍率 (调高极清晰)", 0.5, 2.0, 0.1, 1.0)

    play_state = {"playing": False}

    @btn_play.on_click
    def _(_): play_state["playing"] = True

    @btn_pause.on_click
    def _(_): play_state["playing"] = False

    @gui_sync_cam.on_click
    def _(_):
        gui_free_roam.value = False

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
            native_w = selected_cam.image_width
            native_h = selected_cam.image_height
            render_w = int(native_w * scale)
            render_h = int(native_h * scale)
            
            view_cam = ProxyCam(selected_cam)
            view_cam.override_w = render_w
            view_cam.override_h = render_h

            if not gui_free_roam.value:
                # 导播模式
                c2w_gl = get_c2w(selected_cam)
                client.camera.position = c2w_gl[:3, 3]
                client.camera.wxyz = tf.SO3.from_matrix(c2w_gl[:3, :3]).wxyz
            else:
                # 自由漫游模式
                cam_state = client.camera
                c2w_gl = np.eye(4)
                c2w_gl[:3, :3] = tf.SO3(cam_state.wxyz).as_matrix()
                c2w_gl[:3, 3] = cam_state.position
                
                # 🌟 如果发生极端坐标倒立，提供一键数学急救 (绕局部Z轴旋转180度)
                if gui_invert_cam.value:
                    rot_180 = np.diag([-1, -1, 1, 1])
                    c2w_gl = c2w_gl @ rot_180
                
                # GL 完美转 CV
                c2w_cv = c2w_gl.copy()
                c2w_cv[:, 1:3] *= -1
                w2c_cv = np.linalg.inv(c2w_cv)
                
                R = w2c_cv[:3, :3].T
                T = w2c_cv[:3, 3]
                
                wvt = torch.tensor(getWorld2View2(R, T, np.array([0.,0.,0.]), 1.0), dtype=torch.float32).transpose(0, 1).cuda()
                
                # 锁定 FOV 防止拉伸畸变
                fovx = selected_cam.FoVx
                fovy = selected_cam.FoVy
                proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
                
                view_cam.override_wvt = wvt
                view_cam.override_proj = proj
                view_cam.override_full = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
                view_cam.override_center = wvt.inverse()[3, :3]
                view_cam.override_fovx = fovx
                view_cam.override_fovy = fovy

            # 底层高清渲染
            out = render(view_cam, gaussians, pipe, background)
            img = torch.clamp(out["render"], 0, 1)
            img_np = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            # 自适应防拉伸填缝
            browser_aspect = client.camera.aspect
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

        time.sleep(0.01)

if __name__ == "__main__":
    parser = ArgumentParser()

    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--config", required=True)

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