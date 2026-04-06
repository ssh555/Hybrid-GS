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
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCenterShift, getProjectionMatrixCV, pix2ndc

class RenderCam:
    def __init__(self, base_cam):
        # 1. 拷贝渲染器 (render) 强制需要的核心标量
        self.image_width = base_cam.image_width
        self.image_height = base_cam.image_height
        self.FoVx = base_cam.FoVx
        self.FoVy = base_cam.FoVy
        self.timestamp = base_cam.timestamp
        
        # 2. 核心防御：使用 .clone() 彻底拷贝张量数据，断开与 base_cam 的显存引用
        self.world_view_transform = base_cam.world_view_transform.clone()
        self.projection_matrix = base_cam.projection_matrix.clone()
        self.full_proj_transform = base_cam.full_proj_transform.clone()
        self.camera_center = base_cam.camera_center.clone()
        
        # 3. 拷贝其他杂项以防万一
        self.uid = getattr(base_cam, 'uid', 0)
        self.image_name = getattr(base_cam, 'image_name', 'roam_cam')
        self.gt_alpha_mask = getattr(base_cam, 'gt_alpha_mask', None)

        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0
        self.fl_x = self.cx / math.tan(self.FoVx / 2.0)
        self.fl_y = self.cy / math.tan(self.FoVy / 2.0)
        
    def get_rays(self):
        """
        原生重写射线生成逻辑，使用当前 RenderCam 自身的物理属性，
        彻底杜绝 __getattr__ 带来的隐式上下文穿透！
        """
        # 使用纯 PyTorch 生成网格，避免依赖 Kornia
        y, x = torch.meshgrid(torch.arange(self.image_height), torch.arange(self.image_width), indexing='ij')
        x = (x.float() + 0.5).cuda()
        y = (y.float() + 0.5).cuda()
        
        pts_view = torch.stack([
            (x - self.cx) / self.fl_x, 
            (y - self.cy) / self.fl_y, 
            torch.ones_like(x), 
            torch.ones_like(x)
        ], dim=-1)
        
        c2w = torch.linalg.inv(self.world_view_transform.transpose(0, 1))
        pts_world = pts_view @ c2w.T
        directions = pts_world[..., :3] - self.camera_center[None, None, :]
        
        return self.camera_center[None, None], directions / torch.norm(directions, dim=-1, keepdim=True)

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

    server = viser.ViserServer(port=8080)


    # ==========================================
    # 🎬 UI 控制台
    # ==========================================
    with server.gui.add_folder("🎬 导播台面板"):
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

    @server.on_client_connect
    def on_client_connect(client):
        # 获取初始相机位置（使用选中的相机）
        selected_cam = view_cams[max_cams // 2][0]  # 使用第一帧
        c2w_gl = get_c2w(selected_cam)
        
        # 设置客户端相机位置和姿态
        client.camera.position = c2w_gl[:3, 3]
        client.camera.wxyz = tf.SO3.from_matrix(c2w_gl[:3, :3]).wxyz


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
        selected_cam = view_cams[max_cams // 2][frame_idx % len(view_cams[max_cams // 2])]

        for client in server.get_clients().values():
            scale = gui_res_scale.value
            browser_aspect = client.camera.aspect
            # ==========================================================
            # 🕹️ 自由漫游模式：暴力覆写所有渲染矩阵！
            # ==========================================================
            cam_state = client.camera
            
            render_w = int(selected_cam.image_width * scale)
            render_h = int(selected_cam.image_height * scale)

            c2w_gl = np.eye(4, dtype=np.float32)
            c2w_gl[:3, :3] = tf.SO3(cam_state.wxyz).as_matrix()
            c2w_gl[:3, 3] = cam_state.position

            c2w_cv = c2w_gl.copy()
            c2w_cv[:, 1:3] *= -1 

            w2c_cv = np.linalg.inv(c2w_cv)
            wvt = torch.tensor(w2c_cv, dtype=torch.float32, device="cuda").transpose(0, 1)

            fovy = selected_cam.FoVy
            fovx = selected_cam.FoVx
            if selected_cam.cx > 0:
                proj= getProjectionMatrixCenterShift(0.1, 100.0, selected_cam.cx, selected_cam.cy, selected_cam.fl_x, selected_cam.fl_y, selected_cam.image_width, selected_cam.image_height).transpose(0,1)
            else:
                if selected_cam.cyr != 0.0 :
                    proj = getProjectionMatrixCV(znear=0.1, zfar=100.0, fovX=fovx, fovY=fovy, cx=selected_cam.cxr, cy=selected_cam.cyr).transpose(0,1)
                else: 
                    proj = getProjectionMatrix(znear=0.1, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0,1)
            
            proj = proj.cuda()

            view_cam = RenderCam(selected_cam)
            view_cam.image_width = render_w
            view_cam.image_height = render_h
            view_cam.FoVx = fovx
            view_cam.FoVy = fovy
            view_cam.world_view_transform = wvt
            view_cam.projection_matrix = proj
            view_cam.full_proj_transform = (view_cam.world_view_transform.unsqueeze(0).bmm(view_cam.projection_matrix.unsqueeze(0))).squeeze(0)
            view_cam.camera_center = view_cam.world_view_transform.inverse()[3, :3]

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