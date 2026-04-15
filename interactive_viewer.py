# interactive_viewer.py
import time
import math
import os
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
from utils.camera_utils import get_camera_metadata

import concurrent.futures
import threading

class RenderCam:
    def __init__(self, base_cam):
        self.image_width = base_cam.image_width
        self.image_height = base_cam.image_height
        self.FoVx = base_cam.FoVx
        self.FoVy = base_cam.FoVy
        self.timestamp = base_cam.timestamp
        
        self.world_view_transform = base_cam.world_view_transform.clone()
        self.projection_matrix = base_cam.projection_matrix.clone()
        self.full_proj_transform = base_cam.full_proj_transform.clone()
        self.camera_center = base_cam.camera_center.clone()
        
        self.uid = getattr(base_cam, 'uid', 0)
        self.image_name = getattr(base_cam, 'image_name', 'roam_cam')
        self.gt_alpha_mask = getattr(base_cam, 'gt_alpha_mask', None)

        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0
        self.fl_x = self.cx / math.tan(self.FoVx / 2.0)
        self.fl_y = self.cy / math.tan(self.FoVy / 2.0)
        
    def get_rays(self):
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
# 🚀 异步滑动窗口缓存管理器
# ==============================
class AsyncWindowCache:
    def __init__(self, base_dir, models_dict, max_cache=3):
        self.base_dir = base_dir
        self.models_dict = models_dict
        self.max_cache = max_cache
        self.cache = {}         # 存放就绪的窗口数据 {win_idx: data_tuple}
        self.loading_tasks = {} # 存放正在加载的线程任务 {win_idx: Future}
        self.lock = threading.Lock()
        # 开启 1 个后台独立线程，专门负责磁盘读取和总线传输，绝不阻塞主渲染循环！
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _load_pth(self, win_idx):
        """后台线程执行的实际加载逻辑"""
        rel_path = self.models_dict[win_idx]
        abs_path = os.path.join(self.base_dir, rel_path)
        # 因为瘦身后的模型很小，直接加载到 CUDA 显存中，避免主线程渲染时的 CPU->GPU 拷贝卡顿
        data = torch.load(abs_path, map_location='cuda', weights_only=False)
        with self.lock:
            self.cache[win_idx] = data
            if win_idx in self.loading_tasks:
                del self.loading_tasks[win_idx]
        return data

    def get_window(self, win_idx):
        """主线程调用：获取当前窗口数据，并智能调度后台预加载"""
        # 1. 每次请求时，刷新预加载任务（自动加载前后相邻窗口）
        self._prefetch(win_idx)
        
        # 2. 检查命中情况：如果用户拖动进度条发生了大跳跃（未命中）
        if win_idx not in self.cache:
            print(f"\n[缓存调度] ⚠️ 缓存未命中，发生大跨度跳跃！正在阻塞加载窗口 {win_idx}...")
            # 如果后台还没开始加载它，马上派发任务
            if win_idx not in self.loading_tasks:
                self.loading_tasks[win_idx] = self.executor.submit(self._load_pth, win_idx)
            # 阻塞主线程，直到这个急需的窗口加载完毕
            self.loading_tasks[win_idx].result() 
            print(f"[缓存调度] ⚡ 窗口 {win_idx} 急加载完成，恢复渲染！")
        
        # 3. 完美命中，直接从内存/显存返回，耗时 0.0001 秒
        return self.cache[win_idx]

    def _prefetch(self, current_win_idx):
        """智能缓存替换算法：保留 [current-1, current, current+1]"""
        with self.lock:
            # 定义需要保活的窗口索引（当前、后一个、前一个）
            keep_indices = {current_win_idx - 1, current_win_idx, current_win_idx + 1}
            
            # 1. LRU 淘汰：清理过期缓存，瞬间释放显存
            keys_to_delete = [k for k in list(self.cache.keys()) if k not in keep_indices]
            for k in keys_to_delete:
                del self.cache[k]
                
            # 2. 发起预加载：检查需要的窗口是否在路上，不在就派发任务
            for p_idx in keep_indices:
                if 0 <= p_idx < len(self.models_dict):
                    if p_idx not in self.cache and p_idx not in self.loading_tasks:
                        # 开启后台静默加载
                        self.loading_tasks[p_idx] = self.executor.submit(self._load_pth, p_idx)

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
    train_cams, max_frames, max_cams = get_camera_metadata(scene, dataset.model_path)
    view_cams = []
    for i in range(0, len(train_cams), max_frames):
        view_cams.append(train_cams[i:i+max_frames])
    max_cams = len(view_cams)
    print(f"[渲染器] 数据集解析完成，共 {max_cams} 个视角，每个视角 {max_frames} 帧。")

    # ==========================================
    # 核心修改 1：支持超级大模型的读取
    # ==========================================
    print(f"[渲染器] 正在读取模型权重: {args.start_checkpoint}")
    checkpoint_data, _ = torch.load(args.start_checkpoint, weights_only=False)
    is_swings = isinstance(checkpoint_data, dict) and checkpoint_data.get("is_swings_sequence", False)

    window_cache = None # [新增]

    if is_swings:
        print("[渲染器] 🚀 检测到 SWinGS 长序列超级大模型！将根据时间轴动态加载基底！")
        window_blocks = checkpoint_data["window_blocks"]
        models_dict = checkpoint_data["models"]
        current_loaded_win_idx = -1
        # [新增] 初始化异步缓存，挂载模型路径
        base_model_dir = os.path.dirname(os.path.abspath(args.start_checkpoint))
        window_cache = AsyncWindowCache(base_model_dir, models_dict)
    else:
        print("[渲染器] 📌 检测到传统单体模型，正在直接恢复权重...")
        gaussians.restore(checkpoint_data, None)

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
        
        # ==========================================
        # 核心修改 2：帧级动态参数切换逻辑
        # ==========================================
        if is_swings:
            target_win_idx = 0
            for w_idx, (w_start, w_end) in enumerate(window_blocks):
                if w_start <= frame_idx <= w_end:
                    target_win_idx = w_idx
                    break
                    
            if target_win_idx != current_loaded_win_idx:
                win_data_tuple = window_cache.get_window(target_win_idx)
                gaussians.restore(win_data_tuple, None)
                current_loaded_win_idx = target_win_idx

        selected_cam = view_cams[max_cams // 2][frame_idx % len(view_cams[max_cams // 2])]

        for client in server.get_clients().values():
            scale = gui_res_scale.value
            browser_aspect = client.camera.aspect
            
            # 🕹️ 自由漫游模式：暴力覆写所有渲染矩阵！
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
    parser.add_argument("--start_checkpoint", type=str, default = None)

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