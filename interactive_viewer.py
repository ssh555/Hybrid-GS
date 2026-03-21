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
# 🌟 高质量渲染配置
# ==========================================
class HighQualityRenderer:
    def __init__(self):
        self.render_scale = 1.0  # 渲染缩放因子，可以设置为1.5或2.0进行超采样
        self.enable_antialiasing = True
        self.quality_preset = "high"  # low, medium, high, ultra
        
    def get_render_size(self, client_width, client_height):
        """根据画质预设获取实际渲染尺寸"""
        if self.quality_preset == "ultra":
            scale = 2.0
        elif self.quality_preset == "high":
            scale = 1.5
        elif self.quality_preset == "medium":
            scale = 1.0
        else:  # low
            scale = 0.75
            
        # 确保渲染尺寸是8的倍数（某些CUDA操作要求）
        render_w = int((client_width * scale) // 8 * 8)
        render_h = int((client_height * scale) // 8 * 8)
        return render_w, render_h

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

def get_c2w_from_colmap(cam):
    """获取相机到世界的变换矩阵"""
    w2c = cam.world_view_transform.transpose(0, 1).cpu().numpy()
    c2w_opencv = np.linalg.inv(w2c)
    c2w_opengl = c2w_opencv.copy()
    c2w_opengl[:, 1:3] *= -1
    return c2w_opengl

def apply_tone_mapping(image, gamma=2.2, exposure=1.0):
    """应用色调映射，提升视觉效果"""
    img = image * exposure
    img = torch.clamp(img, 0.0, 1.0)
    # 简单的伽马校正
    img = torch.pow(img, 1.0/gamma)
    return img

def apply_sharpen(image, strength=0.3):
    """简单的锐化滤波器"""
    if strength <= 0:
        return image
    
    kernel = torch.tensor([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]], device=image.device, dtype=image.float32) / 9.0
    
    kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    
    # 对每个通道应用卷积
    img_unsqueezed = image.unsqueeze(0)
    sharpened = torch.nn.functional.conv2d(img_unsqueezed, kernel, padding=1, groups=3)
    sharpened = sharpened.squeeze(0)
    
    # 混合原图和锐化图
    result = image * (1 - strength) + sharpened * strength
    return torch.clamp(result, 0.0, 1.0)

@torch.no_grad()
def main(dataset: ModelParams, pipe: PipelineParams, args):
    print(f"[引擎] 正在初始化 4DGS 高质量渲染引擎...")
    
    # 背景颜色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 初始化高斯模型
    sh_degree = dataset.sh_degree if hasattr(dataset, 'sh_degree') else 3
    gaussians = GaussianModel(sh_degree, 
                             gaussian_dim=args.gaussian_dim, 
                             time_duration=args.time_duration,
                             rot_4d=args.rot_4d, 
                             force_sh_3d=args.force_sh_3d, 
                             sh_degree_t=2 if pipe.eval_shfs_4d else 0)

    # 加载场景
    scene = Scene(dataset, gaussians, shuffle=False)
    train_cameras = [c[1] if isinstance(c, tuple) else c for c in scene.getTrainCameras()]
    max_frames = len(train_cameras) - 1

    # 加载模型权重
    checkpoint = args.start_checkpoint
    (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
    gaussians.restore(model_params, None)

    # 初始化Viser服务器
    server = viser.ViserServer(port=8080)
    quality_renderer = HighQualityRenderer()
    
    print("\n" + "="*60)
    print(f"🚀 4DGS 高质量渲染器启动成功！")
    print(f"📡 访问地址: http://localhost:8080")
    print("="*60 + "\n")

    # ==========================================
    # 🎬 UI 控制台 - 增加画质选项
    # ==========================================
    with server.gui.add_folder("🎬 播放控制"):
        gui_mode = server.gui.add_dropdown(
            "漫游模式", 
            ("原画轨迹播放 (100%安全高清)", "自由漫游 (高质量模式)"), 
            initial_value="原画轨迹播放 (100%安全高清)"
        )
        gui_play = server.gui.add_checkbox("▶️ 自动播放/暂停", initial_value=False)
        gui_frame = server.gui.add_slider("时间与机位进度", min=0, max=max_frames, step=1, initial_value=0)
    
    with server.gui.add_folder("🎨 画质设置"):
        gui_quality = server.gui.add_dropdown(
            "渲染质量",
            ("ultra", "high", "medium", "low"),
            initial_value="high"
        )
        gui_tone_mapping = server.gui.add_checkbox("色调映射", initial_value=True)
        gui_sharpen = server.gui.add_slider("锐化强度", min=0, max=1, step=0.05, initial_value=0.2)
        gui_exposure = server.gui.add_slider("曝光度", min=0.5, max=2.0, step=0.05, initial_value=1.0)
    
    with server.gui.add_folder("⚙️ 相机设置"):
        gui_near_plane = server.gui.add_slider("近平面距离", min=0.01, max=1.0, step=0.01, initial_value=0.01)
        gui_far_plane = server.gui.add_slider("远平面距离", min=50, max=500, step=10, initial_value=100.0)

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.scene.add_grid("grid", visible=False)
        c2w = get_c2w_from_colmap(train_cameras[0])
        client.camera.position = c2w[:3, 3]
        client.camera.wxyz = tf.SO3.from_matrix(c2w[:3, :3]).wxyz
        
        # 设置初始相机参数
        client.camera.fov = 45.0  # 默认FOV
        client.camera.near = 0.01
        client.camera.far = 100.0

    frame_count = 0
    fps = 30
    frame_time = 1.0 / fps
    
    while True:
        start_time = time.time()
        
        clients = server.get_clients()
        if not clients:
            time.sleep(0.01)
            continue

        # 更新画质设置
        quality_renderer.quality_preset = gui_quality.value

        # 播放逻辑
        if gui_play.value:
            gui_frame.value = (gui_frame.value + 1) % (max_frames + 1)

        current_idx = int(gui_frame.value)
        base_cam = train_cameras[current_idx]

        for client_id, client in clients.items():
            if gui_mode.value == "原画轨迹播放 (100%安全高清)":
                # 原画模式：直接使用原始相机参数
                view_cam = base_cam
                
                # 同步相机视角
                c2w = get_c2w_from_colmap(base_cam)
                client.camera.position = c2w[:3, 3]
                client.camera.wxyz = tf.SO3.from_matrix(c2w[:3, :3]).wxyz
                
                # 获取原始渲染尺寸
                render_w = base_cam.image_width
                render_h = base_cam.image_height
                
            else:
                # 自由漫游模式：高质量渲染
                cam_state = client.camera
                
                # 获取高质量渲染尺寸
                render_w, render_h = quality_renderer.get_render_size(
                    int(800 * cam_state.aspect), 800
                )
                
                # 计算精确的FOV
                aspect = render_w / render_h
                fovy = cam_state.fov  # 垂直FOV
                fovx = 2 * math.atan(math.tan(fovy / 2) * aspect)
                
                # 获取相机位姿
                c2w_opengl = np.eye(4)
                c2w_opengl[:3, :3] = tf.SO3(cam_state.wxyz).as_matrix()
                c2w_opengl[:3, 3] = cam_state.position
                
                # OpenCV坐标系转换
                c2w_opencv = c2w_opengl.copy()
                c2w_opencv[:, 1:3] *= -1
                
                # 计算视图矩阵
                w2c = np.linalg.inv(c2w_opencv)
                R = w2c[:3, :3].T
                T = w2c[:3, 3]
                
                # 获取投影矩阵
                znear = gui_near_plane.value
                zfar = gui_far_plane.value
                
                wvt = torch.tensor(getWorld2View2(R, T, np.array([0., 0., 0.]), 1.0), 
                                  dtype=torch.float32).transpose(0, 1).cuda()
                proj = getProjectionMatrix(znear=znear, zfar=zfar, 
                                          fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
                full_proj = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
                cam_center = wvt.inverse()[3, :3]
                
                # 创建代理相机
                view_cam = ProxyCam(base_cam)
                view_cam.override_wvt = wvt
                view_cam.override_proj = proj
                view_cam.override_full = full_proj
                view_cam.override_center = cam_center
                view_cam.override_w = render_w
                view_cam.override_h = render_h
                view_cam.override_fovx = fovx
                view_cam.override_fovy = fovy

            # 执行渲染
            render_pkg = render(view_cam, gaussians, pipe, background)
            rendered_image = render_pkg["render"]
            
            # 后处理
            if gui_tone_mapping.value:
                rendered_image = apply_tone_mapping(rendered_image, exposure=gui_exposure.value)
            else:
                rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
            
            if gui_sharpen.value > 0:
                rendered_image = apply_sharpen(rendered_image, strength=gui_sharpen.value)
            
            # 转换为numpy并发送
            img_np = (rendered_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            
            # 如果渲染尺寸和显示尺寸不同，需要进行高质量下采样
            display_w = int(800 * client.camera.aspect)
            display_h = 800
            
            if img_np.shape[1] != display_w or img_np.shape[0] != display_h:
                # 使用高质量插值进行缩放
                from PIL import Image
                img_pil = Image.fromarray(img_np)
                img_pil = img_pil.resize((display_w, display_h), Image.Resampling.LANCZOS)
                img_np = np.array(img_pil)
            
            # 发送给客户端
            client.scene.set_background_image(img_np, format="jpeg", quality=95)

        # 帧率控制
        elapsed = time.time() - start_time
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

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
            for k in host[key].keys(): 
                recursive_merge(k, host[key])
        elif hasattr(args, key): 
            setattr(args, key, host[key])
    
    for k in cfg.keys(): 
        recursive_merge(k, cfg)
    
    main(lp.extract(args), pp.extract(args), args)