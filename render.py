# 统一渲染与3D漫游入口
# python render.py --config ./configs/n3v/3D4DGS.yaml
# 文件：render.py
import os
import torch
import imageio
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.camera_trajectory import generate_smooth_trajectory
from utils.image_utils import easy_cmap

def get_inference_active_mask(gaussians, frame_id):
    """
    [HybridGS 核心] 推理时的生命周期掩码过滤器。
    确保在漫游视频中，只渲染 3D 静态背景和在当前 frame_id 存活的 4D 前景高斯。
    """
    if not hasattr(gaussians, '_start_frame') or gaussians._start_frame.numel() == 0:
        return None # 如果是原版 3DGS，不过滤

    active_mask = (gaussians._start_frame <= frame_id) & (gaussians._expire_frame >= frame_id)
    
    if hasattr(gaussians, '_mask_dynamic'):
        # 1 是被硬约束冻结的静态背景 (永远可见)，2/0 是动态点 (受生命周期限制)
        final_mask = (gaussians._mask_dynamic == 1) | ((gaussians._mask_dynamic != 1) & active_mask)
        return final_mask
    return active_mask

@torch.no_grad()
def render_video(dataset: ModelParams, pipe: PipelineParams, args):
    """加载模型，生成平滑轨迹，并渲染输出 MP4 视频"""
    print(f"[渲染器] 正在初始化 {args.model_type} 模型渲染管线...")
    
    # 1. 初始化模型并加载权重
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, 
                              rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    
    checkpoint = args.start_checkpoint
    if not checkpoint:
        # 默认加载训练完成的最后一步
        checkpoint = os.path.join(dataset.model_path, "chkpnt_30000.pth")
    
    print(f"[渲染器] 正在加载 Checkpoint: {checkpoint}")
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, None)
    
    # 2. 加载场景 (借用 Scene 获取训练集的相机分布作为运镜关键帧)
    scene = Scene(dataset, gaussians, load_iteration=first_iter, shuffle=False)
    train_cameras = scene.getTrainCameras()
    
    # 3. 规划运镜轨迹
    print("[渲染器] 正在规划 B-Spline 和 Slerp 平滑运镜轨迹...")
    # 选取极值点作为关键帧 (例如：第0帧，中间帧，最后一帧)
    num_cams = len(train_cameras)
    keyframe_indices = [0, num_cams//4, num_cams//2, int(num_cams*0.75), num_cams-1]
    keyframes = [train_cameras[i] for i in keyframe_indices]
    
    # 生成 300 帧的平滑漫游轨迹
    num_render_frames = 300 
    trajectory = generate_smooth_trajectory(keyframes, num_frames=num_render_frames)
    
    # 4. 渲染视频帧
    print("[渲染器] 开始渲染漫游视频...")
    render_dir = os.path.join(dataset.model_path, "roaming_renders")
    os.makedirs(render_dir, exist_ok=True)
    
    frames_rgb = []
    
    for idx, cam in enumerate(tqdm(trajectory, desc="Rendering Frames")):
        # 获取当前帧的 HybridGS 过滤掩码
        active_mask = get_inference_active_mask(gaussians, getattr(cam, 'uid', 0))
        
        # 前向渲染
        render_pkg = render(cam, gaussians, pipe, background, active_dynamic_mask=active_mask)
        
        # 提取 RGB 图像并转换为 0-255 的 numpy 数组
        rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        img_np = (rendered_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        frames_rgb.append(img_np)
        
        # (可选) 将单帧图像保存到磁盘
        # imageio.imwrite(os.path.join(render_dir, f"{cam.image_name}.png"), img_np)

    # 5. 合成 MP4 视频
    video_path = os.path.join(dataset.model_path, f"{args.model_type}_roaming.mp4")
    print(f"\n[渲染器] 正在编码视频: {video_path}")
    imageio.mimwrite(video_path, frames_rgb, fps=30, quality=8)
    print("[渲染器] 视频漫游渲染圆满完成！")

if __name__ == "__main__":
    parser = ArgumentParser(description="HybridGS 混合高斯漫游视频渲染器")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument("--config", type=str, required=True, help="配置文件的路径")
    parser.add_argument("--model_type", type=str, default="hybrid_gs")
    parser.add_argument("--gaussian_dim", type=int, default=4)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--rot_4d", action="store_true", default=True)
    parser.add_argument("--force_sh_3d", action="store_true", default=True)
    parser.add_argument("--start_checkpoint", type=str, default=None, help="指定 pth 权重路径")
    
    args = parser.parse_args()
    
    # 继承 YAML 配置
    cfg = OmegaConf.load(args.config)
    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            if hasattr(args, key):
                setattr(args, key, host[key])
    for k in cfg.keys():
        recursive_merge(k, cfg)
        
    render_video(lp.extract(args), pp.extract(args), args)