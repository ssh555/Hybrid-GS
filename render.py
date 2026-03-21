# 纯净版：自动分组单视角动态渲染 (严格空间判定版)
import os
import torch
import imageio
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render

@torch.no_grad()
def simple_render(dataset: ModelParams, pipe: PipelineParams, args):
    print(f"[渲染器] 正在初始化 4DGS 单视角分离渲染管线...")
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    sh_degree = dataset.sh_degree if hasattr(dataset, 'sh_degree') else 3
    gaussians = GaussianModel(sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, 
                              rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    
    scene = Scene(dataset, gaussians, shuffle=False)
    train_cameras = [c[1] if isinstance(c, tuple) else c for c in scene.getTrainCameras()]

    checkpoint = args.start_checkpoint or os.path.join(dataset.model_path, "chkpnt_6000.pth")
    print(f"[渲染器] 正在覆盖模型权重: {checkpoint}")
    (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
    gaussians.restore(model_params, None)
    
    # ==========================================
    # 核心逻辑：智能分离出【同一个视角】的所有时间帧
    # ==========================================
    print(f"[渲染器] 数据集共有 {len(train_cameras)} 个样本。正在智能聚类单视角...")
    
    # 我们以第 1 个相机的空间位置为基准 (View 0)
    base_cam = train_cameras[0]
    base_T = base_cam.T.cpu().numpy() if hasattr(base_cam.T, 'cpu') else base_cam.T
    base_R = base_cam.R.cpu().numpy() if hasattr(base_cam.R, 'cpu') else base_cam.R # [新增] 获取基准相机的旋转矩阵
    
    view_0_cameras = []
    
    for cam in train_cameras:
        cam_T = cam.T.cpu().numpy() if hasattr(cam.T, 'cpu') else cam.T
        cam_R = cam.R.cpu().numpy() if hasattr(cam.R, 'cpu') else cam.R # [新增] 获取当前相机的旋转矩阵
        
        # [核心修复] 必须位置(T)和旋转角度(R)都极其接近（误差小于1e-5），才被认定是绝对的同一个静态视角
        if np.allclose(base_T, cam_T, atol=1e-5) and np.allclose(base_R, cam_R, atol=1e-5):
            view_0_cameras.append(cam)
            
    # 按时间戳或帧号(fid)排序，确保时间是顺流的
    view_0_cameras.sort(key=lambda x: getattr(x, 'fid', getattr(x, 'timestamp', 0)))
    
    print(f"[渲染器] 成功提取到基准视角的 {len(view_0_cameras)} 帧连续画面！")
    
    # ==========================================
    # 开始渲染并保存
    # ==========================================
    render_dir = os.path.join(dataset.model_path, "single_view_renders")
    os.makedirs(render_dir, exist_ok=True)
    frames_rgb = []
    
    for idx, cam in enumerate(tqdm(view_0_cameras, desc="Rendering Sequence")):
        render_pkg = render(cam, gaussians, pipe, background)
        rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        img_np = (rendered_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # 使用相机自带的 image_name (包含视角和帧号信息) 进行标注保存！
        # cam_name = getattr(cam, 'image_name', f"frame_{idx:03d}")
        # imageio.imwrite(os.path.join(render_dir, f"{cam_name}.png"), img_np)
        frames_rgb.append(img_np)

    video_path = os.path.join(dataset.model_path, "single_view_reconstruction.mp4")
    print(f"\n[渲染器] 正在合成当前视角的动态视频: {video_path}")
    imageio.mimwrite(video_path, frames_rgb, fps=30, quality=8)
    print(f"[渲染器] 圆满完成！请去 {render_dir} 文件夹查看带名称标注的序列帧！")

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
            for k in host[key].keys(): recursive_merge(k, host[key])
        elif hasattr(args, key): setattr(args, key, host[key])
    for k in cfg.keys(): recursive_merge(k, cfg)
        
    simple_render(lp.extract(args), pp.extract(args), args)