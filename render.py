# 纯净版：自动分组单视角动态渲染 (严格空间判定版) + SWinGS 超级模型支持
import os
import json
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

from utils.camera_utils import get_camera_metadata

@torch.no_grad()
def simple_render(dataset: ModelParams, pipe: PipelineParams, args):
    print(f"[渲染器] 正在初始化 4DGS 单视角分离渲染管线...")
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    sh_degree = dataset.sh_degree if hasattr(dataset, 'sh_degree') else 3
    gaussians = GaussianModel(sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, 
                              rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    
    scene = Scene(dataset, gaussians, shuffle=False)

    checkpoint = args.start_checkpoint or os.path.join(dataset.model_path, "chkpnt_6000.pth")
    print(f"[渲染器] 正在读取模型权重: {checkpoint}")
    
    # 【核心修改 1：支持读取超级大模型字典】
    checkpoint_data, first_iter = torch.load(checkpoint, weights_only=False)
    
    is_swings = isinstance(checkpoint_data, dict) and checkpoint_data.get("is_swings_sequence", False)
    
    if is_swings:
        print("[渲染器] 🚀 检测到 SWinGS 长序列超级大模型！将根据时间轴动态加载基底！")
        window_blocks = checkpoint_data["window_blocks"]
        models_dict = checkpoint_data["models"]
        current_loaded_win_idx = -1
        # [新增] 获取主 checkpoint 所在的绝对目录，用于拼接后续子路径
        base_model_dir = os.path.dirname(os.path.abspath(args.start_checkpoint))
    else:
        print("[渲染器] 📌 检测到传统单体模型，正在直接恢复权重...")
        gaussians.restore(checkpoint_data, None)
    
    # ==========================================
    # 核心逻辑：智能分离出【同一个视角】的所有时间帧
    # ==========================================
    all_cams, max_frames, max_cams = get_camera_metadata(scene, dataset.model_path)
    
    # 选定一个基准视角进行渲染（例如正中心视角）
    target_v_idx = max_cams // 2
    view_0_cameras = all_cams[target_v_idx * max_frames : (target_v_idx + 1) * max_frames]
    
    # 按时间或 fid 排序
    view_0_cameras.sort(key=lambda x: getattr(x, 'fid', getattr(x, 'timestamp', 0)))
    
    print(f"[渲染器] 成功提取到基准视角的 {len(view_0_cameras)} 帧连续画面！")
    
    # ==========================================
    # 开始渲染并保存
    # ==========================================
    render_dir = os.path.join(dataset.model_path, "single_view_renders")
    os.makedirs(render_dir, exist_ok=True)
    frames_rgb = []
    
    for idx, cam in enumerate(tqdm(view_0_cameras, desc="Rendering Sequence")):
        
        # 【核心修改 2：帧级动态参数切换逻辑】
        if is_swings:
            # 1. 解析当前帧是第几帧
            try:
                frame_id = int(cam.image_name.split('_')[-1])
            except:
                frame_id = getattr(cam, 'fid', idx)
            
            # 2. 找到当前帧所属的时间窗口
            target_win_idx = 0
            for w_idx, (w_start, w_end) in enumerate(window_blocks):
                if w_start <= frame_id <= w_end:
                    target_win_idx = w_idx
                    break
                    
            # 3. 如果当前帧进入了新的窗口，立刻切换高斯模型参数（毫秒级切换，不影响渲染速度）
            if target_win_idx != current_loaded_win_idx:
                # 🚀 【按需动态加载核心】
                rel_path = models_dict[target_win_idx]
                abs_path = os.path.join(base_model_dir, rel_path)

                # print(f"[渲染引擎] 正在换弹：加载窗口 {target_win_idx} 数据 -> {rel_path}")
                win_data_tuple = torch.load(abs_path, weights_only=False)

                gaussians.restore(win_data_tuple, None)
                current_loaded_win_idx = target_win_idx

        # 4. 执行渲染
        render_pkg = render(cam, gaussians, pipe, background)
        rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        img_np = (rendered_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        save_path = os.path.join(render_dir, f"{cam.image_name}.png")
        imageio.imwrite(save_path, img_np)
        # ==========================================================

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
    parser.add_argument("--start_checkpoint", type=str, default = None) # 修改了默认值，更规范
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for k in host[key].keys(): recursive_merge(k, host[key])
        elif hasattr(args, key): setattr(args, key, host[key])
    for k in cfg.keys(): recursive_merge(k, cfg)
        
    simple_render(lp.extract(args), pp.extract(args), args)