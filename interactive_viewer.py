import os
import time
import copy
import torch
import numpy as np
import viser
from argparse import ArgumentParser
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render

@torch.no_grad()
def main(dataset: ModelParams, pipe: PipelineParams, args):
    print(f"[引擎] 正在初始化 4DGS 安全监视器...")
    
    # 1. 初始化模型
    bg_color = [1, 1, 1] if dataset.white_background else [0.2, 0.2, 0.2]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    sh_degree = dataset.sh_degree if hasattr(dataset, 'sh_degree') else 3
    gaussians = GaussianModel(sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, 
                              rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    
    # 2. 读取原版场景和真实相机
    scene = Scene(dataset, gaussians, shuffle=False)
    train_cameras = [c[1] if isinstance(c, tuple) else c for c in scene.getTrainCameras()]
    max_frames = len(train_cameras) - 1
    
    # 3. 加载完美权重
    checkpoint = args.start_checkpoint
    print(f"[引擎] 正在加载模型权重: {checkpoint}")
    (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
    gaussians.restore(model_params, None)
    
    # 4. 启动 Viser
    server = viser.ViserServer(port=8080)
    print("\n" + "="*50)
    print(f"🚀 4D 安全监视器启动成功！")
    print(f"👉 请在浏览器中打开: http://localhost:8080")
    print("="*50 + "\n")
    
    # ==========================================
    # 🌟 终极 UI：放弃鼠标乱转，用滑动条控制原版相机！
    # ==========================================
    gui_cam_idx = server.gui.add_slider("🎥 切换视角机位", min=0, max=max_frames, step=1, initial_value=0)
    gui_time_idx = server.gui.add_slider("🎬 4D 时间轴", min=0, max=max_frames, step=1, initial_value=0)
    server.gui.add_markdown("⚠️ **注意**：为了防止矩阵爆炸，当前模式已禁用鼠标自由漫游。请使用上方滑块查看原汁原味的 4D 画面！")

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        # 隐藏碍眼的网格
        client.scene.add_grid("grid", visible=False)

    last_cam_idx = -1
    last_time_idx = -1

    while True:
        clients = server.get_clients()
        if not clients:
            time.sleep(0.1)
            continue
            
        current_cam_idx = int(gui_cam_idx.value)
        current_time_idx = int(gui_time_idx.value)
        
        # 只有当滑动条数值改变时，才重新渲染（极其省显存！）
        if current_cam_idx != last_cam_idx or current_time_idx != last_time_idx:
            for client_id, client in clients.items():
                
                # 提取原汁原味的真实相机，绝不自己瞎算矩阵！
                view_cam = copy.deepcopy(train_cameras[current_cam_idx])
                
                # 注入 4D 时间属性
                view_cam.fid = current_time_idx
                view_cam.time = float(current_time_idx / max(1, max_frames))
                if hasattr(view_cam, 'timestamp'):
                    view_cam.timestamp = view_cam.time
                
                # 执行渲染
                render_pkg = render(view_cam, gaussians, pipe, background)
                rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                
                # 发送给网页
                img_np = (rendered_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                client.scene.set_background_image(img_np, format="jpeg")
                
            last_cam_idx = current_cam_idx
            last_time_idx = current_time_idx
            
        time.sleep(0.01)

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
        
    main(lp.extract(args), pp.extract(args), args)