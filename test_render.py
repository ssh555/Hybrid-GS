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
def simple_render(dataset, pipe, args):
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    sh_degree = dataset.sh_degree if hasattr(dataset, 'sh_degree') else 3
    gaussians = GaussianModel(sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, 
                              rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    
    scene = Scene(dataset, gaussians, shuffle=False)
    train_cameras = scene.getTrainCameras() # 直接拿你那 20 张原图的相机！

    print(f"正在加载权重: {args.start_checkpoint}")
    (model_params, first_iter) = torch.load(args.start_checkpoint, weights_only=False)
    gaussians.restore(model_params, None)
    
    out_dir = os.path.join(dataset.model_path, "raw_train_renders")
    os.makedirs(out_dir, exist_ok=True)
    
    print("正在直接渲染原训练集视角...")
    for idx, cam_tuple in enumerate(tqdm(train_cameras)):
        cam = cam_tuple[1] # 获取相机对象
        
        # 直接拿相机去渲染，不加任何插值魔法！
        render_pkg = render(cam, gaussians, pipe, background)
        rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        img_np = (rendered_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        imageio.imwrite(os.path.join(out_dir, f"frame_{idx:03d}.png"), img_np)
    
    print(f"渲染完毕！请去 {out_dir} 文件夹查看图片！")

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