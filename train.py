# 重构的统一执行入口

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import torch
from torch import nn
from utils.loss_utils import l1_loss, ssim, msssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, knn
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, easy_cmap
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import make_grid
import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader

from utils.mesh_utils import GaussianExtractor
from utils.render_utils import generate_path, create_videos

from utils.trainer_4dgs import Trainer4DGS
from utils.trainer_swings import TrainerSWinGS
from utils.trainer_hybrid import TrainerHybrid


def trainer_factory(args, dataset, opt, pipe, testing_iterations, saving_iterations):
    """工厂函数，根据参数动态实例化训练器"""
    if args.model_type == 'baseline_4dgs':
        return Trainer4DGS(dataset, opt, pipe, testing_iterations, saving_iterations, args)
    elif args.model_type == 'swings':
        return TrainerSWinGS(dataset, opt, pipe, testing_iterations, saving_iterations, args)
    elif args.model_type == 'hybrid_gs':
        return TrainerHybrid(dataset, opt, pipe, testing_iterations, saving_iterations, args)
    else:
        raise ValueError(f"未知的模型类型: {args.model_type}")
    
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def validation(dataset, opt, pipe,checkpoint, gaussian_dim, time_duration, rot_4d, force_sh_3d,
               num_pts, num_pts_ratio):
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=gaussian_dim, time_duration=time_duration, 
                              rot_4d=rot_4d, force_sh_3d=force_sh_3d, sh_degree_t=2 if pipe.eval_shfs_4d else 0)
    
    assert checkpoint, "No checkpoint provided for validation"
    scene = Scene(dataset, gaussians, shuffle=False,num_pts=num_pts, num_pts_ratio=num_pts_ratio, time_duration=time_duration)
    
    (model_params, first_iter) = torch.load(checkpoint)
    train_dir = os.path.join(dataset.model_path, 'train', "ours_{}".format(first_iter))
    test_dir = os.path.join(dataset.model_path, 'test', "ours_{}".format(first_iter))
    gaussians.restore(model_params, None)
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)   
    
    #########   1. Validation and Rendering ############

    print("export rendered testing images ...")
    os.makedirs(test_dir, exist_ok=True)
    gaussExtractor.reconstruction(scene.getTestCameras(),test_dir,stage = "validation")
    gaussExtractor.export_image(test_dir,mode = "validation")

    # #########    2. Render Trajectory       ############
    
    # print("rendering trajectory ...")
    # traj_dir = os.path.join(test_dir, 'traj')
    # os.makedirs(traj_dir, exist_ok=True)
    # n_fames = 480
    # cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
    # gaussExtractor.reconstruction(cam_traj, test_dir,stage = "trajectory")
    # gaussExtractor.export_image(traj_dir,mode = "trajectory")
    # create_videos( base_dir =traj_dir,
    #                input_dir=traj_dir, 
    #                out_name='render_traj', 
    #                num_frames=n_fames)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_dict=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        if loss_dict is not None:
            if "Lrigid" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/rigid_loss', loss_dict['Lrigid'].item(), iteration)
            if "Ldepth" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/depth_loss', loss_dict['Ldepth'].item(), iteration)
            if "Ltv" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/tv_loss', loss_dict['Ltv'].item(), iteration)
            if "Lopa" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/opa_loss', loss_dict['Lopa'].item(), iteration)
            if "Lptsopa" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/pts_opa_loss', loss_dict['Lptsopa'].item(), iteration)
            if "Lsmooth" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/smooth_loss', loss_dict['Lsmooth'].item(), iteration)
            if "Llaplacian" in loss_dict:
                tb_writer.add_scalar('train_loss_patches/laplacian_loss', loss_dict['Llaplacian'].item(), iteration)


        tb_writer.add_scalar('gpu/memory_allocated_MB', torch.cuda.memory_allocated() / 1e6, iteration)
        tb_writer.add_scalar('gpu/memory_reserved_MB', torch.cuda.memory_reserved() / 1e6, iteration)

    psnr_test_iter = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        validation_configs = ({'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
                              {'name': 'test', 'cameras' : [scene.getTestCameras()[idx] for idx in range(len(scene.getTestCameras()))]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                msssim_test = 0.0
                for idx, batch_data in enumerate(tqdm(config['cameras'])):
                    gt_image, viewpoint = batch_data
                    gt_image = gt_image.cuda()
                    viewpoint = viewpoint.cuda()
                    
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    
                    depth = easy_cmap(render_pkg['depth'][0])
                    alpha = torch.clamp(render_pkg['alpha'], 0.0, 1.0).repeat(3,1,1)
                    image_4d = torch.clamp(render_pkg["render_4d"], 0.0, 1.0)
                    image_3d = torch.clamp(render_pkg["render_3d"], 0.0, 1.0)

                    if tb_writer and (idx < 5):
                        grid = [gt_image, image, image_4d, image_3d]
                        grid = make_grid(grid, nrow=2)
                        tb_writer.add_images(config['name'] + "_view_{}/gt_vs_render".format(viewpoint.image_name), grid[None], global_step=iteration)
                            
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    msssim_test += msssim(image[None].cpu(), gt_image[None].cpu())
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 
                ssim_test /= len(config['cameras'])     
                msssim_test /= len(config['cameras'])        
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - msssim', msssim_test, iteration)
                if config['name'] == 'test':
                    psnr_test_iter = psnr_test.item()
                    
    torch.cuda.empty_cache()
    return psnr_test_iter


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="混合3D与4D高斯场景重建统一训练框架")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # # 新增模型路由参数
    # parser.add_argument("--model_type", type=str, default="hybrid_gs", 
    #                     choices=["baseline_4dgs", "swings", "hybrid_gs"], 
    #                     help="选择要训练的模型基线")
    # # 预留给 SWinGS 和 HybridGS 的超参数 -> config.yaml文件中若对应项没有被注释，则优先使用config
    # parser.add_argument("--swin_size", type=int, default=50, help="滑动窗口长度")
    # parser.add_argument("--tau_avg", type=float, default=0.01, help="硬约束平均位移阈值")
    # parser.add_argument("--tau_max", type=float, default=0.05, help="硬约束最大瞬时位移阈值")
    # parser.add_argument("--lambda_d", type=float, default=0.1, help="位移收敛软约束惩罚系数")

    parser.add_argument("--config", type=str)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[6_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[6_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = "无效参数，但是删除会影响其他地方的参数解析，暂时保留")
    
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument('--num_pts', type=int, default=100_000)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    parser.add_argument("--val", action="store_true", default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
        
    cfg = OmegaConf.load(args.config)

    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])
    
    for k in cfg.keys():
        recursive_merge(k, cfg)
        
    if args.exhaust_test:
        args.test_iterations = args.test_iterations + [i for i in range(0,args.iterations,500)]
    
    # 系统状态初始化
    setup_seed(args.seed)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # ---------------------------------------------------------
    # 核心路由：使用 OOP 框架替代原有的 training() 
    # ---------------------------------------------------------
    if args.val == False:
        print(f"\n[系统通知] 正在启动训练管线，当前选择的模型架构为: {args.model_type.upper()}")
        # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.start_checkpoint, args.debug_from,
        #         args.gaussian_dim, args.time_duration, args.num_pts, args.num_pts_ratio, args.rot_4d, args.force_sh_3d, args.batch_size)
        # 将原有的 lp.extract(args) 提取出来，传入工厂函数
        trainer = trainer_factory(
            args=args, 
            dataset=lp.extract(args), 
            opt=op.extract(args), 
            pipe=pp.extract(args), 
            testing_iterations=args.test_iterations, 
            saving_iterations=args.save_iterations
        )
        # 启动统一的训练循环
        trainer.train()
    else:
        # 验证模式保持不变
        print("\n[系统通知] 启动验证模式 (Validation)...")
        validation(lp.extract(args), op.extract(args), pp.extract(args),args.start_checkpoint,args.gaussian_dim, 
                   args.time_duration,args.rot_4d, args.force_sh_3d, args.num_pts, args.num_pts_ratio)
        

    print("\nComplete.")
