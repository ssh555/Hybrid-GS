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
import json
import os
import torch

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal


WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height# cam_info.image.size

    if args.resolution in [1, 2, 3, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        scale = resolution_scale * args.resolution
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    try:
        cx = cam_info.cx / scale
        cy = cam_info.cy / scale
        fl_y = cam_info.fl_y / scale
        fl_x = cam_info.fl_x / scale
    except:
        cx,cy,fl_x,fl_y = -1, -1, -1, -1
        cameradirect = cam_info.hpdirecitons
        camerapose = cam_info.pose 
     
        if camerapose is not None:
            rays_o, rays_d = 1, cameradirect
        else :
            rays_o = None
            rays_d = None

    
    loaded_mask = None
    if not args.dataloader:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        gt_image = resized_image_rgb[:3, ...]

        if resized_image_rgb.shape[0] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]
    else:
        gt_image = cam_info.image
    
    if cam_info.depth is not None:
        depth = PILtoTorch(cam_info.depth, resolution) * 255 / 10000
    else:
        depth = None

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                image=gt_image, gt_alpha_mask=loaded_mask,
                image_name=cam_info.image_name, uid=id, data_device=args.data_device, 
                timestamp=cam_info.timestamp,
                cx=cx, cy=cy, fl_x=fl_x, fl_y=fl_y, depth=depth, resolution=resolution, image_path=cam_info.image_path,
                meta_only=args.dataloader,cxr=cam_info.cxr,cyr=cam_info.cyr, far=cam_info.far
                )



def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry



def get_camera_metadata(scene, dataset_path):
    """
    获取相机元数据。如果存在缓存则直接读取，否则进行一次性解析并保存。
    """
    cache_path = os.path.join(dataset_path, "camera_structure_cache.json")
    
    # 1. 获取原始相机列表 (这一步 scene 内部会有一次 IO，但无法完全避免)
    train_cams = [c[1] if isinstance(c, tuple) else c for c in scene.getTrainCameras()]
    total_count = len(train_cams)

    # 2. 检查缓存是否存在且有效
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                meta = json.load(f)
            if meta.get("total_count") == total_count:
                print(f"[缓存] 成功加载相机结构: {meta['max_cams']} 视角 x {meta['max_frames']} 帧")
                return train_cams, meta['max_frames'], meta['max_cams']
        except Exception:
            pass

    # 3. 缓存失效或不存在，执行解析逻辑
    print(f"[解析] 正在分析 {total_count} 个相机的时空结构，请稍候...")
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
    
    max_cams = total_count // max_frames
    
    # 保存结果到 JSON
    with open(cache_path, "w") as f:
        json.dump({
            "max_frames": max_frames,
            "max_cams": max_cams,
            "total_count": total_count
        }, f)
    
    return train_cams, max_frames, max_cams