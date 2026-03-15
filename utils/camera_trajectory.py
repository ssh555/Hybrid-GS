# 用于相机平滑移动
# 文件：utils/camera_trajectory.py
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from scipy.interpolate import CubicSpline
from scene.cameras import Camera
import copy
# 引入 3DGS 投影矩阵计算工具
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scipy.spatial.transform import Slerp

def generate_smooth_trajectory(keyframes, num_frames=300):
    """
    根据离散的关键帧相机，生成一条平滑的连续相机漫游轨迹。
    平移(T)使用 B样条/三次样条插值，旋转(R)使用四元数球面线性插值(Slerp)。
    """
    if len(keyframes) < 2:
        return keyframes

    # 提取关键帧的位姿、内参和时间戳
    times = np.linspace(0, 1, len(keyframes))
    target_times = np.linspace(0, 1, num_frames)

    T_list = []
    R_quats = []
    fovx_list = []
    timestamp_list = []
    frame_id_list = []

    for cam in keyframes:
        if hasattr(cam.T, 'cpu'):
            T_list.append(cam.T.cpu().numpy())
        else:
            T_list.append(cam.T)
            
        if hasattr(cam.R, 'cpu'):
            r_mat = cam.R.cpu().numpy().T
        else:
            r_mat = cam.R.T 
            
        R_quats.append(R_scipy.from_matrix(r_mat).as_quat())
        fovx_list.append(cam.FoVx)
        timestamp_list.append(cam.timestamp)
        frame_id_list.append(getattr(cam, 'uid', 0))

    T_list = np.array(T_list)
    fovx_list = np.array(fovx_list)
    timestamp_list = np.array(timestamp_list)
    frame_id_list = np.array(frame_id_list)

    # 1. 样条插值平移向量 (Translation) 和 视场角 (FOV) 及 时间戳
    spline_T = CubicSpline(times, T_list)
    smooth_T = spline_T(target_times)  # [修复] 加上了遗漏的平移计算
    
    spline_fovx = CubicSpline(times, fovx_list)
    smooth_fovx = spline_fovx(target_times)
    
    spline_timestamp = CubicSpline(times, timestamp_list)
    smooth_timestamp = spline_timestamp(target_times)
    
    interp_frame_ids = np.interp(target_times, times, frame_id_list).astype(int)

    # 2. Slerp 插值旋转四元数 (Rotation)
    # [修复] 彻底理顺 Slerp 插值逻辑，不再重复装箱拆箱
    base_rotations = R_scipy.from_quat(R_quats)
    real_interpolator = Slerp(times, base_rotations) 
    smooth_rotations = real_interpolator(target_times)
    smooth_R_mats = smooth_rotations.as_matrix()

    trajectory = []
    base_cam = keyframes[0] # 用第一个相机作为模板

    for i in range(num_frames):
        new_cam = copy.deepcopy(base_cam)
        
        R_matrix = smooth_R_mats[i].T
        T_vector = smooth_T[i]
        
        # 写回基础位姿
        new_cam.R = torch.tensor(R_matrix, dtype=torch.float32, device="cuda")
        new_cam.T = torch.tensor(T_vector, dtype=torch.float32, device="cuda")
        new_cam.FoVx = float(smooth_fovx[i])
        new_cam.timestamp = float(smooth_timestamp[i])
        new_cam.uid = int(interp_frame_ids[i])
        new_cam.image_name = f"render_frame_{i:04d}"
        
        # 重新计算渲染引擎依赖的底层矩阵
        new_cam.world_view_transform = torch.tensor(
            getWorld2View2(R_matrix, T_vector, np.array([0.0, 0.0, 0.0]), 1.0)
        ).transpose(0, 1).cuda()
        
        new_cam.projection_matrix = getProjectionMatrix(
            znear=new_cam.znear, zfar=new_cam.zfar, fovX=new_cam.FoVx, fovY=new_cam.FoVy
        ).transpose(0, 1).cuda()
        
        new_cam.full_proj_transform = (
            new_cam.world_view_transform.unsqueeze(0).bmm(new_cam.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        
        new_cam.camera_center = new_cam.world_view_transform.inverse()[3, :3]

        trajectory.append(new_cam)

    return trajectory