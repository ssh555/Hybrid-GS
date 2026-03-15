# 用于相机平滑移动
# 文件：utils/camera_trajectory.py
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from scipy.interpolate import CubicSpline
from scene.cameras import Camera
import copy
# [新增导入] 引入 3DGS 投影矩阵计算工具
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

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
            T_list.append(cam.T)  # 它已经是 numpy 数组了，直接 append
        # 3DGS 中的 R 是 W2C 的转置，先转回来，再用 scipy 处理
        if hasattr(cam.R, 'cpu'):
            r_mat = cam.R.cpu().numpy().T
        else:
            r_mat = cam.R.T  # 它已经是 numpy 数组了，直接转置
        R_quats.append(R_scipy.from_matrix(r_mat).as_quat()) # 转为四元数 [x, y, z, w]
        fovx_list.append(cam.FoVx)
        timestamp_list.append(cam.timestamp)
        frame_id_list.append(getattr(cam, 'uid', 0))

    T_list = np.array(T_list)
    fovx_list = np.array(fovx_list)
    timestamp_list = np.array(timestamp_list)
    frame_id_list = np.array(frame_id_list)

    # 1. 样条插值平移向量 (Translation) 和 视场角 (FOV) 及 时间戳
    spline_T = CubicSpline(times, T_list)
    spline_fovx = CubicSpline(times, fovx_list)
    spline_timestamp = CubicSpline(times, timestamp_list)
    
    # 注意：真实帧号 (frame_id) 只能线性插值并取整，用于 HybridGS 生命周期掩码提取
    interp_frame_ids = np.interp(target_times, times, frame_id_list).astype(int)

    # 2. Slerp 插值旋转四元数 (Rotation)
    slerp = R_scipy.from_quat(R_quats)
    spline_R = slerp.as_spline() # scipy 1.10+ 支持直接四元数样条插值

    # 生成平滑轨迹
    smooth_T = spline_T(target_times)
    smooth_R_quats = spline_R(target_times)
    smooth_R_mats = R_scipy.from_quat(smooth_R_quats).as_matrix()
    smooth_fovx = spline_fovx(target_times)
    smooth_timestamp = spline_timestamp(target_times)

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
        
        # ===================================================================
        # [必须新增的修复：手动刷新光栅化引擎依赖的投影矩阵！]
        # 如果不更新这三个矩阵，渲染出的视频将是定格在第一帧的静止画面！
        # ===================================================================
        # 1. 重新计算 World-to-View 矩阵 (W2C)
        new_cam.world_view_transform = torch.tensor(
            getWorld2View2(R_matrix, T_vector, np.array([0.0, 0.0, 0.0]), 1.0)
        ).transpose(0, 1).cuda()
        
        # 2. 重新计算投影矩阵 (考虑可能微变的 FoVx)
        new_cam.projection_matrix = getProjectionMatrix(
            znear=new_cam.znear, zfar=new_cam.zfar, fovX=new_cam.FoVx, fovY=new_cam.FoVy
        ).transpose(0, 1).cuda()
        
        # 3. 重新计算全投影矩阵 (W2C * Proj)
        new_cam.full_proj_transform = (
            new_cam.world_view_transform.unsqueeze(0).bmm(new_cam.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        
        # 4. 更新相机光心坐标
        new_cam.camera_center = new_cam.world_view_transform.inverse()[3, :3]

        trajectory.append(new_cam)

    return trajectory