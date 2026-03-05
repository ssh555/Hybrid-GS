# 用于相机平滑移动
# 文件：utils/camera_trajectory.py
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from scipy.interpolate import CubicSpline
from scene.cameras import Camera
import copy

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
        T_list.append(cam.T.cpu().numpy())
        # 3DGS 中的 R 是 W2C 的转置，先转回来，再用 scipy 处理
        r_mat = cam.R.cpu().numpy().T 
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
        # 写回插值后的位姿
        new_cam.T = torch.tensor(smooth_T[i], dtype=torch.float32, device="cuda")
        # 恢复 3DGS 格式的转置旋转矩阵
        new_cam.R = torch.tensor(smooth_R_mats[i].T, dtype=torch.float32, device="cuda")
        new_cam.FoVx = float(smooth_fovx[i])
        new_cam.timestamp = float(smooth_timestamp[i])
        new_cam.uid = int(interp_frame_ids[i])
        new_cam.image_name = f"render_frame_{i:04d}"
        trajectory.append(new_cam)

    return trajectory