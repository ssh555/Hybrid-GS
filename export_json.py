import bpy
import json
import math
import os

# 配置与你 render_worker.py 一致
CAM_COUNT = 20
TOTAL_FRAMES = 2901
FPS = 30.0 # 假设 30 帧每秒
OUTPUT_JSON = "/home/ssh555/BiShe/DatasetsGenerate/Scene/Datas/Miku_ShuangXue_Daxi/Full_Dataset/transforms_train.json"

def get_nerf_matrix(camera_obj):
    # 获取 Blender 相机到世界矩阵
    matrix_world = camera_obj.matrix_world.copy()
    # 🌟 核心：Blender (X右, Y前, Z上) 转换为 OpenGL/NeRF (X右, Y上, Z后)
    from mathutils import Matrix
    b2n = Matrix(((1, 0, 0, 0), 
                  (0, 0, 1, 0), 
                  (0, -1, 0, 0), 
                  (0, 0, 0, 1)))
    nerf_matrix = matrix_world @ b2n
    return [list(row) for row in nerf_matrix]

def main():
    print("🚀 开始生成 4DGS 完美坐标 JSON...")
    
    # 获取第一台相机来计算内参
    cam_obj = bpy.data.objects.get("Cam_00")
    if not cam_obj:
        print("❌ 找不到 Cam_00，请确保在运行了相机的 .blend 文件中执行！")
        return
        
    cam_data = cam_obj.data
    w = 1920.0
    h = 1080.0
    
    # 计算焦距 (Focal Length)
    fov_x = cam_data.angle_x
    fov_y = cam_data.angle_y
    fl_x = (w / 2.0) / math.tan(fov_x / 2.0)
    fl_y = (h / 2.0) / math.tan(fov_y / 2.0)
    
    frames_data = []
    
    for i in range(CAM_COUNT):
        cam = bpy.data.objects.get(f"Cam_{i:02d}")
        if not cam: continue
        
        transform_matrix = get_nerf_matrix(cam)
        
        for frame in range(TOTAL_FRAMES + 1):
            # 注意：3DGS 读取时通常会自动加上 .png 后缀，这里 file_path 不加后缀
            frames_data.append({
                "file_path": f"images/cam{i:02d}/img_{frame:04d}",
                "time": frame / FPS,  # 🌟 4DGS 的核心时间戳
                "transform_matrix": transform_matrix
            })
            
    json_data = {
        "w": w, "h": h, 
        "fl_x": fl_x, "fl_y": fl_y, 
        "cx": w / 2.0, "cy": h / 2.0,
        "frames": frames_data
    }
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(json_data, f, indent=2)
        
    print(f"✅ 完美！JSON 已保存至: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()