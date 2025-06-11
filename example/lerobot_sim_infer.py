import time
import numpy as np
import random
import os
import pandas as pd
import json
from pathlib import Path
import torch
from genesis import gs
import imageio

from lerobot.common.policies.act.modeling_act import ACTPolicy


# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"


# 加载预训练策略
pretrained_policy_path = Path("/path/to/checkpoints/last/pretrained_model")
policy = ACTPolicy.from_pretrained(pretrained_policy_path)

# 创建输出目录
output_directory = Path("outputs/eval/lerobot_sim_infer")
output_directory.mkdir(parents=True, exist_ok=True)

# 初始化仿真环境
def init_simulation():
    gs.init(backend=gs.gpu)
    
    # 场景设置
    dt = 0.01
    scene = gs.Scene(show_viewer=True, sim_options=gs.options.SimOptions(dt=dt),
                    rigid_options=gs.options.RigidOptions(
                        dt=dt,
                        gravity=(0, 0, -10.0),
                        constraint_solver=gs.constraint_solver.Newton,
                        enable_collision=True,
                        enable_joint_limit=True,
                        enable_self_collision=True,
                    ),
                    vis_options=gs.options.VisOptions(
                        show_world_frame=False,
                    ),
    )
    
    # 添加平面和光源
    plane = scene.add_entity(gs.morphs.Plane())
    light = scene.add_light(gs.morphs.Primitive())
    # 圆柱体材质设置
    cylinder_surface = gs.surfaces.Rough(
        color=(0.5, 0.0, 0.5),  # 灰色
        roughness=0.7,          # 增加粗糙度，提高摩擦力
        metallic=1,           # 非金属材质
        emissive=None,          # 无自发光
        ior=1.5,                # 折射率
        vis_mode="visual",      # 可视化模式
        smooth=True,            # 平滑处理
        double_sided=False,     # 不需要双面渲染
    )
    # 盒子材质设置
    box_surface = gs.surfaces.Rough(
        color=(0.54, 0.27, 0.07),  # 褐色
        roughness=0.8,             # 增加粗糙度，提高摩擦力
        metallic=0.0,              # 非金属材质
        emissive=None,             # 无自发光
        ior=1.5,                   # 折射率
        vis_mode="visual",         # 可视化模式
        smooth=True,               # 平滑处理
        double_sided=False,        # 不需要双面渲染
    )
    # 添加桌子
    tablelegs = scene.add_entity(morph= gs.morphs.Mesh(file= 'lerobot-kinematics/examples/assets/table/tablelegs.obj', pos=(0, 0, 0), euler=(0, 0, 90), fixed= True), surface= gs.surfaces.Default(roughness= 0.7, diffuse_texture= gs.textures.ImageTexture(image_path= 'lerobot-kinematics/examples/assets/table/small_meta_table_diffuse.png')))
    tabletop = scene.add_entity( morph= gs.morphs.Mesh(file= 'lerobot-kinematics/examples/assets/table/tabletop.obj', pos=(0, 0, 0), euler=(0, 0, 90), fixed= True), surface= gs.surfaces.Default(roughness=0.7,  diffuse_texture= gs.textures.ImageTexture(image_path= 'lerobot-kinematics/examples/assets/table/small_meta_table_diffuse.png')))

    # 添加一个圆柱体
    cylinder = scene.add_entity(
        morph=gs.morphs.Cylinder(
            radius=0.013,  # 半径 1.3cm（直径 2.6cm）
            height=0.06,  # 高度 6cm
            pos=(0.007, -0.02, 0.77),  # 平躺放置在桌子上，高度为桌子高度 + 半径
            euler=(90, 0, 0),  # 绕 x 轴旋转 90 度，使圆柱体平躺
            
        ),
        # surface=gs.surfaces.Rough(
        #     color=(0.5, 0.0, 0.5),  # 灰色
        #     vis_mode="visual",
        # ),
        surface=cylinder_surface,
        #collider=True,  # 启用碰撞器
    )

    # 添加木板（总长度15cm，宽度0.02cm，高度2cm）
    board = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.002, 0.15, 0.03),  # 尺寸 15cm x 0.02cm x 2cm
            pos=(0.05, -0.125, 0.76),  # 位置：8.5cm，12cm，高度为桌子高度 + 1cm
            fixed=True,  # 设置为固定
        ),
        surface=gs.surfaces.Rough(
            color=(0.65, 0.45, 0.25),  # 木板颜色
            vis_mode="visual",
        ),
    )

    board2 = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.002, 0.15, 0.03),  # 尺寸 15cm x 0.02cm x 2cm
            pos=(0.20, -0.125, 0.76),  # 位置：8.5cm，12cm，高度为桌子高度 + 1cm
            fixed=True,  # 设置为固定
        ),
        surface=gs.surfaces.Rough(
            color=(0.65, 0.45, 0.25),  # 木板颜色
            vis_mode="visual",
        ),
    )
    board3 = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.15, 0.002, 0.03),  # 尺寸 15cm x 0.02cm x 2cm
            pos=(0.125, -0.05, 0.76),  # 位置：8.5cm，12cm，高度为桌子高度 + 1cm
            fixed=True,  # 设置为固定
        ),
        surface=gs.surfaces.Rough(
            color=(0.65, 0.45, 0.25),  # 木板颜色
            vis_mode="visual",
        ),
    )

    board4 = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.15, 0.002, 0.03),  # 尺寸 15cm x 0.02cm x 2cm
            pos=(0.125, -0.20, 0.76),  # 位置：8.5cm，12cm，高度为桌子高度 + 1cm
            fixed=True,  # 设置为固定
        ),
        surface=gs.surfaces.Rough(
            color=(0.65, 0.45, 0.25),  # 木板颜色
            vis_mode="visual",
        ),
    )


    # 添加褐色镂空盒子（中心点在8.5cm，12.5cm，长宽15cm，边高度2cm，内部高度0.1cm）
    # 底板
    box_base = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.149, 0.149, 0.02),  # 底板尺寸 15cm x 15cm x 1mm
            pos=(0.125, -0.125, 0.752),  # 中心点位置
            fixed=True,
        ),
        # surface=gs.surfaces.Rough(
        #     color=(0.54, 0.27, 0.07),  # 褐色
        #     vis_mode="visual",
        # ),
        surface=box_surface,
    )

    black_region_pos = (0.0, 0.0, 0.75)  # 区域的中心位置
    black_region_size = (0.4, 0.4, 0.002)  # 区域的尺寸（40cm x 40cm）

    # 创建黑色材质
    black_surface = gs.surfaces.Rough(
        color=(0.0, 0.0, 0.0),  # 黑色
        vis_mode="visual",
    )

    # 添加黑色区域到桌子
    black_region = scene.add_entity(
        morph=gs.morphs.Box(
            size=black_region_size,
            pos=black_region_pos,
            fixed=True,
        ),
        surface=black_surface,
    )

    # 添加机械臂
    so_100_left = scene.add_entity(gs.morphs.MJCF(file='lerobot-kinematics/examples/so_100.xml', pos=(0.30, 0, 0.75), euler=(0, 0, -90)))
    
    # 添加桌子和其他物体

    # 添加相机
    cam1 = scene.add_camera(res=(640, 480), pos=(0.035, 0.08, 1.4), lookat=(0.036, 0.08, 0), fov=45, GUI=True)
    cam2 = scene.add_camera(res=(640, 480), pos=(0.075, -0.52, 1.02), lookat=(0.14, 0.20, 0.75), fov=45, GUI=True)
    
    # 构建场景
    scene.build()
    
    # 设置机械臂初始位姿
    joint_names = ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']
    left_joint_idx = [so_100_left.get_joint(name).dof_idx_local for name in joint_names]
    init_pos = np.array([0, -3.14, 3.14, 0.817, 0, -0.157])
    so_100_left.set_dofs_position(init_pos, left_joint_idx)
    
    # 设置PD控制参数
    kp = np.array([2500, 2500, 1500, 1500, 800, 100])
    kv = np.array([250, 250, 150, 150, 80, 10])
    force_upper = np.array([50, 50, 50, 50, 12, 100])
    force_lower = np.array([-50, -50, -50, -50, -12, -100])
    so_100_left.set_dofs_kp(kp=kp, dofs_idx_local=left_joint_idx)
    so_100_left.set_dofs_kv(kv=kv, dofs_idx_local=left_joint_idx)
    so_100_left.set_dofs_force_range(lower=force_lower, upper=force_upper, dofs_idx_local=left_joint_idx)
    
    return scene, so_100_left, left_joint_idx, cam1, cam2

# 获取观测数据
def get_observation(cam1, cam2, so_100_left, left_joint_idx):
    # 获取机械臂的关节位置
    arm_qpos = so_100_left.get_dofs_position(left_joint_idx)
    arm_qpos = torch.rad2deg(arm_qpos)

    # 获取相机图像
    cam1_image = cam1.render()
    cam2_image = cam2.render()
    
    # 将图像数据转换为模型所需的格式
    cam1_image = process_image(cam1_image)
    cam2_image = process_image(cam2_image)

    arm_qpos = arm_qpos.unsqueeze(0)  # 添加批次维度

    # 将观测数据组合成模型所需的格式
    observation = {
        "observation.state": arm_qpos.to(torch.float32).to(device),
        "observation.images.laptop": cam1_image.to(device),
        "observation.images.phone": cam2_image.to(device)
    }
    
    return observation

# 处理图像数据
def process_image(image):
    array_data = image[0]
    image = array_data.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    return image


# 应用动作到机械臂
def apply_action(action, so_100_left, left_joint_idx):
    numpy_action = action.squeeze(0).to("cpu").numpy()
    so_100_left.control_dofs_position(numpy_action, left_joint_idx)

# 重置环境
def reset_environment(scene, so_100_left, left_joint_idx, cam1, cam2):
    # 重置机械臂
    global i
    init_pos = np.array([0, -3.14, 3.14, 0.817, 0, -0.157])
    so_100_left.set_dofs_position(init_pos, left_joint_idx)

    # 重置圆柱体位置
    #cylinder = scene.get_entity_by_name("cylinder")
    # x = random.uniform(0, 0.02)
    # y = random.uniform(0, 0.02)
    # scene.cylinder.set_pos((x, y, 0.77), zero_velocity=True)
    
    # 重新开始录制相机
    if(i != 0):
        cam1.stop_recording()
        cam2.stop_recording()
        cam1.start_recording()
        cam2.start_recording()
    else:
        cam1.start_recording()
        cam2.start_recording()
        i=1
i = 0
# 主推理循环
def main_inference(num_steps=1000):
    # 初始化仿真环境
    scene, so_100_left, left_joint_idx, cam1, cam2 = init_simulation()
    
    # 准备录制视频
    frames = []
    fps = 30  # 假设视频帧率为30fps
    
    # 重置环境
    reset_environment(scene, so_100_left, left_joint_idx, cam1, cam2)
    
    # 推理循环
    for step in range(num_steps):
        # 获取当前观测数据
    
        observation = get_observation(cam1, cam2, so_100_left, left_joint_idx)
        
        # 使用模型生成动作
        with torch.no_grad():
            action = policy.select_action(observation)
        print(action)
        
        action = torch.deg2rad(action)
        if(action[0,-1]<0.2):
            action[0,-1]=0.1
        print(action)
        # 应用动作到机械臂
        apply_action(action, so_100_left, left_joint_idx)
        
        # 执行仿真步
        scene.step()
        
        # 渲染相机视图并保存帧
        cam1.render()
        cam2.render()
        #frame = cam1.get_recording_frame()
        #if frame is not None:
        #    frames.append(frame)
        
        # 打印当前进度
        print(f"Step {step + 1}/{num_steps} completed")
    
    # 保存视频
    #video_path = output_directory / "lerobot_inference.mp4"
    #imageio.mimsave(str(video_path), np.stack(frames), fps=fps)
    #print(f"Video saved to {video_path}")

if __name__ == "__main__":
    main_inference()