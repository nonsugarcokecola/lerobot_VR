import time
import numpy as np
import random
import os
import argparse
import genesis as gs
import pandas as pd
import json
from pathlib import Path
import torch

from lerobot.common.policies.act.modeling_act import ACTPolicy
data = []
# image_data = {}
# image_data['observation.images.laptop'] = []
# image_data['observation.images.phone'] = [],
frame_index=0
episode_index=0
index=0
count=0

gs.init(backend=gs.gpu)
def float_to_int32(data):
    if isinstance(data, dict):
        return {k: float_to_int32(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [float_to_int32(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.astype(np.int32).tolist()
    elif isinstance(data, float):
        return int(np.int32(data))
    else:
        return data

def reset_robot():
    init_pos = np.array([0, -3.14, 3.14, 0.817, 0, -0.157])
    so_100_left.set_dofs_position(init_pos, left_joint_idx)
    scene.step()

def return_to_initial_position():
    init_pos = np.array([0, -3.14, 3.14, 0.817, 0, -0.157])
    move_to_target(
        target_pos=np.array(so_100_left.get_link('Fixed_Jaw').get_pos().cpu()),
        target_quat=np.array([0.707, 0.707, 0, 0]),
        steps=200,
        effector=so_100_left.get_link('Fixed_Jaw'),
        arm=so_100_left,
        joint_idx=left_joint_idx,
        jaw_open=False
    )
    so_100_left.set_dofs_position(init_pos, left_joint_idx)
    scene.step()
# 处理图像uint8输入
def process_image(image):
    array_data = image[0]
    image = array_data.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0)

    return image

def calculate_image_stats(image_data, index):
    """
    计算图像数据的统计值（最小值、最大值、均值、标准差）。

    参数:
        image_data (numpy.ndarray): 输入的图像数据，形状为 (帧数, 1, 通道数, 高度, 宽度)

    返回:
        dict: 包含统计值的字典，格式如下：
        {
            "observation.images.phone": {
                "min": [[[min_channel_0]], [[min_channel_1]], [[min_channel_2]]],
                "max": [[[max_channel_0]], [[max_channel_1]], [[max_channel_2]]],
                "mean": [[[mean_channel_0]], [[mean_channel_1]], [[mean_channel_2]]],
                "std": [[[std_channel_0]], [[std_channel_1]], [[std_channel_2]]],
                "count": [总帧数]
            }
        }
    """
    # 确保输入是 NumPy 数组
    if not isinstance(image_data, np.ndarray):
        image_data = np.array(image_data)
    
    # 计算每个通道的最小值
    min_values = np.min(image_data, axis=(0, 1, 3, 4)).tolist()
    
    # 计算每个通道的最大值
    max_values = np.max(image_data, axis=(0, 1, 3, 4)).tolist()
    
    # 计算每个通道的均值
    mean_values = np.mean(image_data, axis=(0, 1, 3, 4)).tolist()
    
    # 计算每个通道的标准差
    std_values = np.std(image_data, axis=(0, 1, 3, 4)).tolist()
    
    # 计算总帧数
    #count = image_data.shape[0]
    count = [163]
    return [min_values,  max_values, mean_values, std_values, count]

# 基础场景
dt = 0.01
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
scene = gs.Scene(show_viewer= True, sim_options= gs.options.SimOptions(dt = dt),
                rigid_options= gs.options.RigidOptions(
                dt= dt,
                gravity=(0, 0, -10.0),
                constraint_solver= gs.constraint_solver.Newton,
                enable_collision= True,
                enable_joint_limit= True,
                enable_self_collision= True,
            ),
                vis_options = gs.options.VisOptions(
                show_world_frame = False, # 显示原点坐标系
    ),)
plane = scene.add_entity(gs.morphs.Plane())
light = scene.add_light(gs.morphs.Primitive()) 

#添加机械臂

so_100_left = scene.add_entity(gs.morphs.MJCF(file= 'lerobot-kinematics/examples/so_100.xml', pos=(0.30, 0, 0.75), euler= (0, 0, -90)))

# 添加桌子
tablelegs = scene.add_entity(morph= gs.morphs.Mesh(file= 'lerobot-kinematics/examples/assets/table/tablelegs.obj', pos=(0, 0, 0), euler=(0, 0, 90), fixed= True), surface= gs.surfaces.Default(roughness= 0.7, diffuse_texture= gs.textures.ImageTexture(image_path= 'lerobot-kinematics/examples/assets/table/small_meta_table_diffuse.png')))
tabletop = scene.add_entity( morph= gs.morphs.Mesh(file= 'lerobot-kinematics/examples/assets/table/tabletop.obj', pos=(0, 0, 0), euler=(0, 0, 90), fixed= True), surface= gs.surfaces.Default(roughness=0.7,  diffuse_texture= gs.textures.ImageTexture(image_path= 'lerobot-kinematics/examples/assets/table/small_meta_table_diffuse.png')))

# 添加一个圆柱体
cylinder = scene.add_entity(
    morph=gs.morphs.Cylinder(
        radius=0.013,  # 半径 1.3cm（直径 2.6cm）
        height=0.06,  # 高度 6cm
        pos=(0.0, 0.0, 0.77),  # 平躺放置在桌子上，高度为桌子高度 + 半径
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

cam1 = scene.add_camera(
    res    = (640, 480),
    pos    = (0.035, 0.08, 1.4),
    lookat = (0.036, 0.08, 0), 
    fov    = 45,
   
    GUI    = True
)
cam2 = scene.add_camera(
    res    = (640, 480),
    pos    = (0.075, -0.52, 1.02),
    lookat = (0.14, 0.20, 0.75), 
    fov    = 45,
   
    GUI    = True
)
#cam1 laptop
#cam2 phone

# 场景构建
scene.build()

# cam1.start_recording()
# cam2.start_recording()
#机械臂关节名称，从机械臂xml文件获取
joint_names = [
    'Rotation',
    'Pitch',
    'Elbow',
    'Wrist_Pitch',
    'Wrist_Roll',
    'Jaw',
]
# print(red_square.get_mass())
#设置机械臂初始位姿
left_joint_idx = [so_100_left.get_joint(name).dof_idx_local for name in joint_names]


init_pos = np.array([0, -3.14, 3.14, 0.817, 0, -0.157])
so_100_left.set_dofs_position(init_pos, left_joint_idx)



#PD控制
#机械臂PD控制参数和力、力矩范围限制
kp = np.array([2500, 2500, 1500, 1500, 800, 100])
kv = np.array([250, 250, 150, 150, 80, 10])
force_upper = np.array([50, 50, 50, 50, 12, 100])
force_lower = np.array([-50, -50, -50, -50, -12, -100])
#左臂
so_100_left.set_dofs_kp(kp= kp, dofs_idx_local= left_joint_idx)
so_100_left.set_dofs_kv(kv= kv, dofs_idx_local= left_joint_idx)
so_100_left.set_dofs_force_range(lower= force_lower, upper= force_upper, dofs_idx_local= left_joint_idx)


#逆运动学控制
so_100_left.control_dofs_position(np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), left_joint_idx)
scene.step()
left_end_effector = so_100_left.get_link('Fixed_Jaw')
left_trajectory = []


def randomize_cylinder_position():
    """随机化圆柱体位置并重置速度"""
    x = random.uniform(0, 0.015)  # 随机 x 坐标
    y = random.uniform(-0.05, 0.05)  # 随机 y 坐标
    cylinder.set_pos((x, y, 0.77), zero_velocity=True)  # 更新圆柱体位置并重置速度
    scene.step()
    

def collect_data(action_qpos, observed_qpos):
    """采集数据"""
    # cylinder_pos = cylinder.get_pos().cpu().numpy()
    # cylinder_force = cylinder.get_dofs_force()
    # left_force = so_100_left.get_dofs_force(left_joint_idx)
    global frame_index, index
    action_qpos_degrees = np.degrees(action_qpos.cpu().numpy())
    observed_qpos_degrees = np.degrees(observed_qpos.cpu().numpy())
    #print(action_qpos)
    # print(action_qpos_degrees)
    action_qpos_degrees[0] = -action_qpos_degrees[0]
    action_qpos_degrees[1] = -action_qpos_degrees[1]
    action_qpos_degrees[4] = -action_qpos_degrees[4]
    observed_qpos_degrees[0] = -observed_qpos_degrees[0]
    observed_qpos_degrees[1] = -observed_qpos_degrees[1]
    observed_qpos_degrees[4] = -observed_qpos_degrees[4]
    # print(action_qpos_degrees)
    #cam1_image = cam1.get_color_image().cpu().numpy()  # 假设相机的API支持获取图像数据
    #cam2_image = cam2.get_color_image().cpu().numpy()
    laptop = process_image(cam1.render())
    phone = process_image(cam2.render())
    data.append({
        'action_qpos': action_qpos_degrees,  # 保存action的qpos
        'observation.state': observed_qpos_degrees,  # 保存观测到的qpos
        'observation.images.laptop': laptop,
        'observation.images.phone': phone,
        'timestamp':frame_index/30,
        'frame_index':frame_index,
        'episode_index':episode_index,
        'index':index,
        'task_index':0,
    })
    # image_data['observation.images.laptop'].append(laptop),
    # image_data['observation.images.phone'].append(phone),
    frame_index = frame_index + 1
    index = index + 1
def move_to_target(target_pos, target_quat, steps, effector, arm, joint_idx, jaw_open=True):
    """控制机械臂移动到目标位置和姿态"""
   
    steps = int(steps/2)
    for i in range(steps):
        cur_pos = np.array(effector.get_pos().cpu())
        cur_quat = np.array(effector.get_quat().cpu())
        t_frac = i / steps
        next_pos = cur_pos + (target_pos - cur_pos) * t_frac
        next_quat = cur_quat + (target_quat - cur_quat) * t_frac
        next_qpos = arm.inverse_kinematics(
            link=effector,
            pos=next_pos,
            quat=next_quat
        )
        if jaw_open:
            next_qpos[-1] = 1.5  # 设置夹爪为打开状态
        else:
            next_qpos[-1] = 0  # 设置夹爪为闭合状态
        arm.control_dofs_position(next_qpos, joint_idx)
        scene.step()

        # 获取action的qpos（即目标qpos）
        action_qpos = next_qpos
        # 获取观测到的qpos
        observed_qpos = arm.get_dofs_position(joint_idx)
        # .numpy().astype(np.int32)
        
        collect_data(action_qpos, observed_qpos)


def automated_data_collection(success_num, total_episodes, save_dir='./data'):
    global episode_index, frame_index, index, count
    
    successful_episodes = success_num  # 成功采集的次数
    index = successful_episodes*894
    while successful_episodes<total_episodes:
        
        randomize_cylinder_position()
        reset_robot()
        cam1.start_recording()
        cam2.start_recording()
        episode_index = successful_episodes
        frame_index = 0
        temp_index = index
        temp_episode = episode_index
        
        # 获取圆柱体位置
        target_pos = np.array(cylinder.get_pos().cpu())
        target_quat = np.array([0.707, 0.707, 0, 0])  # 目标姿态

        # 获取机械臂当前末端执行器的位置
        current_pos = np.array(left_end_effector.get_pos().cpu())
        current_quat = np.array(left_end_effector.get_quat().cpu())

        # 机械臂先抬高一定距离，仅在垂直方向上抬高
        lift_pos = np.array([current_pos[0], current_pos[1], current_pos[2] + 0.05])
        move_to_target(lift_pos, target_quat, 200, left_end_effector, so_100_left, left_joint_idx, jaw_open=True)

        # 末端下降到合适位置
        grab_pos = np.array([target_pos[0]+0.01, target_pos[1], target_pos[2]+0.09])
       
        move_to_target(grab_pos, target_quat, 200, left_end_effector, so_100_left, left_joint_idx, jaw_open=True)

        # 闭合夹爪
        move_to_target(grab_pos, target_quat, 50, left_end_effector, so_100_left, left_joint_idx, jaw_open=False)
        move_to_target(grab_pos, target_quat, 50, left_end_effector, so_100_left, left_joint_idx, jaw_open=False)

        # 抬起机械臂
        lift_pos_after_grab = np.array([grab_pos[0], grab_pos[1], grab_pos[2] + 0.05])
        move_to_target(lift_pos_after_grab, target_quat, 200, left_end_effector, so_100_left, left_joint_idx, jaw_open=False)

        box_center = np.array(box_base.get_pos().cpu())
        target_release_pos = np.array([box_center[0], box_center[1], grab_pos[2] + 0.06])
        move_to_target(target_release_pos, target_quat, 200, left_end_effector, so_100_left, left_joint_idx, jaw_open=False)
        target_release_pos = np.array([box_center[0]+0.02, box_center[1]+0.02, box_center[2] + 0.12])
        move_to_target(target_release_pos, target_quat, 200, left_end_effector, so_100_left, left_joint_idx, jaw_open=False)

        # 松开夹爪
        move_to_target(target_release_pos, target_quat, 50, left_end_effector, so_100_left, left_joint_idx, jaw_open=True)

        # 抬起机械臂
        lift_pos_after_release = np.array([target_release_pos[0], target_release_pos[1], target_release_pos[2] + 0.06])
        move_to_target(lift_pos_after_release, target_quat, 200, left_end_effector, so_100_left, left_joint_idx, jaw_open=True)

        move_to_target(current_pos, current_quat, 200, left_end_effector, so_100_left, left_joint_idx, jaw_open=False)

        # 获取圆柱体和盒子的位置信息
        cylinder_pos = np.array(cylinder.get_pos().cpu())
        box_center = np.array(box_base.get_pos().cpu())
        box_size = np.array([0.15, 0.15, 0.001])  # 盒子的尺寸

        # 计算盒子的边界范围
        box_min = box_center - box_size / 2
        box_max = box_center + box_size / 2
        #cyclinder_pos = cylinder_pos - box_center
        # 判断圆柱体是否在盒子内
        is_inside = (
            
            box_min[0] <= cylinder_pos[0] <= box_max[0] and
            box_min[1] <= cylinder_pos[1] <= box_max[1] 
            
        )
        if is_inside:
            # 如果圆柱体在盒子内，则保存视频
            # 补帧以确保视频为30秒
            total_frames = 30 * 30  # 30秒，每秒30帧
            current_frames = frame_index  # 当前已录制的帧数
            needed_frames = 894 - current_frames   # 需要补的帧数

            for _ in range(needed_frames):
                scene.step()
                
                # 获取当前机械臂的关节位置
                current_qpos = so_100_left.get_dofs_position(left_joint_idx)
                
                # 调用 collect_data 函数记录数据
                collect_data(current_qpos, current_qpos)
            import json

            def format_data(data):
                formatted_data = []
                image_data = {'observation.images.laptop':[],'observation.images.phone':[]}
                for item in data:
                    formatted_item = {
                        'action': [num for num in item['action_qpos'].tolist()],
                        'observation.state': [num for num in item['observation.state'].tolist()],
                        'timestamp': item['timestamp'],
                        'frame_index': item['frame_index'],
                        'episode_index': item['episode_index'],
                        'index':item['index'],
                        'task_index': item['task_index'],
                    }

                    image_data['observation.images.laptop'].append(item['observation.images.laptop'].numpy())
                    image_data['observation.images.phone'].append(item['observation.images.phone'].numpy())
                    formatted_data.append(formatted_item)
                   
                return formatted_data, image_data
            formatted_data, image_data= format_data(data)
            result_laptop = calculate_image_stats(image_data=image_data['observation.images.laptop'],index='observation.images.laptop')
            result_phone = calculate_image_stats(image_data=image_data['observation.images.phone'],index='observation.images.phone')

            os.makedirs(f"{save_dir}/episode", exist_ok=True)
            df = pd.DataFrame(formatted_data)
            episode_str = f"{episode_index:06d}"
            df.to_parquet(f'{save_dir}/episode/episode_{episode_str}.parquet',  engine='pyarrow')

            def cal(df, index):
                df[index] = df[index].apply(lambda x: [x] if isinstance(x, (int, float)) else x)
                min_values = [min(values) for values in zip(*df[index])]
                max_values = [max(values) for values in zip(*df[index])]
                mean_values = [np.mean(values) for values in zip(*df[index])]
                std_values = [np.std(values) for values in zip(*df[index])]
                count_values = [len(df[index])]

                return [min_values, max_values, mean_values, std_values, count_values]

            action_values = cal(df, "action")
            observation_values = cal(df, 'observation.state')
            timestamp_values = cal(df, 'timestamp')
            frame_index_values = cal(df, 'frame_index')
            episode_index_values = cal(df, 'episode_index')
            index_values = cal(df, 'index')
            task_index_values = cal(df, 'task_index')
            stats = {
                
                "action": {
                    "min": action_values[0],
                    "max": action_values[1],
                    "mean": action_values[2],
                    "std":action_values[3],
                    "count": action_values[4]
                },
                "observation.state": {
                    "min": observation_values[0],
                    "max": observation_values[1],
                    "mean": observation_values[2],
                    "std":observation_values[3],
                    "count": observation_values[4]
                },
                "observation.images.laptop":{ 
                        "min": [[[value]] for value in result_laptop[0]],
                        "max": [[[value]] for value in result_laptop[1]],
                        "mean": [[[value]] for value in result_laptop[2]],
                        "std": [[[value]] for value in result_laptop[3]],
                        "count": result_laptop[4]

                },
                "observation.images.phone":{ 
                        "min": [[[value]] for value in result_phone[0]],
                        "max": [[[value]] for value in result_phone[1]],
                        "mean": [[[value]] for value in result_phone[2]],
                        "std": [[[value]] for value in result_phone[3]],
                        "count": result_phone[4]
                },
                "timestamp": {
                    "min": timestamp_values[0],
                    "max": timestamp_values[1],
                    "mean": timestamp_values[2],
                    "std": timestamp_values[3],
                    "count": timestamp_values[4]
                },
                "frame_index": {
                    "min": frame_index_values[0],
                    "max": frame_index_values[1],
                    "mean": frame_index_values[2],
                    "std": frame_index_values[3],
                    "count": frame_index_values[4]
                },
                "episode_index": {
                    "min": episode_index_values[0],
                    "max": episode_index_values[1],
                    "mean": episode_index_values[2],
                    "std": episode_index_values[3],
                    "count": episode_index_values[4]
                },
                "index": {
                    "min": index_values[0],
                    "max": index_values[1],
                    "mean": index_values[2],
                    "std": index_values[3],
                    "count": index_values[4]
                },
                "task_index": {
                    "min": task_index_values[0],
                    "max": task_index_values[1],
                    "mean": task_index_values[2],
                    "std": task_index_values[3],
                    "count": task_index_values[4]
                }
            }
            result = {
                    "episode_index": episode_index,
                    "stats": stats
                }
            with open(f'{save_dir}/episodes_stats.jsonl', 'a') as f:
                #json.dump(result, f, indent=4)
                json.dump(result, f)
                f.write('\n') 

            #打印统计数据
            print(f"统计数据 - Episode {episode_index}:")
            #print(json.dumps(stats, indent=4))
            print(f"数据已保存到{save_dir}/episode_{episode_str}.parquet文件中。")
            save_video(successful_episodes,save_dir=save_dir)
            successful_episodes += 1
        else:
            # 如果不在盒子内，则不保存视频
            randomize_cylinder_position()
            reset_robot()

            print("圆柱体不在盒子内，不保存视频。")
            cam1.stop_recording()
            cam2.stop_recording()
            episode_index = temp_episode
            index = temp_index
        data.clear()
def save_video(episode,save_dir):
    # 确保视频保存目录存在
    episode_str = f"{episode_index:06d}"
    video_dir_cam1 = f"./{save_dir}/video/cam1/"
    video_dir_cam2 = f"./{save_dir}/video/cam2/"
    os.makedirs(video_dir_cam1, exist_ok=True)
    os.makedirs(video_dir_cam2, exist_ok=True)

    # 保存视频
    cam1.stop_recording(save_to_filename=f"{video_dir_cam1}episode_{episode_str}.mp4",fps=30)
    cam2.stop_recording(save_to_filename=f"{video_dir_cam2}episode_{episode_str}.mp4",fps=30)

    print(f"Video saved to {video_dir_cam1}episode_{episode_str}.mp4 and {video_dir_cam2}episode_{episode_str}.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据采集程序")
    parser.add_argument('--start', type=int, default=0, help='采集数据起点')
    parser.add_argument('--last', type=int, default=1, help='采集数据终点')
    parser.add_argument('--save_dir', type=str, default="./data", help='采集数据保存目录')
    args = parser.parse_args()
    automated_data_collection(args.start, args.last, save_dir=args.save_dir)

