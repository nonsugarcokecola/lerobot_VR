# code by LinCC111 Boxjod 2025.1.13 Box2AI-Robotics copyright 盒桥智能 版权所有
import time
import numpy as np
import random
import cv2
# If you install genesis it may break the lerobot python environment.
import genesis as gs
gs.init(backend=gs.gpu)

# 基础场景
dt = 0.01
scene = gs.Scene(show_viewer= True, sim_options= gs.options.SimOptions(dt = dt),
                rigid_options= gs.options.RigidOptions(
                dt= dt,
                constraint_solver= gs.constraint_solver.Newton,
                enable_collision= True,
                enable_joint_limit= True,
                enable_self_collision= True,
            ),)
plane = scene.add_entity(gs.morphs.Plane())
light = scene.add_light(gs.morphs.Primitive()) 

#添加机械臂

so_100_left = scene.add_entity(gs.morphs.MJCF(file= 'examples/so_100.xml', pos=(0, 0, 0.75), euler= (0, 0, -90)))

# 添加桌子
tablelegs = scene.add_entity(morph= gs.morphs.Mesh(file= 'examples/assets/table/tablelegs.obj', pos=(0, 0, 0), euler=(0, 0, 90), fixed= True), surface= gs.surfaces.Default(roughness= 0.7, diffuse_texture= gs.textures.ImageTexture(image_path= 'examples/assets/table/small_meta_table_diffuse.png')))
tabletop = scene.add_entity( morph= gs.morphs.Mesh(file= 'examples/assets/table/tabletop.obj', pos=(0, 0, 0), euler=(0, 0, 90), fixed= True), surface= gs.surfaces.Default(roughness=0.7,  diffuse_texture= gs.textures.ImageTexture(image_path= 'examples/assets/table/small_meta_table_diffuse.png')))

# 添加一个圆柱体
cylinder = scene.add_entity(
    morph=gs.morphs.Cylinder(
        radius=0.013,  # 半径 1.3cm（直径 2.6cm）
        height=0.06,  # 高度 6cm
        pos=(-0.285, 0.0, 0.77),  # 平躺放置在桌子上，高度为桌子高度 + 半径
        euler=(90, 0, 0),  # 绕 x 轴旋转 90 度，使圆柱体平躺
        
    ),
    surface=gs.surfaces.Rough(
        color=(0.5, 0.0, 0.5),  # 灰色
        vis_mode="visual",
    ),
)

# 添加木板（总长度15cm，宽度0.02cm，高度2cm）
board = scene.add_entity(
    morph=gs.morphs.Box(
        size=(0.002, 0.15, 0.02),  # 尺寸 15cm x 0.02cm x 2cm
        pos=(-0.085, -0.125, 0.76),  # 位置：8.5cm，12cm，高度为桌子高度 + 1cm
        fixed=True,  # 设置为固定
    ),
    surface=gs.surfaces.Rough(
        color=(0.65, 0.45, 0.25),  # 木板颜色
        vis_mode="visual",
    ),
)

board2 = scene.add_entity(
    morph=gs.morphs.Box(
        size=(0.002, 0.15, 0.02),  # 尺寸 15cm x 0.02cm x 2cm
        pos=(-0.235, -0.125, 0.76),  # 位置：8.5cm，12cm，高度为桌子高度 + 1cm
        fixed=True,  # 设置为固定
    ),
    surface=gs.surfaces.Rough(
        color=(0.65, 0.45, 0.25),  # 木板颜色
        vis_mode="visual",
    ),
)
board3 = scene.add_entity(
    morph=gs.morphs.Box(
        size=(0.15, 0.002, 0.02),  # 尺寸 15cm x 0.02cm x 2cm
        pos=(-0.16, -0.05, 0.76),  # 位置：8.5cm，12cm，高度为桌子高度 + 1cm
        fixed=True,  # 设置为固定
    ),
    surface=gs.surfaces.Rough(
        color=(0.65, 0.45, 0.25),  # 木板颜色
        vis_mode="visual",
    ),
)

board4 = scene.add_entity(
    morph=gs.morphs.Box(
        size=(0.15, 0.002, 0.02),  # 尺寸 15cm x 0.02cm x 2cm
        pos=(-0.16, -0.20, 0.76),  # 位置：8.5cm，12cm，高度为桌子高度 + 1cm
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
        size=(0.15, 0.15, 0.001),  # 底板尺寸 15cm x 15cm x 1mm
        pos=(-0.16, -0.125, 0.75),  # 中心点位置
        fixed=True,
    ),
    surface=gs.surfaces.Rough(
        color=(0.54, 0.27, 0.07),  # 褐色
        vis_mode="visual",
    ),
)
cam1 = scene.add_camera(
    res    = (640, 480),
    pos    = (-0.335, 0.08, 1.7),
    lookat = (-0.335, 0.08, 0), 
    fov    = 30,
    GUI    = True
)



# 场景构建
scene.build()
rgb, depth, segmentation, normal = cam1.render(normal=True)
cam1.start_recording()
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

data = []

def randomize_cylinder_position():
    """随机化圆柱体位置并重置速度"""
    x = random.uniform(-0.285, -0.287)  # 随机 x 坐标
    y = random.uniform(0.0, 0.02)  # 随机 y 坐标
    cylinder.set_pos((x, y, 0.77), zero_velocity=True)  # 更新圆柱体位置并重置速度
  
def collect_data():
    """采集数据"""
    cylinder_pos = cylinder.get_pos().cpu().numpy()
    cylinder_force = cylinder.get_dofs_force()
    left_force = so_100_left.get_dofs_force(left_joint_idx)
    
    data.append({
        'cylinder_pos': cylinder_pos,
        'cylinder_force': cylinder_force,
        'left_force': left_force,
        
    })
    
def move_to_target(target_pos, target_quat, steps, effector, arm, joint_idx, jaw_open=True):
    """控制机械臂移动到目标位置和姿态"""
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
        collect_data()
        cam1.render()

# while True:
#     episode_len = 10000
#     for i in range(episode_len):
        # 随机化圆柱体位置
        #if i % 1000 == 0:  # 每 1000 步随机化一次位置
randomize_cylinder_position()

# 获取圆柱体位置
target_pos = np.array(cylinder.get_pos().cpu())
target_quat = np.array([0.707, 0.707, 0, 0])  # 目标姿态

# 获取机械臂当前末端执行器的位置
current_pos = np.array(left_end_effector.get_pos().cpu())

# 机械臂先抬高一定距离，仅在垂直方向上抬高
lift_pos = np.array([current_pos[0]+0.4, current_pos[1], current_pos[2] + 0.05])
move_to_target(lift_pos, target_quat, 200, left_end_effector, so_100_left, left_joint_idx, jaw_open=True)

# 末端垂直移动到物体上方
above_pos = np.array([target_pos[0]+0.03, target_pos[1], target_pos[2] + 0.05])
move_to_target(above_pos, target_quat, 200, left_end_effector, so_100_left, left_joint_idx, jaw_open=True)

# 末端下降到合适位置
grab_pos = np.array([target_pos[0]+0.02, target_pos[1], target_pos[2]])
move_to_target(grab_pos, target_quat, 200, left_end_effector, so_100_left, left_joint_idx, jaw_open=False)

# 闭合夹爪
move_to_target(grab_pos, target_quat, 50, left_end_effector, so_100_left, left_joint_idx, jaw_open=False)

#for _ in range(100):  # 100 步，每步 0.01 秒，总共 1 秒
#    scene.step()
#   collect_data()

# 抬起机械臂
lift_pos_after_grab = np.array([grab_pos[0], grab_pos[1], grab_pos[2] + 0.15])
move_to_target(lift_pos_after_grab, target_quat, 200, left_end_effector, so_100_left, left_joint_idx, jaw_open=False)

box_center = np.array(box_base.get_pos().cpu())
target_release_pos = np.array([box_center[0], box_center[1], grab_pos[2] + 0.15])
move_to_target(target_release_pos, target_quat, 200, left_end_effector, so_100_left, left_joint_idx, jaw_open=False)
target_release_pos = np.array([box_center[0], box_center[1], box_center[2] + 0.06])
move_to_target(target_release_pos, target_quat, 200, left_end_effector, so_100_left, left_joint_idx, jaw_open=False)

# 松开夹爪
move_to_target(target_release_pos, target_quat, 50, left_end_effector, so_100_left, left_joint_idx, jaw_open=True)

# 抬起机械臂
lift_pos_after_release = np.array([target_release_pos[0], target_release_pos[1], target_release_pos[2] + 0.1])
move_to_target(lift_pos_after_release, target_quat, 200, left_end_effector, so_100_left, left_joint_idx, jaw_open=True)

move_to_target(current_pos, target_quat, 200, left_end_effector, so_100_left, left_joint_idx, jaw_open=False)
cam1.render()
#so_100_left.set_dofs_position(init_pos, left_joint_idx)
time.sleep(3)

cam1.stop_recording(save_to_filename='video1.mp4', fps=60)
