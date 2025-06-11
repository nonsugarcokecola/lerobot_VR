# code by LinCC111 Boxjod 2025.1.13 Box2AI-Robotics copyright 盒桥智能 版权所有

import numpy as np

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
so_100_right = scene.add_entity(gs.morphs.MJCF(file= 'examples/so_100.xml', pos=(-0.30, 0, 0.75), euler= (0, 0, 90)))
so_100_left = scene.add_entity(gs.morphs.MJCF(file= 'examples/so_100.xml', pos=(0.30, 0, 0.75), euler= (0, 0, -90)))

# 添加桌子
tablelegs = scene.add_entity(morph= gs.morphs.Mesh(file= 'examples/assets/table/tablelegs.obj', pos=(0, 0, 0), euler=(0, 0, 90), fixed= True), surface= gs.surfaces.Default(roughness= 0.7, diffuse_texture= gs.textures.ImageTexture(image_path= 'examples/assets/table/small_meta_table_diffuse.png')))
tabletop = scene.add_entity( morph= gs.morphs.Mesh(file= 'examples/assets/table/tabletop.obj', pos=(0, 0, 0), euler=(0, 0, 90), fixed= True), surface= gs.surfaces.Default(roughness=0.7,  diffuse_texture= gs.textures.ImageTexture(image_path= 'examples/assets/table/small_meta_table_diffuse.png')))

# 添加一个盒子
red_square = scene.add_entity(
    morph = gs.morphs.Box(
        size = (0.02, 0.02, 0.02),
        pos  = (0.05, 0.0, 0.8),
    ),
    surface= gs.surfaces.Rough(
                        color= (1.0, 0.2, 0.2),
                        vis_mode= "visual",
                    ),
)

red_square = scene.add_entity(
    morph = gs.morphs.Box(
        size = (0.02, 0.02, 0.02),
        pos  = (-0.05, 0.0, 0.8),
    ),
    surface= gs.surfaces.Rough(
                        color= (0.2, 1.0, 0.2),
                        vis_mode= "visual",
                    ),
)

# 场景构建
scene.build()

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
right_joint_idx = [so_100_right.get_joint(name).dof_idx_local for name in joint_names]

init_pos = np.array([0, -3.14, 3.14, 0.817, 0, -0.157])
so_100_left.set_dofs_position(init_pos, left_joint_idx)
so_100_right.set_dofs_position(init_pos, right_joint_idx)


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
#右臂
so_100_right.set_dofs_kp(kp= kp, dofs_idx_local= right_joint_idx)
so_100_right.set_dofs_kv(kv =kv, dofs_idx_local= right_joint_idx)
so_100_right.set_dofs_force_range(lower= force_lower, upper= force_upper, dofs_idx_local= right_joint_idx)


#逆运动学控制
so_100_left.control_dofs_position(np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), left_joint_idx)
scene.step()
left_end_effector = so_100_left.get_link('Fixed_Jaw')
left_trajectory = []
right_end_effector = so_100_right.get_link('Fixed_Jaw')
right_trajectory = []

while(True):
    
    episode_len = 10000
    for i in range(episode_len):
        print(red_square.get_dofs_force())
        if i < 200:
            left_target_pos = np.array([0.05, 0.00, 0.85])
            left_target_quat = np.array([0.707, 0.707, 0, 0])
            t_frac = i / 200
            cur_pos = np.array(left_end_effector.get_pos().cpu())
            cur_quat = np.array(left_end_effector.get_quat().cpu())
            print(cur_pos, cur_quat)
            next_pos = cur_pos + (left_target_pos - cur_pos) * t_frac
            next_quat = cur_quat + (left_target_quat - cur_quat) * t_frac
            next_qpos = so_100_left.inverse_kinematics(
                link= left_end_effector,
                pos = next_pos,
                quat = next_quat
            )
            next_qpos[-1] = 1.5
            so_100_left.control_dofs_position(next_qpos, left_joint_idx)
            scene.step()
        elif i < 250:
            left_target_pos = np.array([0.05, 0.0, 0.83])
            left_target_quat = np.array([0.707, 0.707, 0, 0])
            t_frac = (i - 200) / (250 - 200)
            cur_pos = np.array(left_end_effector.get_pos().cpu())
            cur_quat = np.array(left_end_effector.get_quat().cpu())
            print(cur_pos)
            next_pos = cur_pos + (left_target_pos - cur_pos) * t_frac
            next_quat = cur_quat + (left_target_quat - cur_quat) * t_frac
            next_qpos = so_100_left.inverse_kinematics(
                link= left_end_effector,
                pos = next_pos,
                quat = next_quat
            )
            next_qpos[-1] = 0
            so_100_left.control_dofs_position(next_qpos, left_joint_idx)
            scene.step()

        elif i < 320:
            left_target_pos = np.array([0.05, 0.0, 0.90])
            left_target_quat = np.array([0.707, 0.707, 0, 0])
            t_frac = (i - 250) / (320 - 250)
            cur_pos = np.array(left_end_effector.get_pos().cpu())
            cur_quat = np.array(left_end_effector.get_quat().cpu())
            print(cur_pos)
            next_pos = cur_pos + (left_target_pos - cur_pos) * t_frac
            next_quat = cur_quat + (left_target_quat - cur_quat) * t_frac
            next_qpos = so_100_left.inverse_kinematics(
                link= left_end_effector,
                pos = next_pos,
                quat = next_quat
            )
            next_qpos[-1] = 0
            so_100_left.control_dofs_position(next_qpos, left_joint_idx)
            scene.step()
        
        elif i < 400:
            right_target_pos = np.array([-0.05, 0.0, 0.85])
            right_target_quat = np.array([0, 0, 0.707, 0.707])
            t_frac = (i - 320) / (400 - 320)
            cur_pos = np.array(right_end_effector.get_pos().cpu())
            cur_quat = np.array(right_end_effector.get_quat().cpu())
            print(cur_pos)
            next_pos = cur_pos + (right_target_pos - cur_pos) * t_frac
            next_quat = cur_quat + (right_target_quat - cur_quat) * t_frac
            next_qpos = so_100_right.inverse_kinematics(
                link= right_end_effector,
                pos = next_pos,
                quat = next_quat
            )
            next_qpos[-1] = 1.0
            so_100_right.control_dofs_position(next_qpos, right_joint_idx)
            scene.step()
        elif i < 450:
            right_target_pos = np.array([-0.05, 0.0, 0.83])
            right_target_quat = np.array([0, 0, 0.707, 0.707])
            t_frac = (i - 400) / (450 - 400)
            cur_pos = np.array(right_end_effector.get_pos().cpu())
            cur_quat = np.array(right_end_effector.get_quat().cpu())
            print(cur_pos)
            next_pos = cur_pos + (right_target_pos - cur_pos) * t_frac
            next_quat = cur_quat + (right_target_quat - cur_quat) * t_frac
            next_qpos = so_100_right.inverse_kinematics(
                link= right_end_effector,
                pos = next_pos,
                quat = next_quat
            )
            next_qpos[-1] = 0
            so_100_right.control_dofs_position(next_qpos, right_joint_idx)
            scene.step()
        elif i < 500:
            right_target_pos = np.array([-0.05, 0.0, 0.90])
            right_target_quat = np.array([0, 0, 0.707, 0.707])
            t_frac = (i - 450) / (500 - 450)
            cur_pos = np.array(right_end_effector.get_pos().cpu())
            cur_quat = np.array(right_end_effector.get_quat().cpu())
            print(cur_pos)
            next_pos = cur_pos + (right_target_pos - cur_pos) * t_frac
            next_quat = cur_quat + (right_target_quat - cur_quat) * t_frac
            next_qpos = so_100_right.inverse_kinematics(
                link= right_end_effector,
                pos = next_pos,
                quat = next_quat
            )
            next_qpos[-1] = 0
            so_100_right.control_dofs_position(next_qpos, right_joint_idx)
            scene.step()

