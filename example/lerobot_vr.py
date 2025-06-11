# code by LinCC111 Boxjod 2025.1.13 Box2AI-Robotics copyright 盒桥智能 版权所有
# Modified by Manus to include VR controller input from controller_data.json
# Further modified by Manus for delta position control, new axis mapping, coordinate mapping fix,
# rotation sensitivity adjustment, and VR attitude control (absolute mapping, corrected Euler angle usage).

import os
import mujoco
import mujoco.viewer
import numpy as np
import time
from lerobot_kinematics import lerobot_IK, lerobot_FK, get_robot # Ensure this module is available
from pynput import keyboard # For keyboard fallback/reset
import threading
import json
import math # For Euler angle conversion

np.set_printoptions(linewidth=200, precision=3, suppress=True)

# --- VR Controller Setup ---
DATA_FILE = 'controller_data.json' 
VR_AXIS_THRESHOLD = 0.5
VR_DELTA_POS_SENSITIVITY = 0.5 # For end-effector position control (fwd/back, up/down)
VR_ROTATION_SENSITIVITY = 1  # For base rotation control (left/right)
# --- End VR Controller Setup ---

# Set up the MuJoCo render backend
if os.name != 'nt':
    os.environ["MUJOCO_GL"] = "egl"
else:
    if "MUJOCO_GL" in os.environ and os.environ["MUJOCO_GL"].lower() == "egl":
        print("Warning: MUJOCO_GL is set to EGL on Windows. This might not be intended for local viewing.")

# Define joint names
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir,  "scene_with_table.xml")

if not os.path.exists(xml_path):
    print(f"Error: MuJoCo XML model not found at {xml_path}")
    exit(1)

mjmodel = mujoco.MjModel.from_xml_path(xml_path)
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
mjdata = mujoco.MjData(mjmodel)

JOINT_INCREMENT = 0.005
KEYBOARD_POS_INCREMENT = 0.0008 

robot = get_robot('so100')

control_qlimit = np.array([[-2.1, -3.1, -0.0, -1.57,  -1.57, -0.15], 
                           [ 2.1,  0.0,  3.1,  1.57,   1.57,  1.5]])

control_glimit = np.array([[0.125, -0.4,  0.046, -3.1, -0.75, -1.57], 
                           [0.340,  0.4,  0.230,  2.0,  1.57,  1.57]])

init_qpos = np.array([0.0, -3.14, 3.14, 0.0, 0.0, -0.157]) 
target_qpos = init_qpos.copy()
init_gpos = lerobot_FK(init_qpos[1:5], robot=robot) 
target_gpos = init_gpos.copy()

lock = threading.Lock()

# --- Keyboard Control ---
key_to_gpos_idx_increase = { 'w': 0, 'r': 2, 'q': 3, 'g': 4 } 
key_to_gpos_idx_decrease = { 's': 0, 'f': 2, 'e': 3, 't': 4 }
key_to_qpos_idx_increase = { 'a': 0, 'z': 5 } 
key_to_qpos_idx_decrease = { 'd': 0, 'c': 5 }
keys_pressed = {}

def on_press(key):
    try:
        k = key.char.lower()
        if k in key_to_gpos_idx_increase or k in key_to_qpos_idx_increase or \
           k in key_to_gpos_idx_decrease or k in key_to_qpos_idx_decrease:
            with lock: keys_pressed[k] = 1
        elif k == "0":
            with lock:
                global target_qpos, target_gpos, target_gpos_last, target_qpos_last
                global vr_delta_control_active, vr_initial_controller_pos, vr_target_gpos_at_activation
                global vr_target_qpos_at_activation 
                target_qpos = init_qpos.copy()
                target_gpos = init_gpos.copy()
                target_gpos_last = init_gpos.copy()
                target_qpos_last = init_qpos.copy()
                vr_delta_control_active = False 
                vr_initial_controller_pos = None
                vr_target_gpos_at_activation = None
                vr_target_qpos_at_activation = None
                print("Robot and VR control reset to initial position.")
    except AttributeError: pass

def on_release(key):
    try:
        k = key.char.lower()
        if k in keys_pressed: 
            with lock: del keys_pressed[k]
    except AttributeError: pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

target_gpos_last = init_gpos.copy()
target_qpos_last = init_qpos.copy()

# --- VR Delta Control State ---
vr_delta_control_active = False
vr_initial_controller_pos = None  
vr_target_gpos_at_activation = None 
vr_target_qpos_at_activation = None 
# --- End VR Delta Control State ---

def quaternion_to_euler(x, y, z, w):
    """Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise) -> physical pitch of controller
    pitch is rotation around y in radians (counterclockwise) -> physical yaw of controller
    yaw is rotation around z in radians (counterclockwise) -> physical roll of controller
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1) # This is physical PITCH of the controller
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2) # This is physical YAW of the controller
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4) # This is physical ROLL of the controller
    
    return roll_x, pitch_y, yaw_z 

def read_vr_controller_data():
    data_file_path = os.path.join(script_dir, DATA_FILE)
    if not os.path.exists(data_file_path):
        return None
    try:
        with open(data_file_path, 'r') as f:
            content = f.read()
            if not content: return None
            data = json.loads(content)
            return data
    except: return None

try:
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        print("MuJoCo viewer launched. VR Control (Hold Button 0):")
        print("  - Fwd/Back/Up/Down movement -> End-effector X/Z position (Delta).")
        print("  - Left/Right movement -> Base rotation (Delta).")
        print("  - Controller Physical Pitch (Absolute) -> Wrist Pitch (qpos[3]). Up = qpos[3] small, Down = qpos[3] large.")
        print("  - Controller Physical Roll (Absolute) -> Wrist Roll (qpos[4]). Left = qpos[4] large, Right = qpos[4] small.")
        print("VR Gripper Control: Use Axis 2 on VR controller.")
        print(f"Sensitivities: Pos_Delta:{VR_DELTA_POS_SENSITIVITY}, Rot_Delta:{VR_ROTATION_SENSITIVITY}")
        print("Keyboard: Press '0' to reset robot.")
        
        start_time = time.time()
        while viewer.is_running() and time.time() - start_time < 10000: 
            step_start_time = time.time()

            with lock:
                # --- Keyboard input processing ---
                for k in list(keys_pressed.keys()): 
                    increment_factor = KEYBOARD_POS_INCREMENT
                    q_increment_factor = JOINT_INCREMENT
                    if k in key_to_gpos_idx_increase:
                        idx = key_to_gpos_idx_increase[k]
                        target_gpos[idx] += increment_factor * (4 if idx ==3 or idx==4 else 1)
                    elif k in key_to_gpos_idx_decrease:
                        idx = key_to_gpos_idx_decrease[k]
                        target_gpos[idx] -= increment_factor * (4 if idx ==3 or idx==4 else 1)
                    elif k in key_to_qpos_idx_increase:
                        idx = key_to_qpos_idx_increase[k]
                        target_qpos[idx] += q_increment_factor
                    elif k in key_to_qpos_idx_decrease:
                        idx = key_to_qpos_idx_decrease[k]
                        target_qpos[idx] -= q_increment_factor
                
                for i in range(len(target_gpos)): target_gpos[i] = np.clip(target_gpos[i], control_glimit[0][i], control_glimit[1][i])
                for i in range(len(target_qpos)): target_qpos[i] = np.clip(target_qpos[i], control_qlimit[0][i], control_qlimit[1][i])
                # --- End Keyboard input processing ---

                # --- VR Controller Input Processing ---
                vr_data = read_vr_controller_data()
                active_ctrl = None
                if vr_data and vr_data.get('controllers'):
                    controllers_list = vr_data.get('controllers', [])
                    for c in controllers_list: 
                        if c.get('handedness') == 'right': active_ctrl = c; break
                    if not active_ctrl: 
                        for c_left in controllers_list: 
                            if c_left.get('handedness') == 'left': active_ctrl = c_left; break
                    if not active_ctrl and controllers_list: active_ctrl = controllers_list[0]

                if active_ctrl:
                    current_vr_pos_dict = active_ctrl.get('position')
                    current_vr_orient_dict = active_ctrl.get('orientation')
                    buttons = active_ctrl.get('buttons', [])
                    axes = active_ctrl.get('axes', [])

                    button0_pressed = False
                    for btn in buttons: 
                        if btn.get('index') == 0 and btn.get('pressed'): button0_pressed = True; break

                    if button0_pressed:
                        if not vr_delta_control_active and current_vr_pos_dict and current_vr_orient_dict:
                            vr_delta_control_active = True
                            vr_initial_controller_pos = np.array([current_vr_pos_dict['x'], current_vr_pos_dict['y'], current_vr_pos_dict['z']])
                            vr_target_gpos_at_activation = target_gpos.copy() 
                            vr_target_qpos_at_activation = target_qpos.copy() 
                            print(f"VR Control ACTIVATED.")

                        elif vr_delta_control_active and vr_initial_controller_pos is not None and current_vr_pos_dict and current_vr_orient_dict:
                            current_vr_pos_np = np.array([current_vr_pos_dict['x'], current_vr_pos_dict['y'], current_vr_pos_dict['z']])
                            delta_vr_pos = current_vr_pos_np - vr_initial_controller_pos 
                            
                            # roll_x is controller's physical pitch, pitch_y is controller's physical yaw, yaw_z is controller's physical roll
                            controller_physical_pitch, _, controller_physical_roll = quaternion_to_euler(current_vr_orient_dict['x'], current_vr_orient_dict['y'], current_vr_orient_dict['z'], current_vr_orient_dict['w'])

                            # Delta Position Control (gpos for X,Z) & Delta Rotation Control (qpos for base rotation)
                            target_gpos[0] = vr_target_gpos_at_activation[0] - delta_vr_pos[2] * VR_DELTA_POS_SENSITIVITY # Fwd/Back (VR Z -> Robot X)
                            target_gpos[2] = vr_target_gpos_at_activation[2] + delta_vr_pos[1] * VR_DELTA_POS_SENSITIVITY # Up/Down (VR Y -> Robot Z)
                            target_qpos[0] = vr_target_qpos_at_activation[0] - delta_vr_pos[0] * VR_ROTATION_SENSITIVITY  # Base Rotation (VR X -> Robot qpos[0])
                            
                            # Absolute Attitude Control (qpos for wrist pitch/roll)
                            # qpos[3] (Wrist Pitch): Controlled by controller's physical PITCH (roll_x from quaternion_to_euler)
                            # Controller physical pitch up (positive roll_x) -> qpos[3] becomes more negative (smaller)
                            target_qpos[3] = -controller_physical_pitch 
                            
                            # qpos[4] (Wrist Roll): Controlled by controller's physical ROLL (yaw_z from quaternion_to_euler)
                            # Controller physical roll left (positive yaw_z) -> qpos[4] becomes more positive (larger)
                            target_qpos[4] = controller_physical_roll
                            
                            # print(f"Debug: PhysPitch:{math.degrees(controller_physical_pitch):.1f} -> T_qPos[3]:{target_qpos[3]:.2f}, PhysRoll:{math.degrees(controller_physical_roll):.1f} -> T_qPos[4]:{target_qpos[4]:.2f}")

                    else: 
                        if vr_delta_control_active:
                            vr_delta_control_active = False
                            print("VR Control DEACTIVATED.")

                    # Gripper Control (Axis 2)
                    axis2_val = 0.0
                    for axis_data in axes: 
                        if axis_data.get('index') == 2: axis2_val = axis_data.get('value', 0.0); break
                    
                    if axis2_val < -VR_AXIS_THRESHOLD: 
                        target_qpos[5] += JOINT_INCREMENT
                    elif axis2_val > VR_AXIS_THRESHOLD: 
                        target_qpos[5] -= JOINT_INCREMENT
                    
                    for i in range(3): target_gpos[i] = np.clip(target_gpos[i], control_glimit[0][i], control_glimit[1][i])
                    target_qpos[0] = np.clip(target_qpos[0], control_qlimit[0][0], control_qlimit[1][0]) 
                    target_qpos[3] = np.clip(target_qpos[3], control_qlimit[0][3], control_qlimit[1][3]) 
                    target_qpos[4] = np.clip(target_qpos[4], control_qlimit[0][4], control_qlimit[1][4]) 
                    target_qpos[5] = np.clip(target_qpos[5], control_qlimit[0][5], control_qlimit[1][5]) 
                # --- End VR Controller Input Processing ---
            
            fd_qpos = mjdata.qpos[qpos_indices][1:5] 
            qpos_inv, ik_success = lerobot_IK(fd_qpos, target_gpos, robot=robot)
            
            if ik_success:
                target_qpos_from_ik = target_qpos.copy() 
                target_qpos_from_ik[1:5] = qpos_inv[:4] 
                
                mjdata.qpos[qpos_indices] = target_qpos_from_ik 
                mjdata.qpos[qpos_indices[0]] = target_qpos[0]    
                mjdata.qpos[qpos_indices[3]] = target_qpos[3]    
                mjdata.qpos[qpos_indices[4]] = target_qpos[4]    
                mjdata.qpos[qpos_indices[5]] = target_qpos[5]    

                target_qpos_last = mjdata.qpos[qpos_indices].copy() 
                current_achieved_gpos = lerobot_FK(mjdata.qpos[qpos_indices][1:5], robot=robot)
                target_gpos_last = current_achieved_gpos.copy() 

            else: 
                target_gpos = target_gpos_last.copy()
                mjdata.qpos[qpos_indices] = target_qpos 
                target_qpos_last = target_qpos.copy()

            mujoco.mj_step(mjmodel, mjdata)
            viewer.sync()
            
            time_elapsed_step = time.time() - step_start_time
            time_to_wait = mjmodel.opt.timestep - time_elapsed_step
            if time_to_wait > 0:
                time.sleep(time_to_wait)

except KeyboardInterrupt:
    print("User interrupted the simulation.")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("Stopping keyboard listener...")
    listener.stop()
    listener.join()
    if 'viewer' in locals() and viewer.is_running():
        print("Closing MuJoCo viewer...")
        viewer.close()
    print("Simulation ended.")

