# code by LinCC111 Boxjod 2025.1.13 Box2AI-Robotics copyright 盒桥智能 版权所有

import os
import mujoco
import mujoco.viewer
import numpy as np
import time

from lerobot_kinematics import lerobot_IK, lerobot_FK, get_robot
from pynput import keyboard
import threading

np.set_printoptions(linewidth=200)

# Set the MuJoCo render backend
os.environ["MUJOCO_GL"] = "egl"

# Define joint names
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# Absolute path of the XML model
xml_path = "./examples/scene.xml"
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
mjdata = mujoco.MjData(mjmodel)

# Define joint control increment (in radians)
JOINT_INCREMENT = 0.01  # Can be adjusted as needed

# create robot
robot = get_robot('so100')

# Define joint limits
control_qlimit = [[-2.1, -3.1, -0.0, -1.4,    -1.57, -0.15], 
                  [ 2.1,  0.0,  3.1,  1.475,   3.1,  1.5]]

# Initialize target joint positions
init_qpos = np.array([0.0, -3.14, 3.14, 0.0, -1.57, -0.157])
target_qpos = init_qpos.copy()  # Copy of initial joint positions
init_gpos = lerobot_FK(init_qpos[1:5], robot=robot)

# Thread-safe lock
lock = threading.Lock()

# Define key mappings for joint control
key_to_joint_increase = {
    '1': 0,  # Rotation
    '2': 1,  # Pitch
    '3': 2,  # Elbow
    '4': 3,  # Wrist_Pitch
    '5': 4,  # Wrist_Roll
    '6': 5,  # Jaw
}

key_to_joint_decrease = {
    'q': 0,  # Rotation
    'w': 1,  # Pitch
    'e': 2,  # Elbow
    'r': 3,  # Wrist_Pitch
    't': 4,  # Wrist_Roll
    'y': 5,  # Jaw
}

# Dictionary to track currently pressed keys and their direction
keys_pressed = {}

# Handle key press events
def on_press(key):
    try:
        k = key.char.lower()  # Convert to lowercase to handle both upper and lower case inputs
        if k in key_to_joint_increase:
            with lock:
                keys_pressed[k] = 1  # Increase direction
        elif k in key_to_joint_decrease:
            with lock:
                keys_pressed[k] = -1  # Decrease direction
        elif k == "0":
            # Reset to initial joint positions
            print(f'{k=}')
            with lock:
                global target_qpos
                target_qpos = init_qpos.copy()
        
    except AttributeError:
        pass  # Handle special keys if necessary

# Handle key release events
def on_release(key):
    try:
        k = key.char.lower()
        if k in keys_pressed:
            with lock:
                del keys_pressed[k]
    except AttributeError:
        pass  # Handle special keys if necessary

# Start the keyboard listener in a separate thread
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

try:
    # Launch the MuJoCo viewer
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        
        start = time.time()
        while viewer.is_running() and time.time() - start < 1000:
            step_start = time.time()

            # Update target_qpos based on the keys currently pressed
            with lock:
                for k, direction in keys_pressed.items():
                    if k in key_to_joint_increase:
                        joint_idx = key_to_joint_increase[k]
                        if target_qpos[joint_idx] < control_qlimit[1][joint_idx]:
                            target_qpos[joint_idx] += JOINT_INCREMENT * direction
                        
                    elif k in key_to_joint_decrease:
                        joint_idx = key_to_joint_decrease[k]
                        if target_qpos[joint_idx] > control_qlimit[0][joint_idx]:
                            target_qpos[joint_idx] += JOINT_INCREMENT * direction  # direction is -1

            # Forward and inverse kinematics
            position = lerobot_FK(target_qpos[1:5], robot=robot)
            print("Target qpos:", [f"{x:.3f}" for x in target_qpos])
            
            qpos_inv, ik_success = lerobot_IK(target_qpos[1:5], position, robot=robot)
            # Use inverse kinematics solution with validation
            if ik_success:  # Check if the inverse kinematics solution is valid
                target_qpos = np.concatenate((target_qpos[0:1], qpos_inv[:4], target_qpos[5:]))

                # Step the simulation
                # mjdata.ctrl[qpos_indices] = target_qpos
                mjdata.qpos[qpos_indices] = target_qpos
                mujoco.mj_step(mjmodel, mjdata)

                # Update viewer options (e.g., toggle contact point display every second)
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                viewer.sync()
                
                target_gpos_last = target_qpos.copy()  # Backup the valid target_qpos
            else:
                target_qpos = target_gpos_last.copy()  # Revert to the last valid position if IK fails
            
            # Time management to maintain simulation timestep
            time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

except KeyboardInterrupt:
    print("User interrupted the simulation.")
finally:
    listener.stop()  # Stop the keyboard listener
    viewer.close()
