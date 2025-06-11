# code by Boxjod 2025.1.13 Box2AI-Robotics copyright 盒桥智能 版权所有

# For Feetech Motors
from lerobot_kinematics.lerobot.feetech import FeetechMotorsBus
import json
import numpy as np

class feetech_arm:
  def __init__(self, driver_port, calibration_file):
      
    self.driver_port = driver_port
    self.calibration_file = calibration_file
    self.arm_hardware = self.connect_arm()
          
  def connect_arm(self):
    # Connect to the robotic arm motors
    motors = {"shoulder_pan": (1, "sts3215"),
              "shoulder_lift": (2, "sts3215"),
              "elbow_flex": (3, "sts3215"),
              "wrist_flex": (4, "sts3215"),
              "wrist_roll": (5, "sts3215"),
              "gripper": (6, "sts3215")}

    follower_arm = FeetechMotorsBus(port=self.driver_port, motors=motors)
    follower_arm.connect()
    print('Robot arm connected successfully')

    # Load the robot arm calibration parameters
    arm_calib_path = self.calibration_file
    with open(arm_calib_path) as f:
        calibration = json.load(f)
    follower_arm.set_calibration(calibration)
    
    return follower_arm
  
  def action(self, qpos_rad):
    joint_angles = np.rad2deg(qpos_rad)
    joint_angles[1] = -joint_angles[1]
    joint_angles[0] = -joint_angles[0]
    joint_angles[4] = -joint_angles[4]
    self.arm_hardware.write("Goal_Position", joint_angles)
    qpos = self.arm_hardware.read("Present_Position")
    return qpos
  
  def feedback(self):
    qpos_degree = self.arm_hardware.read("Present_Position")
    qpos_degree[1] = -qpos_degree[1]
    qpos_degree[0] = -qpos_degree[0]
    qpos_degree[4] = -qpos_degree[4]
    
    joint_angles = np.deg2rad(qpos_degree)
    return joint_angles
  
  def disconnect(self):
    self.arm_hardware.disconnect()






