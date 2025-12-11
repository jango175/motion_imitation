"""Apply the same action to the simulated and real A1 robot.


As a basic debug tool, this script allows you to execute the same action
(which you choose from the pybullet GUI) on the simulation and real robot
simultaneouly. Make sure to put the real robot on rack before testing.
"""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import time
from tqdm import tqdm
FREQ = 0.5

from absl import app
from absl import logging
import numpy as np
import pybullet as p  # pytype: disable=import-error

from motion_imitation.envs import env_builder
from motion_imitation.robots import a1
from motion_imitation.robots import a1_robot
from motion_imitation.robots import robot_config


def main(_):
  logging.info("WARNING: this code executes low-level controller on the robot.")
  logging.info("Make sure the robot is hang on rack before proceeding.")
  input("Press enter to continue...")
  # Construct sim env and real robot
  sim_env = env_builder.build_regular_env(
      robot_class=a1.A1,
      motor_control_mode=robot_config.MotorControlMode.POSITION,
      on_rack=False,
      enable_rendering=True,
      wrap_trajectory_generator=False)
  real_env = env_builder.build_regular_env(
      robot_class=a1_robot.A1Robot,
      motor_control_mode=robot_config.MotorControlMode.POSITION,
      on_rack=False,
      enable_rendering=False,
      wrap_trajectory_generator=False)

  # Add debug sliders
  action_low, action_high = sim_env.action_space.low, sim_env.action_space.high
  dim_action = action_low.shape[0]
  action_selector_ids = []
  robot_motor_angles = real_env.robot.GetMotorAngles()

  for dim in range(dim_action):
    action_selector_id = p.addUserDebugParameter(
        paramName='dim{}'.format(dim),
        rangeMin=action_low[dim],
        rangeMax=action_high[dim],
        startValue=robot_motor_angles[dim])
    action_selector_ids.append(action_selector_id)

  # Move the motors slowly to initial position
  sim_env.robot.ReceiveObservation()
  current_motor_angle = np.array(sim_env.robot.GetMotorAngles())
  desired_motor_angle = np.array([0., 0.9, -1.8] * 4)
  for t in tqdm(range(300)):
    blend_ratio = np.minimum(t / 200., 1)
    action = (1 - blend_ratio) * current_motor_angle + blend_ratio * desired_motor_angle
    sim_env.robot.Step(action, robot_config.MotorControlMode.POSITION)
    time.sleep(0.005)

  # Move the legs in a sinusoidal curve
  for t in tqdm(range(1000)):
    angle_hip = 0.9 + 0.2 * np.sin(2 * np.pi * FREQ * 0.01 * t)
    angle_calf = -2 * angle_hip
    action = np.array([0., angle_hip, angle_calf] * 4)
    sim_env.robot.Step(action, robot_config.MotorControlMode.POSITION)
    time.sleep(0.007)
    # print(sim_env.robot.GetFootContacts())
    print(sim_env.robot.GetBaseVelocity())

  # Visualize debug slider in sim
  for _ in range(10000):
    # Get user action input
    action = np.zeros(dim_action)
    for dim in range(dim_action):
      action[dim] = sim_env.pybullet_client.readUserDebugParameter(
          action_selector_ids[dim])

    real_env.step(action)
    sim_env.step(action)

  real_env.Terminate()


if __name__ == '__main__':
  app.run(main)
