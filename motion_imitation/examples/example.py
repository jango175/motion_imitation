# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import argparse
import numpy as np
import os
import random
import time

from motion_imitation.envs import env_builder as env_builder
from motion_imitation.robots import robot_config

from motion_imitation.robots import laikago
from motion_imitation.robots import a1

ROBOT_CLASS_MAP = {'A1': a1.A1, 'Laikago': laikago.Laikago}

def test(env, robot):
  o = env.reset()
  while 1:
    if isinstance(env.robot, laikago.Laikago):
        a = laikago.INIT_MOTOR_ANGLES
    else:
        a = a1.INIT_MOTOR_ANGLES

    o, r, done, info = env.step(a)
    if done:
        o = env.reset()
  return

def main():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
  arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=True)
  arg_parser.add_argument("--robot", dest="robot", default="Laikago", choices=["A1", "Laikago"])

  args = arg_parser.parse_args()
  robot = ROBOT_CLASS_MAP[args.robot]

#   env = env_builder.build_laikago_env( motor_control_mode = robot_config.MotorControlMode.POSITION, enable_rendering=args.visualize)
  env = env_builder.build_regular_env(robot, motor_control_mode=robot_config.MotorControlMode.POSITION, enable_rendering=args.visualize)

  test(env, robot)
  
  return

if __name__ == '__main__':
  main()
