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

"""This file implements the robot specific pose tools."""

import math
import attr
import numpy as np

from blind_walking.robots import laikago_pose_utils
from blind_walking.robots import laikago

_ABDUCTION_ACTION_INDEXES = [0, 3, 6, 9]

# The default values used to give a neutral pose for minitaur.
_MINITAUR_DEFAULT_EXTENSION_POS = math.pi / 2
_MINITAUR_DEFAULT_SWING_POS = 0

_LAIKAGO_NEUTRAL_POSE_HIP_ANGLE = math.pi / 4
_LAIKAGO_NEUTRAL_POSE_KNEE_ANGLE = -math.pi / 2
_LAIKAGO_EXTENSION_CONVERSION_MULTIPLIER = 1.0
_LAIKAGO_SWING_CONVERSION_MULTIPLIER = -1.0

_MINI_CHEETAH_NEUTRAL_POSE_HIP_ANGLE = -math.pi / 4
_MINI_CHEETAH_NEUTRAL_POSE_KNEE_ANGLE = math.pi / 2
_MINI_CHEETAH_EXTENSION_CONVERSION_MULTIPLIER = -1.0
_MINI_CHEETAH_SWING_CONVERSION_MULTIPLIER = 1.0


def get_neutral_motor_angles(robot_class):
  """Return a neutral (standing) pose for a given robot type.

  Args:
    robot_class: This returns the class (not the instance) for the robot.
      Currently it supports minitaur, laikago and mini-cheetah.

  Returns:
    Pose object for the given robot. It's either MinitaurPose, LaikagoPose or
    MiniCheetahPose.

  Raises:
    ValueError: If the given robot_class is different than the supported robots.
  """
  if str(robot_class) == str(laikago.Laikago):
    init_pose = np.array(
        attr.astuple(
            laikago_pose_utils.LaikagoPose(
                abduction_angle_0=0,
                hip_angle_0=_LAIKAGO_NEUTRAL_POSE_HIP_ANGLE,
                knee_angle_0=_LAIKAGO_NEUTRAL_POSE_KNEE_ANGLE,
                abduction_angle_1=0,
                hip_angle_1=_LAIKAGO_NEUTRAL_POSE_HIP_ANGLE,
                knee_angle_1=_LAIKAGO_NEUTRAL_POSE_KNEE_ANGLE,
                abduction_angle_2=0,
                hip_angle_2=_LAIKAGO_NEUTRAL_POSE_HIP_ANGLE,
                knee_angle_2=_LAIKAGO_NEUTRAL_POSE_KNEE_ANGLE,
                abduction_angle_3=0,
                hip_angle_3=_LAIKAGO_NEUTRAL_POSE_HIP_ANGLE,
                knee_angle_3=_LAIKAGO_NEUTRAL_POSE_KNEE_ANGLE)))
  else:
    init_pose = robot_class.get_neutral_motor_angles()
  return init_pose


def convert_leg_pose_to_motor_angles(robot_class, leg_poses):
  """Convert swing-extend coordinate space to motor angles for a robot type.

  Args:
    robot_class: This returns the class (not the instance) for the robot.
      Currently it supports minitaur, laikago and mini-cheetah.
    leg_poses: A list of leg poses in [swing,extend] or [abduction, swing,
      extend] space for all 4 legs. The order is [abd_0, swing_0, extend_0,
      abd_1, swing_1, extend_1, ...] or [swing_0, extend_0, swing_1, extend_1,
      ...]. Zero swing and zero extend gives a neutral standing pose for all the
      robots. For minitaur, the conversion is fully accurate, for laikago and
      mini-cheetah the conversion is approximate where swing is reflected to hip
      and extend is reflected to both knee and the hip.

  Returns:
    List of motor positions for the selected robot. The list include 8 or 12
    motor angles depending on the given robot type as an argument. Currently
    laikago and mini-cheetah has motors for abduction which does not exist for
    minitaur robot.

  Raises:
    ValueError: Conversion fails due to wrong inputs.
  """
  if len(leg_poses) not in [8, 12]:
    raise ValueError("Dimension of the leg pose provided is not 8 or 12.")
  neutral_motor_angles = get_neutral_motor_angles(robot_class)
  motor_angles = leg_poses
  # If it is a robot with 12 motors but the provided leg pose does not contain
  # abduction, extend the pose to include abduction.
  if len(neutral_motor_angles) == 12 and len(leg_poses) == 8:
    for i in _ABDUCTION_ACTION_INDEXES:
      motor_angles.insert(i, 0)
  # If the robot does not have abduction (minitaur) but the input contains them,
  # ignore the abduction angles for the conversion.
  elif len(neutral_motor_angles) == 8 and len(leg_poses) == 12:
    del leg_poses[::3]

  motor_angles = robot_class.convert_leg_pose_to_motor_angles(leg_poses)

  return motor_angles
