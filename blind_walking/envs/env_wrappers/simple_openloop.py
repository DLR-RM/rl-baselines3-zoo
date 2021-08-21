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
"""Simple openloop trajectory generators."""
import attr
from gym import spaces
import numpy as np

from blind_walking.robots import laikago_pose_utils
from blind_walking.robots import minitaur_pose_utils


class MinitaurPoseOffsetGenerator(object):
  """A trajectory generator that return a constant leg pose."""
  def __init__(self,
               init_swing=0,
               init_extension=2.0,
               init_pose=None,
               action_scale=1.0,
               action_limit=0.5):
    """Initializes the controller.

    Args:
      init_swing: the swing of the default pose offset
      init_extension: the extension of the default pose offset
      init_pose: the default pose offset, which is None by default. If not None,
        it will define the default pose offset while ignoring init_swing and
        init_extension.
      action_scale: changes the magnitudes of actions
      action_limit: clips actions
    """
    if init_pose is None:
      self._pose = np.array(
          attr.astuple(
              minitaur_pose_utils.MinitaurPose(
                  swing_angle_0=init_swing,
                  swing_angle_1=init_swing,
                  swing_angle_2=init_swing,
                  swing_angle_3=init_swing,
                  extension_angle_0=init_extension,
                  extension_angle_1=init_extension,
                  extension_angle_2=init_extension,
                  extension_angle_3=init_extension)))
    else:  # Ignore init_swing and init_extension
      self._pose = np.array(init_pose)
    action_high = np.array([action_limit] * minitaur_pose_utils.NUM_MOTORS)
    self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    self._action_scale = action_scale

  def reset(self):
    pass

  def get_action(self, current_time=None, input_action=None):
    """Computes the trajectory according to input time and action.

    Args:
      current_time: The time in gym env since reset.
      input_action: A numpy array. The input leg pose from a NN controller.

    Returns:
      A numpy array. The desired motor angles.
    """
    del current_time
    return minitaur_pose_utils.leg_pose_to_motor_angles(self._pose +
                                                        self._action_scale *
                                                        np.array(input_action))

  def get_observation(self, input_observation):
    """Get the trajectory generator's observation."""

    return input_observation


class LaikagoPoseOffsetGenerator(object):
  """A trajectory generator that return constant motor angles."""
  def __init__(
      self,
      init_abduction=laikago_pose_utils.LAIKAGO_DEFAULT_ABDUCTION_ANGLE,
      init_hip=laikago_pose_utils.LAIKAGO_DEFAULT_HIP_ANGLE,
      init_knee=laikago_pose_utils.LAIKAGO_DEFAULT_KNEE_ANGLE,
      action_limit=(0.5, 0.5, 0.5),
  ):
    """Initializes the controller.
    Args:
      action_limit: a tuple of [limit_abduction, limit_hip, limit_knee]
    """
    self._pose = np.array(
        attr.astuple(
            laikago_pose_utils.LaikagoPose(abduction_angle_0=init_abduction,
                                           hip_angle_0=init_hip,
                                           knee_angle_0=init_knee,
                                           abduction_angle_1=init_abduction,
                                           hip_angle_1=init_hip,
                                           knee_angle_1=init_knee,
                                           abduction_angle_2=init_abduction,
                                           hip_angle_2=init_hip,
                                           knee_angle_2=init_knee,
                                           abduction_angle_3=init_abduction,
                                           hip_angle_3=init_hip,
                                           knee_angle_3=init_knee)))
    action_high = np.hstack([action_limit] * 4)
    self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

  def reset(self):
    pass

  def get_action(self, current_time=None, input_action=None):
    """Computes the trajectory according to input time and action.

    Args:
      current_time: The time in gym env since reset.
      input_action: A numpy array. The input leg pose from a NN controller.

    Returns:
      A numpy array. The desired motor angles.
    """
    del current_time
    return self._pose + input_action

  def get_observation(self, input_observation):
    """Get the trajectory generator's observation."""

    return input_observation
