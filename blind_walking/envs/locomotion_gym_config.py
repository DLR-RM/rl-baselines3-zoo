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
"""A gin-config class for locomotion_gym_env.

This should be identical to locomotion_gym_config.proto.
"""
import attr
import typing
from blind_walking.robots import robot_config


@attr.s
class SimulationParameters(object):
  """Parameters specific for the pyBullet simulation."""
  sim_time_step_s = attr.ib(type=float, default=0.001)
  num_action_repeat = attr.ib(type=int, default=33)
  enable_hard_reset = attr.ib(type=bool, default=False)
  enable_rendering = attr.ib(type=bool, default=False)
  enable_rendering_gui = attr.ib(type=bool, default=True)
  robot_on_rack = attr.ib(type=bool, default=False)
  camera_distance = attr.ib(type=float, default=1.0)
  camera_yaw = attr.ib(type=float, default=0)
  camera_pitch = attr.ib(type=float, default=-30)
  render_width = attr.ib(type=int, default=480)
  render_height = attr.ib(type=int, default=360)
  egl_rendering = attr.ib(type=bool, default=False)
  motor_control_mode = attr.ib(type=int,
                               default=robot_config.MotorControlMode.POSITION)
  reset_time = attr.ib(type=float, default=-1)
  enable_action_filter = attr.ib(type=bool, default=True)
  enable_action_interpolation = attr.ib(type=bool, default=True)
  allow_knee_contact = attr.ib(type=bool, default=False)
  enable_clip_motor_commands = attr.ib(type=bool, default=True)

  # 0-plain, 1-heightfield, 2-collapsible_tile, 3-collapsible
  terrain_type = attr.ib(type=int, default=0)
  height_field_iters = attr.ib(type=int, default=1)
  height_field_friction = attr.ib(type=float, default=1.0)
  height_field_perturbation_range = attr.ib(type=float, default=0.08)

@attr.s
class ScalarField(object):
  """A named scalar space with bounds."""
  name = attr.ib(type=str)
  upper_bound = attr.ib(type=float)
  lower_bound = attr.ib(type=float)

@attr.s
class LocomotionGymConfig(object):
  """Grouped Config Parameters for LocomotionGym."""
  simulation_parameters = attr.ib(type=SimulationParameters)
  log_path = attr.ib(type=typing.Text, default=None)
  profiling_path = attr.ib(type=typing.Text, default=None)
