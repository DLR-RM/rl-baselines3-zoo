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
"""Converts a list of sensors to gym space."""
import gym
from gym import spaces
import numpy as np
import typing

from blind_walking.envs.sensors import sensor


class UnsupportedConversionError(Exception):
  """An exception when the function cannot convert sensors to gym space."""


class AmbiguousDataTypeError(Exception):
  """An exception when the function cannot determine the data type."""


def convert_sensors_to_gym_space(
    sensors: typing.List[sensor.Sensor]) -> gym.Space:
  """Convert a list of sensors to the corresponding gym space.

  Args:
    sensors: a list of the current sensors

  Returns:
    space: the converted gym space

  Raises:
    UnsupportedConversionError: raises when the function cannot convert the
      given list of sensors.
  """

  if all([
      isinstance(s, sensor.BoxSpaceSensor) and s.get_dimension() == 1
      for s in sensors
  ]):
    return convert_1d_box_sensors_to_gym_space(sensors)
  raise UnsupportedConversionError('sensors = ' + str(sensors))


def convert_1d_box_sensors_to_gym_space(
    sensors: typing.List[sensor.Sensor]) -> gym.Space:
  """Convert a list of 1D BoxSpaceSensors to the corresponding gym space.

  Args:
    sensors: a list of the current sensors

  Returns:
    space: the converted gym space

  Raises:
    UnsupportedConversionError: raises when the function cannot convert the
      given list of sensors.
    AmbiguousDataTypeError: raises when the function cannot determine the
      data types because they are not uniform.
  """
  # Check if all sensors are 1D BoxSpaceSensors
  if not all([
      isinstance(s, sensor.BoxSpaceSensor) and s.get_dimension() == 1
      for s in sensors
  ]):
    raise UnsupportedConversionError('sensors = ' + str(sensors))

  # Check if all sensors have the same data type
  sensor_dtypes = [s.get_dtype() for s in sensors]
  if sensor_dtypes.count(sensor_dtypes[0]) != len(sensor_dtypes):
    raise AmbiguousDataTypeError('sensor datatypes are inhomogeneous')

  lower_bound = np.concatenate([s.get_lower_bound() for s in sensors])
  upper_bound = np.concatenate([s.get_upper_bound() for s in sensors])
  observation_space = spaces.Box(np.array(lower_bound),
                                 np.array(upper_bound),
                                 dtype=np.float32)
  return observation_space


def convert_sensors_to_gym_space_dictionary(
    sensors: typing.List[sensor.Sensor]) -> gym.Space:
  """Convert a list of sensors to the corresponding gym space dictionary.

  Args:
    sensors: a list of the current sensors

  Returns:
    space: the converted gym space dictionary

  Raises:
    UnsupportedConversionError: raises when the function cannot convert the
      given list of sensors.
  """
  gym_space_dict = {}
  for s in sensors:
    if isinstance(s, sensor.BoxSpaceSensor):
      gym_space_dict[s.get_name()] = spaces.Box(np.array(s.get_lower_bound()),
                                                np.array(s.get_upper_bound()),
                                                dtype=np.float32)
    else:
      raise UnsupportedConversionError('sensors = ' + str(sensors))
  return spaces.Dict(gym_space_dict)
