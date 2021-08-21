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

"""A sensor prototype class.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import typing


_ARRAY = typing.Iterable[float] # pylint: disable=invalid-name
_FLOAT_OR_ARRAY = typing.Union[float, _ARRAY] # pylint: disable=invalid-name
_DATATYPE_LIST = typing.Iterable[typing.Any] # pylint: disable=invalid-name


class Sensor(object):
  """A prototype class of sensors."""

  def __init__(self,
               name: typing.Text):
    """A basic constructor of the sensor.

    This initialized a robot as none. This instance may be regularly updated
    by the environment, when it resets the simulation environment.

    Args:
      name: the name of the sensor
    """
    self._robot = None
    self._name = name

  def get_name(self) -> typing.Text:
    return self._name

  def get_dtype(self):
    pass

  def get_observation_datatype(self):
    """Returns the data type for the numpy structured array.

    It is recommended to define a list of tuples: (name, datatype, shape)
    Reference: https://docs.scipy.org/doc/numpy-1.15.0/user/basics.rec.html
    Ex:
      return [('motor_angles', np.float64, (8, ))]  # motor angle sensor
      return [('IMU_x', np.float64), ('IMU_z', np.float64), ] # IMU

    Returns:
      datatype: a list of data types.
    """
    pass

  def get_lower_bound(self):
    """Returns the lower bound of the observation.


    Returns:
      lower_bound: the lower bound of sensor values in np.array format
    """
    pass

  def get_upper_bound(self):
    """Returns the upper bound of the observation.

    Returns:
      upper_bound: the upper bound of sensor values in np.array format
    """
    pass

  def get_observation(self):
    """Returns the observation data.

    Returns:
      observation: the observed sensor values in np.array format
    """
    pass

  def set_robot(self, robot):
    """Set a robot instance."""
    self._robot = robot

  def get_robot(self):
    """Returns the robot instance."""
    return self._robot

  def on_reset(self, env):
    """A callback function for the reset event.

    Args:
      env: the environment who invokes this callback function.
    """
    pass

  def on_step(self, env):
    """A callback function for the step event.

    Args:
      env: the environment who invokes this callback function.
    """
    pass

  def on_terminate(self, env):
    """A callback function for the terminate event.

    Args:
      env: the environment who invokes this callback function.
    """
    pass


class BoxSpaceSensor(Sensor):
  """A prototype class of sensors with Box shapes."""

  def __init__(self,
               name: typing.Text,
               shape: typing.Tuple[int, ...],
               lower_bound: _FLOAT_OR_ARRAY = -np.pi,
               upper_bound: _FLOAT_OR_ARRAY = np.pi,
               dtype=np.float64) -> None:
    """Constructs a box type sensor.

    Args:
      name: the name of the sensor
      shape: the shape of the sensor values
      lower_bound: the lower_bound of sensor value, in float or np.array.
      upper_bound: the upper_bound of sensor value, in float or np.array.
      dtype: data type of sensor value
    """
    super(BoxSpaceSensor, self).__init__(name)

    self._shape = shape
    self._dtype = dtype

    if isinstance(lower_bound, (float, int)):
      self._lower_bound = np.full(shape, lower_bound, dtype=dtype)
    else:
      self._lower_bound = np.array(lower_bound)

    if isinstance(upper_bound, (float, int)):
      self._upper_bound = np.full(shape, upper_bound, dtype=dtype)
    else:
      self._upper_bound = np.array(upper_bound)

  def get_shape(self) -> typing.Tuple[int, ...]:
    return self._shape

  def get_dimension(self) -> int:
    return len(self._shape)

  def get_dtype(self):
    pass

  def get_observation_datatype(self) -> _DATATYPE_LIST:
    """Returns box-shape data type."""
    return [(self._name, self._dtype, self._shape)]

  def get_lower_bound(self) -> _ARRAY:
    """Returns the computed lower bound."""
    return self._lower_bound

  def get_upper_bound(self) -> _ARRAY:
    """Returns the computed upper bound."""
    return self._upper_bound

  def _get_observation(self) -> _ARRAY:
    """Returns raw observation"""
    raise NotImplementedError()

  def get_observation(self) -> np.ndarray:
    return np.asarray(self._get_observation(), dtype=self._dtype)
