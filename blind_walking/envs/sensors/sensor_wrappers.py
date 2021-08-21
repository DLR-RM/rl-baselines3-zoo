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
"""Wrapper classes for extending sensor information."""
import collections

import numpy as np
import typing

from blind_walking.envs.sensors import sensor

_ARRAY = typing.Iterable[float] # pylint: disable=invalid-name


class SensorWrapper(sensor.BoxSpaceSensor):
  """A base interface for sensor wrappers."""
  def __init__(self, wrapped_sensor: sensor.BoxSpaceSensor, **kwargs) -> None:
    """A base wrapper interface.

    Args:
      wrapped_sensor: an inner sensor that you wrap around
      **kwargs: keyword arguments to the parent class
    """
    super(SensorWrapper, self).__init__(**kwargs)
    self._wrapped_sensor = wrapped_sensor

  def __getattr__(self, attr):
    return getattr(self._wrapped_sensor, attr)

  def set_robot(self, robot) -> None:
    """Set a robot instance."""
    self._wrapped_sensor.set_robot(robot)

  def get_robot(self):
    """Returns the robot instance."""
    return self._wrapped_sensor.get_robot()

  def on_reset(self, env) -> None:
    """A callback function for the reset event.

    Args:
      env: the environment who invokes this callback function.
    """
    self._wrapped_sensor.on_reset(env)

  def on_step(self, env) -> None:
    """A callback function for the step event.

    Args:
      env: the environment who invokes this callback function.
    """
    self._wrapped_sensor.on_step(env)

  def on_terminate(self, env) -> None:
    """A callback function for the terminate event.

    Args:
      env: the environment who invokes this callback function.
    """
    self._wrapped_sensor.on_terminate(env)


class HistoricSensorWrapper(SensorWrapper):
  """A sensor wrapper for maintaining the history of the sensor."""
  def __init__(self,
               wrapped_sensor: sensor.BoxSpaceSensor,
               num_history: int,
               append_history_axis: bool = False,
               name: typing.Text = None) -> None:
    """Constructs HistoricSensorWrapper.

    Note that the history begins with the recent one and becomes older. In
    other world, the most recent observation is the first item in the
    history buffer.

    Args:
      wrapped_sensor: an inner sensor that you wrap around
      num_history: the history of sensors want to maintain
      append_history_axis: if True, add an extra axis at the end of the
        observation array for history. If False, stack the historical
        observations without adding an axis.
      name: label for the sensor. Defaults to HistoricSensorWrapper(<wrapped
        sensor name>).
    """
    self._num_history = num_history
    self._append_history_axis = append_history_axis
    name = name or "HistoricSensorWrapper(%s)" % wrapped_sensor.get_name()
    if self._append_history_axis:
      lower_bound = np.tile(
          np.expand_dims(wrapped_sensor.get_lower_bound(), -1),
          (1, self._num_history))
      upper_bound = np.tile(
          np.expand_dims(wrapped_sensor.get_upper_bound(), -1),
          (1, self._num_history))
    else:
      lower_bound = np.tile(wrapped_sensor.get_lower_bound(),
                            self._num_history)
      upper_bound = np.tile(wrapped_sensor.get_upper_bound(),
                            self._num_history)
    shape = lower_bound.shape

    self._history_buffer = None
    super(HistoricSensorWrapper, self).__init__(name=name,
                                                shape=shape,
                                                lower_bound=lower_bound,
                                                upper_bound=upper_bound,
                                                wrapped_sensor=wrapped_sensor)

  def on_reset(self, env) -> None:
    """A callback for the reset event that initializes the history buffer.

    Args:
      env: the environment who invokes this callback function (unused)
    """
    super(HistoricSensorWrapper, self).on_reset(env)

    self._history_buffer = collections.deque(maxlen=self._num_history)
    for _ in range(self._num_history):
      self._history_buffer.appendleft(self._wrapped_sensor.get_observation())

  def on_step(self, env):
    """A callback for the step event that updates the history buffer.

    Args:
      env: the environment who invokes this callback function (unused)
    """
    super(HistoricSensorWrapper, self).on_step(env)
    self._history_buffer.appendleft(self._wrapped_sensor.get_observation())

  def get_observation(self) -> _ARRAY:
    """Returns the observation by concatenating the history buffer."""
    if self._append_history_axis:
      return np.stack(self._history_buffer, axis=-1)
    else:
      return np.concatenate(self._history_buffer)

  @property
  def history_buffer(self):
    """Returns the raw history buffer."""
    return self._history_buffer
