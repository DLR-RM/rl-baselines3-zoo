"""State estimator."""

import numpy as np
from typing import Any, Sequence

from blind_walking.utilities.moving_window_filter import MovingWindowFilter

_DEFAULT_WINDOW_SIZE = 20

class COMVelocityEstimator(object):
  """Estimate the CoM velocity using on board sensors.


  Requires knowledge about the base velocity in world frame, which for example
  can be obtained from a MoCap system. This estimator will filter out the high
  frequency noises in the velocity so the results can be used with controllers
  reliably.

  """

  def __init__(
      self,
      robot: Any,
      window_size: int = _DEFAULT_WINDOW_SIZE,
  ):
    self._robot = robot
    self._window_size = window_size
    self.reset(0)

  @property
  def com_velocity_body_frame(self) -> Sequence[float]:
    """The base velocity projected in the body aligned inertial frame.

    The body aligned frame is a intertia frame that coincides with the body
    frame, but has a zero relative velocity/angular velocity to the world frame.

    Returns:
      The com velocity in body aligned frame.
    """
    return self._com_velocity_body_frame

  @property
  def com_velocity_world_frame(self) -> Sequence[float]:
    return self._com_velocity_world_frame

  def reset(self, current_time):
    del current_time
    # We use a moving window filter to reduce the noise in velocity estimation.
    self._velocity_filter_x = MovingWindowFilter(
        window_size=self._window_size)
    self._velocity_filter_y = MovingWindowFilter(
        window_size=self._window_size)
    self._velocity_filter_z = MovingWindowFilter(
        window_size=self._window_size)
    self._com_velocity_world_frame = np.array((0, 0, 0))
    self._com_velocity_body_frame = np.array((0, 0, 0))

  def update(self, current_time):
    del current_time
    velocity = self._robot.GetBaseVelocity()

    vx = self._velocity_filter_x.calculate_average(velocity[0])
    vy = self._velocity_filter_y.calculate_average(velocity[1])
    vz = self._velocity_filter_z.calculate_average(velocity[2])
    self._com_velocity_world_frame = np.array((vx, vy, vz))

    base_orientation = self._robot.GetTrueBaseOrientation()
    _, inverse_rotation = self._robot.pybullet_client.invertTransform(
        (0, 0, 0), base_orientation)

    self._com_velocity_body_frame, _ = (
        self._robot.pybullet_client.multiplyTransforms(
            (0, 0, 0), inverse_rotation, self._com_velocity_world_frame,
            (0, 0, 0, 1)))
