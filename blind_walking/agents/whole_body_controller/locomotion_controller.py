"""A model based controller framework."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any
import numpy as np
# import time

class LocomotionController(object):
  """Generates the quadruped locomotion.

  The actual effect of this controller depends on the composition of each
  individual subcomponent.

  """
  def __init__(
      self,
      robot: Any,
      gait_generator,
      state_estimator,
      swing_leg_controller,
      stance_leg_controller,
      clock,
  ):
    """Initializes the class.

    Args:
      robot: A robot instance.
      gait_generator: Generates the leg swing/stance pattern.
      state_estimator: Estimates the state of the robot (e.g. center of mass
        position or velocity that may not be observable from sensors).
      swing_leg_controller: Generates motor actions for swing legs.
      stance_leg_controller: Generates motor actions for stance legs.
      clock: A real or fake clock source.
    """
    self._robot = robot
    self._clock = clock
    self._reset_time = self._clock()
    self._time_since_reset = 0
    self._gait_generator = gait_generator
    self._state_estimator = state_estimator
    self._swing_leg_controller = swing_leg_controller
    self._stance_leg_controller = stance_leg_controller

  @property
  def swing_leg_controller(self):
    return self._swing_leg_controller

  @property
  def stance_leg_controller(self):
    return self._stance_leg_controller

  @property
  def gait_generator(self):
    return self._gait_generator

  @property
  def state_estimator(self):
    return self._state_estimator

  def reset(self):
    self._reset_time = self._clock()
    self._time_since_reset = 0
    self._gait_generator.reset(self._time_since_reset)
    self._state_estimator.reset(self._time_since_reset)
    self._swing_leg_controller.reset(self._time_since_reset)
    self._stance_leg_controller.reset(self._time_since_reset)

  def update(self):
    self._time_since_reset = self._clock() - self._reset_time
    self._gait_generator.update(self._time_since_reset)
    self._state_estimator.update(self._time_since_reset)
    self._swing_leg_controller.update(self._time_since_reset)
    self._stance_leg_controller.update(self._time_since_reset)

  def get_action(self):
    """Returns the control ouputs (e.g. positions/torques) for all motors."""
    swing_action = self._swing_leg_controller.get_action()
    stance_action, qp_sol = self._stance_leg_controller.get_action()
    action = []
    for joint_id in range(self._robot.num_motors):
      if joint_id in swing_action:
        action.extend(swing_action[joint_id])
      else:
        assert joint_id in stance_action
        action.extend(stance_action[joint_id])
    action = np.array(action, dtype=np.float32)

    return action, dict(qp_sol=qp_sol)
