"""The swing leg controller class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
from typing import Any, Mapping, Sequence, Tuple

from blind_walking.agents.whole_body_controller import gait_generator as gait_generator_lib
from blind_walking.agents.whole_body_controller import leg_controller

# The position correction coefficients in Raibert's formula.
_KP = np.array([0.01, 0.01, 0.01]) * 3
# At the end of swing, we leave a small clearance to prevent unexpected foot
# collision.
_FOOT_CLEARANCE_M = 0.01


def _gen_parabola(phase: float, start: float, mid: float, end: float) -> float:
  """Gets a point on a parabola y = a x^2 + b x + c.

  The Parabola is determined by three points (0, start), (0.5, mid), (1, end) in
  the plane.

  Args:
    phase: Normalized to [0, 1]. A point on the x-axis of the parabola.
    start: The y value at x == 0.
    mid: The y value at x == 0.5.
    end: The y value at x == 1.

  Returns:
    The y value at x == phase.
  """
  mid_phase = 0.5
  delta_1 = mid - start
  delta_2 = end - start
  delta_3 = mid_phase**2 - mid_phase
  coef_a = (delta_1 - delta_2 * mid_phase) / delta_3
  coef_b = (delta_2 * mid_phase**2 - delta_1) / delta_3
  coef_c = start

  return coef_a * phase**2 + coef_b * phase + coef_c


def _gen_swing_foot_trajectory(input_phase: float, start_pos: Sequence[float],
                               end_pos: Sequence[float]) -> Tuple[float]:
  """Generates the swing trajectory using a parabola.

  Args:
    input_phase: the swing/stance phase value between [0, 1].
    start_pos: The foot's position at the beginning of swing cycle.
    end_pos: The foot's desired position at the end of swing cycle.

  Returns:
    The desired foot position at the current phase.
  """
  # We augment the swing speed using the below formula. For the first half of
  # the swing cycle, the swing leg moves faster and finishes 80% of the full
  # swing trajectory. The rest 20% of trajectory takes another half swing
  # cycle. Intuitely, we want to move the swing foot quickly to the target
  # landing location and stay above the ground, in this way the control is more
  # robust to perturbations to the body that may cause the swing foot to drop
  # onto the ground earlier than expected. This is a common practice similar
  # to the MIT cheetah and Marc Raibert's original controllers.
  phase = input_phase
  if input_phase <= 0.5:
    phase = 0.8 * math.sin(input_phase * math.pi)
  else:
    phase = 0.8 + (input_phase - 0.5) * 0.4

  x = (1 - phase) * start_pos[0] + phase * end_pos[0]
  y = (1 - phase) * start_pos[1] + phase * end_pos[1]
  max_clearance = 0.1
  mid = max(end_pos[2], start_pos[2]) + max_clearance
  z = _gen_parabola(phase, start_pos[2], mid, end_pos[2])

  # PyType detects the wrong return type here.
  return (x, y, z)  # pytype: disable=bad-return-type


class RaibertSwingLegController(leg_controller.LegController):
  """Controls the swing leg position using Raibert's formula.

  For details, please refer to chapter 2 in "Legged robbots that balance" by
  Marc Raibert. The key idea is to stablize the swing foot's location based on
  the CoM moving speed.

  """
  def __init__(
      self,
      robot: Any,
      gait_generator: Any,
      state_estimator: Any,
      desired_speed: Tuple[float, float],
      desired_twisting_speed: float,
      desired_height: float,
      foot_clearance: float,
  ):
    """Initializes the class.

    Args:
      robot: A robot instance.
      gait_generator: Generates the stance/swing pattern.
      state_estimator: Estiamtes the CoM speeds.
      desired_speed: Behavior parameters. X-Y speed.
      desired_twisting_speed: Behavior control parameters.
      desired_height: Desired standing height.
      foot_clearance: The foot clearance on the ground at the end of the swing
        cycle.
    """
    self._robot = robot
    self._state_estimator = state_estimator
    self._gait_generator = gait_generator
    self._last_leg_state = gait_generator.desired_leg_state
    self.desired_speed = np.array((desired_speed[0], desired_speed[1], 0))
    self.desired_twisting_speed = desired_twisting_speed
    self._desired_height = np.array((0, 0, desired_height - foot_clearance))

    self._joint_angles = None
    self._phase_switch_foot_local_position = None
    self.reset(0)

  def reset(self, current_time: float) -> None:
    """Called during the start of a swing cycle.

    Args:
      current_time: The wall time in seconds.
    """
    del current_time
    self._last_leg_state = self._gait_generator.desired_leg_state
    self._phase_switch_foot_local_position = (
        self._robot.GetFootPositionsInBaseFrame())
    self._joint_angles = {}

  def update(self, current_time: float) -> None:
    """Called at each control step.

    Args:
      current_time: The wall time in seconds.
    """
    del current_time
    new_leg_state = self._gait_generator.desired_leg_state

    # Detects phase switch for each leg so we can remember the feet position at
    # the beginning of the swing phase.
    for leg_id, state in enumerate(new_leg_state):
      if (state == gait_generator_lib.LegState.SWING
          and state != self._last_leg_state[leg_id]):
        self._phase_switch_foot_local_position[leg_id] = (
            self._robot.GetFootPositionsInBaseFrame()[leg_id])

    self._last_leg_state = copy.deepcopy(new_leg_state)

  def get_action(self) -> Mapping[Any, Any]:
    com_velocity = self._state_estimator.com_velocity_body_frame
    com_velocity = np.array((com_velocity[0], com_velocity[1], 0))

    _, _, yaw_dot = self._robot.GetBaseRollPitchYawRate()
    hip_positions = self._robot.GetHipPositionsInBaseFrame()

    for leg_id, leg_state in enumerate(self._gait_generator.leg_state):
      if leg_state in (gait_generator_lib.LegState.STANCE,
                       gait_generator_lib.LegState.EARLY_CONTACT):
        continue

      # For now we did not consider the body pitch/roll and all calculation is
      # in the body frame. TODO(b/143378213): Calculate the foot_target_position
      # in world frame and then project back to calculate the joint angles.
      hip_offset = hip_positions[leg_id]
      twisting_vector = np.array((-hip_offset[1], hip_offset[0], 0))
      hip_horizontal_velocity = com_velocity + yaw_dot * twisting_vector
      # print("Leg: {}, ComVel: {}, Yaw_dot: {}".format(leg_id, com_velocity,
      #                                                 yaw_dot))
      # print(hip_horizontal_velocity)
      target_hip_horizontal_velocity = (
          self.desired_speed + self.desired_twisting_speed * twisting_vector)
      foot_target_position = (
          hip_horizontal_velocity *
          self._gait_generator.stance_duration[leg_id] / 2 - _KP *
          (target_hip_horizontal_velocity - hip_horizontal_velocity)
      ) - self._desired_height + np.array((hip_offset[0], hip_offset[1], 0))
      foot_position = _gen_swing_foot_trajectory(
          self._gait_generator.normalized_phase[leg_id],
          self._phase_switch_foot_local_position[leg_id], foot_target_position)
      joint_ids, joint_angles = (
          self._robot.ComputeMotorAnglesFromFootLocalPosition(
              leg_id, foot_position))
      # Update the stored joint angles as needed.
      for joint_id, joint_angle in zip(joint_ids, joint_angles):
        self._joint_angles[joint_id] = (joint_angle, leg_id)

    action = {}
    kps = self._robot.GetMotorPositionGains()
    kds = self._robot.GetMotorVelocityGains()
    for joint_id, joint_angle_leg_id in self._joint_angles.items():
      leg_id = joint_angle_leg_id[1]
      if self._gait_generator.desired_leg_state[
          leg_id] == gait_generator_lib.LegState.SWING:
        # This is a hybrid action for PD control.
        action[joint_id] = (joint_angle_leg_id[0], kps[joint_id], 0,
                            kds[joint_id], 0)

    return action
