"""Gait pattern planning module."""

from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function

import os
import sys
import inspect
import logging
import math

import numpy as np
from typing import Any, Sequence
from blind_walking.agents.whole_body_controller import gait_generator

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

LAIKAGO_TROTTING = (
    gait_generator.LegState.SWING,
    gait_generator.LegState.STANCE,
    gait_generator.LegState.STANCE,
    gait_generator.LegState.SWING,
)

_NOMINAL_STANCE_DURATION = (0.3, 0.3, 0.3, 0.3)
_NOMINAL_DUTY_FACTOR = (0.5, 0.5, 0.5, 0.5)
_NOMINAL_CONTACT_DETECTION_PHASE = 0.1


class OpenloopGaitGenerator(gait_generator.GaitGenerator):
  """Generates openloop gaits for quadruped robots.

  A flexible open-loop gait generator. Each leg has its own cycle and duty
  factor. And the state of each leg alternates between stance and swing. One can
  easily formuate a set of common quadruped gaits like trotting, pacing,
  pronking, bounding, etc by tweaking the input parameters.
  """
  def __init__(
      self,
      robot: Any,
      stance_duration: Sequence[float] = _NOMINAL_STANCE_DURATION,
      duty_factor: Sequence[float] = _NOMINAL_DUTY_FACTOR,
      initial_leg_state: Sequence[gait_generator.LegState] = LAIKAGO_TROTTING,
      initial_leg_phase: Sequence[float] = (0, 0, 0, 0),
      contact_detection_phase_threshold:
      float = _NOMINAL_CONTACT_DETECTION_PHASE,
  ):
    """Initializes the class.

    Args:
      robot: A quadruped robot that at least implements the GetFootContacts API
        and num_legs property.
      stance_duration: The desired stance duration.
      duty_factor: The ratio  stance_duration / total_gait_cycle.
      initial_leg_state: The desired initial swing/stance state of legs indexed
        by their id.
      initial_leg_phase: The desired initial phase [0, 1] of the legs within the
        full swing + stance cycle.
      contact_detection_phase_threshold: Updates the state of each leg based on
        contact info, when the current normalized phase is greater than this
        threshold. This is essential to remove false positives in contact
        detection when phase switches. For example, a swing foot at at the
        beginning of the gait cycle might be still on the ground.
    """
    self._robot = robot
    self._stance_duration = stance_duration
    self._duty_factor = duty_factor
    self._swing_duration = np.array(stance_duration) / np.array(
        duty_factor) - np.array(stance_duration)
    if len(initial_leg_phase) != self._robot.num_legs:
      raise ValueError(
          "The number of leg phases should be the same as number of legs.")
    self._initial_leg_phase = initial_leg_phase
    if len(initial_leg_state) != self._robot.num_legs:
      raise ValueError(
          "The number of leg states should be the same of number of legs.")
    self._initial_leg_state = initial_leg_state
    self._next_leg_state = []
    # The ratio in cycle is duty factor if initial state of the leg is STANCE,
    # and 1 - duty_factory if the initial state of the leg is SWING.
    self._initial_state_ratio_in_cycle = []
    for state, duty in zip(initial_leg_state, duty_factor):
      if state == gait_generator.LegState.SWING:
        self._initial_state_ratio_in_cycle.append(1 - duty)
        self._next_leg_state.append(gait_generator.LegState.STANCE)
      else:
        self._initial_state_ratio_in_cycle.append(duty)
        self._next_leg_state.append(gait_generator.LegState.SWING)

    self._contact_detection_phase_threshold = contact_detection_phase_threshold

    # The normalized phase within swing or stance duration.
    self._normalized_phase = None
    self._leg_state = None
    self._desired_leg_state = None

    self.reset(0)

  def reset(self, current_time):
    # The normalized phase within swing or stance duration.
    self._normalized_phase = np.zeros(self._robot.num_legs)
    self._leg_state = list(self._initial_leg_state)
    self._desired_leg_state = list(self._initial_leg_state)

  @property
  def desired_leg_state(self) -> Sequence[gait_generator.LegState]:
    """The desired leg SWING/STANCE states.

    Returns:
      The SWING/STANCE states for all legs.

    """
    return self._desired_leg_state

  @property
  def leg_state(self) -> Sequence[gait_generator.LegState]:
    """The leg state after considering contact with ground.

    Returns:
      The actual state of each leg after accounting for contacts.
    """
    return self._leg_state

  @property
  def swing_duration(self) -> Sequence[float]:
    return self._swing_duration

  @property
  def stance_duration(self) -> Sequence[float]:
    return self._stance_duration

  @property
  def normalized_phase(self) -> Sequence[float]:
    """The phase within the current swing or stance cycle.

    Reflects the leg's phase within the curren swing or stance stage. For
    example, at the end of the current swing duration, the phase will
    be set to 1 for all swing legs. Same for stance legs.

    Returns:
      Normalized leg phase for all legs.

    """
    return self._normalized_phase

  def update(self, current_time):
    contact_state = self._robot.GetFootContacts()
    for leg_id in range(self._robot.num_legs):
      # Here is the explanation behind this logic: We use the phase within the
      # full swing/stance cycle to determine if a swing/stance switch occurs
      # for a leg. The threshold value is the "initial_state_ratio_in_cycle" as
      # explained before. If the current phase is less than the initial state
      # ratio, the leg is either in the initial state or has switched back after
      # one or more full cycles.
      full_cycle_period = (self._stance_duration[leg_id] /
                           self._duty_factor[leg_id])
      # To account for the non-zero initial phase, we offset the time duration
      # with the effect time contribution from the initial leg phase.
      augmented_time = current_time + self._initial_leg_phase[
          leg_id] * full_cycle_period
      phase_in_full_cycle = math.fmod(augmented_time,
                                      full_cycle_period) / full_cycle_period
      ratio = self._initial_state_ratio_in_cycle[leg_id]
      if phase_in_full_cycle < ratio:
        self._desired_leg_state[leg_id] = self._initial_leg_state[leg_id]
        self._normalized_phase[leg_id] = phase_in_full_cycle / ratio
      else:
        # A phase switch happens for this leg.
        self._desired_leg_state[leg_id] = self._next_leg_state[leg_id]
        self._normalized_phase[leg_id] = (phase_in_full_cycle -
                                          ratio) / (1 - ratio)

      self._leg_state[leg_id] = self._desired_leg_state[leg_id]

      # No contact detection at the beginning of each SWING/STANCE phase.
      if (self._normalized_phase[leg_id] <
          self._contact_detection_phase_threshold):
        continue

      if (self._leg_state[leg_id] == gait_generator.LegState.SWING
          and contact_state[leg_id]):
        logging.info("early touch down detected.")
        self._leg_state[leg_id] = gait_generator.LegState.EARLY_CONTACT
      if (self._leg_state[leg_id] == gait_generator.LegState.STANCE
          and not contact_state[leg_id]):
        logging.info("lost contact detected.")
        self._leg_state[leg_id] = gait_generator.LegState.LOSE_CONTACT
