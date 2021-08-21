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
"""A controllable env randomizer that randomizes physical params from config."""
from absl import logging
import copy
import functools
import numpy as np

from blind_walking.envs.utilities import controllable_env_randomizer_base
from blind_walking.envs.utilities import minitaur_env_randomizer_config

SIMULATION_TIME_STEP = 0.001
NUM_LEGS = 4


class ControllableEnvRandomizerFromConfig(
    controllable_env_randomizer_base.ControllableEnvRandomizerBase):
  """A randomizer that change the minitaur_gym_env during every reset."""
  def __init__(self,
               config=None,
               verbose=True,
               param_bounds=(-1., 1.),
               randomization_seed=None):
    if config is None:
      config = "all_params"
    try:
      config = getattr(minitaur_env_randomizer_config, config)
    except AttributeError:
      raise ValueError("Config {} is not found.".format(config))
    self._randomization_param_dict = config()
    logging.info("Randomization config is: {}".format(
        self._randomization_param_dict))

    self._randomization_param_value_dict = {}
    self._randomization_seed = randomization_seed
    self._param_bounds = param_bounds
    self._suspend_randomization = False
    self._verbose = verbose
    self._rejection_param_range = {}

    self._np_random = np.random.RandomState()

  @property
  def suspend_randomization(self):
    return self._suspend_randomization

  @suspend_randomization.setter
  def suspend_randomization(self, suspend_rand):
    self._suspend_randomization = suspend_rand

  @property
  def randomization_seed(self):
    """Area of the square."""
    return self._randomization_seed

  @randomization_seed.setter
  def randomization_seed(self, seed):
    self._randomization_seed = seed

  def _check_all_randomization_parameter_in_rejection_range(self):
    """Check if current parameters are in the region to be rejected."""

    for param_name, reject_random_range in sorted(
        self._rejection_param_range.items()):
      randomized_value = self._randomization_param_value_dict[param_name]
      if np.any(randomized_value < reject_random_range[0]) or np.any(
          randomized_value > reject_random_range[1]):
        return False
    return True

  def randomize_env(self, env):
    """Randomize various physical properties of the environment.

    It randomizes the physical parameters according to input configuration.

    Args:
      env: A minitaur gym environment.
    """

    if not self.suspend_randomization:
      # Use a specific seed for controllable randomization.
      if self._randomization_seed is not None:
        self._np_random.seed(self._randomization_seed)

      self._randomization_function_dict = self._build_randomization_func_dict(
          env)

      self._rejection_param_range = {}
      for param_name, random_range in sorted(
          self._randomization_param_dict.items()):
        self._randomization_function_dict[param_name](
            lower_bound=random_range[0], upper_bound=random_range[1])
        if len(random_range) == 4:
          self._rejection_param_range[param_name] = [
              random_range[2], random_range[3]
          ]
      if self._rejection_param_range:
        while self._check_all_randomization_parameter_in_rejection_range():
          for param_name, random_range in sorted(
              self._randomization_param_dict.items()):
            self._randomization_function_dict[param_name](
                lower_bound=random_range[0], upper_bound=random_range[1])
    elif self._randomization_param_value_dict:
      # Re-apply the randomization because hard_reset might change previously
      # randomized parameters.
      self.set_env_from_randomization_parameters(
          env, self._randomization_param_value_dict)

  def get_randomization_parameters(self):
    return copy.deepcopy(self._randomization_param_value_dict)

  def set_env_from_randomization_parameters(self, env,
                                            randomization_parameters):
    self._randomization_param_value_dict = randomization_parameters
    # Run the randomization function to propgate the parameters.
    self._randomization_function_dict = self._build_randomization_func_dict(
        env)
    for param_name, random_range in self._randomization_param_dict.items():
      self._randomization_function_dict[param_name](
          lower_bound=random_range[0],
          upper_bound=random_range[1],
          parameters=randomization_parameters[param_name])

  def _get_robot_from_env(self, env):
    if hasattr(env, "minitaur"):  #  Compabible with v1 envs.
      return env.minitaur
    elif hasattr(env, "robot"):  # Compatible with v2 envs.
      return env.robot
    else:
      return None

  def _build_randomization_func_dict(self, env):
    func_dict = {}
    robot = self._get_robot_from_env(env)
    func_dict["mass"] = functools.partial(self._randomize_masses,
                                          minitaur=robot)
    func_dict["individual mass"] = functools.partial(
        self._randomize_individual_masses, minitaur=robot)
    func_dict["base mass"] = functools.partial(self._randomize_basemass,
                                               minitaur=robot)
    func_dict["inertia"] = functools.partial(self._randomize_inertia,
                                             minitaur=robot)
    func_dict["individual inertia"] = functools.partial(
        self._randomize_individual_inertia, minitaur=robot)
    func_dict["latency"] = functools.partial(self._randomize_latency,
                                             minitaur=robot)
    func_dict["joint friction"] = functools.partial(
        self._randomize_joint_friction, minitaur=robot)
    func_dict["motor friction"] = functools.partial(
        self._randomize_motor_friction, minitaur=robot)
    func_dict["restitution"] = functools.partial(
        self._randomize_contact_restitution, minitaur=robot)
    func_dict["lateral friction"] = functools.partial(
        self._randomize_contact_friction, minitaur=robot)
    func_dict["battery"] = functools.partial(self._randomize_battery_level,
                                             minitaur=robot)
    func_dict["motor strength"] = functools.partial(
        self._randomize_motor_strength, minitaur=robot)
    func_dict["global motor strength"] = functools.partial(
        self._randomize_global_motor_strength, minitaur=robot)
    # Setting control step needs access to the environment.
    func_dict["control step"] = functools.partial(self._randomize_control_step,
                                                  env=env)
    func_dict["leg weaken"] = functools.partial(self._randomize_leg_weakening,
                                                minitaur=robot)
    func_dict["single leg weaken"] = functools.partial(
        self._randomize_single_leg_weakening, minitaur=robot)
    return func_dict

  def _randomize_control_step(self,
                              env,
                              lower_bound,
                              upper_bound,
                              parameters=None):
    if parameters is None:
      sample = self._np_random.uniform(self._param_bounds[0],
                                       self._param_bounds[1])
    else:
      sample = parameters
    self._randomization_param_value_dict["control step"] = sample
    randomized_control_step = (sample - self._param_bounds[0]) / (
        self._param_bounds[1] -
        self._param_bounds[0]) * (upper_bound - lower_bound) + lower_bound
    randomized_control_step = int(randomized_control_step)
    env.set_time_step(randomized_control_step)
    if self._verbose:
      logging.info("control step is: {}".format(randomized_control_step))

  def _randomize_masses(self,
                        minitaur,
                        lower_bound,
                        upper_bound,
                        parameters=None):
    if parameters is None:
      sample = self._np_random.uniform([self._param_bounds[0]] * 2,
                                       [self._param_bounds[1]] * 2)
    else:
      sample = parameters

    self._randomization_param_value_dict["mass"] = sample
    randomized_mass_ratios = (sample - self._param_bounds[0]) / (
        self._param_bounds[1] -
        self._param_bounds[0]) * (upper_bound - lower_bound) + lower_bound

    base_mass = minitaur.GetBaseMassesFromURDF()
    random_base_ratio = randomized_mass_ratios[0]
    randomized_base_mass = random_base_ratio * np.array(base_mass)
    minitaur.SetBaseMasses(randomized_base_mass)
    if self._verbose:
      logging.info("base mass is: {}".format(randomized_base_mass))

    leg_masses = minitaur.GetLegMassesFromURDF()
    random_leg_ratio = randomized_mass_ratios[1]
    randomized_leg_masses = random_leg_ratio * np.array(leg_masses)
    minitaur.SetLegMasses(randomized_leg_masses)
    if self._verbose:
      logging.info("leg mass is: {}".format(randomized_leg_masses))

  def _randomize_individual_masses(self,
                                   minitaur,
                                   lower_bound,
                                   upper_bound,
                                   parameters=None):
    base_mass = minitaur.GetBaseMassesFromURDF()
    leg_masses = minitaur.GetLegMassesFromURDF()
    param_dim = len(base_mass) + len(leg_masses)
    if parameters is None:
      sample = self._np_random.uniform([self._param_bounds[0]] * param_dim,
                                       [self._param_bounds[1]] * param_dim)
    else:
      sample = parameters
    self._randomization_param_value_dict["individual mass"] = sample
    randomized_mass_ratios = (sample - self._param_bounds[0]) / (
        self._param_bounds[1] -
        self._param_bounds[0]) * (upper_bound - lower_bound) + lower_bound

    random_base_ratio = randomized_mass_ratios[0:len(base_mass)]
    randomized_base_mass = random_base_ratio * np.array(base_mass)
    minitaur.SetBaseMasses(randomized_base_mass)
    if self._verbose:
      logging.info("base mass is: {}".format(randomized_base_mass))

    random_leg_ratio = randomized_mass_ratios[len(base_mass):]
    randomized_leg_masses = random_leg_ratio * np.array(leg_masses)
    minitaur.SetLegMasses(randomized_leg_masses)
    if self._verbose:
      logging.info("randomization dim: {}".format(param_dim))
      logging.info("leg mass is: {}".format(randomized_leg_masses))

  def _randomize_basemass(self,
                          minitaur,
                          lower_bound,
                          upper_bound,
                          parameters=None):
    if parameters is None:
      sample = self._np_random.uniform(self._param_bounds[0],
                                       self._param_bounds[1])
    else:
      sample = parameters
    self._randomization_param_value_dict["base mass"] = sample
    randomized_mass_ratios = (sample - self._param_bounds[0]) / (
        self._param_bounds[1] -
        self._param_bounds[0]) * (upper_bound - lower_bound) + lower_bound

    base_mass = minitaur.GetBaseMassesFromURDF()
    random_base_ratio = randomized_mass_ratios
    randomized_base_mass = random_base_ratio * np.array(base_mass)
    minitaur.SetBaseMasses(randomized_base_mass)
    if self._verbose:
      logging.info("base mass is: {}".format(randomized_base_mass))

  def _randomize_individual_inertia(self,
                                    minitaur,
                                    lower_bound,
                                    upper_bound,
                                    parameters=None):
    base_inertia = minitaur.GetBaseInertiasFromURDF()
    leg_inertia = minitaur.GetLegInertiasFromURDF()
    param_dim = (len(base_inertia) + len(leg_inertia)) * 3

    if parameters is None:
      sample = self._np_random.uniform([self._param_bounds[0]] * param_dim,
                                       [self._param_bounds[1]] * param_dim)
    else:
      sample = parameters
    self._randomization_param_value_dict["individual inertia"] = sample
    randomized_inertia_ratios = (sample - self._param_bounds[0]) / (
        self._param_bounds[1] -
        self._param_bounds[0]) * (upper_bound - lower_bound) + lower_bound
    random_base_ratio = np.reshape(
        randomized_inertia_ratios[0:len(base_inertia) * 3],
        (len(base_inertia), 3))
    randomized_base_inertia = random_base_ratio * np.array(base_inertia)
    minitaur.SetBaseInertias(randomized_base_inertia)
    if self._verbose:
      logging.info("base inertia is: {}".format(randomized_base_inertia))
    random_leg_ratio = np.reshape(
        randomized_inertia_ratios[len(base_inertia) * 3:],
        (len(leg_inertia), 3))
    randomized_leg_inertia = random_leg_ratio * np.array(leg_inertia)
    minitaur.SetLegInertias(randomized_leg_inertia)
    if self._verbose:
      logging.info("leg inertia is: {}".format(randomized_leg_inertia))

  def _randomize_inertia(self,
                         minitaur,
                         lower_bound,
                         upper_bound,
                         parameters=None):
    if parameters is None:
      sample = self._np_random.uniform([self._param_bounds[0]] * 2,
                                       [self._param_bounds[1]] * 2)
    else:
      sample = parameters
    self._randomization_param_value_dict["inertia"] = sample
    randomized_inertia_ratios = (sample - self._param_bounds[0]) / (
        self._param_bounds[1] -
        self._param_bounds[0]) * (upper_bound - lower_bound) + lower_bound

    base_inertia = minitaur.GetBaseInertiasFromURDF()
    random_base_ratio = randomized_inertia_ratios[0]
    randomized_base_inertia = random_base_ratio * np.array(base_inertia)
    minitaur.SetBaseInertias(randomized_base_inertia)
    if self._verbose:
      logging.info("base inertia is: {}".format(randomized_base_inertia))
    leg_inertia = minitaur.GetLegInertiasFromURDF()
    random_leg_ratio = randomized_inertia_ratios[1]
    randomized_leg_inertia = random_leg_ratio * np.array(leg_inertia)
    minitaur.SetLegInertias(randomized_leg_inertia)
    if self._verbose:
      logging.info("leg inertia is: {}".format(randomized_leg_inertia))

  def _randomize_latency(self,
                         minitaur,
                         lower_bound,
                         upper_bound,
                         parameters=None):
    if parameters is None:
      sample = self._np_random.uniform(self._param_bounds[0],
                                       self._param_bounds[1])
    else:
      sample = parameters
    self._randomization_param_value_dict["latency"] = sample
    randomized_latency = (sample - self._param_bounds[0]) / (
        self._param_bounds[1] -
        self._param_bounds[0]) * (upper_bound - lower_bound) + lower_bound

    minitaur.SetControlLatency(randomized_latency)
    if self._verbose:
      logging.info("control latency is: {}".format(randomized_latency))

  def _randomize_joint_friction(self,
                                minitaur,
                                lower_bound,
                                upper_bound,
                                parameters=None):
    num_knee_joints = minitaur.GetNumKneeJoints()

    if parameters is None:
      sample = self._np_random.uniform(
          [self._param_bounds[0]] * num_knee_joints,
          [self._param_bounds[1]] * num_knee_joints)
    else:
      sample = parameters
    self._randomization_param_value_dict["joint friction"] = sample
    randomized_joint_frictions = (sample - self._param_bounds[0]) / (
        self._param_bounds[1] -
        self._param_bounds[0]) * (upper_bound - lower_bound) + lower_bound

    minitaur.SetJointFriction(randomized_joint_frictions)
    if self._verbose:
      logging.info("joint friction is: {}".format(randomized_joint_frictions))

  def _randomize_motor_friction(self,
                                minitaur,
                                lower_bound,
                                upper_bound,
                                parameters=None):
    if parameters is None:
      sample = self._np_random.uniform(self._param_bounds[0],
                                       self._param_bounds[1])
    else:
      sample = parameters
    self._randomization_param_value_dict["motor friction"] = sample
    randomized_motor_damping = (sample - self._param_bounds[0]) / (
        self._param_bounds[1] -
        self._param_bounds[0]) * (upper_bound - lower_bound) + lower_bound

    minitaur.SetMotorViscousDamping(randomized_motor_damping)
    if self._verbose:
      logging.info("motor friction is: {}".format(randomized_motor_damping))

  def _randomize_contact_restitution(self,
                                     minitaur,
                                     lower_bound,
                                     upper_bound,
                                     parameters=None):
    if parameters is None:
      sample = self._np_random.uniform(self._param_bounds[0],
                                       self._param_bounds[1])
    else:
      sample = parameters
    self._randomization_param_value_dict["restitution"] = sample
    randomized_restitution = (sample - self._param_bounds[0]) / (
        self._param_bounds[1] -
        self._param_bounds[0]) * (upper_bound - lower_bound) + lower_bound

    minitaur.SetFootRestitution(randomized_restitution)
    if self._verbose:
      logging.info("foot restitution is: {}".format(randomized_restitution))

  def _randomize_contact_friction(self,
                                  minitaur,
                                  lower_bound,
                                  upper_bound,
                                  parameters=None):
    if parameters is None:
      sample = self._np_random.uniform(self._param_bounds[0],
                                       self._param_bounds[1])
    else:
      sample = parameters
    self._randomization_param_value_dict["lateral friction"] = sample
    randomized_foot_friction = (sample - self._param_bounds[0]) / (
        self._param_bounds[1] -
        self._param_bounds[0]) * (upper_bound - lower_bound) + lower_bound

    minitaur.SetFootFriction(randomized_foot_friction)
    if self._verbose:
      logging.info("foot friction is: {}".format(randomized_foot_friction))

  def _randomize_battery_level(self,
                               minitaur,
                               lower_bound,
                               upper_bound,
                               parameters=None):
    if parameters is None:
      sample = self._np_random.uniform(self._param_bounds[0],
                                       self._param_bounds[1])
    else:
      sample = parameters
    self._randomization_param_value_dict["battery"] = sample
    randomized_battery_voltage = (sample - self._param_bounds[0]) / (
        self._param_bounds[1] -
        self._param_bounds[0]) * (upper_bound - lower_bound) + lower_bound

    minitaur.SetBatteryVoltage(randomized_battery_voltage)
    if self._verbose:
      logging.info("battery voltage is: {}".format(randomized_battery_voltage))

  def _randomize_global_motor_strength(self,
                                       minitaur,
                                       lower_bound,
                                       upper_bound,
                                       parameters=None):
    if parameters is None:
      sample = self._np_random.uniform(self._param_bounds[0],
                                       self._param_bounds[1])
    else:
      sample = parameters
    self._randomization_param_value_dict["global motor strength"] = sample
    randomized_motor_strength_ratio = (sample - self._param_bounds[0]) / (
        self._param_bounds[1] -
        self._param_bounds[0]) * (upper_bound - lower_bound) + lower_bound

    minitaur.SetMotorStrengthRatios([randomized_motor_strength_ratio] *
                                    minitaur.num_motors)
    if self._verbose:
      logging.info("global motor strength is: {}".format(
          randomized_motor_strength_ratio))

  def _randomize_motor_strength(self,
                                minitaur,
                                lower_bound,
                                upper_bound,
                                parameters=None):
    if parameters is None:
      sample = self._np_random.uniform(
          [self._param_bounds[0]] * minitaur.num_motors,
          [self._param_bounds[1]] * minitaur.num_motors)
    else:
      sample = parameters
    self._randomization_param_value_dict["motor strength"] = sample
    randomized_motor_strength_ratios = (sample - self._param_bounds[0]) / (
        self._param_bounds[1] -
        self._param_bounds[0]) * (upper_bound - lower_bound) + lower_bound

    minitaur.SetMotorStrengthRatios(randomized_motor_strength_ratios)
    if self._verbose:
      logging.info(
          "motor strength is: {}".format(randomized_motor_strength_ratios))

  def _randomize_leg_weakening(self,
                               minitaur,
                               lower_bound,
                               upper_bound,
                               parameters=None):
    motor_per_leg = int(minitaur.num_motors / NUM_LEGS)
    if parameters is None:
      # First choose which leg to weaken
      leg_to_weaken = self._np_random.randint(NUM_LEGS)

      # Choose what ratio to randomize
      normalized_ratio = self._np_random.uniform(self._param_bounds[0],
                                                 self._param_bounds[1])
      sample = [leg_to_weaken, normalized_ratio]
    else:
      sample = [parameters[0], parameters[1]]
      leg_to_weaken = sample[0]
      normalized_ratio = sample[1]

    self._randomization_param_value_dict["leg weaken"] = sample

    leg_weaken_ratio = (normalized_ratio - self._param_bounds[0]) / (
        self._param_bounds[1] -
        self._param_bounds[0]) * (upper_bound - lower_bound) + lower_bound

    motor_strength_ratios = np.ones(minitaur.num_motors)
    motor_strength_ratios[leg_to_weaken * motor_per_leg:(leg_to_weaken + 1) *
                          motor_per_leg] = leg_weaken_ratio
    minitaur.SetMotorStrengthRatios(motor_strength_ratios)
    if self._verbose:
      logging.info("weakening leg {} with ratio: {}".format(
          leg_to_weaken, leg_weaken_ratio))

  def _randomize_single_leg_weakening(self,
                                      minitaur,
                                      lower_bound,
                                      upper_bound,
                                      parameters=None):
    motor_per_leg = int(minitaur.num_motors / NUM_LEGS)
    leg_to_weaken = 0
    if parameters is None:
      # Choose what ratio to randomize
      normalized_ratio = self._np_random.uniform(self._param_bounds[0],
                                                 self._param_bounds[1])
    else:
      normalized_ratio = parameters

    self._randomization_param_value_dict[
        "single leg weaken"] = normalized_ratio

    leg_weaken_ratio = (normalized_ratio - self._param_bounds[0]) / (
        self._param_bounds[1] -
        self._param_bounds[0]) * (upper_bound - lower_bound) + lower_bound

    motor_strength_ratios = np.ones(minitaur.num_motors)
    motor_strength_ratios[leg_to_weaken * motor_per_leg:(leg_to_weaken + 1) *
                          motor_per_leg] = leg_weaken_ratio
    minitaur.SetMotorStrengthRatios(motor_strength_ratios)
    if self._verbose:
      logging.info("weakening leg {} with ratio: {}".format(
          leg_to_weaken, leg_weaken_ratio))
