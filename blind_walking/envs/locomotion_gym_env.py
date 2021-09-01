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
"""This file implements the locomotion gym env."""
import collections
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet  # pytype: disable=import-error
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd

from blind_walking.robots import robot_config
from blind_walking.envs.sensors import sensor
from blind_walking.envs.sensors import space_utils
from blind_walking.envs.env_wrappers.heightfield import HeightField
from blind_walking.envs.env_wrappers.collapsibleplatform import CollapsiblePlatform


_ACTION_EPS = 0.01
_NUM_SIMULATION_ITERATION_STEPS = 300
_LOG_BUFFER_LENGTH = 5000


class LocomotionGymEnv(gym.Env):
  """The gym environment for the locomotion tasks."""
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second': 100
  }

  def __init__(self,
               gym_config,
               robot_class=None,
               env_sensors=None,
               robot_sensors=None,
               task=None,
               env_randomizers=None):
    """Initializes the locomotion gym environment.

    Args:
      gym_config: An instance of LocomotionGymConfig.
      robot_class: A class of a robot. We provide a class rather than an
        instance due to hard_reset functionality. Parameters are expected to be
        configured with gin.
      sensors: A list of environmental sensors for observation.
      task: A callable function/class to calculate the reward and termination
        condition. Takes the gym env as the argument when calling.
      env_randomizers: A list of EnvRandomizer(s). An EnvRandomizer may
        randomize the physical property of minitaur, change the terrrain during
        reset(), or add perturbation forces during step().

    Raises:
      ValueError: If the num_action_repeat is less than 1.

    """

    self.seed()
    self._gym_config = gym_config
    self._robot_class = robot_class
    self._robot_sensors = robot_sensors

    self._sensors = env_sensors if env_sensors is not None else list()
    if self._robot_class is None:
      raise ValueError('robot_class cannot be None.')

    # A dictionary containing the objects in the world other than the robot.
    self._world_dict = {}
    self._task = task

    self._env_randomizers = env_randomizers if env_randomizers else []

    # This is a workaround due to the issue in b/130128505#comment5
    if isinstance(self._task, sensor.Sensor):
      self._sensors.append(self._task)

    # Simulation related parameters.
    self._num_action_repeat = gym_config.simulation_parameters.num_action_repeat
    self._on_rack = gym_config.simulation_parameters.robot_on_rack
    if self._num_action_repeat < 1:
      raise ValueError('number of action repeats should be at least 1.')
    self._sim_time_step = gym_config.simulation_parameters.sim_time_step_s
    self._env_time_step = self._num_action_repeat * self._sim_time_step
    self._env_step_counter = 0

    self._num_bullet_solver_iterations = int(_NUM_SIMULATION_ITERATION_STEPS /
                                             self._num_action_repeat)
    self._is_render = gym_config.simulation_parameters.enable_rendering

    # The wall-clock time at which the last frame is rendered.
    self._last_frame_time = 0.0
    self._show_reference_id = -1

    if self._is_render:
      self._pybullet_client = bullet_client.BulletClient(
          connection_mode=pybullet.GUI)
      pybullet.configureDebugVisualizer(
          pybullet.COV_ENABLE_GUI,
          gym_config.simulation_parameters.enable_rendering_gui)
    else:
      self._pybullet_client = bullet_client.BulletClient(
          connection_mode=pybullet.DIRECT)
    self._pybullet_client.setAdditionalSearchPath(pd.getDataPath())
    if gym_config.simulation_parameters.egl_rendering:
      self._pybullet_client.loadPlugin('eglRendererPlugin')

    # The action list contains the name of all actions.
    self._build_action_space()

    # Set the default render options.
    self._camera_dist = gym_config.simulation_parameters.camera_distance
    self._camera_yaw = gym_config.simulation_parameters.camera_yaw
    self._camera_pitch = gym_config.simulation_parameters.camera_pitch
    self._render_width = gym_config.simulation_parameters.render_width
    self._render_height = gym_config.simulation_parameters.render_height

    self.height_field = False
    self.collapsible_tile = False
    self.collapsible_platform = False

    self._hard_reset = True
    self.reset()
    self._hard_reset = gym_config.simulation_parameters.enable_hard_reset

    # Construct the observation space from the list of sensors. Note that we
    # will reconstruct the observation_space after the robot is created.
    self.observation_space = (
        space_utils.convert_sensors_to_gym_space_dictionary(
            self.all_sensors()))

    # Set terrain type
    self.terrain_type = gym_config.simulation_parameters.terrain_type
    self.height_field = self.terrain_type == 1
    self.collapsible_tile = self.terrain_type == 2
    self.collapsible_platform = self.terrain_type == 3
    # Terrain generators
    self.hf = HeightField()
    self.cp = CollapsiblePlatform()

    # Set the default height field options.
    self.height_field_iters = gym_config.simulation_parameters.height_field_iters
    self.height_field_friction = gym_config.simulation_parameters.height_field_friction
    self.height_field_perturbation_range = gym_config.simulation_parameters.height_field_perturbation_range
    # Generate terrain
    if self.height_field:
      # Extra roughness generated with each iteration
      for _ in range(self.height_field_iters):
        self.hf._generate_field(self,
                                friction=self.height_field_friction,
                                heightPerturbationRange=self.height_field_perturbation_range)
    # If collapsible tile or platform, enable hard reset
    elif self.collapsible_tile:
      self.cp._generate_field(self)
      self._hard_reset = True
    elif self.collapsible_platform:
      self.cp._generate_soft_env(self)
      self._hard_reset = True

  def _build_action_space(self):
    """Builds action space based on motor control mode."""
    motor_mode = self._gym_config.simulation_parameters.motor_control_mode
    if motor_mode == robot_config.MotorControlMode.HYBRID:
      action_upper_bound = []
      action_lower_bound = []
      action_config = self._robot_class.ACTION_CONFIG
      for action in action_config:
        action_upper_bound.extend([6.28] * 5)
        action_lower_bound.extend([-6.28] * 5)
      self.action_space = spaces.Box(np.array(action_lower_bound),
                                     np.array(action_upper_bound),
                                     dtype=np.float32)
    elif motor_mode == robot_config.MotorControlMode.TORQUE:
      # TODO (yuxiangy): figure out the torque limits of robots.
      torque_limits = np.array([100] * len(self._robot_class.ACTION_CONFIG))
      self.action_space = spaces.Box(-torque_limits,
                                     torque_limits,
                                     dtype=np.float32)
    else:
      # Position mode
      action_upper_bound = []
      action_lower_bound = []
      action_config = self._robot_class.ACTION_CONFIG
      for action in action_config:
        action_upper_bound.append(action.upper_bound)
        action_lower_bound.append(action.lower_bound)

      self.action_space = spaces.Box(np.array(action_lower_bound),
                                     np.array(action_upper_bound),
                                     dtype=np.float32)

  def close(self):
    if hasattr(self, '_robot') and self._robot:
      self._robot.Terminate()

  def seed(self, seed=None):
    self.np_random, self.np_random_seed = seeding.np_random(seed)
    return [self.np_random_seed]

  def all_sensors(self):
    """Returns all robot and environmental sensors."""
    return self._robot.GetAllSensors() + self._sensors

  def sensor_by_name(self, name):
    """Returns the sensor with the given name, or None if not exist."""
    for sensor_ in self.all_sensors():
      if sensor_.get_name() == name:
        return sensor_
    return None

  def reset(self,
            initial_motor_angles=None,
            reset_duration=0.0,
            reset_visualization_camera=True):
    """Resets the robot's position in the world or rebuild the sim world.

    The simulation world will be rebuilt if self._hard_reset is True.

    Args:
      initial_motor_angles: A list of Floats. The desired joint angles after
        reset. If None, the robot will use its built-in value.
      reset_duration: Float. The time (in seconds) needed to rotate all motors
        to the desired initial values.
      reset_visualization_camera: Whether to reset debug visualization camera on
        reset.

    Returns:
      A numpy array contains the initial observation after reset.
    """
    if self._is_render:
      self._pybullet_client.configureDebugVisualizer(
          self._pybullet_client.COV_ENABLE_RENDERING, 0)

    # Clear the simulation world and rebuild the robot interface.
    if self._hard_reset:
      if self.collapsible_tile or self.collapsible_platform:
        self._pybullet_client.resetSimulation(self._pybullet_client.RESET_USE_DEFORMABLE_WORLD)
      else:
        self._pybullet_client.resetSimulation()
      self._pybullet_client.setPhysicsEngineParameter(
          numSolverIterations=self._num_bullet_solver_iterations)
      self._pybullet_client.setTimeStep(self._sim_time_step)
      self._pybullet_client.setGravity(0, 0, -10)

      # Rebuild the world.
      self._world_dict = {
          "ground": self._pybullet_client.loadURDF("plane_implicit.urdf")
      }

      adjust_position = [0, 0, 0]
      # Adjust position if there is a collapsible platform
      # Generate collapsible platform
      if self.collapsible_tile:
        adjust_position = [0, 0, 0.5]
        self.cp._generate_field(self)
      elif self.collapsible_platform:
        adjust_position = [0, 0, 0.15]
        self.cp._generate_soft_env(self)

      # Rebuild the robot
      self._robot = self._robot_class(
          pybullet_client=self._pybullet_client,
          sensors=self._robot_sensors,
          on_rack=self._on_rack,
          action_repeat=self._gym_config.simulation_parameters.
          num_action_repeat,
          motor_control_mode=self._gym_config.simulation_parameters.
          motor_control_mode,
          reset_time=self._gym_config.simulation_parameters.reset_time,
          enable_clip_motor_commands=self._gym_config.simulation_parameters.
          enable_clip_motor_commands,
          enable_action_filter=self._gym_config.simulation_parameters.
          enable_action_filter,
          enable_action_interpolation=self._gym_config.simulation_parameters.
          enable_action_interpolation,
          allow_knee_contact=self._gym_config.simulation_parameters.
          allow_knee_contact,
          adjust_position=adjust_position)

    # Reset the pose of the robot.
    self._robot.Reset(reload_urdf=False,
                      default_motor_angles=initial_motor_angles,
                      reset_time=reset_duration)

    self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
    self._env_step_counter = 0
    if reset_visualization_camera:
      self._pybullet_client.resetDebugVisualizerCamera(self._camera_dist,
                                                       self._camera_yaw,
                                                       self._camera_pitch,
                                                       [0, 0, 0])
    self._last_action = np.zeros(self.action_space.shape)

    if self._is_render:
      self._pybullet_client.configureDebugVisualizer(
          self._pybullet_client.COV_ENABLE_RENDERING, 1)

    for s in self.all_sensors():
      s.on_reset(self)

    if self._task and hasattr(self._task, 'reset'):
      self._task.reset(self)

    # Loop over all env randomizers.
    for env_randomizer in self._env_randomizers:
      env_randomizer.randomize_env(self)

    return self._get_observation()

  def step(self, action):
    """Step forward the simulation, given the action.

    Args:
      action: Can be a list of desired motor angles for all motors when the
        robot is in position control mode; A list of desired motor torques. Or a
        list of tuples (q, qdot, kp, kd, tau) for hybrid control mode. The
        action must be compatible with the robot's motor control mode. Also, we
        are not going to use the leg space (swing/extension) definition at the
        gym level, since they are specific to Minitaur.

    Returns:
      observations: The observation dictionary. The keys are the sensor names
        and the values are the sensor readings.
      reward: The reward for the current state-action pair.
      done: Whether the episode has ended.
      info: A dictionary that stores diagnostic information.

    Raises:
      ValueError: The action dimension is not the same as the number of motors.
      ValueError: The magnitude of actions is out of bounds.
    """
    self._last_base_position = self._robot.GetBasePosition()
    self._last_action = action

    if self._is_render:
      # Sleep, otherwise the computation takes less time than real time,
      # which will make the visualization like a fast-forward video.
      time_spent = time.time() - self._last_frame_time
      self._last_frame_time = time.time()
      time_to_sleep = self._env_time_step - time_spent
      if time_to_sleep > 0:
        time.sleep(time_to_sleep)
      base_pos = self._robot.GetBasePosition()

      # Also keep the previous orientation of the camera set by the user.
      [yaw, pitch,
       dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
      self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch,
                                                       base_pos)
      self._pybullet_client.configureDebugVisualizer(
          self._pybullet_client.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

    for env_randomizer in self._env_randomizers:
      env_randomizer.randomize_step(self)

    # robot class and put the logics here.
    self._robot.Step(action)

    for s in self.all_sensors():
      s.on_step(self)

    if self._task and hasattr(self._task, 'update'):
      self._task.update(self)

    reward = self._reward()

    done = self._termination()
    self._env_step_counter += 1
    if done:
      self._robot.Terminate()
    return self._get_observation(), reward, done, {}

  def render(self, mode='rgb_array'):
    if mode != 'rgb_array':
      raise ValueError('Unsupported render mode:{}'.format(mode))
    base_pos = self._robot.GetBasePosition()
    view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._camera_dist,
        yaw=self._camera_yaw,
        pitch=self._camera_pitch,
        roll=0,
        upAxisIndex=2)
    proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
        fov=60,
        aspect=float(self._render_width) / self._render_height,
        nearVal=0.1,
        farVal=100.0)
    (w, h, px, _, _) = self._pybullet_client.getCameraImage(
        width=self._render_width,
        height=self._render_height,
        renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix)
    
    rgb_array = np.array(px)
    rgb_array = rgb_array.reshape(h,w,4)
    rgb_array = rgb_array.astype(np.uint8)
    return rgb_array

  def get_ground(self):
    """Get simulation ground model."""
    return self._world_dict['ground']

  def set_ground(self, ground_id):
    """Set simulation ground model."""
    self._world_dict['ground'] = ground_id

  @property
  def rendering_enabled(self):
    return self._is_render

  @property
  def last_base_position(self):
    return self._last_base_position

  @property
  def world_dict(self):
    return self._world_dict.copy()

  @world_dict.setter
  def world_dict(self, new_dict):
    self._world_dict = new_dict.copy()

  def _termination(self):
    if not self._robot.is_safe:
      return True

    if self._task and hasattr(self._task, 'done'):
      return self._task.done(self)

    for s in self.all_sensors():
      s.on_terminate(self)

    return False

  def _reward(self):
    if self._task:
      return self._task(self)
    return 0

  def _get_observation(self):
    """Get observation of this environment from a list of sensors.

    Returns:
      observations: sensory observation in the numpy array format
    """
    sensors_dict = {}
    for s in self.all_sensors():
      sensors_dict[s.get_name()] = s.get_observation()

    observations = collections.OrderedDict(list(sensors_dict.items()))
    return observations

  def _get_env_state(self):
    """Get simulation environment state

    Returns:
      state: carry mass, carry mass position, motor strength, friction, terrain height
        in numpy array format
    """
    carry_mass = self.carry_mass
    carry_mass_position = self.carry_mass_pos
    motor_strength = self.robot._motor_model._strength_ratios
    friction = self.height_field_friction
    terrain_height = self.height_field_perturbation_range
    return np.hstack([carry_mass, carry_mass_position, motor_strength, friction, terrain_height])

  def set_time_step(self, num_action_repeat, sim_step=0.001):
    """Sets the time step of the environment.

    Args:
      num_action_repeat: The number of simulation steps/action repeats to be
        executed when calling env.step().
      sim_step: The simulation time step in PyBullet. By default, the simulation
        step is 0.001s, which is a good trade-off between simulation speed and
        accuracy.

    Raises:
      ValueError: If the num_action_repeat is less than 1.
    """
    if num_action_repeat < 1:
      raise ValueError('number of action repeats should be at least 1.')
    self._sim_time_step = sim_step
    self._num_action_repeat = num_action_repeat
    self._env_time_step = sim_step * num_action_repeat
    self._num_bullet_solver_iterations = (_NUM_SIMULATION_ITERATION_STEPS /
                                          self._num_action_repeat)
    self._pybullet_client.setPhysicsEngineParameter(
        numSolverIterations=int(np.round(self._num_bullet_solver_iterations)))
    self._pybullet_client.setTimeStep(self._sim_time_step)
    self._robot.SetTimeSteps(self._num_action_repeat, self._sim_time_step)

  def get_time_since_reset(self):
    """Get the time passed (in seconds) since the last reset.

    Returns:
      Time in seconds since the last reset.
    """
    return self._robot.GetTimeSinceReset()

  @property
  def pybullet_client(self):
    return self._pybullet_client

  @property
  def robot(self):
    return self._robot

  @property
  def env_step_counter(self):
    return self._env_step_counter

  @property
  def hard_reset(self):
    return self._hard_reset

  @property
  def last_action(self):
    return self._last_action

  @property
  def env_time_step(self):
    return self._env_time_step

  @property
  def task(self):
    return self._task

  @property
  def robot_class(self):
    return self._robot_class
