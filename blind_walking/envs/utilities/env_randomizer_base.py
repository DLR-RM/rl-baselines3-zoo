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

"""Abstract base class for environment randomizer."""



class EnvRandomizerBase(object):
  """Abstract base class for environment randomizer.

  Randomizes physical parameters of the objects in the simulation and adds
  perturbations to the stepping of the simulation.
  """

  def randomize_env(self, env):
    """Randomize the simulated_objects in the environment.

    Will be called at when env is reset. The physical parameters will be fixed
    for that episode and be randomized again in the next environment.reset().

    Args:
      env: The Minitaur gym environment to be randomized.
    """
    pass

  def randomize_step(self, env):
    """Randomize simulation steps.

    Will be called at every timestep. May add random forces/torques to Minitaur.

    Args:
      env: The Minitaur gym environment to be randomized.
    """
    pass
