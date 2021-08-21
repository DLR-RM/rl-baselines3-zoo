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
"""Base class for controllable environment randomizer."""
from blind_walking.envs.utilities import env_randomizer_base


class ControllableEnvRandomizerBase(env_randomizer_base.EnvRandomizerBase):
  """Base class for environment randomizer that can be manipulated explicitly.

  Randomizes physical parameters of the objects in the simulation and adds
  perturbations to the stepping of the simulation.
  """
  def get_randomization_parameters(self):
    """Get the parameters of the randomization."""
    raise NotImplementedError

  def set_randomization_from_parameters(self, env, randomization_parameters):
    """Set the parameters of the randomization."""
    raise NotImplementedError
