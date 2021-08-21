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

"""Utility functions to manipulate environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from gym import spaces
import numpy as np

def flatten_observations(observation_dict, observation_excluded=()):
  """Flattens the observation dictionary to an array.

  If observation_excluded is passed in, it will still return a dictionary,
  which includes all the (key, observation_dict[key]) in observation_excluded,
  and ('other': the flattened array).

  Args:
    observation_dict: A dictionary of all the observations.
    observation_excluded: A list/tuple of all the keys of the observations to be
      ignored during flattening.

  Returns:
    An array or a dictionary of observations based on whether
      observation_excluded is empty.
  """
  if not isinstance(observation_excluded, (list, tuple)):
    observation_excluded = [observation_excluded]
  observations = []
  for key, value in observation_dict.items():
    if key not in observation_excluded:
      observations.append(np.asarray(value).flatten())
  flat_observations = np.concatenate(observations)
  if not observation_excluded:
    return flat_observations
  else:
    observation_dict_after_flatten = {"other": flat_observations}
    for key in observation_excluded:
      observation_dict_after_flatten[key] = observation_dict[key]
    return collections.OrderedDict(
        sorted(list(observation_dict_after_flatten.items())))


def flatten_observation_spaces(observation_spaces, observation_excluded=()):
  """Flattens the dictionary observation spaces to gym.spaces.Box.

  If observation_excluded is passed in, it will still return a dictionary,
  which includes all the (key, observation_spaces[key]) in observation_excluded,
  and ('other': the flattened Box space).

  Args:
    observation_spaces: A dictionary of all the observation spaces.
    observation_excluded: A list/tuple of all the keys of the observations to be
      ignored during flattening.

  Returns:
    A box space or a dictionary of observation spaces based on whether
      observation_excluded is empty.
  """
  if not isinstance(observation_excluded, (list, tuple)):
    observation_excluded = [observation_excluded]
  lower_bound = []
  upper_bound = []
  for key, value in observation_spaces.spaces.items():
    if key not in observation_excluded:
      lower_bound.append(np.asarray(value.low).flatten())
      upper_bound.append(np.asarray(value.high).flatten())
  lower_bound = np.concatenate(lower_bound)
  upper_bound = np.concatenate(upper_bound)
  observation_space = spaces.Box(
      np.array(lower_bound), np.array(upper_bound), dtype=np.float32)
  if not observation_excluded:
    return observation_space
  else:
    observation_spaces_after_flatten = {"other": observation_space}
    for key in observation_excluded:
      observation_spaces_after_flatten[key] = observation_spaces[key]
    return spaces.Dict(observation_spaces_after_flatten)
