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

"""A config file for parameters and their ranges in dynamics randomization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def all_params():
  """Randomize all the physical parameters."""
  param_range = {
      # The following ranges are in percentage. e.g. 0.8 means 80%.
      "mass": [0.8, 1.2],
      "inertia": [0.5, 1.5],
      "motor strength": [0.8, 1.2],
      # The following ranges are the physical values, in SI unit.
      "motor friction": [0, 0.05],  # Viscous damping (Nm s/rad).
      "latency": [0.0, 0.04],  # Time inteval (s).
      "lateral friction": [0.5, 1.25],  # Friction coefficient (dimensionless).
      "battery": [14.0, 16.8],  # Voltage (V).
      "joint friction": [0, 0.05],  # Coulomb friction torque (Nm).
  }
  return param_range
