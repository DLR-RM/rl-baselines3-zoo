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
"""A simple locomotion task and termination condition."""
class DummyTask(object):
  """Default empy task."""
  def __init__(self):
    """Initializes the task."""
    self._draw_ref_model_alpha = 1.
    self._ref_model = -1

  def __call__(self, env):
    return self.reward(env)

  def reset(self, env):
    """Resets the internal state of the task."""
    self._env = env

  def update(self, env):
    """Updates the internal state of the task."""
    del env

  def done(self, env):
    """Checks if the episode is over."""
    del env
    return False

  def reward(self, env):
    """Get the reward without side effects."""
    del env
    return 1
