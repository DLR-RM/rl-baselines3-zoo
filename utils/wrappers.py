import gym
from gym.wrappers import TimeLimit
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3.common.atari_wrappers import AtariWrapper  # pytype: disable=import-error


class DoneOnSuccessWrapper(gym.Wrapper):
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.
    """

    def __init__(self, env, reward_offset=1.0):
        super(DoneOnSuccessWrapper, self).__init__(env)
        self.reward_offset = reward_offset

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done = done or info.get('is_success', False)
        reward += self.reward_offset
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward + self.reward_offset


class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.

    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """

    def __init__(self, env, max_steps=1000, test_mode=False):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high = np.concatenate((low, [0])), np.concatenate((high, [1.]))
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        super(TimeFeatureWrapper, self).__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self):
        self._current_step = 0
        return self._get_obs(self.env.reset())

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.

        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionnaly: concatenate [time_feature, time_feature ** 2]
        return np.concatenate((obs, [time_feature]))


class ActionNoiseWrapper(gym.Wrapper):
    """
    Add gaussian noise to the action (without telling the agent),
    to test the robustness of the control.

    :param env: (gym.Env)
    :param noise_std: (float) Standard deviation of the noise
    """

    def __init__(self, env, noise_std=0.1):
        super(ActionNoiseWrapper, self).__init__(env)
        self.noise_std = noise_std

    def step(self, action):
        noise = np.random.normal(np.zeros_like(action), np.ones_like(action) * self.noise_std)
        noisy_action = action + noise
        return self.env.step(noisy_action)


class DelayedRewardWrapper(gym.Wrapper):
    """
    Delay the reward by `delay` steps, it makes the task harder but more realistic.
    The reward is accumulated during those steps.

    :param env: (gym.Env)
    :param delay: (int) Number of steps the reward should be delayed.
    """

    def __init__(self, env, delay=10):
        super(DelayedRewardWrapper, self).__init__(env)
        self.delay = delay
        self.current_step = 0
        self.accumulated_reward = 0.0

    def reset(self):
        self.current_step = 0
        self.accumulated_reward = 0.0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.accumulated_reward += reward
        self.current_step += 1

        if self.current_step % self.delay == 0 or done:
            reward = self.accumulated_reward
            self.accumulated_reward = 0.0
        else:
            reward = 0.0
        return obs, reward, done, info


class HistoryWrapper(gym.Wrapper):
    """
    Delay the reward by `delay` steps, it makes the task harder but more realistic.
    The reward is accumulated during those steps.

    :param env: (gym.Env)
    :param horizon: (int) Number of steps to keep in the history.
    """

    def __init__(self, env, horizon=5):
        assert isinstance(env.observation_space, gym.spaces.Box)

        wrapped_obs_space = env.observation_space
        wrapped_action_space = env.action_space

        # TODO: double check, it seems wrong when we have different low and highs
        low_obs = np.repeat(wrapped_obs_space.low, horizon, axis=-1)
        high_obs = np.repeat(wrapped_obs_space.high, horizon, axis=-1)

        low_action = np.repeat(wrapped_action_space.low, horizon, axis=-1)
        high_action = np.repeat(wrapped_action_space.high, horizon, axis=-1)

        low = np.concatenate((low_obs, low_action))
        high = np.concatenate((high_obs, high_action))

        # Overwrite the observation space
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=wrapped_obs_space.dtype)

        super(HistoryWrapper, self).__init__(env)

        self.horizon = horizon
        self.low_action, self.high_action = low_action, high_action
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)
        self.action_history = np.zeros(low_action.shape, low_action.dtype)

    def _create_obs_from_history(self):
        return np.concatenate((self.obs_history, self.action_history))

    def reset(self):
        # Flush the history
        self.obs_history[...] = 0
        self.action_history[...] = 0
        obs = self.env.reset()
        self.obs_history[..., -obs.shape[-1]:] = obs
        return self._create_obs_from_history()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1]:] = obs

        self.action_history = np.roll(self.action_history, shift=-action.shape[-1], axis=-1)
        self.action_history[..., -action.shape[-1]:] = action
        return self._create_obs_from_history(), reward, done, info


class PlotActionWrapper(gym.Wrapper):
    """
    Wrapper for plotting the taken actions.
    Only works with 1D actions for now.
    Optionally, it can be used to plot the observations too.

    :param env: (gym.Env)
    :param plot_freq: (int) Plot every `plot_freq` episodes
    :param fft_plot: (bool) Whether to plot the fft plot of the actions
        (to see the frequency) needs some tuning (regarding the sampling frequency)
    """

    def __init__(self, env, plot_freq=5, fft_plot=False):
        super(PlotActionWrapper, self).__init__(env)
        self.plot_freq = plot_freq
        self.fft_plot = fft_plot
        self.current_episode = 0
        # Observation buffer (Optional)
        # self.observations = []
        # Action buffer
        self.actions = []

    def reset(self):
        self.current_episode += 1
        if self.current_episode % self.plot_freq == 0:
            self.plot()
            # Reset
            self.actions = []
        obs = self.env.reset()
        self.actions.append([])
        # self.observations.append(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.actions[-1].append(action)
        # self.observations.append(obs)

        return obs, reward, done, info

    def plot(self):
        actions = self.actions
        x = np.arange(sum([len(episode) for episode in actions]))
        plt.figure("Actions")
        plt.title('Actions during exploration', fontsize=14)
        plt.xlabel('Timesteps', fontsize=14)
        plt.ylabel('Action', fontsize=14)

        start = 0
        for i in range(len(self.actions)):
            end = start + len(self.actions[i])
            plt.plot(x[start:end], self.actions[i])
            # Clipped actions: real behavior, note that it is between [-2, 2] for the Pendulum
            # plt.scatter(x[start:end], np.clip(self.actions[i], -1, 1), s=1)
            # plt.scatter(x[start:end], self.actions[i], s=1)
            start = end

        # Plot Frequency plot
        if self.fft_plot:
            signal = np.concatenate(tuple(self.actions))
            n_samples = len(signal)
            # TODO: change the time_delta to match the real one
            time_delta = 1 / 4e4
            # Sanity check: sinusoidal signal
            # signal = np.sin(10 * 2 * np.pi * np.arange(n_samples) * time_delta)
            signal_fft = np.fft.fft(signal)
            freq = np.fft.fftfreq(n_samples, time_delta)
            plt.figure("FFT")
            plt.plot(freq[:n_samples // 2], np.abs(signal_fft[:n_samples // 2]))

        plt.show()
