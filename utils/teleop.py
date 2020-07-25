import time
from typing import List, Optional, Tuple

import numpy as np
import pygame
from pygame.locals import *  # noqa: F403
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl

TELEOP_RATE = 1 / 60

UP = (1, 0)
LEFT = (0, 1)
RIGHT = (0, -1)
DOWN = (-1, 0)
STOP = (0, 0)
KEY_CODE_SPACE = 32

MAX_TURN = 1
# Smoothing constants
STEP_THROTTLE = 0.3
STEP_TURN = 0.3

GREEN = (72, 205, 40)
RED = (205, 39, 46)
GREY = (187, 179, 179)
BLACK = (36, 36, 36)
WHITE = (230, 230, 230)
ORANGE = (200, 110, 0)

# pytype: disable=name-error
moveBindingsGame = {K_UP: UP, K_LEFT: LEFT, K_RIGHT: RIGHT, K_DOWN: DOWN}
# pytype: enable=name-error

pygame.font.init()
FONT = pygame.font.SysFont("Open Sans", 25)
SMALL_FONT = pygame.font.SysFont("Open Sans", 20)
KEY_MIN_DELAY = 0.4


def control(x, theta, control_throttle, control_steering):
    """
    Smooth control.

    :param x: (float)
    :param theta: (float)
    :param control_throttle: (float)
    :param control_steering: (float)
    :return: (float, float)
    """
    target_throttle = x
    target_steering = MAX_TURN * theta
    if target_throttle > control_throttle:
        control_throttle = min(target_throttle, control_throttle + STEP_THROTTLE)
    elif target_throttle < control_throttle:
        control_throttle = max(target_throttle, control_throttle - STEP_THROTTLE)
    else:
        control_throttle = target_throttle

    if target_steering > control_steering:
        control_steering = min(target_steering, control_steering + STEP_TURN)
    elif target_steering < control_steering:
        control_steering = max(target_steering, control_steering - STEP_TURN)
    else:
        control_steering = target_steering
    return control_throttle, control_steering


class HumanTeleop(BaseAlgorithm):
    def __init__(self, policy, env, buffer_size=50000, tensorboard_log=None, verbose=0, seed=None, delay=0):
        super(HumanTeleop, self).__init__(
            policy=None, env=env, policy_base=None, learning_rate=0.0, verbose=verbose, seed=seed
        )

        # pytype: disable=name-error
        # self.button_switch_mode = K_m
        # self.button_toggle_train_mode = K_t
        # pytype: enable=name-error

        # Used to prevent from multiple successive key press
        self.last_time_pressed = {}
        self.event_buttons = None
        self.action = np.zeros((2,))
        self.exit_thread = False
        self.process = None
        self.window = None
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(
            buffer_size, self.observation_space, self.action_space, self.device, optimize_memory_usage=False
        )
        self.delay = delay
        # self._reset_buffer()
        self.episode_steps = 0
        # self.start_process()
        # self.model = SAC.load("logs/sac/donkey-generated-track-v0_113/donkey-generated-track-v0.zip")
        self.model = None

    # def _reset_buffer(self):
    #     if self.delay > 0:
    #         obs_space_shape = (self.delay + 1,) + self.observation_space.shape
    #         self.obs_history = np.zeros(obs_space_shape)
    #         self.new_obs_history = np.zeros(obs_space_shape)
    #         self.action_history = np.zeros((self.delay + 1, 2))
    #         self.reward_history = np.zeros((self.delay + 1, 1))
    #         self.done_history = np.zeros((self.delay + 1, 1))
    #         self.episode_steps = 0

    def excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded by default
        when saving the model.

        :return: (List[str]) List of parameters that should be excluded from save
        """
        # Exclude aliases
        return super().excluded_save_params() + ["process", "window", "model"]

    def _setup_model(self):
        pass

    def init_buttons(self):
        """
        Initialize the last_time_pressed timers that prevent
        successive key press.
        """
        self.event_buttons = [
            # self.button_switch_mode,
            # self.button_toggle_train_mode,
        ]
        for key in self.event_buttons:
            self.last_time_pressed[key] = 0

    # def start_process(self):
    #     """Start main loop process."""
    #     # Reset last time pressed
    #     self.init_buttons()
    #     self.process = Thread(target=self.main_loop)
    #     # Make it a deamon, so it will be deleted at the same time
    #     # of the main process
    #     self.process.daemon = True
    #     self.process.start()

    def check_key(self, keys, key):
        """
        Check if a key was pressed and update associated timer.

        :param keys: (dict)
        :param key: (any hashable type)
        :return: (bool) Returns true when a given key was pressed, False otherwise
        """
        if key is None:
            return False
        if keys[key] and (time.time() - self.last_time_pressed[key]) > KEY_MIN_DELAY:
            # avoid multiple key press
            self.last_time_pressed[key] = time.time()
            return True
        return False

    def handle_keys_event(self, keys):
        """
        Handle the events induced by key press:
        e.g. change of mode, toggling recording, ...
        """

        # Switch from "MANUAL" to "AUTONOMOUS" mode
        # if self.check_key(keys, self.button_switch_mode) or self.check_key(keys, self.button_pause):
        #     self.is_manual = not self.is_manual

    def main_loop(self, total_timesteps=-1):
        """
        Pygame loop that listens to keyboard events.
        """
        pygame.init()
        # Create a pygame window
        self.window = pygame.display.set_mode((800, 500), RESIZABLE)  # pytype: disable=name-error

        # Init values and fill the screen
        control_throttle, control_steering = 0, 0
        action = [control_steering, control_throttle]
        self.update_screen(action)
        n_steps = 0
        buffer_action = np.array([[0.0, 0.0]])

        while not self.exit_thread:
            new_obs, reward, done, _ = self.env.step(buffer_action)

            x, theta = 0, 0
            # Record pressed keys
            keys = pygame.key.get_pressed()
            for keycode in moveBindingsGame.keys():
                if keys[keycode]:
                    x_tmp, th_tmp = moveBindingsGame[keycode]
                    x += x_tmp
                    theta += th_tmp

            self.handle_keys_event(keys)

            # Smooth control for teleoperation
            control_throttle, control_steering = control(x, theta, control_throttle, control_steering)
            self.action = np.array([-control_steering, control_throttle]).astype(np.float32)
            buffer_action = np.array([self.action])

            if self.model is not None:
                buffer_action, _ = self.model.predict(new_obs)
                self.action = buffer_action[0]

            self.episode_steps += 1

            # if self.delay > 0:
            #     self.action_history = np.roll(self.action_history, shift=-1, axis=0)
            #     self.action_history[-1, :] = buffer_action
            #     self.obs_history = np.roll(self.obs_history, shift=-1, axis=0)
            #     self.obs_history[-1, :] = self._last_obs
            #     self.new_obs_history = np.roll(self.new_obs_history, shift=-1, axis=0)
            #     self.new_obs_history[-1, :] = new_obs
            #     self.reward_history = np.roll(self.reward_history, shift=-1, axis=0)
            #     self.reward_history[-1, :] = reward
            #     self.done_history = np.roll(self.done_history, shift=-1, axis=0)
            #     self.done_history[-1, :] = done
            #
            #     if self.episode_steps > self.delay:
            #         obs_ = self.obs_history[0, :]
            #         next_obs_ = self.new_obs_history[0, :]
            #         action_ = buffer_action
            #         reward_ = self.reward_history[0, :]
            #         done_ = self.done_history[0, :]
            #
            #         self.replay_buffer.add(obs_, next_obs_, action_, reward_, done_)
            # else:
            #     self.replay_buffer.add(self._last_obs, new_obs, buffer_action, reward, done)
            self.replay_buffer.add(self._last_obs, new_obs, buffer_action, reward, done)

            self._last_obs = new_obs

            self.update_screen(self.action)

            n_steps += 1
            if total_timesteps > 0:
                self.exit_thread = n_steps >= total_timesteps

            if done:
                print(f"{n_steps} steps")
                # self._reset_buffer()

            for event in pygame.event.get():
                if (event.type == QUIT or event.type == KEYDOWN) and event.key in [  # pytype: disable=name-error
                    K_ESCAPE,  # pytype: disable=name-error
                    K_q,  # pytype: disable=name-error
                ]:
                    self.exit_thread = True
            pygame.display.flip()
            # Limit FPS
            pygame.time.Clock().tick(1 / TELEOP_RATE)

    def write_text(self, text, x, y, font, color=GREY):
        """
        :param text: (str)
        :param x: (int)
        :param y: (int)
        :param font: (str)
        :param color: (tuple)
        """
        text = str(text)
        text = font.render(text, True, color)
        self.window.blit(text, (x, y))

    def clear(self):
        self.window.fill((0, 0, 0))

    def update_screen(self, action):
        """
        Update pygame window.

        :param action: ([float])
        """
        self.clear()
        steering, throttle = action
        self.write_text("Throttle: {:.2f}, Steering: {:.2f}".format(throttle, steering), 20, 0, FONT, WHITE)

    def get_torch_variables(self) -> Tuple[List[str], List[str]]:
        """
        Get the name of the torch variables that will be saved.
        ``th.save`` and ``th.load`` will be used with the right device
        instead of the default pickling strategy.

        :return: (Tuple[List[str], List[str]])
            name of the variables with state dicts to save, name of additional torch tensors,
        """
        return [], []

    def learn(
        self,
        total_timesteps,
        callback=None,
        log_interval=100,
        tb_log_name="run",
        eval_env=None,
        eval_freq=-1,
        n_eval_episodes=5,
        eval_log_path=None,
        reset_num_timesteps=True,
    ) -> "HumanTeleop":
        self._last_obs = self.env.reset()
        # Wait for teleop process
        # time.sleep(3)
        self.main_loop(total_timesteps)
        # with threading:
        # for _ in range(total_timesteps):
        #     print(np.array([self.action]))
        #     self.env.step(np.array([self.action]))
        #     if self.exit_thread:
        #         break
        return self

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the model's action(s) from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (Optional[np.ndarray]) The last states (can be None, used in recurrent policies)
        :param mask: (Optional[np.ndarray]) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (Tuple[np.ndarray, Optional[np.ndarray]]) the model's action and the next state
            (used in recurrent policies)
        """
        return self.action, None

    def save_replay_buffer(self, path) -> None:
        """
        Save the replay buffer as a pickle file.

        :param path: (Union[str,pathlib.Path, io.BufferedIOBase]) Path to the file where the replay buffer should be saved.
            if path is a str or pathlib.Path, the path is automatically created if necessary.
        """
        assert self.replay_buffer is not None, "The replay buffer is not defined"
        save_to_pkl(path, self.replay_buffer, self.verbose)

    def load_replay_buffer(self, path) -> None:
        """
        Load a replay buffer from a pickle file.

        :param path: (Union[str, pathlib.Path, io.BufferedIOBase]) Path to the pickled replay buffer.
        """
        self.replay_buffer = load_from_pkl(path, self.verbose)
        assert isinstance(self.replay_buffer, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"
