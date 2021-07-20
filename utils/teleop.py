import os
import time
from typing import List, Optional, Tuple

import numpy as np
import pygame
from gym_space_engineers.envs.walking_robot_ik import Task
from pygame.locals import *  # noqa: F403
from sb3_contrib import TQC
from stable_baselines3.common.base_class import BaseAlgorithm

# TELEOP_RATE = 1 / 60

UP = (1, 0)
LEFT = (0, 1)
RIGHT = (0, -1)
DOWN = (-1, 0)
STOP = (0, 0)
KEY_CODE_SPACE = 32

# MAX_TURN = 1
# # Smoothing constants
# STEP_THROTTLE = 0.8
# STEP_TURN = 0.8

GREEN = (72, 205, 40)
RED = (205, 39, 46)
GREY = (187, 179, 179)
BLACK = (36, 36, 36)
WHITE = (230, 230, 230)
ORANGE = (200, 110, 0)

# pytype: disable=name-error
moveBindingsGame = {K_UP: UP, K_LEFT: LEFT, K_RIGHT: RIGHT, K_DOWN: DOWN}  # noqa: F405
# pytype: enable=name-error
pygame.font.init()
FONT = pygame.font.SysFont("Open Sans", 25)
SMALL_FONT = pygame.font.SysFont("Open Sans", 20)
KEY_MIN_DELAY = 0.4


class HumanTeleop(BaseAlgorithm):
    def __init__(
        self,
        policy,
        env,
        tensorboard_log=None,
        verbose=0,
        seed=None,
        device=None,
        _init_setup_model: bool = False,
        forward_controller_path: str = os.environ.get("FORWARD_CONTROLLER_PATH"),  # noqa: B008
        backward_controller_path: str = os.environ.get("BACKWARD_CONTROLLER_PATH"),  # noqa: B008
        turn_left_controller_path: str = os.environ.get("TURN_LEFT_CONTROLLER_PATH"),  # noqa: B008
        turn_right_controller_path: str = os.environ.get("TURN_RIGHT_CONTROLLER_PATH"),  # noqa: B008
        multi_controller_path: str = os.environ.get("MULTI_CONTROLLER_PATH"),  # noqa: B008
        deterministic: bool = True,
    ):
        self.multi_controller_path = multi_controller_path
        if multi_controller_path is None:
            assert forward_controller_path is not None
            assert backward_controller_path is not None
            assert turn_left_controller_path is not None
            assert turn_right_controller_path is not None
            # Pretrained model
            # set BACKWARD_CONTROLLER_PATH=logs\pretrained-tqc\SE-Symmetric-v1_2\SE-Symmetric-v1.zip
            # set FORWARD_CONTROLLER_PATH=logs\pretrained-tqc\SE-Symmetric-v1_1\SE-Symmetric-v1.zip
            # set TURN_LEFT_CONTROLLER_PATH=logs\pretrained-tqc\SE-TurnLeft-v1_1\SE-TurnLeft-v1.zip
            # set TURN_RIGHT_CONTROLLER_PATH=logs\pretrained-tqc\SE-TurnLeft-v1_2\SE-TurnLeft-v1.zip
            self.forward_controller = TQC.load(forward_controller_path)
            self.backward_controller = TQC.load(backward_controller_path)
            self.turn_left_controller = TQC.load(turn_left_controller_path)
            self.turn_right_controller = TQC.load(turn_right_controller_path)
        else:
            # set MULTI_CONTROLLER_PATH=logs\pretrained-tqc\SE-MultiTask-v1_9/rl_model_250000_steps.zip
            self.forward_controller = TQC.load(multi_controller_path)
            self.backward_controller = self.forward_controller
            self.turn_left_controller = self.forward_controller
            self.turn_right_controller = self.forward_controller

        super(HumanTeleop, self).__init__(
            policy=None, env=env, policy_base=None, learning_rate=0.0, verbose=verbose, seed=seed
        )

        # Used to prevent from multiple successive key press
        self.last_time_pressed = {}
        self.event_buttons = None
        self.exit_thread = False
        self.process = None
        self.window = None
        self.max_speed = 0.0

        self.deterministic = deterministic

    def _excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded by default
        when saving the model.

        :return: (List[str]) List of parameters that should be excluded from save
        """
        # Exclude aliases
        return super()._excluded_save_params() + ["process", "window", "forward_controller", "turn_controller", "exit_thread"]

    def _setup_model(self):
        self.exit_thread = False

    def init_buttons(self):
        """
        Initialize the last_time_pressed timers that prevent
        successive key press.
        """
        self.event_buttons = []
        for key in self.event_buttons:
            self.last_time_pressed[key] = 0

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
        self.window = pygame.display.set_mode((200, 200), RESIZABLE)  # pytype: disable=name-error

        # Init values and fill the screen
        move, task = "stay", None
        # TODO: implement "stay"
        self.update_screen(move)

        n_steps = 0
        action = np.array([self.env.action_space.sample()]) * 0.0
        self.max_speed = self.env.get_attr("max_speed")

        while not self.exit_thread:
            x, theta = 0, 0
            # Record pressed keys
            keys = pygame.key.get_pressed()
            for keycode in moveBindingsGame.keys():
                if keys[keycode]:
                    x_tmp, th_tmp = moveBindingsGame[keycode]
                    x += x_tmp
                    theta += th_tmp

            self.handle_keys_event(keys)
            # For now only handle one model at once
            if x > 0:
                move = "forward"
            elif x < 0:
                move = "backward"
            elif theta < 0:
                move = "turn_right"
            elif theta > 0:
                move = "turn_left"
            else:
                move = "stay"

            if move != "stay":
                task = Task(move)
                # TODO: check if the task has changed
                self.env.env_method("change_task", task)
                self.env.set_attr("max_speed", self.max_speed)
                # TODO: update for the frame stack by stepping fast in the env?
                # self._last_obs = self.env.env_method("change_task", task)

                controller = {
                    Task.FORWARD: self.forward_controller,
                    Task.BACKWARD: self.backward_controller,
                    Task.TURN_LEFT: self.turn_left_controller,
                    Task.TURN_RIGHT: self.turn_right_controller,
                }[task]

                action = controller.predict(self._last_obs, deterministic=self.deterministic)
                # TODO for multi policy: display proba for each expert
            else:
                task = None
                self.env.set_attr("max_speed", 0.0)

            self._last_obs, reward, done, infos = self.env.step(action)

            self.update_screen(move)

            n_steps += 1
            if total_timesteps > 0:
                self.exit_thread = n_steps >= total_timesteps

            for event in pygame.event.get():
                if (event.type == QUIT or event.type == KEYDOWN) and event.key in [  # pytype: disable=name-error
                    K_ESCAPE,  # pytype: disable=name-error
                    K_q,  # pytype: disable=name-error
                ]:
                    self.exit_thread = True
            pygame.display.flip()
            # Limit FPS
            # pygame.time.Clock().tick(1 / TELEOP_RATE)

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

    def clear(self) -> None:
        self.window.fill((0, 0, 0))

    def update_screen(self, move: str) -> None:
        """
        Update pygame window.

        :param action:
        """
        self.clear()
        self.write_text(f"Task: {move}", 50, 50, FONT, WHITE)

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
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
        self.main_loop(total_timesteps)

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
        # TODO: launch separate thread to handle user keyboard events
        return self.model.predict(observation, deterministic)
