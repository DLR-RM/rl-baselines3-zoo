from stable_baselines3 import PPO
from pettingzoo.butterfly import pistonball_v4
import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
import numpy as np
import os
import sys
from array2gif import write_gif

os.environ["SDL_VIDEODRIVER"] = "dummy"
num = sys.argv[1]


def image_transpose(env):
    if is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space):
        env = VecTransposeImage(env)
    return env


env = pistonball_v4.env()
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v0(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 3)
env = image_transpose(env)

model = PPO.load("./logs/" + num + '/' + "best_model.zip")

obs_list = []
i = 0
env.reset()

while True:
    for agent in env.agent_iter():
        observation, _, done, _ = env.last()
        action = model.predict(observation, deterministic=True)[0] if not done else None

        env.step(action)
        i += 1
        if i % (len(env.possible_agents) + 1) == 0:
            obs_list.append(np.transpose(env.render(mode='rgb_array'), axes=(1, 0, 2)))
    env.close()
    break

print('writing gif')
write_gif(obs_list, str("./logs/" + num + '/' + num + '.gif', fps=15)
