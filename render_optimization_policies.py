import os
from os.path import exists
import numpy as np
import supersuit as ss
from array2gif import write_gif
from pettingzoo.butterfly import knights_archers_zombies_v7
from stable_baselines3 import PPO

n_agents = 20
n_envs = 4

env = knights_archers_zombies_v7.env()
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v0(env, x_size=84, y_size=84)
env = ss.pad_action_space_v0(env)
env = ss.frame_stack_v1(env, 3)
env = ss.black_death_v2(env)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(env, n_envs, num_cpus=1, base_class="stable_baselines3")

policies = os.listdir("./optimization_policies/")

for policy in policies:
    filepath = "./optimization_policies/" + policy + "/best_model"
    if not exists(filepath + '.zip'):
        continue
    print("Loading new policy ", filepath)
    model = PPO.load(filepath)

    obs_list = []
    i = 0
    env.reset()
    reward = 0

    while True:
        for agent in env.agent_iter():
            observation, _, done, reward = env.last()
            action = (model.predict(observation, deterministic=True)[0] if not done else None)
            reward += reward

            reward = reward / n_agents

            env.step(action)
            i += 1
            if i % (len(env.possible_agents) + 1) == 0:
                obs_list.append(
                    np.transpose(env.render(mode="rgb_array"), axes=(1, 0, 2))
                )

        break

    print("writing gif")
    write_gif(
        obs_list, "./optimization_gifs/" + policy + "_" + "reward" + ".gif", fps=5
    )
