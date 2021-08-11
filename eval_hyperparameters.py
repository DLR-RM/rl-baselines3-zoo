import sys
import json
from stable_baselines3 import PPO
from pettingzoo.butterfly import pistonball_v4
import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first

num = sys.argv[1]
n_evaluations = 20
n_agents = 20
n_envs = 4
n_timesteps = 2000000

with open('./hyperparameter_jsons/' + 'hyperparameters_' + num + ".json") as f:
    params = json.load(f)

print(params)


def image_transpose(env):
    if is_image_space(env.observation_space) and not is_image_space_channels_first(env.observation_space):
        env = VecTransposeImage(env)
    return env


env = pistonball_v4.parallel_env()
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v0(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 3)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(env, n_envs, num_cpus=1, base_class='stable_baselines3')
env = VecMonitor(env)
env = image_transpose(env)

eval_env = pistonball_v4.parallel_env()
eval_env = ss.color_reduction_v0(eval_env, mode='B')
eval_env = ss.resize_v0(eval_env, x_size=84, y_size=84)
eval_env = ss.frame_stack_v1(eval_env, 3)
eval_env = ss.pettingzoo_env_to_vec_env_v0(eval_env)
eval_env = ss.concat_vec_envs_v0(eval_env, 1, num_cpus=1, base_class='stable_baselines3')
eval_env = VecMonitor(eval_env)
eval_env = image_transpose(eval_env)

eval_freq = int(n_timesteps / n_evaluations)
eval_freq = max(eval_freq // (n_envs * n_agents), 1)

all_mean_rewards = []
for i in range(10):
    model = PPO("CnnPolicy", env, verbose=3, **params)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./eval_logs/' + num + '/', log_path='./eval_logs/' + num + '/' , eval_freq=eval_freq, deterministic=True, render=False)
    model.learn(total_timesteps=n_timesteps, callback=eval_callback) 
    model = PPO.load('./eval_logs/' + num + '/' + 'best_model')
    mean_reward, std_reward = evaluate_policy(model, eval_env, deterministic=True, n_eval_episodes=25)
    print(mean_reward)
    print(std_reward)
    all_mean_rewards.append(mean_reward)
    if mean_reward > 90:
        model.save('./mature_policies/' + str(num) + '/' + str(i) + '_' + str(mean_reward).split('.')[0] + '.zip')


print(sum(all_mean_rewards) / len(all_mean_rewards))
