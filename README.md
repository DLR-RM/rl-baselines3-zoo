[![pipeline status](https://gitlab.com/araffin/rl-baselines3-zoo/badges/master/pipeline.svg)](https://gitlab.com/araffin/rl-baselines3-zoo/-/commits/master) [![coverage report](https://gitlab.com/araffin/rl-baselines3-zoo/badges/master/coverage.svg)](https://gitlab.com/araffin/rl-baselines3-zoo/-/commits/master) [![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)



# RL Baselines3 Zoo: A Training Framework for Stable Baselines3 Reinforcement Learning Agents

<!-- <img src="images/BipedalWalkerHardcorePPO.gif" align="right" width="35%"/> -->

RL Baselines3 Zoo is a training framework for Reinforcement Learning (RL), using [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3).

It provides scripts for training, evaluating agents, tuning hyperparameters, plotting results and recording videos.

In addition, it includes a collection of tuned hyperparameters for common environments and RL algorithms, and agents trained with those settings.


We are **looking for contributors** to complete the collection!

Goals of this repository:

1. Provide a simple interface to train and enjoy RL agents
2. Benchmark the different Reinforcement Learning algorithms
3. Provide tuned hyperparameters for each environment and RL algorithm
4. Have fun with the trained agents!

This is the SB3 version of the original SB2 [rl-zoo](https://github.com/araffin/rl-baselines-zoo).

## Train an Agent

The hyperparameters for each environment are defined in `hyperparameters/algo_name.yml`.

If the environment exists in this file, then you can train an agent using:
```
python train.py --algo algo_name --env env_id
```

For example (with tensorboard support):
```
python train.py --algo ppo --env CartPole-v1 --tensorboard-log /tmp/stable-baselines/
```

Evaluate the agent every 10000 steps using 10 episodes for evaluation:
```
python train.py --algo sac --env HalfCheetahBulletEnv-v0 --eval-freq 10000 --eval-episodes 10
```

Save a checkpoint of the agent every 100000 steps:
```
python train.py --algo td3 --env HalfCheetahBulletEnv-v0 --save-freq 100000
```

Continue training (here, load pretrained agent for Breakout and continue training for 5000 steps):
```
python train.py --algo a2c --env BreakoutNoFrameskip-v4 -i rl-trained-agents/a2c/BreakoutNoFrameskip-v4_1/BreakoutNoFrameskip-v4.zip -n 5000
```

When using off-policy algorithms, you can also save the replay buffer after training:
```
python train.py --algo sac --env Pendulum-v0 --save-replay-buffer
```
It will be automatically loaded if present when continuing training.

## Plot Scripts

Plot scripts (to be documented, see "Results" sections in SB3 documentation):
- `scripts/all_plots.py`/`scripts/plot_from_file.py` for plotting evaluations
- `scripts/plot_train.py` for plotting training reward/success

*Examples (on the current collection)*

Plot training success (y-axis) w.r.t. timesteps (x-axis) with a moving window of 500 episodes for all the `Fetch` environment with `HER` algorithm:

```
python scripts/plot_train.py -a her -e Fetch -y success -f rl-trained-agents/ -w 500 -x steps
```

Plot evaluation reward curve for TQC, SAC and TD3 on the HalfCheetah and Ant PyBullet environments:

```
python scripts/all_plots.py -a sac td3 tqc --env HalfCheetah Ant -f rl-trained-agents/
```

## Custom Environment

The easiest way to add support for a custom environment is to edit `utils/import_envs.py` and register your environment here. Then, you need to add a section for it in the hyperparameters file (`hyperparams/algo.yml`).

## Enjoy a Trained Agent

**Note: to download the repo with the trained agents, you must use `git clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo`** in order to clone the submodule too.


If the trained agent exists, then you can see it in action using:
```
python enjoy.py --algo algo_name --env env_id
```

For example, enjoy A2C on Breakout during 5000 timesteps:
```
python enjoy.py --algo a2c --env BreakoutNoFrameskip-v4 --folder rl-trained-agents/ -n 5000
```

If you have trained an agent yourself, you need to do:
```
# exp-id 0 corresponds to the last experiment, otherwise, you can specify another ID
python enjoy.py --algo algo_name --env env_id -f logs/ --exp-id 0
```

To load the best model (when using evaluation environment):
```
python enjoy.py --algo algo_name --env env_id -f logs/ --exp-id 1 --load-best
```

To load a checkpoint (here the checkpoint name is `rl_model_10000_steps.zip`):
```
python enjoy.py --algo algo_name --env env_id -f logs/ --exp-id 1 --load-checkpoint 10000
```

## Hyperparameter yaml syntax

The syntax used in `hyperparameters/algo_name.yml` for setting hyperparameters (likewise the syntax to [overwrite hyperparameters](https://github.com/DLR-RM/rl-baselines3-zoo#overwrite-hyperparameters) on the cli) may be specialized if the argument is a function.  See examples in the `hyperparameters/` directory. For example:

- Specify a linear schedule for the learning rate:

```yaml
  learning_rate: lin_0.012486195510232303
```

Specify a different activation function for the network:

```yaml
  policy_kwargs: "dict(activation_fn=nn.ReLU)"
```

## Hyperparameter Tuning

We use [Optuna](https://optuna.org/) for optimizing the hyperparameters.
Not all hyperparameters are tuned, and tuning enforces certain default hyperparameter settings that may be different from the official defaults. See [utils/hyperparams_opt.py](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/hyperparams_opt.py) for the current settings for each agent.

Hyperparameters not specified in [utils/hyperparams_opt.py](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/hyperparams_opt.py) are taken from the associated YAML file and fallback to the default values of SB3 if not present.

Note: hyperparameters search is not implemented for DQN for now.
when using SuccessiveHalvingPruner ("halving"), you must specify `--n-jobs > 1`

Budget of 1000 trials with a maximum of 50000 steps:

```
python train.py --algo ppo --env MountainCar-v0 -n 50000 -optimize --n-trials 1000 --n-jobs 2 \
  --sampler tpe --pruner median
```

Distributed optimization using a shared database is also possible (see the corresponding [Optuna documentation](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html)):
```
python train.py --algo ppo --env MountainCar-v0 -optimize --study-name test --storage sqlite:///example.db
```

### Hyperparameters search space

Note that the default hyperparameters used in the zoo when tuning are not always the same as the defaults provided in [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/base.html). Consult the latest source code to be sure of these settings. For example:

- PPO tuning assumes a network architecture with `ortho_init = False` when tuning, though it is `True` by [default](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#ppo-policies). You can change that by updating [utils/hyperparams_opt.py](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/hyperparams_opt.py).

- Non-epsodic rollout in TD3 and DDPG assumes `gradient_steps = train_freq` and so tunes only `train_freq` to reduce the search space.  

When working with continuous actions, we recommend to enable [gSDE](https://arxiv.org/abs/2005.05719) by uncommenting lines in [utils/hyperparams_opt.py](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/hyperparams_opt.py).

## Env normalization

In the hyperparameter file, `normalize: True` means that the training environment will be wrapped in a [VecNormalize](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_normalize.py#L13) wrapper.

[Normalization uses](https://github.com/DLR-RM/rl-baselines3-zoo/issues/64) the default parameters of `VecNormalize`, with the exception of `gamma` which is set to match that of the agent.  This can be [overridden](https://github.com/DLR-RM/rl-baselines3-zoo/blob/v0.10.0/hyperparams/sac.yml#L239) using the appropriate `hyperparameters/algo_name.yml`, e.g.

```yaml
  normalize: "{'norm_obs': True, 'norm_reward': False}"
```


## Env Wrappers

You can specify in the hyperparameter config one or more wrapper to use around the environment:

for one wrapper:
```yaml
env_wrapper: gym_minigrid.wrappers.FlatObsWrapper
```

for multiple, specify a list:

```yaml
env_wrapper:
    - utils.wrappers.DoneOnSuccessWrapper:
        reward_offset: 1.0
    - sb3_contrib.common.wrappers.TimeFeatureWrapper
```

Note that you can easily specify parameters too.

## Env keyword arguments

You can specify keyword arguments to pass to the env constructor in the command line, using `--env-kwargs`:

```
python enjoy.py --algo ppo --env MountainCar-v0 --env-kwargs goal_velocity:10
```

## Overwrite hyperparameters

You can easily overwrite hyperparameters in the command line, using ``--hyperparams``:

```
python train.py --algo a2c --env MountainCarContinuous-v0 --hyperparams learning_rate:0.001 policy_kwargs:"dict(net_arch=[64, 64])"
```

Note: if you want to pass a string, you need to escape it like that: `my_string:"'value'"`

## Record a Video of a Trained Agent

Record 1000 steps with the latest saved model:

```
python -m utils.record_video --algo ppo --env BipedalWalkerHardcore-v3 -n 1000
```

Use the best saved model instead:

```
python -m utils.record_video --algo ppo --env BipedalWalkerHardcore-v3 -n 1000 --load-best
```

Record a video of a checkpoint saved during training (here the checkpoint name is `rl_model_10000_steps.zip`):

```
python -m utils.record_video --algo ppo --env BipedalWalkerHardcore-v3 -n 1000 --load-checkpoint 10000
```

## Record a Video of a Training Experiment

Apart from recording videos of specific saved models, it is also possible to record a video of a training experiment where checkpoints have been saved.

Record 1000 steps for each checkpoint, latest and best saved models:

```
python -m utils.record_training --algo ppo --env CartPole-v1 -n 1000 -f logs --deterministic
```

The previous command will create a `mp4` file. To convert this file to `gif` format as well:

```
python -m utils.record_training --algo ppo --env CartPole-v1 -n 1000 -f logs --deterministic --gif
```

## Current Collection: 100+ Trained Agents!

Final performance of the trained agents can be found in [`benchmark.md`](./benchmark.md). To compute them, simply run `python -m utils.benchmark`.

*NOTE: this is not a quantitative benchmark as it corresponds to only one run (cf [issue #38](https://github.com/araffin/rl-baselines-zoo/issues/38)). This benchmark is meant to check algorithm (maximal) performance, find potential bugs and also allow users to have access to pretrained agents.*

### Atari Games

7 atari games from OpenAI benchmark (NoFrameskip-v4 versions).

|  RL Algo |  BeamRider         | Breakout           | Enduro             |  Pong | Qbert | Seaquest           | SpaceInvaders      |
|----------|--------------------|--------------------|--------------------|-------|-------|--------------------|--------------------|
| A2C      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| PPO      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| DQN      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| QR-DQN   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

Additional Atari Games (to be completed):

|  RL Algo |  MsPacman   | Asteroids | RoadRunner |
|----------|-------------|-----------|------------|
| A2C      |  | :heavy_check_mark: | :heavy_check_mark: |
| PPO      |  | :heavy_check_mark: | :heavy_check_mark: |
| DQN      |  | :heavy_check_mark: | :heavy_check_mark: |
| QR-DQN   |  | :heavy_check_mark: | :heavy_check_mark: |


### Classic Control Environments

|  RL Algo |  CartPole-v1 | MountainCar-v0 | Acrobot-v1 |  Pendulum-v0 | MountainCarContinuous-v0 |
|----------|--------------|----------------|------------|--------------|--------------------------|
| A2C      | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: |
| PPO      | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: |
| DQN      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | N/A | N/A |
| QR-DQN   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | N/A | N/A |
| DDPG     |  N/A |  N/A  | N/A | :heavy_check_mark: | :heavy_check_mark: |
| SAC      |  N/A |  N/A  | N/A | :heavy_check_mark: | :heavy_check_mark: |
| TD3      |  N/A |  N/A  | N/A | :heavy_check_mark: | :heavy_check_mark: |
| TQC      |  N/A |  N/A  | N/A | :heavy_check_mark: | :heavy_check_mark: |


### Box2D Environments

|  RL Algo |  BipedalWalker-v3 | LunarLander-v2 | LunarLanderContinuous-v2 |  BipedalWalkerHardcore-v3 | CarRacing-v0 |
|----------|--------------|----------------|------------|--------------|--------------------------|
| A2C      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| PPO      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| DQN      | N/A | :heavy_check_mark: | N/A | N/A | N/A |
| QR-DQN   | N/A | :heavy_check_mark: | N/A | N/A | N/A |
| DDPG     | :heavy_check_mark: | N/A | :heavy_check_mark: | | |
| SAC      | :heavy_check_mark: | N/A | :heavy_check_mark: | :heavy_check_mark: | |
| TD3      | :heavy_check_mark: | N/A | :heavy_check_mark: | :heavy_check_mark: | |
| TQC      | :heavy_check_mark: | N/A | :heavy_check_mark: | :heavy_check_mark: | |

### PyBullet Environments

See https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs.
Similar to [MuJoCo Envs](https://gym.openai.com/envs/#mujoco) but with a free simulator: pybullet. We are using `BulletEnv-v0` version.

Note: those environments are derived from [Roboschool](https://github.com/openai/roboschool) and are much harder than the Mujoco version (see [Pybullet issue](https://github.com/bulletphysics/bullet3/issues/1718#issuecomment-393198883))

|  RL Algo |  Walker2D | HalfCheetah | Ant | Reacher |  Hopper | Humanoid |
|----------|-----------|-------------|-----|---------|---------|----------|
| A2C      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| PPO      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| DDPG     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| SAC      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| TD3      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |
| TQC      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | |

PyBullet Envs (Continued)

|  RL Algo |  Minitaur | MinitaurDuck | InvertedDoublePendulum | InvertedPendulumSwingup |
|----------|-----------|-------------|-----|---------|
| A2C      | | | | |
| PPO      | | | | |
| DDPG     | | | | |
| SAC      | | | | |
| TD3      | | | | |
| TQC      | | | | |


### Robotics Environments

See https://gym.openai.com/envs/#robotics and https://github.com/DLR-RM/rl-baselines3-zoo/pull/71

MuJoCo version: 1.50.1.0
Gym version: 0.18.0

We used the v1 environments.

|  RL Algo |  FetchReach | FetchPickAndPlace | FetchPush | FetchSlide |
|----------|-------------|-------------------|-----------|------------|
| HER+TQC  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |



### MiniGrid Envs

See https://github.com/maximecb/gym-minigrid
A simple, lightweight and fast Gym environments implementation of the famous gridworld.

|  RL Algo | Empty | FourRooms | DoorKey | MultiRoom | Fetch |
|----------|-------|-----------|---------|-----------|-------|
| A2C      | | | | | |
| PPO      | |  |  | | |
| DDPG     | | | | | |
| SAC      | | | | | |
| TRPO     | | | | | |

There are 19 environment groups (variations for each) in total.

Note that you need to specify `--gym-packages gym_minigrid` with `enjoy.py` and `train.py` as it is not a standard Gym environment, as well as installing the custom Gym package module or putting it in python path.

```
pip install gym-minigrid
python train.py --algo ppo --env MiniGrid-DoorKey-5x5-v0 --gym-packages gym_minigrid
```

This does the same thing as:

```python
import gym_minigrid
```

Also, you may need to specify a Gym environment wrapper in hyperparameters, as MiniGrid environments have Dict observation space, which is not supported by StableBaselines for now.

```
MiniGrid-DoorKey-5x5-v0:
  env_wrapper: gym_minigrid.wrappers.FlatObsWrapper
```


## Colab Notebook: Try it Online!

You can train agents online using [colab notebook](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/rl-baselines-zoo.ipynb).

## Installation

### Stable-Baselines3 PyPi Package

Min version: stable-baselines3[extra] >= 1.0
and sb3_contrib >= 1.0

```
apt-get install swig cmake ffmpeg
pip install -r requirements.txt
```

Please see [Stable Baselines3 README](https://github.com/DLR-RM/stable-baselines3) for alternatives.

### Docker Images

Build docker image (CPU):
```
make docker-cpu
```

GPU:
```
USE_GPU=True make docker-gpu
```

Pull built docker image (CPU):
```
docker pull stablebaselines/rl-baselines3-zoo-cpu
```

GPU image:
```
docker pull stablebaselines/rl-baselines3-zoo
```

Run script in the docker image:

```
./scripts/run_docker_cpu.sh python train.py --algo ppo --env CartPole-v1
```

## Tests

To run tests, first install pytest, then:
```
make pytest
```

Same for type checking with pytype:
```
make type
```


## Citing the Project

To cite this repository in publications:

```bibtex
@misc{rl-zoo3,
  author = {Raffin, Antonin},
  title = {RL Baselines3 Zoo},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DLR-RM/rl-baselines3-zoo}},
}
```

## Contributing

If you trained an agent that is not present in the rl zoo, please submit a Pull Request (containing the hyperparameters and the score too).

## Contributors

We would like to thanks our contributors: [@iandanforth](https://github.com/iandanforth), [@tatsubori](https://github.com/tatsubori) [@Shade5](https://github.com/Shade5)
