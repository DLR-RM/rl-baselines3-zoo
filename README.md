[![pipeline status](https://gitlab.com/araffin/rl-baselines3-zoo/badges/master/pipeline.svg)](https://gitlab.com/araffin/rl-baselines3-zoo/-/commits/master) [![coverage report](https://gitlab.com/araffin/rl-baselines3-zoo/badges/master/coverage.svg)](https://gitlab.com/araffin/rl-baselines3-zoo/-/commits/master) [![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)



# RL Baselines3 Zoo: a Collection of Pre-Trained Reinforcement Learning Agents

<!-- <img src="images/BipedalWalkerHardcorePPO.gif" align="right" width="35%"/> -->

A collection of trained Reinforcement Learning (RL) agents, with tuned hyperparameters, using [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3).

We are **looking for contributors** to complete the collection!

Goals of this repository:

1. Provide a simple interface to train and enjoy RL agents
2. Benchmark the different Reinforcement Learning algorithms
3. Provide tuned hyperparameters for each environment and RL algorithm
4. Have fun with the trained agents!

This is the SB3 version of the original SB2 [rl-zoo](https://github.com/araffin/rl-baselines-zoo).

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


## Hyperparameter Tuning

We use [Optuna](https://optuna.org/) for optimizing the hyperparameters.

Note: hyperparameters search is not implemented for DQN for now.
when using SuccessiveHalvingPruner ("halving"), you must specify `--n-jobs > 1`

Budget of 1000 trials with a maximum of 50000 steps:

```
python train.py --algo ppo --env MountainCar-v0 -n 50000 -optimize --n-trials 1000 --n-jobs 2 \
  --sampler tpe --pruner median
```

Distributed optimization using a shared database is also possible (see the corresponding [Optuna documentation](https://optuna.readthedocs.io/en/latest/tutorial/distributed.html)):
```
python train.py --algo ppo --env MountainCar-v0 -optimize --study-name test --storage sqlite:///example.db
```

## Env Wrappers

You can specify in the hyperparameter config one or more wrapper to use around the environment:

for one wrapper:
```
env_wrapper: gym_minigrid.wrappers.FlatObsWrapper
```

for multiple, specify a list:

```
env_wrapper:
    - utils.wrappers.DoneOnSuccessWrapper:
        reward_offset: 1.0
    - utils.wrappers.TimeFeatureWrapper
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

## Record a Video of a Trained Agent

Record 1000 steps:

```
python -m utils.record_video --algo ppo --env BipedalWalkerHardcore-v2 -n 1000
```


## Current Collection: 120+ Trained Agents!

Scores can be found in `benchmark.md`. To compute them, simply run `python -m utils.benchmark`.

### Atari Games

7 atari games from OpenAI benchmark (NoFrameskip-v4 versions).

|  RL Algo |  BeamRider         | Breakout           | Enduro             |  Pong | Qbert | Seaquest           | SpaceInvaders      |
|----------|--------------------|--------------------|--------------------|-------|-------|--------------------|--------------------|
| A2C      |  |  |  |  |  | | |
| PPO      |  | |  |  |  | |   |
| DQN     |  |  |  | |  | |  |


Additional Atari Games (to be completed):

|  RL Algo |  MsPacman   |
|----------|-------------|
| A2C      |  |
| PPO      |  |
| DQN      |  |

### Classic Control Environments

|  RL Algo |  CartPole-v1 | MountainCar-v0 | Acrobot-v1 |  Pendulum-v0 | MountainCarContinuous-v0 |
|----------|--------------|----------------|------------|--------------|--------------------------|
| A2C      |  |  |  |  |  |
| PPO      |  |  |  |  |  |
| DQN      |  |  |  | N/A | N/A |
| DDPG     |  N/A |  N/A  | N/A |  |  |
| SAC      |  N/A |  N/A  | N/A |  |  |
| TD3      |  N/A |  N/A  | N/A |  |  |


### Box2D Environments

|  RL Algo |  BipedalWalker-v2 | LunarLander-v2 | LunarLanderContinuous-v2 |  BipedalWalkerHardcore-v2 | CarRacing-v0 |
|----------|--------------|----------------|------------|--------------|--------------------------|
| A2C      |  |  |  |  | |
| PPO      |  |  |  |  | |
| DQN      | N/A |  | N/A | N/A | N/A |
| DDPG     |  | N/A |  | | |
| SAC      |  | N/A |  |  | |
| TD3      |  | N/A |  | | |
| TRPO     |  |  |  | | |

### PyBullet Environments

See https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs.
Similar to [MuJoCo Envs](https://gym.openai.com/envs/#mujoco) but with a free simulator: pybullet. We are using `BulletEnv-v0` version.

Note: those environments are derived from [Roboschool](https://github.com/openai/roboschool) and are much harder than the Mujoco version (see [Pybullet issue](https://github.com/bulletphysics/bullet3/issues/1718#issuecomment-393198883))

|  RL Algo |  Walker2D | HalfCheetah | Ant | Reacher |  Hopper | Humanoid |
|----------|-----------|-------------|-----|---------|---------|----------|
| A2C      |  |  |  | |  | |
| PPO      |  |  |  |  |  |  |
| DDPG     |  |  |  | | | |
| SAC      |  |  |  |  |  |  |
| TD3      |  |  |  | |  |  |
| TRPO     |  |  |  | |  | |

PyBullet Envs (Continued)

|  RL Algo |  Minitaur | MinitaurDuck | InvertedDoublePendulum | InvertedPendulumSwingup |
|----------|-----------|-------------|-----|---------|
| A2C      | | | | |
| PPO      |  |  |  |  |
| DDPG     | | | | |
| SAC      | | |  |  |
| TD3      | | |  |  |
| TRPO     | | | |  |

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

Note that you need to specify --gym-packages gym_minigrid with enjoy.py and train.py as it is not a standard Gym environment, as well as installing the custom Gym package module or putting it in python path.

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

Min version: stable-baselines3[extra] >= 0.6.0

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

```
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
