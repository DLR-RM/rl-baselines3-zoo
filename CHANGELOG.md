## Release 2.2.1 (2023-11-17)

### Breaking Changes
- Removed `gym` dependency, the package is still required for some pretrained agents.
- Upgraded to SB3 >= 2.2.1
- Upgraded to Huggingface-SB3 >= 3.0
- Upgraded to pytablewriter >= 1.0

### New Features
- Added `--eval-env-kwargs` to `train.py` (@Quentin18)
- Added `ppo_lstm` to hyperparams_opt.py (@technocrat13)

### Bug fixes
- Upgraded to `pybullet_envs_gymnasium>=0.4.0`
- Removed old hacks (for instance limiting offpolicy algorithms to one env at test time)

### Documentation

### Other
- Updated docker image, removed support for X server
- Replaced deprecated `optuna.suggest_uniform(...)` by `optuna.suggest_float(..., low=..., high=...)`
- Switched to ruff for sorting imports
- Updated tests to use `shlex.split()`
- Fixed `rl_zoo3/hyperparams_opt.py` type hints
- Fixed `rl_zoo3/exp_manager.py` type hints

## Release 2.1.0 (2023-08-17)

### Breaking Changes
- Dropped python 3.7 support
- SB3 now requires PyTorch 1.13+
- Upgraded to SB3 >= 2.1.0
- Upgraded to Huggingface-SB3 >= 2.3
- Upgraded to Optuna >= 3.0
- Upgraded to cloudpickle >= 2.2.1

### New Features
- Added python 3.11 support

### Bug fixes

### Documentation

### Other


## Release 2.0.0 (2023-06-22)

**Gymnasium support**

> **Warning**
> Stable-Baselines3 (SB3) v2.0.0 will be the last one supporting python 3.7

### Breaking Changes
- Fixed bug in HistoryWrapper, now returns the correct obs space limits
- Upgraded to SB3 >= 2.0.0
- Upgraded to Huggingface-SB3 >= 2.2.5
- Upgraded to Gym API 0.26+, RL Zoo3 doesn't work anymore with Gym 0.21

### New Features
- Added Gymnasium support
- Gym 0.26+ patches to continue working with pybullet and TimeLimit wrapper

### Bug fixes
- Renamed `CarRacing-v1` to `CarRacing-v2` in hyperparameters
- Huggingface push to hub now accepts a `--n-timesteps` argument to adjust the length of the video
- Fixed `record_video` steps (before it was stepping in a closed env)

## Release 1.8.0 (2023-04-07)

**New Documentation, Multi-Env HerReplayBuffer**

> **Warning**
> Stable-Baselines3 (SB3) v1.8.0 will be the last one to use Gym as a backend.
  Starting with v2.0.0, Gymnasium will be the default backend (though SB3 will have compatibility layers for Gym envs).
  You can find a migration guide [here](https://gymnasium.farama.org/content/migration-guide/).
  If you want to try the SB3 v2.0 alpha version, you can take a look at [PR #1327](https://github.com/DLR-RM/stable-baselines3/pull/1327).

### Breaking Changes
- Upgraded to SB3 >= 1.8.0
- Upgraded to new `HerReplayBuffer` implementation that supports multiple envs
- Removed `TimeFeatureWrapper` for Panda and Fetch envs, as the new replay buffer should handle timeout.

### New Features
- Tuned hyperparameters for RecurrentPPO on Swimmer
- Documentation is now built using Sphinx and hosted on read the doc
- Added hyperparameters pre-trained agents for PPO on 11 MiniGrid envs

### Bug fixes
- Set ``highway-env`` version to 1.5 and ``setuptools to`` v65.5 for the CI
- Removed `use_auth_token` for push to hub util
- Reverted from v3 to v2 for HumanoidStandup, Reacher, InvertedPendulum and InvertedDoublePendulum since they were not part of the mujoco refactoring (see https://github.com/openai/gym/pull/1304)
- Fixed `gym-minigrid` policy (from `MlpPolicy` to `MultiInputPolicy`)

### Documentation

### Other
- Added support for `ruff` (fast alternative to flake8) in the Makefile
- Removed Gitlab CI file
- Replaced deprecated `optuna.suggest_loguniform(...)` by `optuna.suggest_float(..., log=True)`
- Switched to `ruff` and `pyproject.toml`
- Removed `online_sampling` and `max_episode_length` argument when using `HerReplayBuffer`

## Release 1.7.0 (2023-01-10)

**SB3 v1.7.0, added support for python config files**

### Breaking Changes
- `--yaml-file` argument was renamed to `-conf` (`--conf-file`) as now python file are supported too
- Upgraded to SB3 >= 1.7.0 (changed `net_arch=[dict(pi=.., vf=..)]` to `net_arch=dict(pi=.., vf=..)`)

### New Features
- Specifying custom policies in yaml file is now supported (@Rick-v-E)
- Added ``monitor_kwargs`` parameter
- Handle the `env_kwargs` of `render:True` under the hood for panda-gym v1 envs in `enjoy` replay to match visualzation behavior of other envs
- Added support for python config file
- Tuned hyperparameters for PPO on Swimmer
- Added ``-tags/--wandb-tags`` argument to ``train.py`` to add tags to the wandb run
- Added a sb3 version tag to the wandb run

### Bug fixes
- Allow `python -m rl_zoo3.cli` to be called directly
- Fixed a bug where custom environments were not found despite passing ``--gym-package`` when using subprocesses
- Fixed TRPO hyperparameters for MinitaurBulletEnv-v0, MinitaurBulletDuckEnv-v0, HumanoidBulletEnv-v0, InvertedDoublePendulumBulletEnv-v0 and InvertedPendulumSwingupBulletEnv

### Documentation

### Other
- `scripts/plot_train.py` plots models such that newer models appear on top of older ones.
- Added additional type checking using mypy
- Standardized the use of ``from gym import spaces``


## Release 1.6.3 (2022-10-13)

### Breaking Changes

### New Features

### Bug fixes
- `python3 -m rl_zoo3.train` now works as expected

### Documentation
- Added instructions and examples on passing arguments in an interactive session (@richter43)

### Other
- Used issue forms instead of issue templates


## Release 1.6.2.post2 (2022-10-10)

### Breaking Changes
- RL Zoo is now a python package
- low pass filter was removed
- Upgraded to Stable-Baselines3 (SB3) >= 1.6.2
- Upgraded to sb3-contrib >= 1.6.2
- Use now built-in SB3 `ProgressBarCallback` instead of `TQDMCallback`

### New Features
- RL Zoo cli: `rl_zoo3 train` and `rl_zoo3 enjoy`

### Bug fixes

### Documentation

### Other

## Release 1.6.1 (2022-09-30)

**Progress bar and custom yaml file**

### Breaking Changes
- Upgraded to Stable-Baselines3 (SB3) >= 1.6.1
- Upgraded to sb3-contrib >= 1.6.1

### New Features
- Added `--yaml-file` argument option for `train.py` to read hyperparameters from custom yaml files (@JohannesUl)

### Bug fixes
- Added `custom_object` parameter on record_video.py (@Affonso-Gui)
- Changed `optimize_memory_usage` to `False` for DQN/QR-DQN on record_video.py (@Affonso-Gui)
- In `ExperimentManager` `_maybe_normalize` set `training` to `False` for eval envs,
  to prevent normalization stats from being updated in eval envs (e.g. in EvalCallback) (@pchalasani).
- Only one env is used to get the action space while optimizing hyperparameters and it is correctly closed (@SammyRamone)
- Added progress bar via the `-P` argument using tqdm and rich

### Documentation

### Other

## Release 1.6.0 (2022-08-05)

**RecurrentPPO (ppo_lstm) and Huggingface integration**

### Breaking Changes
- Change default value for number of hyperparameter optimization trials from 10 to 500. (@ernestum)
- Derive number of intermediate pruning evaluations from number of time steps (1 evaluation per 100k time steps.) (@ernestum)
- Updated default --eval-freq from 10k to 25k steps
- Update default horizon to 2 for the `HistoryWrapper`
- Upgrade to Stable-Baselines3 (SB3) >= 1.6.0
- Upgrade to sb3-contrib >= 1.6.0

### New Features
- Support setting PyTorch's device with thye `--device` flag (@gregwar)
- Add `--max-total-trials` parameter to help with distributed optimization. (@ernestum)
- Added `vec_env_wrapper` support in the config (works the same as `env_wrapper`)
- Added Huggingface hub integration
- Added `RecurrentPPO` support (aka `ppo_lstm`)
- Added autodownload for "official" sb3 models from the hub
- Added Humanoid-v3, Ant-v3, Walker2d-v3 models for A2C (@pseudo-rnd-thoughts)
- Added MsPacman models

### Bug fixes
- Fix `Reacher-v3` name in PPO hyperparameter file
- Pinned ale-py==0.7.4 until new SB3 version is released
- Fix enjoy / record videos with LSTM policy
- Fix bug with environments that have a slash in their name (@ernestum)
- Changed `optimize_memory_usage` to `False` for DQN/QR-DQN on Atari games,
  if you want to save RAM, you need to deactivate `handle_timeout_termination`
  in the `replay_buffer_kwargs`

### Documentation

### Other
- When pruner is set to `"none"`, use `NopPruner` instead of diverted `MedianPruner` (@qgallouedec)

## Release 1.5.0 (2022-03-25)

**Support for Weight and Biases experiment tracking**

### Breaking Changes
- Upgrade to Stable-Baselines3 (SB3) >= 1.5.0
- Upgrade to sb3-contrib >= 1.5.0
- Upgraded to gym 0.21

### New Features
- Verbose mode for each trial (when doing hyperparam optimization) can now be activated using the debug mode (verbose == 2)
- Support experiment tracking via Weights and Biases via the `--track` flag (@vwxyzjn)
- Support tracking raw episodic stats via `RawStatisticsCallback` (@vwxyzjn, see https://github.com/DLR-RM/rl-baselines3-zoo/pull/216)

### Bug fixes
- Policies saved during during optimization with distributed Optuna load on new systems (@jkterry)
- Fixed script for recording video that was not up to date with the enjoy script

### Documentation

### Other

## Release 1.4.0 (2022-01-19)

### Breaking Changes
- Dropped python 3.6 support
- Upgrade to Stable-Baselines3 (SB3) >= 1.4.0
- Upgrade to sb3-contrib >= 1.4.0

### New Features
- Added mujoco hyperparameters
- Added MuJoCo pre-trained agents
- Added script to parse best hyperparameters of an optuna study
- Added TRPO support
- Added ARS support and pre-trained agents

### Bug fixes

### Documentation
- Replace front image

### Other


## Release 1.3.0 (2021-10-23)

**rliable plots and bug fixes**

**WARNING: This version will be the last one supporting Python 3.6 (end of life in Dec 2021). We highly recommended you to upgrade to Python >= 3.7.**

### Breaking Changes
- Upgrade to panda-gym 1.1.1
- Upgrade to Stable-Baselines3 (SB3) >= 1.3.0
- Upgrade to sb3-contrib >= 1.3.0

### New Features
- Added support for using rliable for performance comparison

### Bug fixes
- Fix training with Dict obs and channel last images

### Documentation

### Other
- Updated docker image
- constrained gym version: gym>=0.17,<0.20
- Better hyperparameters for A2C/PPO on Pendulum

## Release 1.2.0 (2021-09-08)

### Breaking Changes
- Upgrade to Stable-Baselines3 (SB3) >= 1.2.0
- Upgrade to sb3-contrib >= 1.2.0

### New Features
- Added support for Python 3.10

### Bug fixes
- Fix `--load-last-checkpoint` (@SammyRamone)
- Fix `TypeError` for `gym.Env` class entry points in `ExperimentManager` (@schuderer)
- Fix usage of callbacks during hyperparameter optimization (@SammyRamone)

### Documentation

### Other
- Added python 3.9 to Github CI
- Increased DQN replay buffer size for Atari games (@nikhilrayaprolu)

## Release 1.1.0 (2021-07-01)

### Breaking Changes
- Upgrade to Stable-Baselines3 (SB3) >= 1.1.0
- Upgrade to sb3-contrib >= 1.1.0
- Add timeout handling (cf SB3 doc)
- `HER` is now a replay buffer class and no more an algorithm
- Removed `PlotNoiseRatioCallback`
- Removed `PlotActionWrapper`
- Changed `'lr'` key in Optuna param dict to `'learning_rate'` so the dict can be directly passed to SB3 methods (@jkterry)

### New Features
- Add support for recording videos of best models and checkpoints (@mcres)
- Add support for recording videos of training experiments (@mcres)
- Add support for dictionary observations
- Added experimental parallel training (with `utils.callbacks.ParallelTrainCallback`)
- Added support for using multiple envs for evaluation
- Added `--load-last-checkpoint` option for the enjoy script
- Save Optuna study object at the end of hyperparameter optimization and plot the results (`plotly` package required)
- Allow to pass multiple folders to `scripts/plot_train.py`
- Flag to save logs and optimal policies from each training run (@jkterry)

### Bug fixes
- Fixed video rendering for PyBullet envs on Linux
- Fixed `get_latest_run_id()` so it works in Windows too (@NicolasHaeffner)
- Fixed video record when using `HER` replay buffer

### Documentation
- Updated README (dict obs are now supported)

### Other
- Added `is_bullet()` to `ExperimentManager`
- Simplify `close()` for the enjoy script
- Updated docker image to include latest black version
- Updated TD3 Walker2D model (thanks @modanesh)
- Fixed typo in plot title (@scottemmons)
- Minimum cloudpickle version added to `requirements.txt` (@amy12xx)
- Fixed atari-py version (ROM missing in newest release)
- Updated `SAC` and `TD3` search spaces
- Cleanup eval_freq documentation and variable name changes (@jkterry)
- Add clarifying print statement when printing saved hyperparameters during optimization (@jkterry)
- Clarify n_evaluations help text (@jkterry)
- Simplified hyperparameters files making use of defaults
- Added new TQC+HER agents
- Add `panda-gym` environments (@qgallouedec)

## Release 1.0 (2021-03-17)

### Breaking Changes
- Upgrade to SB3 >= 1.0
- Upgrade to sb3-contrib >= 1.0

### New Features
- Added 100+ trained agents + benchmark file
- Add support for loading saved model under python 3.8+ (no retraining possible)
- Added Robotics pre-trained agents (@sgillen)

### Bug fixes
- Bug fixes for `HER` handling action noise
- Fixed double reset bug with `HER` and enjoy script

### Documentation
- Added doc about plotting scripts

### Other
- Updated `HER` hyperparameters

## Pre-Release 0.11.1 (2021-02-27)

### Breaking Changes
- Removed `LinearNormalActionNoise`
- Evaluation is now deterministic by default, except for Atari games
- `sb3_contrib` is now required
- `TimeFeatureWrapper` was moved to the contrib repo
- Replaced old `plot_train.py` script with updated `plot_training_success.py`
- Renamed ``n_episodes_rollout`` to ``train_freq`` tuple to match latest version of SB3

### New Features
- Added option to choose which `VecEnv` class to use for multiprocessing
- Added hyperparameter optimization support for `TQC`
- Added support for `QR-DQN` from SB3 contrib

### Bug fixes
- Improved detection of Atari games
- Fix potential bug in plotting script when there is not enough timesteps
- Fixed a bug when using HER + DQN/TQC for hyperparam optimization

### Documentation
- Improved documentation (@cboettig)

### Other
- Refactored train script, now uses a `ExperimentManager` class
- Replaced `make_env` with SB3 built-in `make_vec_env`
- Add more type hints (`utils/utils.py` done)
- Use f-strings when possible
- Changed `PPO` atari hyperparameters (removed vf clipping)
- Changed `A2C` atari hyperparameters (eps value of the optimizer)
- Updated benchmark script
- Updated hyperparameter optim search space (commented gSDE for A2C/PPO)
- Updated `DQN` hyperparameters for CartPole
- Do not wrap channel-first image env (now natively supported by SB3)
- Removed hack to log success rate
- Simplify plot script

## Pre-Release 0.10.0 (2020-10-28)

### Breaking Changes

### New Features
- Added support for `HER`
- Added low-pass filter wrappers in `utils/wrappers.py`
- Added `TQC` support, implementation from sb3-contrib

### Bug fixes
- Fixed `TimeFeatureWrapper` inferring max timesteps
- Fixed ``flatten_dict_observations`` in `utils/utils.py` for recent Gym versions (@ManifoldFR)
- `VecNormalize` now takes `gamma` hyperparameter into account
- Fix loading of `VecNormalize` when continuing training or using trained agent

### Documentation

### Other
- Added tests for the wrappers
- Updated plotting script


## Release 0.8.0 (2020-08-04)

### Breaking Changes

### New Features
- Distributed optimization (@SammyRamone)
- Added ``--load-checkpoints`` to load particular checkpoints
- Added ``--num-threads`` to enjoy script
- Added DQN support
- Added saving of command line args (@SammyRamone)
- Added DDPG support
- Added version
- Added ``RMSpropTFLike`` support

### Bug fixes
- Fixed optuna warning (@SammyRamone)
- Fixed `--save-freq` which was not taking parallel env into account
- Set `buffer_size` to 1 when testing an Off-Policy model (e.g. SAC/DQN) to avoid memory allocation issue
- Fixed seed at load time for `enjoy.py`
- Non-deterministic eval when doing hyperparameter optimization on atari games
- Use 'maximize' for hyperparameter optimization (@SammyRamone)
- Fixed a bug where reward where not normalized when doing hyperparameter optimization (@caburu)
- Removed `nminibatches` from `ppo.yml` for `MountainCar-v0` and `Acrobot-v1`. (@blurLake)
- Fixed `--save-replay-buffer` to be compatible with latest SB3 version
- Close environment at the end of training
- Updated DQN hyperparameters on simpler gym env (due to an update in the implementation)

### Documentation

### Other
- Reformat `enjoy.py`, `test_enjoy.py`, `test_hyperparams_opt.py`, `test_train.py`, `train.py`, `callbacks.py`, `hyperparams_opt.py`, `utils.py`, `wrappers.py` (@salmannotkhan)
- Reformat `record_video.py` (@salmannotkhan)
- Added codestyle check `make lint` using flake8
- Reformat `benchmark.py` (@salmannotkhan)
- Added github ci
- Fixes most linter warnings
- Now using black and isort for auto-formatting
- Updated plots
