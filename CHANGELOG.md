## Release 1.1.0a10 (WIP)

### Breaking Changes
- Upgrade to SB3 >= 1.1.0a10 (master version)
- Upgrade to sb3-contrib >= 1.1.0a6
- Add timeout handling (cf SB3 doc)
- `HER` is now a replay buffer class and no more an algorithm
- Removed `PlotNoiseRatioCallback`
- Removed `PlotActionWrapper`

### New Features
- Add support for recording videos of best models and checkpoints (@mcres)
- Add support for recording videos of training experiments (@mcres)
- Add support for dictionary observations
- Added experimental parallel training (with `utils.callbacks.ParallelTrainCallback`)
- Added support for using multiple envs for evaluation
- Added `--load-last-checkpoint` option for the enjoy script

### Bug fixes
- Fixed video rendering for PyBullet envs on Linux
- Fixed `get_latest_run_id()` so it works in Windows too (@NicolasHaeffner)

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
