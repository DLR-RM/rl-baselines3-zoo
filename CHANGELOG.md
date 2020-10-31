## Pre-Release 0.11.0a0 (WIP)

### Breaking Changes
- Removed `LinearNormalActionNoise`

### New Features

### Bug fixes
- Improved detection of Atari games

### Documentation

### Other
- Refactored train script, now uses a `ExperimentManager` class
- Replaced `make_env` with SB3 built-in `make_vec_env`
- Add more type hints (`utils/utils.py` done)
- Use f-strings when possible

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
