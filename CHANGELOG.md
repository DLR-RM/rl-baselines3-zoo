## Release ... (WIP)

### Breaking Changes

### New Features
- Distributed optimization (@SammyRamone)
- Added ``--load-checkpoints`` to load particular checkpoints
- Added ``--num-threads`` to enjoy script
- Added DQN support

### Bug fixes
- Fixed optuna warning (@SammyRamone)
- Fixed `--save-freq` which was not taking parallel env into account
- Set `buffer_size` to 1 when testing an Off-Policy model (e.g. SAC/DQN) to avoid memory allocation issue
- Fixed seed at load time for `enjoy.py`
- Non-deterministic eval when doing hyperparameter optimization on atari games
- Use 'maximize' for hyperparameter optimization (@SammyRamone)
- Fixed a bug where reward where not normalized when doing hyperparameter optimization (@caburu)
- Removed `nminibatches` from `ppo.yml` for `MountainCar-v0` and `Acrobot-v1`. (@blurLake)

### Documentation

### Other
- Reformat `enjoy.py`, `test_enjoy.py`, `test_hyperparams_opt.py`, `test_train.py`, `train.py`, `callbacks.py`, `hyperparams_opt.py`, `utils.py`, `wrappers.py` (@salmannotkhan)
- Reformat `record_video.py` (@salmannotkhan)
- Added codestyle check `make lint` using flake8
- Reformat `benchmark.py` (@salmannotkhan)
- Added github ci
- Fixes most linter warnings
