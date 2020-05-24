## Release ... (WIP)

### Breaking Changes

### New Features
- Added ``--load-checkpoints`` to load particular checkpoints
- Added ``--num-threads`` to enjoy script

### Bug fixes
- Fixed optuna warning (@SammyRamone)
- Fixed `--save-freq` which was not taking parallel env into account

### Documentation

### Other
- Reformat `enjoy.py`, `test_enjoy.py`, `test_hyperparams_opt.py`, `test_train.py`, `train.py`, `callbacks.py`, `hyperparams_opt.py`, `utils.py`, `wrappers.py` (@salmannotkhan)
- Reformat `record_video.py` (@salmannotkhan)
- Added codestyle check `make lint` using flake8
- Reformat `benchmark.py` (@salmannotkhan)
- Added github ci
