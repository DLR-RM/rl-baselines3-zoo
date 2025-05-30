#!/bin/bash
# Launch an experiment using the docker gpu image

cmd_line="$@"

echo "Executing in the docker (gpu image):"
echo $cmd_line

# Note: --user=root is needed, as the current user id/group id will be used
# to mount the log directory (and $MAMBAUSER is not root)
docker run -it --user=root --gpus all --rm --network host --ipc=host \
  --mount src=$(pwd),target=/home/mambauser/code/rl_zoo3,type=bind stablebaselines/rl-baselines3-zoo:latest\
  bash -c "cd /home/mambauser/code/rl_zoo3/ && $cmd_line"
