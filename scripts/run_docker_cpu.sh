#!/bin/bash
# Launch an experiment using the docker cpu image

cmd_line="$@"

echo "Executing in the docker (cpu image):"
echo $cmd_line

docker run -it --rm --network host --ipc=host \
 --mount src=$(pwd),target=/root/code/rl_zoo,type=bind stablebaselines/rl-baselines3-zoo-cpu:latest\
  bash -c "cd /root/code/rl_zoo/ && $cmd_line"
