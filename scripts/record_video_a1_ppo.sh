#!/bin/bash

CMD="python -m utils.record_video --algo ppo --env A1GymEnv-v0 -f logs --load-best"
echo "Executing command: ${CMD}"
eval $CMD
