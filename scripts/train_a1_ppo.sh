#!/bin/bash

CMD="python train.py --algo ppo --env A1GymEnv-v0 --save-freq 100000 | tee run.log"
echo "Executing command: ${CMD}"
eval $CMD

