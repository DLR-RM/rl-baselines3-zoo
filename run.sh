#!/bin/bash
OMP_NUM_THREADS=1 python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 2000000 -optimize --n-trials 150 --sampler tpe --pruner median --study-name $1 --storage mysql://root:dummy@10.128.0.28/$1

