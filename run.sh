#!/usr/bin/bash
for lambda in 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001
do
  for seed in 2550 998692 129612 428635 8351
  do
    python train.py --algo ppo --seed $seed --log-folder logs/11_01_2022  --env cwcf-v0 --n-timesteps 2048000 --eval-freq 10240 --eval-episodes 990 --n-eval-envs 1 --env-kwargs lambda_coefficient:$lambda mode:0 terminal_reward:[[0,-0.3],[-0.7,0]] --eval-env-kwargs lambda_coefficient:$lambda random_mode:False mode:1  terminal_reward:[[0,-0.3],[-0.7,0]] --tensorboard-log /tmp/stable-baselines/
  done
 done

