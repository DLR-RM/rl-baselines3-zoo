mkdir -p ./optimize_logs

#!/bin/bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 2000000 -optimize --n-trials 150 --sampler tpe --pruner median --study-name $1 --storage mysql://root:dummy@10.128.0.28/$1 &> ./optimize_logs/optimize_0.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 2000000 -optimize --n-trials 150 --sampler tpe --pruner median --study-name $1 --storage mysql://root:dummy@10.128.0.28/$1 &> ./optimize_logs/optimize_1.out &
sleep 3
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 2000000 -optimize --n-trials 150 --sampler tpe --pruner median --study-name $1 --storage mysql://root:dummy@10.128.0.28/$1 &> ./optimize_logs/optimize_2.out &
sleep 3
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 2000000 -optimize --n-trials 150 --sampler tpe --pruner median --study-name $1 --storage mysql://root:dummy@10.128.0.28/$1 &> ./optimize_logs/optimize_3.out &
sleep 3
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 2000000 -optimize --n-trials 150 --sampler tpe --pruner median --study-name $1 --storage mysql://root:dummy@10.128.0.28/$1 &> ./optimize_logs/optimize_4.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 2000000 -optimize --n-trials 150 --sampler tpe --pruner median --study-name $1 --storage mysql://root:dummy@10.128.0.28/$1 &> ./optimize_logs/optimize_5.out &
sleep 3
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 2000000 -optimize --n-trials 150 --sampler tpe --pruner median --study-name $1 --storage mysql://root:dummy@10.128.0.28/$1 &> ./optimize_logs/optimize_6.out &
sleep 3
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 2000000 -optimize --n-trials 150 --sampler tpe --pruner median --study-name $1 --storage mysql://root:dummy@10.128.0.28/$1 &> ./optimize_logs/optimize_7.out &
sleep 3