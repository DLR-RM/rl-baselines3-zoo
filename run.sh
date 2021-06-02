CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 4000000 -optimize --n-trials 1000 --sampler tpe --pruner median --study-name pistonball18 --storage mysql://root:dummy@10.128.0.28/pistonball18 &> 0.out &

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 4000000 -optimize --n-trials 1000 --sampler tpe --pruner median --study-name pistonball18 --storage mysql://root:dummy@10.128.0.28/pistonball18 &> 1.out &

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 4000000 -optimize --n-trials 1000 --sampler tpe --pruner median --study-name pistonball18 --storage mysql://root:dummy@10.128.0.28/pistonball18 &> 2.out &

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 4000000 -optimize --n-trials 1000 --sampler tpe --pruner median --study-name pistonball18 --storage mysql://root:dummy@10.128.0.28/pistonball18 &> 3.out &

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 4000000 -optimize --n-trials 1000 --sampler tpe --pruner median --study-name pistonball18 --storage mysql://root:dummy@10.128.0.28/pistonball18 &> 4.out &

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 4000000 -optimize --n-trials 1000 --sampler tpe --pruner median --study-name pistonball18 --storage mysql://root:dummy@10.128.0.28/pistonball18 &> 5.out &

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 4000000 -optimize --n-trials 1000 --sampler tpe --pruner median --study-name pistonball18 --storage mysql://root:dummy@10.128.0.28/pistonball18 &> 6.out &

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 4000000 -optimize --n-trials 1000 --sampler tpe --pruner median --study-name pistonball18 --storage mysql://root:dummy@10.128.0.28/pistonball18 &> 7.out &
