CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 2000000 -optimize --n-trials 1000 --sampler tpe --pruner median --study-name pistonball21 --storage mysql://root:dummy@10.128.0.28/pistonball21 &> 0.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 2000000 -optimize --n-trials 1000 --sampler tpe --pruner median --study-name pistonball21 --storage mysql://root:dummy@10.128.0.28/pistonball21 &> 1.out &
sleep 3
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 2000000 -optimize --n-trials 1000 --sampler tpe --pruner median --study-name pistonball21 --storage mysql://root:dummy@10.128.0.28/pistonball21 &> 2.out &
sleep 3
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 2000000 -optimize --n-trials 1000 --sampler tpe --pruner median --study-name pistonball21 --storage mysql://root:dummy@10.128.0.28/pistonball21 &> 3.out &
sleep 3
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 2000000 -optimize --n-trials 1000 --sampler tpe --pruner median --study-name pistonball21 --storage mysql://root:dummy@10.128.0.28/pistonball21 &> 4.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 2000000 -optimize --n-trials 1000 --sampler tpe --pruner median --study-name pistonball21 --storage mysql://root:dummy@10.128.0.28/pistonball21 &> 5.out &
sleep 3
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 2000000 -optimize --n-trials 1000 --sampler tpe --pruner median --study-name pistonball21 --storage mysql://root:dummy@10.128.0.28/pistonball21 &> 6.out &
sleep 3
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 2000000 -optimize --n-trials 1000 --sampler tpe --pruner median --study-name pistonball21 --storage mysql://root:dummy@10.128.0.28/pistonball21 &> 7.out &
sleep 3