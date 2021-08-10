mkdir -p ./eval_logs
mkdir -p ./mature_policies

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 8 &> ./eval_logs/eval_8.out &
sleep 3
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 9 &> ./eval_logs/eval_9.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 10 &> ./eval_logs/eval_10.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 11 &> ./eval_logs/eval_11.out &
sleep 3
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 12 &> ./eval_logs/eval_12.out &
sleep 3
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 13 &> ./eval_logs/eval_13.out &
sleep 3
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 14 &> ./eval_logs/eval_14.out &
sleep 3
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 15 &> ./eval_logs/eval_15.out &