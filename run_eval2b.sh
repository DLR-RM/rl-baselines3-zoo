CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 12 &> ./eval_logs/eval_12.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 13 &> ./eval_logs/eval_13.out &
sleep 3
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 14 &> ./eval_logs/eval_14.out &
sleep 3
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 15 &> ./eval_logs/eval_15.out &