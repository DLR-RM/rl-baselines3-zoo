CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 4 &> ./eval_logs/eval_4.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 5 &> ./eval_logs/eval_5.out &
sleep 3
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 6 &> ./eval_logs/eval_6.out &
sleep 3
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 7 &> ./eval_logs/eval_7.out &
