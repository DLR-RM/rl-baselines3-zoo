CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 test_hyperparameters.py 0 &> ./eval_logs/eval_0.out &
sleep 3
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 test_hyperparameters.py 1 &> ./eval_logs/eval_1.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 test_hyperparameters.py 2 &> ./eval_logs/eval_2.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 test_hyperparameters.py 3 &> ./eval_logs/eval_3.out &
sleep 3
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 test_hyperparameters.py 4 &> ./eval_logs/eval_4.out &
sleep 3
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 test_hyperparameters.py 5 &> ./eval_logs/eval_5.out &
sleep 3
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 test_hyperparameters.py 6 &> ./eval_logs/eval_6.out &
sleep 3
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 test_hyperparameters.py 7 &> ./eval_logs/eval_7.out &
