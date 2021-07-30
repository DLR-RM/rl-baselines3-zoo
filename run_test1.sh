CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 test_hyperparameters.py 0 &> eval_0.out &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 test_hyperparameters.py 1 &> eval_1.out &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 test_hyperparameters.py 2 &> eval_2.out &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 test_hyperparameters.py 3 &> eval_3.out &
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 test_hyperparameters.py 4 &> eval_4.out &
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 test_hyperparameters.py 5 &> eval_5.out &
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 test_hyperparameters.py 6 &> eval_6.out &
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 test_hyperparameters.py 7 &> eval_7.out &

