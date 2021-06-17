CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 best_trials.py 0 &> eval_0.out &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 best_trials.py 1 &> eval_1.out &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 best_trials.py 2 &> eval_2.out &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 best_trials.py 3 &> eval_3.out &
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 best_trials.py 4 &> eval_4.out &
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 best_trials.py 5 &> eval_5.out &
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 best_trials.py 6 &> eval_6.out &
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 best_trials.py 7 &> eval_7.out &

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 best_trials.py 8 &> eval_8.out &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 best_trials.py 9 &> eval_9.out &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 best_trials.py 10 &> eval_10.out &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 best_trials.py 11 &> eval_11.out &
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 best_trials.py 12 &> eval_12.out &
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 best_trials.py 13 &> eval_13.out &
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 best_trials.py 14 &> eval_14.out &
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 best_trials.py 15 &> eval_15.out &