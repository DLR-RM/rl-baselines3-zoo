mkdir -p ./eval_logs
mkdir -p ./mature_policies
rm -rf eval_logs/*
rm -rf mature_policies/*

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 0 &> ./eval_logs/eval_0.out &
sleep 3
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 1 &> ./eval_logs/eval_1.out &
sleep 3
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 2 &> ./eval_logs/eval_2.out &
sleep 3
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 eval_hyperparameters.py 3 &> ./eval_logs/eval_3.out &
sleep 3
