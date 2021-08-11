mkdir -p ./mature_gifs
mkdir -p ./render_logs
rm -rf mature_gifs/*
rm -rf render_logs/*

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 render.py 8 &> ./render_logs/render_8.out &
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 render.py 9 &> ./render_logs/render_9.out &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 render.py 10 &> ./render_logs/render_10.out &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 nohup python3 render.py 11 &> ./render_logs/render_11.out &
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 render.py 12 &> ./render_logs/render_12.out &
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 nohup python3 render.py 13 &> ./render_logs/render_13.out &
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 render.py 14 &> ./render_logs/render_14.out &
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 nohup python3 render.py 15 &> ./render_logs/render_15.out &
