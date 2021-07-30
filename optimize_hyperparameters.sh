#!/bin/bash
nohup docker run -t --gpus 0 flocking &> 0.out &
sleep 3
nohup docker run -t --gpus 0 flocking &> 1.out &
sleep 3
nohup docker run -t --gpus 1 flocking &> 2.out &
sleep 3
nohup docker run -t --gpus 1 flocking &> 3.out &
sleep 3
nohup docker run -t --gpus 2 flocking &> 4.out &
sleep 3
nohup docker run -t --gpus 2 flocking &> 5.out &
sleep 3
nohup docker run -t --gpus 3 flocking &> 6.out &
sleep 3
nohup docker run -t --gpus 3 flocking &> 7.out &
