mkdir -p ./hyperparameter_jsons
rm -rf hyperparameter_jsons/*

python3 best_hyperparameters.py --study-name $1 --storage mysql://root:dummy@35.194.57.226/$1 --save-n-best-hyperparameters 16  --print-n-best-trials 100
