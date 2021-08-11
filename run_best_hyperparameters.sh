mkdir -p ./hyperparameter_jsons
rm -rf hyperparameter_jsons/*

python3 best_hyperparameters.py --study-name $1 --storage mysql://root:dummy@10.128.0.28/$1 --save-n-best-hyperparameters 16