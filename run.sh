

# Train a model with CrossEntropyLoss on CIFAR10 dataset for 2 epochs to initialize the model with CustomLoss
python main.py --loss CrossEntropyLoss --save_path ./results/CIFAR10/exp1/ --dataset CIFAR10 --max_num_epochs 200

# Train a model with CustomLoss on CIFAR10 dataset
python main.py --loss CustomLoss --save_path ./results/CIFAR10/exp2/ --model_path ./results/CIFAR10/exp1/checkpoint_1000.pth --dataset CIFAR100 --resume --max_num_epochs 200

# Train a model with CrossEntropyLoss on CIFAR100 dataset for 2 epochs to initialize the model with CustomLoss
python main.py --loss CrossEntropyLoss --save_path ./results/CIFAR100/exp1/ --dataset CIFAR100 --max_num_epochs 200

# Train a model with CustomLoss on CIFAR100 dataset
python main.py --loss CustomLoss --save_path ./results/CIFAR100/exp2/ --model_path ./results/CIFAR100/exp1/checkpoint_8000.pth --dataset CIFAR100 --resume --max_num_epochs 200