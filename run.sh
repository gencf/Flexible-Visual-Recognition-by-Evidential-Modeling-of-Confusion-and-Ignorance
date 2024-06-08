# SETUP
conda create -n custom_loss python=3.9
conda activate custom_loss
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
bash download_data.sh

# USAGE

# Train a model with CrossEntropyLoss on CIFAR10 dataset for 2 epochs to initialize the model with CustomLoss
python main.py --loss CrossEntropyLoss --save_path ./results/CIFAR10/exp1/ --dataset CIFAR10 --max_num_epochs 200

# Train a model with CustomLoss on CIFAR10 dataset
python main.py --loss CustomLoss --save_path ./results/CIFAR10/exp2/ --model_path ./results/CIFAR10/exp1/checkpoint_1000.pth --dataset CIFAR10 --resume --max_num_epochs 200

# Train a model with CrossEntropyLoss on CIFAR100 dataset for 2 epochs to initialize the model with CustomLoss
python main.py --loss CrossEntropyLoss --save_path ./results/CIFAR100/exp1/ --dataset CIFAR100 --max_num_epochs 200

# Train a model with CustomLoss on CIFAR100 dataset
python main.py --loss CustomLoss --save_path ./results/CIFAR100/exp2/ --model_path ./results/CIFAR100/exp1/checkpoint_1000.pth --dataset CIFAR100 --resume --max_num_epochs 200