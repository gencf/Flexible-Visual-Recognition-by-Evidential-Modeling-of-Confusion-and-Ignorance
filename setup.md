# [Flexible Visual Recognition by Evidential Modeling of Confusion and Ignorance](https://openaccess.thecvf.com/content/ICCV2023/papers/Fan_Flexible_Visual_Recognition_by_Evidential_Modeling_of_Confusion_and_Ignorance_ICCV_2023_paper.pdf)

Lei Fan, Bo Liu, Haoxiang Li, Ying Wu, and Gang Hua

*ICCV 2023*

This folder provides a re-implementation of this paper in PyTorch, developed as part of the course METU CENG 796 - Deep Generative Models. The re-implementation is provided by:
* Furkan Genç, genc.furkan@metu.edu.tr 
* Umut Özyurt, umut.ozyurt@metu.edu.tr

Please see the Jupyter notebook file [train_toy_dataset.ipynb](train_toy_dataset.ipynb) for the implementation of the Synthetic Experiments in the paper.

## Project Setup

### Setup Environment

1. **Create a new conda environment** named `custom_loss` with Python 3.9:
    ```bash
    conda create -n custom_loss python=3.9
    ```

2. **Activate** the newly created environment:
    ```bash
    conda activate custom_loss
    ```

3. **Install PyTorch** and related libraries with CUDA support:
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

4. **Install additional required Python packages** from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Train a model with CrossEntropyLoss on CIFAR10 dataset for 200 epochs
```bash
python main.py --loss CrossEntropyLoss --save_path ./results/CIFAR10/exp1/ --dataset CIFAR10 --max_num_epochs 200
```

### Train a model with CustomLoss on CIFAR10 dataset for 200 epochs. 
```bash
python main.py --loss CustomLoss --save_path ./results/CIFAR10/exp2/ --model_path ./results/CIFAR10/exp1/checkpoint_1000.pth --dataset CIFAR10 --resume --max_num_epochs 200
```

### Train a model with CrossEntropyLoss on CIFAR100 dataset for 200 epochs
```bash
python main.py --loss CrossEntropyLoss --save_path ./results/CIFAR100/exp1/ --dataset CIFAR100 --max_num_epochs 200
```

### Train a model with CustomLoss on CIFAR100 dataset for 200 epochs
```bash
python main.py --loss CustomLoss --save_path ./results/CIFAR100/exp2/ --model_path ./results/CIFAR100/exp1/checkpoint_1000.pth --dataset CIFAR100 --resume --max_num_epochs 200
```

### Pre-trained Models

The model trained on the CIFAR10 and CIFAR100 datasets for 200 epochs can be found in this [link](https://drive.google.com/drive/folders/1vjfo9FlwEuImDOGvFxDm_he1nAwZCKO2?usp=sharing).


