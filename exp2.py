import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torcheval.metrics import MulticlassAUROC
from sklearn.metrics import roc_auc_score

import argparse
import yaml
import wandb
from tqdm import tqdm

from codes.resnet18_custom_classes import ResNet18_custom_class_number

from codes.dataloaders import get_cifar10_train_dataloader, get_cifar10_test_dataloader, get_cifar100_train_dataloader, get_cifar100_test_dataloader

from codes.loss import custom_loss, calculate_only_belief_uncertainity_ignorance_confusion

from train_exp import Train


config_dict = {
    "set_seed"                        : False,
    "seed"                            : 42,
    "use_wandb"                       : False,
    "online"                          : True,
    "exp_name"                        : 'exp',
    "use_pretrained_resnet"           : False,
    "pretrained_resnet_cache_path"    : '',
    "pretrained_with_BCE_resnet_path" : '',
    "dataset_name"                    : 'CIFAR10',
    "train_batch_size"                : 128,
    "test_batch_size"                 : 128,
    "max_epochs"                      : 5000,
    "optimizer"                       : 'Adam',
    "learning_rate"                   : 0.004,
    "momentum"                        : 0.9,
    "weight_decay"                    : 0.0005,
    "max_lambda_kl"                   : 0.05,
    "annealing_last_value"            : 0.0,
    "lambda_reg"                      : 1,
    "num_workers"                     : 2,
    "save_every_for_model"            : 10000,
    "logging_interval"                : 100,
    "use_BCE_classic_training"        : False,
    "output_losses_separately"        : False,
    "auroc_metric_lib"                : 'sklearn',
    "project_root"                    : os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "model_save_path"                 : 'exps',
    "dataset_download_path"           : './data',
    "verbose"                         : True,
    "device"                          : 'cuda',
}

# change some parameters

config_dict["use_wandb"]                          = False
config_dict["verbose"]                            = True
config_dict["device"]                             = "cuda"
config_dict["set_seed"]                           = True
config_dict["seed"]                               = 42
config_dict["num_workers"]                        = 4
config_dict["max_epochs"]                         = 1000
config_dict["train_batch_size"]                   = 128
config_dict["test_batch_size"]                    = 3333
config_dict["dataset_name"]                       = "CIFAR10"
config_dict["save_every_for_model"]               = 10000
config_dict["output_losses_separately"]           = True
config_dict["optimizer"]                          = "SGD"
config_dict["learning_rate"]                      = 0.004
config_dict["auroc_metric_lib"]                   = "sklearn"
config_dict["pretrained_with_BCE_resnet_path"]    = "exps/experiment_17/full_model_experiment_17_999.pt"
config_dict["exp_name"]                           = "local_exp"

# create a config to call Train class
config = argparse.Namespace()

for key, value in config_dict.items():
    setattr(config, key, value)


# create train object
train = Train(config)

# start train
train.train()


