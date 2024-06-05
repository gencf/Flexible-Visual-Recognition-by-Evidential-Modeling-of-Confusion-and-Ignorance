import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

import argparse
import yaml
import wandb
from tqdm import tqdm

from codes.resnet18_custom_classes import ResNet18_custom_class_number





class Train:
    def __init__(self, config):
        if config.set_seed:
            np.random.seed(config.seed)
            random.seed(config.seed)
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            os.environ["PYTHONHASHSEED"] = str(config.seed)

            if config.verbose:
                print(f'Using seed {config.seed}')


        self.set_seed = config.set_seed
        self.seed = config.seed

        self.use_wandb = config.use_wandb
        self.online = config.online
        self.exp_name = config.exp_name

        self.use_pretrained_resnet = config.use_pretrained_resnet
        self.pretrained_resnet_cache_path = config.pretrained_resnet_cache_path

        self.verbose = config.verbose
        self.project_root = config.project_root
        self.dataset_name = config.dataset_name
        self.device = config.device
        self.model_save_path = config.model_save_path
        self.save_every_for_model = config.save_every_for_model


        # check the dataset name
        if self.dataset_name == 'CIFAR10':
            self.num_classes = 10
        elif self.dataset_name == 'CIFAR100':
            self.num_classes = 100
        else:
            raise ValueError('Invalid dataset name. Choose from CIFAR10, CIFAR100')
        


        # load the ResNet18 model
        self.master_style_transformer = ResNet18_custom_class_number(
            num_classes=self.num_classes,
            project_root=self.project_root,
            use_pretrained=self.use_pretrained_resnet,
            model_cache_path=self.pretrained_resnet_cache_path,
            verbose=self.verbose
        )



        # Make sure model saving path exists
        if not os.path.exists(os.path.join(self.model_save_path, self.exp_name)):
            os.makedirs(os.path.join(self.model_save_path, self.exp_name))
        else:
            # If the model saving path already exists, create a new folder with a new name and change experiment name
            print(f"Model saving path already exists: {os.path.join(self.model_save_path, self.exp_name)}")

            self.exp_name = self.exp_name + "_new_0"
            while os.path.exists(os.path.join(self.model_save_path, self.exp_name)):
                self.exp_name = self.exp_name[:-1] + str(int(self.exp_name[-1]) + 1)
            
            print(f"New experiment name: {self.exp_name}")

            os.makedirs(os.path.join(self.model_save_path, self.exp_name))
            
        
        # save config file as a yaml file
        with open(os.path.join(self.project_root, self.model_save_path, self.exp_name, f"{self.exp_name}_config.yaml"), 'w') as file:
            yaml.dump(vars(self), file)


    # initialize wandb
    def initialize_wandb(self):
        mode = 'online' if self.online else 'offline'
        kwargs = {'name': self.exp_name, 'project': 'Flexible_Visual_Recognition',
                'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode, 'save_code': True}
        wandb.init(**kwargs)



    def save_whole_model(self, iter):
        full_model_save_path = os.path.join(self.project_root, self.model_save_path, self.exp_name, f"full_model_{self.exp_name}_{iter}.pt")

        torch.save(self.master_style_transformer.state_dict(), full_model_save_path)




    def train(self):
        pass



if __name__ == '__main__':

    # define str2bool function for argparse
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    def str2listint(v):
        # strip the string and split it by comma
        v = v.strip().split(',')
        # convert the string to integer
        v = [int(i) for i in v]
        return v

        
    # create the parser
    parser = argparse.ArgumentParser(description='Train the model')


    # add the arguments


    # Seed configuration.
    parser.add_argument('--set_seed', type=str2bool, nargs='?', const=True, default=False,
                        help='set seed for reproducibility')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for reproducibility')


    # wandb configuration.
    parser.add_argument('--use_wandb', type=str2bool, nargs='?', const=True, default=False,
                        help='use wandb for logging')
    
    parser.add_argument('--online', type=str2bool, nargs='?', const=True, default=True,
                        help='use wandb online')
    
    parser.add_argument('--exp_name', type=str, default='master',
                        help='experiment name')
    

    # ResNet18 configuration

    
    parser.add_argument('--use_pretrained_resnet', type=str2bool, default=False,
                        help='Whether to use the pretrained weights.')
    
    parser.add_argument('--pretrained_resnet_cache_path', type=str, default='',
                        help='The relative path to save the pretrained model.')


    # other configurations

    parser.add_argument('--verbose', type=str2bool, default=True,
                        help='Whether to print the informations.')

    parser.add_argument('--project_root', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
                        help='The absolute path of the project root directory.')
    
    parser.add_argument('--dataset_name', type=str, default='CIFAR10',
                        help='The dataset name to use (CIFAR10, CIFAR100)')
    
    parser.add_argument('--device', type=str, default='cuda',
                        help='The device to use (cuda, cpu)')
    
    parser.add_argument('--model_save_path', type=str, default='weights',
                        help='The relative path to save the model.')
    
    parser.add_argument('--save_every_for_model', type=int, default=5000,
                        help='The number of iterations to save the model.')



    config = parser.parse_args()



    train = Train(config)
    train.train()

    

