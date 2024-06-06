import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import argparse
import yaml
import wandb
from tqdm import tqdm

from codes.resnet18_custom_classes import ResNet18_custom_class_number

from codes.dataloaders import get_cifar10_train_dataloader, get_cifar10_test_dataloader, get_cifar100_train_dataloader, get_cifar100_test_dataloader

from codes.loss import custom_loss, calculate_only_belief_uncertainity_ignorance_confusion




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
        self.dataset_name = config.dataset_name
        self.train_batch_size = config.train_batch_size
        self.test_batch_size = config.test_batch_size
        self.max_epochs = config.max_epochs
        self.learning_rate = config.learning_rate
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.max_lambda_kl = config.max_lambda_kl
        self.annealing_last_value = config.annealing_last_value
        self.lambda_reg = config.lambda_reg
        self.num_workers = config.num_workers
        self.save_every_for_model = config.save_every_for_model
        self.logging_interval = config.logging_interval
        self.project_root = config.project_root
        self.model_save_path = config.model_save_path
        self.dataset_download_path = config.dataset_download_path
        self.verbose = config.verbose
        self.device = config.device


        # check the dataset name
        if self.dataset_name == 'CIFAR10':
            self.n_classes = 10
        elif self.dataset_name == 'CIFAR100':
            self.n_classes = 100
        else:
            raise ValueError('Invalid dataset name. Choose from CIFAR10, CIFAR100')
        


        # load the ResNet18 model
        self.resnet18_classifier = ResNet18_custom_class_number(
            num_classes=self.n_classes,
            project_root=self.project_root,
            use_pretrained=self.use_pretrained_resnet,
            model_cache_path=self.pretrained_resnet_cache_path,
            verbose=self.verbose
        )

        # print the model information
        if self.verbose:
            print(self.resnet18_classifier)

            # print the number of parameters
            print(f"Number of parameters: {sum(p.numel() for p in self.resnet18_classifier.parameters() if p.requires_grad)}")



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
        if self.use_wandb:
            self.initialize_wandb()


    # initialize wandb
    def initialize_wandb(self):
        mode = 'online' if self.online else 'offline'
        kwargs = {'name': self.exp_name, 'project': 'Flexible_Visual_Recognition',
                'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode, 'save_code': True}
        wandb.init(**kwargs)



    def save_whole_model(self, iter):
        if not os.path.exists(os.path.join(self.project_root, self.model_save_path, self.exp_name)):
            os.makedirs(os.path.join(self.project_root, self.model_save_path, self.exp_name))

        full_model_save_path = os.path.join(self.project_root, self.model_save_path, self.exp_name, f"full_model_{self.exp_name}_{iter}.pt")

        torch.save(self.resnet18_classifier.state_dict(), full_model_save_path)




    def train(self):

        # set the seed again to guarantee the same results in terms of used data
        if self.set_seed:
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            os.environ["PYTHONHASHSEED"] = str(self.seed)

            if self.verbose:
                print(f'Using seed {self.seed}')
            
        # get the train and test dataloaders
        if self.dataset_name == 'CIFAR10':
            trainloader = get_cifar10_train_dataloader(
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
                dataset_relative_path=self.dataset_download_path
            )

            testloader = get_cifar10_test_dataloader(
                batch_size=self.test_batch_size,
                num_workers=self.num_workers,
                dataset_relative_path=self.dataset_download_path
            )
        elif self.dataset_name == 'CIFAR100':
            trainloader = get_cifar100_train_dataloader(
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
                dataset_relative_path=self.dataset_download_path
            )

            testloader = get_cifar100_test_dataloader(
                batch_size=self.test_batch_size,
                num_workers=self.num_workers,
                dataset_relative_path=self.dataset_download_path
            )
        
        # define the optimizer
        optimizer = optim.Adam(self.resnet18_classifier.parameters(),
                               lr=self.learning_rate,
                               betas=(self.momentum, 0.999),
                               weight_decay=self.weight_decay)
        
        # define the loss function
        criterion = custom_loss(
            max_epochs=self.max_epochs,
            max_lambda_kl=self.max_lambda_kl,
            annealing_last_value=self.annealing_last_value,
            n_classes=self.n_classes,
            lambda_reg=self.lambda_reg
        )

        # get the length of the dataset
        len_dataset = len(trainloader.dataset)

        dataset_batch_len = len_dataset//self.train_batch_size

        # set the model to the device
        self.resnet18_classifier.to(self.device)

        # set the model to train mode
        self.resnet18_classifier.train()

        # print info
        print(f"Starting training for {self.max_epochs} epochs with {len_dataset} samples.")


        # train the model
        for current_epoch in tqdm(range(self.max_epochs), desc='Epochs', total=self.max_epochs, dynamic_ncols=True):
            for batch_idx, (inputs, y_true) in enumerate(trainloader):
                # move the inputs and y_true to the device
                inputs, y_true = inputs.to(self.device), y_true.to(self.device)

                # turn y_true to one-hot
                y_true = nn.functional.one_hot(y_true, num_classes=self.n_classes).float()

                # zero the gradients
                optimizer.zero_grad()

                # forward pass
                y_pred = self.resnet18_classifier(inputs)

                # calculate the loss
                loss = criterion(plausibility=y_pred,
                                 y_true=y_true,
                                 epoch=current_epoch,)

                # backward pass
                loss.backward()
                
                # update the weights
                optimizer.step()

                    
                # calculate the top-1 accuracy
                train_accuracy = torch.sum(y_pred.argmax(dim=1) == y_true.argmax(dim=1)).detach().cpu() / self.train_batch_size

                # calculate the step number
                step_num = current_epoch * dataset_batch_len + batch_idx



                # save the model
                if ((current_epoch) % (self.save_every_for_model) == 0) and (batch_idx == 0):
                    self.save_whole_model(batch_idx)
                    if self.verbose:
                        print(f"Model saved at epoch {current_epoch} and step {step_num}")

                

                if (step_num+1) % self.logging_interval == 0:
                    # log the losses
                    if self.use_wandb:
                        wandb.log({'train_loss': loss.item(),
                                'step': step_num,
                                'train_accuracy': train_accuracy,
                                'lr': optimizer.param_groups[0]['lr'],
                                'lambda_kl': criterion.lambda_kl,})

                    # print the loss and accuracy
                    if self.verbose:
                        # update the progress bar
                        print(f"Epoch: {current_epoch}, Step: {step_num}, Loss: {loss.item():.3f}, Accuracy: {train_accuracy}")


            






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
    
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    

    # ResNet18 configuration

    
    parser.add_argument('--use_pretrained_resnet', type=str2bool, default=False,
                        help='Whether to use the pretrained weights.')
    
    parser.add_argument('--pretrained_resnet_cache_path', type=str, default='',
                        help='The relative path to save the pretrained model.')


    # Dataset configuration

    parser.add_argument('--dataset_name', type=str, default='CIFAR10',
                        help='The dataset name to use (CIFAR10, CIFAR100)')
    

    # Training configuration
    
    parser.add_argument('--train_batch_size', type=int, default=128,
                        help='The batch size for training.')
    
    parser.add_argument('--test_batch_size', type=int, default=128,
                        help='The batch size for testing.')
    
    parser.add_argument('--max_epochs', type=int, default=5000,
                        help='The number of max epochs for training.')
    
    parser.add_argument('--learning_rate', type=float, default=0.004,
                        help='The learning rate for training.')
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='The momentum for training.')
    
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='The weight decay for training.')
    
    parser.add_argument('--max_lambda_kl', type=float, default=0.05,
                        help='The maximum value for the KL divergence (for KL loss).')
    
    parser.add_argument('--annealing_last_value', type=float, default=0.0,
                        help='The last value for the annealing (for KL loss).')
    
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='The regularization coefficient (for REG loss).')
    
    parser.add_argument('--num_workers', type=int, default=2,
                        help='The number of workers for the dataloaders.')

    parser.add_argument('--save_every_for_model', type=int, default=50,
                        help='The number of epochs to save the model.')
    
    parser.add_argument('--logging_interval', type=int, default=100,
                        help='The number of iterations to log the informations.')



    # paths configurations

    parser.add_argument('--project_root', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
                        help='The absolute path of the project root directory.')
    
    parser.add_argument('--model_save_path', type=str, default='weights',
                        help='The relative path to save the model.')
    
    parser.add_argument('--dataset_download_path', type=str, default='./data',
                        help='The relative path to download the dataset.')


    # other configurations

    parser.add_argument('--verbose', type=str2bool, default=True,
                        help='Whether to print the informations.')

    
    parser.add_argument('--device', type=str, default='cuda',
                        help='The device to use (cuda, cpu)')
    

    



    config = parser.parse_args()



    train = Train(config)
    train.train()

    

