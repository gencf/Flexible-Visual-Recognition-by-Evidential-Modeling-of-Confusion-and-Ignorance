'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--loss', default='CrossEntropyLoss', type=str, help='loss function')
parser.add_argument('--mode' , default='train', type=str, help='train or test')
parser.add_argument('--max_num_epochs', default=1000, type=int, help='max number of epochs')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--save_path', default='results/exp1/CrossEntropyLoss', type=str, help='save path')
parser.add_argument('--model_path', default='results/exp1/CrossEntropyLoss/best.pth', type=str, help='model path')
parser.add_argument('--embedding_size', default=64, type=int, help='embedding size')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_acc_epoch = 0 # best test accuracy epoch
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
save_path = args.save_path
model_path = args.model_path
embedding_size = args.embedding_size
dataset = args.dataset

print('Save path: {}'.format(save_path))
print('Embedding size: {}'.format(embedding_size))
print('Dataset: {}'.format(dataset))

if not os.path.exists(save_path):
    os.makedirs(save_path)
# Train and test transforms using mean and std of the dataset as input
def get_transforms(mean, std):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return transform_train, transform_test


# Data
print('==> Preparing data..')

if dataset == 'CIFAR10':
    # Mean and std for CIFAR10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)

    # Get transforms
    transform_train, transform_test = get_transforms(mean, std)

    # Use CIFAR10 classes
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    
    num_classes = 10

# Model
print('==> Building model..')

net = ResNet18(num_classes=num_classes)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)
criterion = nn.CrossEntropyLoss()

print('Length of testloader: {}'.format(len(testloader)))
print('Length of trainloader: {}'.format(len(trainloader)))

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.mode == 'train' and args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'])
    acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    best_acc_epoch = checkpoint['best_acc_epoch']

if args.mode == 'test':
    # Load checkpoint.
    print('==> Testing..')
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'])
    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    best_acc_epoch = checkpoint['best_acc_epoch']

# criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

stop_training = False

# Training
def train(epoch):
    global stop_training
    print('\nTraining Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # If loss becomes NaN, stop training
        if np.isnan(loss.item()):
            stop_training = True
            print("Loss is NaN. Stopping training..")
            break

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    s = 'Train Epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' \
            % (epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total)

    print(s)
    with open(os.path.join(save_path, 'train_log.txt'), 'a') as f:
        f.write(s)
        f.write('\n')


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
