from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_cifar10_train_transform():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])
    return transform_train


def get_cifar10_test_transform():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])
    return transform_test


def get_cifar100_train_transform():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    return transform_train


def get_cifar100_test_transform():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    return transform_test


def get_cifar10_train_dataloader(batch_size=128, num_workers=2, dataset_relative_path=""):
    transform_train = get_cifar10_train_transform()

    trainset = datasets.CIFAR10(root=dataset_relative_path, train=True, download=True, transform=transform_train)

    dataloader_train = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    return dataloader_train

def get_cifar10_test_dataloader(batch_size=128, num_workers=2, dataset_relative_path=""):
    transform_test = get_cifar10_test_transform()

    testset = datasets.CIFAR10(root=dataset_relative_path, train=False, download=True, transform=transform_test)

    dataloader_test = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return dataloader_test

def get_cifar100_train_dataloader(batch_size=128, num_workers=2, dataset_relative_path=""):
    transform_train = get_cifar100_train_transform()

    trainset = datasets.CIFAR100(root=dataset_relative_path, train=True, download=True, transform=transform_train)

    dataloader_train = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    return dataloader_train

def get_cifar100_test_dataloader(batch_size=128, num_workers=2, dataset_relative_path=""):
    transform_test = get_cifar100_test_transform()

    testset = datasets.CIFAR100(root=dataset_relative_path, train=False, download=True, transform=transform_test)

    dataloader_test = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return dataloader_test

