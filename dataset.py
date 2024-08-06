import torch
from data.cifar100 import load_cifar100
from data.cifar100H import load_cifar100H
from data.cub200 import load_cub200
from data.fgvc100 import load_fgvc100
from data.cars196 import load_cars196
from data.dogs120 import load_dogs120

def get_loader(args, k=None):

    if args.ds == 'cifar100':
        num_classes = 100
        train_loader, train_partialY_matrix, test_loader = load_cifar100(args)

    elif args.ds == 'cifar100H':
        num_classes = 100
        train_loader, train_partialY_matrix, test_loader = load_cifar100H(args)

    elif args.ds == 'cub200':
        num_classes = 200
        train_loader, train_partialY_matrix, test_loader = load_cub200(args)

    elif args.ds == 'fgvc100':
        num_classes = 100
        train_loader, train_partialY_matrix, test_loader = load_fgvc100(args)

    elif args.ds == 'cars196':
        num_classes = 196
        train_loader, train_partialY_matrix, test_loader = load_cars196(args)

    elif args.ds == 'dogs120':
        num_classes = 120
        train_loader, train_partialY_matrix, test_loader = load_dogs120(args)

    print(torch.sum(train_partialY_matrix)/train_partialY_matrix.shape[0])
    return train_loader, train_partialY_matrix, test_loader, num_classes
