import torch
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import models
from augment.randaugment import RandomAugment
from utils.util import generate_instancedependent_candidate_labels
import torch.nn as nn
from augment.autoaugment_extra import ImageNetPolicy
from torchvision.datasets import FGVCAircraft
import PIL.Image
import torchvision.datasets as dsets


def load_fgvc100(args):
    test_transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    original_train = dsets.FGVCAircraft(root='../data', split='trainval', transform=test_transform, download=False)
    original_full_loader = torch.utils.data.DataLoader(dataset=original_train, batch_size=len(original_train),
                                                       shuffle=False, num_workers=20)
    ori_data, ori_labels = next(iter(original_full_loader))
    ori_labels = ori_labels.long()

    test_dataset = dsets.FGVCAircraft(root='../data', split='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=8)

    model = models.wide_resnet50_2()
    model.fc = nn.Linear(model.fc.in_features, max(ori_labels) + 1)
    model = model.cuda()
    model.load_state_dict(
        torch.load(os.path.expanduser('weight/fgvc100/fgvc100.pt'))['model_state_dict'])
    partialY_matrix = generate_instancedependent_candidate_labels(model, ori_data, ori_labels, args.rate)

    temp = torch.zeros(partialY_matrix.shape)
    temp[torch.arange(partialY_matrix.shape[0]), ori_labels] = 1

    if torch.sum(partialY_matrix * temp) == partialY_matrix.shape[0]:
        print('data loading done !')

    partial_training_dataset = FGVC100_Partialize(original_train._image_files, original_train._labels,
                                                  partialY_matrix.float(), ori_labels.float())

    partial_training_dataloader = torch.utils.data.DataLoader(
        dataset=partial_training_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )
    return partial_training_dataloader, partialY_matrix, test_loader


class FGVC100_Partialize(Dataset):
    def __init__(self, image_files, image_labels, given_partial_label_matrix, true_labels):
        self.image_files = image_files
        self.labels = image_labels
        self.given_partial_label_matrix = given_partial_label_matrix
        self.true_labels = true_labels
        self.weak_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        self.strong_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        image_file, label = self.image_files[index], self.labels[index]
        image = PIL.Image.open(image_file).convert("RGB")

        each_image_w = self.weak_transform(image)
        each_image_s = self.strong_transform(image)
        each_label = self.given_partial_label_matrix[index]
        each_true_label = self.true_labels[index]

        return each_image_w, each_image_s, each_label, each_true_label, index
