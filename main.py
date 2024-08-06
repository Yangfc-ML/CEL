import os.path
import random
import torch.nn.functional as F
import argparse
import numpy as np
from loss import Label_Disambiguation_Loss, Class_Associative_Loss, Prototype_Discriminative_Loss
from resnet import CWENet
from dataset import get_loader
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-2, help='optimizer\'s learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='learning rate\'s decay rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('--bs', type=int, default=128, help='batch size')
parser.add_argument('--ep', type=int, default=500, help='number of total epochs')
parser.add_argument('--wp_epoch', type=int, default=250, help='number of first stage epochs')
parser.add_argument('--embed_dim', type=int, default=512, help='class-wise embedding dim')
parser.add_argument('--ds', type=str, default='cub200', help='specify a dataset')
parser.add_argument('--model', type=str, default='resnet34', help='model name')
parser.add_argument('--gamma1', type=float, default=1, help='parameters gamma1')
parser.add_argument('--alpha', type=float, default=0.5, help='parameters alpha')
parser.add_argument('--gamma2', type=float, default=1, help='parameters gamma2')
parser.add_argument('--beta', type=float, default=0.5, help='parameters beta')
parser.add_argument('--nw', type=int, default=8, help='multi-process data loading')
parser.add_argument('--dir', type=str, default='results/', help='result save path')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--gpu', type=str, default='1', help='gpu')
parser.add_argument('--rate', default=0.1, type=float, help='partial rate')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


def adjust_learning_rate(args, optimizer, epoch):
    import math
    lr = args.lr
    eta_min = lr * (args.lr_decay_rate ** 3)
    lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / args.ep)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluate(loader, model):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        feature, output1 = model(images, eval_only=True)
        output = F.softmax(output1, dim=1)
        _, pred = torch.max(output.data, 1)
        total += images.size(0)
        correct += (pred == labels).sum().item()
    acc = 100 * float(correct) / float(total)
    return acc


def training_one_epoch(net, optimizer, loss_cls, loss_cal, loss_pdl, train_loader, epoch):
    net.train()
    adjust_learning_rate(args, optimizer, epoch)
    for i, (image_w, image_s, labels, true_labels, index) in enumerate(train_loader):
        image_w = image_w.to(device)
        image_s = image_s.to(device)
        labels = labels.to(device)
        trues = true_labels.to(device)

        output_w, output_s, feature_w, feature_s = net(image_w, image_s, labels)
        cls_loss = loss_cls(output_w, output_s, index)
        ole_loss_sample = loss_cal(feature_w, feature_s, labels)
        if epoch < args.wp_epoch:
            loss = cls_loss + args.lambda1_ * ole_loss_sample
        else:
            ole_loss_prototype = loss_pdl(feature_w, feature_s, net.class_wise_prototypes.clone().detach(), output_w,
                                            labels)
            loss = cls_loss + args.lambda1_ * ole_loss_sample + args.lambda2_ * ole_loss_prototype
        cls_loss.update_predicted_score(output_w, index, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    train_loader, train_partialY_matrix, test_loader, num_class = get_loader(args)
    args.num_classes = num_class

    tempY = train_partialY_matrix.sum(dim=1).unsqueeze(1).repeat(1, train_partialY_matrix.shape[1])
    init_confidence = train_partialY_matrix.float() / tempY
    init_confidence = init_confidence.cuda()

    loss_cls = Label_Disambiguation_Loss(predicted_score=init_confidence, predicted_score_weight=0)
    loss_cal = Class_Associative_Loss(gamma=args.gamma1)
    loss_pdl = Prototype_Discriminative_Loss(gamma=args.gamma2)

    print('building model...')
    net = CWENet(args)
    net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

    for epoch in range(0, args.ep):
        training_one_epoch(net, optimizer, loss_cls, loss_cal, loss_pdl, train_loader, epoch)
        test_acc = evaluate(test_loader, net)
        print("epoch: " + str(epoch + 1) + " test acc: " + str(test_acc))


if __name__ == '__main__':
    main()
