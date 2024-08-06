import torch
import torch.nn.functional as F
import torch.nn as nn
from copy import *


class Class_Associative_Loss(nn.Module):
    def __init__(self, gamma=2):
        super(Class_Associative_Loss, self).__init__()
        self.gamma = gamma

    def forward(self, features_w, features_s, labels):
        labels_cad_expanded = labels.unsqueeze(1)
        labels_ncad_expanded = (1 - labels).unsqueeze(1)

        mask_cad = labels_cad_expanded * labels_cad_expanded.transpose(1, 2).float()
        mask_ncad = labels_ncad_expanded * labels_ncad_expanded.transpose(1, 2).float()

        mask_pos = mask_cad
        mask_neg = 1 - mask_cad - mask_ncad

        indices = torch.arange(labels.shape[1])
        mask_pos[:, indices, indices] = 0
        mask_neg[:, indices, indices] = 0

        dot_prod_w = torch.matmul(features_w, features_w.transpose(-1, -2))
        dot_prod_s = torch.matmul(features_s, features_s.transpose(-1, -2))

        pos_pairs_mean_w = (mask_pos * dot_prod_w).sum(dim=(1, 2)) / (mask_pos.sum(dim=(1, 2)) + 1e-6)
        neg_pairs_mean_w = torch.abs(mask_neg * dot_prod_w).sum(dim=(1, 2)) / (mask_neg.sum(dim=(1, 2)) + 1e-6)

        pos_pairs_mean_s = (mask_pos * dot_prod_s).sum(dim=(1, 2)) / (mask_pos.sum(dim=(1, 2)) + 1e-6)
        neg_pairs_mean_s = torch.abs(mask_neg * dot_prod_s).sum(dim=(1, 2)) / (mask_neg.sum(dim=(1, 2)) + 1e-6)

        loss_w = ((1.0 - pos_pairs_mean_w) + (self.gamma * neg_pairs_mean_w)).mean()
        loss_s = ((1.0 - pos_pairs_mean_s) + (self.gamma * neg_pairs_mean_s)).mean()

        return loss_w + loss_s


class Prototype_Discriminative_Loss(nn.Module):
    def __init__(self, gamma):
        super(Prototype_Discriminative_Loss, self).__init__()
        self.gamma = gamma

    def forward(self, features_w, features_s, class_wise_prototypes, output, candidate_label):
        mask_pos = torch.zeros((output.shape[0], output.shape[1], output.shape[1])).cuda()
        mask_neg = torch.zeros((output.shape[0], output.shape[1], output.shape[1])).cuda()
        output = output * candidate_label
        value, index = torch.max(output, 1)

        for i in range(output.shape[0]):
            mask_pos[i, index[i], index[i]] = 1
            mask_neg[i, index[i], :] = 1
            mask_neg[i, index[i], index[i]] = 0

        dot_prod_w = torch.matmul(features_w, class_wise_prototypes.transpose(-1, -2))
        dot_prod_s = torch.matmul(features_s, class_wise_prototypes.transpose(-1, -2))

        pos_pairs_mean_w = (mask_pos * dot_prod_w).sum(dim=(1, 2)) / (mask_pos.sum(dim=(1, 2)) + 1e-6)
        neg_pairs_mean_w = torch.abs(mask_neg * dot_prod_w).sum(dim=(1, 2)) / (mask_neg.sum(dim=(1, 2)) + 1e-6)

        pos_pairs_mean_s = (mask_pos * dot_prod_s).sum(dim=(1, 2)) / (mask_pos.sum(dim=(1, 2)) + 1e-6)
        neg_pairs_mean_s = torch.abs(mask_neg * dot_prod_s).sum(dim=(1, 2)) / (mask_neg.sum(dim=(1, 2)) + 1e-6)

        loss_w = ((1.0 - pos_pairs_mean_w) + (self.gamma * neg_pairs_mean_w)).mean()
        loss_s = ((1.0 - pos_pairs_mean_s) + (self.gamma * neg_pairs_mean_s)).mean()

        return loss_w + loss_s


class Label_Disambiguation_Loss(nn.Module):
    def __init__(self, predicted_score, predicted_score_weight=0):
        super().__init__()
        self.predicted_score_w = deepcopy(predicted_score)
        self.predicted_score_weight = predicted_score_weight

    def update_predicted_score(self, output_w, index, Y):
        with torch.no_grad():
            new_w = torch.softmax(output_w, dim=1)

            revisedY = Y.clone()
            revisedY_w = revisedY * new_w
            revisedY_w = revisedY_w / revisedY_w.sum(dim=1).repeat(revisedY.size(1), 1).transpose(0, 1)

            predicted_score_w = revisedY_w.detach()

            score = self.predicted_score_weight
            self.predicted_score_w[index, :] = score * self.predicted_score_w[index, :] + (
                    1 - score) * predicted_score_w

    def forward(self, output_w, output_s, index):
        soft_positive_label_w = self.predicted_score_w[index, :].clone().detach()

        output_w = F.softmax(output_w, dim=1)
        l_w = soft_positive_label_w * torch.log(output_w)
        loss_w = (-torch.sum(l_w)) / l_w.size(0)

        output_s = F.softmax(output_s, dim=1)
        l_s = soft_positive_label_w * torch.log(output_s)
        loss_s = (-torch.sum(l_s)) / l_s.size(0)

        return (loss_w + loss_s)
