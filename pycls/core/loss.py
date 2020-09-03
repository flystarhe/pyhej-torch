import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pycls.core.config import cfg


def smooth_l1_loss(input: Tensor, target: Tensor, beta: float = 0.5) -> Tensor:
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return loss.mean()


class AbnormalL1Loss(nn.Module):

    def __init__(self) -> None:
        super(AbnormalL1Loss, self).__init__()
        self.softmax = cfg.SOFTMAX

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.to(input.dtype)
        if self.softmax:
            input = F.softmax(input, dim=1)
        else:
            input = torch.sigmoid(input)
        return F.l1_loss(input, target, reduction="mean")


class AbnormalMSELoss(nn.Module):

    def __init__(self) -> None:
        super(AbnormalMSELoss, self).__init__()
        self.softmax = cfg.SOFTMAX

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.to(input.dtype)
        if self.softmax:
            input = F.softmax(input, dim=1)
        else:
            input = torch.sigmoid(input)
        return F.mse_loss(input, target, reduction="mean")


class AbnormalSmoothL1Loss(nn.Module):

    def __init__(self) -> None:
        super(AbnormalSmoothL1Loss, self).__init__()
        self.softmax = cfg.SOFTMAX

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.to(input.dtype)
        if self.softmax:
            input = F.softmax(input, dim=1)
        else:
            input = torch.sigmoid(input)
        return smooth_l1_loss(input, target, 0.5)


class AbnormalBalancedL1Loss(nn.Module):

    def __init__(self) -> None:
        super(AbnormalBalancedL1Loss, self).__init__()
        self.softmax = cfg.SOFTMAX

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.to(input.dtype)
        if self.softmax:
            input = F.softmax(input, dim=1)
        else:
            input = torch.sigmoid(input)

        pos = torch.nonzero(target).t()
        indice = np.arange(pos.size(1))
        np.random.shuffle(indice)
        pos = pos[:, indice]

        neg = torch.nonzero(1 - target).t()
        indice = np.arange(neg.size(1))
        np.random.shuffle(indice)
        neg = neg[:, indice]

        n = min(pos.size(1), neg.size(1)) * 2 + 3

        loss_pos = 0
        if pos.size(1) > 0:
            pos = pos[:, :n]
            input_pos = input[pos[0], pos[1]]
            target_pos = target[pos[0], pos[1]]
            loss_pos = smooth_l1_loss(input_pos, target_pos, 0.5)

        loss_neg = 0
        if neg.size(1) > 0:
            neg = neg[:, :n]
            input_neg = input[neg[0], neg[1]]
            target_neg = target[neg[0], neg[1]]
            loss_neg = smooth_l1_loss(input_neg, target_neg, 0.5)

        return loss_pos + loss_neg
