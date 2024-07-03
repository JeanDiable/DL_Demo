import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """initialization of FocalLoss clss

        Args:
            alpha (tensor, optional): weights of classes. Defaults to None.
            gamma (float, optional): focal config, decrese the impact of easy classes. Defaults to 2.0.
            reduction (str, optional): how to return loss, 'none','mean','sum'. Defaults to 'mean'.
        """
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.tensor([1.0])
        else:
            self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (tensor): (N*C)
            targets (tensor): (N,)
        """

        # compute Softmax
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prob of true class

        alpha_t = self.alpha.to(inputs.device)
        alpha_t = alpha_t.gather(0, targets.data.view(-1))  # get alpha for each class
        loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


if __name__ == "__main__":
    criterion = FocalLoss(alpha=[0.25, 0.5, 1], gamma=2.0, reduction='mean')
    inputs = torch.randn(10, 3, requires_grad=True)
    labels = torch.randint(0, 3, (10,))

    loss = criterion(inputs, labels)
    print(loss)
