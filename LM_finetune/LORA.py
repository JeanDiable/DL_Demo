import torch
import torch.nn as nn
import torch.nn.functional as F


class LORALinear(nn.Module):
    def __init__(
        self, in_features, out_features, merge, rank=16, lora_alpha=16, dropout=0.5
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.merge = merge
        self.rank = rank
        self.merge = merge
        self.lora_alpha = lora_alpha
        self.dropout_rate = dropout

        self.linear = nn.Linear(in_features, out_features)
        if rank > 0:
            self.lora_b = nn.Parameter(torch.zeros(self.out_features, self.rank))
            self.lora_a = nn.Parameter(torch.zeros(self.rank, self.in_features))
            nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
            self.scale = self.lora_alpha / self.rank
            self.linear.weight.requires_grad = False

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        if self.rank > 0 and self.merge:
            x = F.linear(
                x,
                self.linear.weight + self.lora_b @ self.lora_a * self.scale,
                self.linear.bias,
            )
            x = self.dropout(x)
            return x
        else:
            x = self.dropout(self.linear(x))
            return x


if __name__ == "__main__":
    loralinear = LORALinear(10, 5, True, 16, 16, 0.5)
    x = torch.randn(5, 10)
    print(loralinear(x).shape)
    print(loralinear.lora_a.shape)
    print(loralinear.lora_b.shape)
