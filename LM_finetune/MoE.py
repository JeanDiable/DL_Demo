import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)


class MoELayer(nn.Module):
    def __init__(self, num_experts, in_features, out_features):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [Linear(in_features, out_features) for _ in range(num_experts)]
        )
        self.gate = Linear(in_features, num_experts)

    def forward(self, x):
        gate_score = F.softmax(self.gate(x), dim=-1)  # shape: (batch_size, num_experts)
        expert_out = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # shape: (batch_size, num_experts, out_features)
        output = torch.bmm(gate_score.unsqueeze(1), expert_out).squeeze(1)

        return output


if __name__ == "__main__":
    input_size = 5
    out_put_size = 10
    num_experts = 3
    batch_size = 8

    model = MoELayer(num_experts, input_size, out_put_size)
    input = torch.randn(batch_size, input_size)
    output = model(input)

    print(output.shape)
