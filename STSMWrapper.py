from typing import List
import torch
import torch.nn as nn


class STSMWrapper(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 H: int = 64,
                 Q: float = 0.01,
                 weight: float = 1,
                 eps: float = 1e-8):
        '''
        H: number of halton points
        Q: Q-union size
        '''
        super().__init__()
        self.model = model
        self.H = H
        self.Q = Q
        self.weight = weight
        self.eps = eps
        self.STSMLoss = None

    def forward(self, input: torch.Tensor, *ext_data: torch.Tensor):
        y = self.model(input, *ext_data)
        delta_x = torch.rand(self.H, *input.shape, device=input.device) * 2 * self.Q - self.Q
        xs = input.unsqueeze(dim=0) + delta_x
        self.delta_ys = [(self.model(x, *ext_data) - y) for x in xs]
        delta_y_sqr = [torch.square(self.model(x, *ext_data) - y + self.eps) for x in xs]
        stsm = torch.mean(torch.stack(delta_y_sqr)) # H x B x D* -> B x D*
        stsm = torch.sqrt(torch.mean(stsm.flatten())) # flatten (B x D* -> B*prod(D*)), mean, then sqrt
        self.STSMLoss = self.weight * stsm
        return y
