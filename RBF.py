import math
import numpy as np

import torch
import torch.nn as nn

# https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer/blob/master/Torch%20RBF/torch_rbf.py


class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})

    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size

    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.

        log_sigmas: logarithm of the learnable scaling factors of shape (out_features).

        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, in_features, n_centers, out_features):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.n_centers = n_centers
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(n_centers, in_features), requires_grad=True)
        # self.log_sigmas = nn.Parameter(torch.Tensor(n_centers), requires_grad=True)
        self.log_sigmas = nn.Parameter(torch.Tensor(n_centers, in_features), requires_grad=True)
        self.linear = nn.Linear(n_centers, out_features)
        # self.reset_parameters(-30, 30, math.log(1./50))
        nn.init.normal_(self.centres)
        nn.init.constant_(self.log_sigmas, 0)

    def reset_parameters(self, q1, q9, log_sigma) -> None:
        nn.init.uniform_(self.centres, q1, q9)
        nn.init.constant_(self.log_sigmas, log_sigma)

    def reset_parameters_by(self, xs: np.ndarray) -> None:
        q1, q9 = np.quantile(xs, [.1, .9])
        nn.init.uniform_(self.centres, q1, q9)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = (x.size(0), self.n_centers, self.in_features)
        x = x.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        # d = (x - c).pow(2).sum(-1).mul(-torch.exp(self.log_sigmas).unsqueeze(0))
        # d = (x - c).pow(2).mean(-1).mul(-torch.exp(self.log_sigmas).unsqueeze(0))
        d = (x - c).pow(2).mul(-torch.exp(self.log_sigmas).unsqueeze(0)).mean(-1)
        h = torch.exp(d)
        out = self.linear(h)
        return out
