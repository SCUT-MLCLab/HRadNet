import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HRadBlock(nn.Module):
    def __init__(self,
                 inp: int,
                 outp: int,
                 size: int,
                 gsize: int,
                 dropout_ratio: float,
                 linear_scale: int = 1) -> None:
        super(HRadBlock, self).__init__()
        if size == 1:
            conv = nn.Conv2d(inp, outp, size)
        else:
            conv = nn.Conv2d(inp, outp, 7, 2, 3, groups=inp)
        self.downsample = nn.Sequential(
            conv,
            nn.BatchNorm2d(outp),
            # nn.ELU(),
            nn.SELU()
        )
        self.dropout = nn.Dropout(dropout_ratio)
        # self.gconv = nn.Conv2d(outp, outp, gsize, groups=outp)
        self.gconv = nn.Linear(gsize * gsize, linear_scale)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.downsample(x)
        # f = self.gconv(x)
        f = self.gconv(x.flatten(2))
        x = self.dropout(x)
        return x, f


class HRadNet(nn.Module):
    def __init__(self,
                 size: int,
                 meta_size: int,
                 num_classes: int,
                 in_channels: int = 3,
                 channels: List[int] = None,
                 layers: List[int] = None,
                 dropout_ratio: float = .0,
                 linear_scale: int = 1) -> None:
        super(HRadNet, self).__init__()

        self.size = size
        # self.depth = int(math.log2(self.size)) # depth
        # assert size > 0 and ((size & (size - 1)) == 0), "size should be power of 2"
        self.depth = int(math.log2(size & (-size))) # depth, 2^n
        self.scale = int(size / (size & (-size)))

        self.meta_size = meta_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.channels = channels or [2**(i + 2) for i in range(self.depth + 1)]

        assert len(self.channels) == self.depth + 1, f"len(channels) shoud be {self.depth + 1}"

        self.layers = layers if layers is not None else list(range(self.depth + 1))
        self.prune_layers = []

        self.dropout_ratio = dropout_ratio
        self.linear_scale = linear_scale
        self.blocks = self._build_blocks()

        out_features = sum(self.channels[i] for i in self.layers)
        self.fuse = nn.Bilinear(out_features * linear_scale + 1, meta_size + 1, self.num_classes, False)

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, math.sqrt(1 / m.weight.numel()))

    def _build_blocks(self) -> nn.ModuleList:
        blocks = [HRadBlock(self.in_channels, self.channels[0], 1, self.size, self.dropout_ratio, self.linear_scale)]
        blocks.extend(
            HRadBlock(self.channels[i],
                      self.channels[i + 1],
                      2**(self.depth - i - 1) + 1,
                      2**(self.depth - i - 1) * self.scale,
                      self.dropout_ratio,
                      self.linear_scale)
            for i in range(self.depth)
        )
        return nn.ModuleList(blocks)

    def prune(self, layers: List[int]) -> None:
        # for p in self.blocks.parameters():
        #     p.requires_grad = False
        self.prune_layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, Tuple):
            x, m = x
            meta = [m]
        else:
            meta = []
        features = [torch.ones(x.shape[0], 1, device=x.device)]

        for i in range(self.depth + 1):
            x, f = self.blocks[i](x)
            if i in self.layers:
                if i in self.prune_layers:
                    f.zero_()
                features.append(f.flatten(start_dim=1))
        
        features = torch.cat(features, dim=1)

        m = [torch.ones(x.shape[0], 1, device=x.device)]
        m.extend(v.view((x.shape[0], -1)) for v in meta)
        m = torch.cat(m, dim=1)
        x = self.fuse(features, m)

        return x
