import os
import copy

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

class GramMatrix(object):
    def __call__(self, x: torch.Tensor):
        batch_size , h, w, map_num = x.size()
        f = x.view(batch_size * h, w * map_num)

        G = torch.mm(f, f.t()) # calculate the Gram matrix

        return G.div(x.numel()) # normalize


class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x

class StyleLoss(nn.Module):
    def __init__(self, target_feature, use_mask=False, invert_mask=None):
        super(StyleLoss, self).__init__()
        self.gram = GramMatrix()
        
        if use_mask:
            self.mask = torch.zeros_like(target_feature, dtype=torch.bool)
            self.mask[:, :, :, :target_feature.shape[-1] // 2] = True

            if invert_mask:
                self.mask = ~self.mask

            self.mask = self.mask.detach()
            self.target = self.gram(target_feature * self.mask).detach()
        else:
            self.mask = None
            self.target = self.gram(target_feature).detach()
 
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        if self.mask is not None:
            G = self.gram(input * self.mask)
        else:
            G = self.gram(input)
        
        self.loss = F.mse_loss(G, self.target)
        return input

