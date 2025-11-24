from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    
    def __init__(self, latent_dims=128, h=80, w=172):
        super().__init__();

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.nograd():
            dmmy = torch.zeros(1, 1, h, w)
            h_feat = self.conv(dmmy)
        
        self.flat_dim = h_feat.numel()

        self.fc_mu = nn.Linear(self.flat_dim, latent_dims)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dims)


    def forward(self, x):
        h = self.conv(x)
        h = h.view(x.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar




class Decoder(nn.Module):
    ...

    def forward(self, x):
        ...




class VariationalAutoEncoder(nn.Module):
    ...

    def forward(self, x):
        ...
