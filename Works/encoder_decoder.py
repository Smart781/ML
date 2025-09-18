import torch
from torch import nn
from torch.nn import functional as F


# Task 1

class Encoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, start_channels=16, downsamplings=5):
        super(Encoder, self).__init__()

        channels = start_channels
        layers = []

        layers.append(nn.Conv2d(3, channels, kernel_size=1, stride=1, padding=0))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(downsamplings):
            layers.append(nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            channels *= 2
        
        layers.append(nn.Flatten())
        
        layers.append(nn.Linear(channels * (img_size // (2 ** downsamplings)) ** 2, 256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(256, 2 * latent_size))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        mu, log_sigma_sq = torch.chunk(x, 2, dim=-1)
        return (mu + torch.exp(log_sigma_sq) * torch.randn_like(torch.exp(log_sigma_sq))), (mu, torch.exp(log_sigma_sq))
    
    
# Task 2

class Decoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, end_channels=16, upsamplings=5):
        super(Decoder, self).__init__()

        upsampling_blocks = []

        channels = end_channels * 2 ** upsamplings

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, channels * (img_size // (2 ** upsamplings)) ** 2)
        )

        upsampling_blocks.append(self.decoder)
        upsampling_blocks.append(nn.Unflatten(1, (channels, img_size // (2 ** upsamplings), img_size // (2 ** upsamplings))))
        
        for _ in range(upsamplings):
            upsampling_blocks.append(nn.ConvTranspose2d(channels, channels // 2, kernel_size=4, stride=2, padding=1))
            upsampling_blocks.append(nn.BatchNorm2d(channels // 2))
            upsampling_blocks.append(nn.ReLU(inplace=True))
            channels //= 2

        upsampling_blocks.append(nn.Conv2d(channels, 3, kernel_size=1, stride=1, padding=0))
        upsampling_blocks.append(nn.Tanh())

        self.upsampling = nn.Sequential(*upsampling_blocks)

    def forward(self, z):
        return self.upsampling(z)