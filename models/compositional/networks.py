import math

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.GELU(),
    )


class FieldEncoder(nn.Module):
    """
    Convolutional encoder E: flow field (u, v, p) -> block-structured latent
    z = [z_mu || z_g || z_xi].
    """

    def __init__(self, in_channels=3, resolution=256, base_channels=32,
                 latent_mu=4, latent_g=32, latent_xi=16, max_channels=256):
        super().__init__()
        n_down = int(math.log2(resolution // 8))  # downsample to 8x8
        channels = [min(base_channels * 2 ** i, max_channels) for i in range(n_down)]

        layers = [conv_block(in_channels, channels[0])]
        for i in range(n_down):
            out_ch = channels[min(i + 1, n_down - 1)]
            layers.append(conv_block(channels[i], out_ch, stride=2))
            channels[min(i + 1, n_down - 1)] = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(4)

        feat_dim = channels[-1] * 4 * 4
        self.head_mu = nn.Linear(feat_dim, latent_mu)
        # latent_g = 0 means the geometry block comes from a separate static
        # encoder (SDFEncoder) instead of the flow field
        self.head_g = nn.Linear(feat_dim, latent_g) if latent_g > 0 else None
        self.head_xi = nn.Linear(feat_dim, latent_xi)

    def forward(self, fields):
        h = self.pool(self.conv(fields)).flatten(1)
        z_g = self.head_g(h) if self.head_g is not None else None
        return self.head_mu(h), z_g, self.head_xi(h)


class SDFEncoder(nn.Module):
    """
    Static geometry encoder E_g: SDF -> z_g. Because the SDF does not depend
    on the operating condition, z_g is Reynolds-invariant by construction.
    """

    def __init__(self, resolution=256, base_channels=32, latent_g=32,
                 max_channels=256):
        super().__init__()
        n_down = int(math.log2(resolution // 8))
        channels = [min(base_channels * 2 ** i, max_channels) for i in range(n_down)]

        layers = [conv_block(1, channels[0])]
        for i in range(n_down):
            out_ch = channels[min(i + 1, n_down - 1)]
            layers.append(conv_block(channels[i], out_ch, stride=2))
            channels[min(i + 1, n_down - 1)] = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(channels[-1] * 4 * 4, latent_g)

    def forward(self, sdf):
        return self.fc(self.pool(self.conv(sdf)).flatten(1))


class FieldDecoder(nn.Module):
    """
    Convolutional decoder D: z = [z_mu || z_g || z_xi] -> flow field (u, v, p).
    """

    def __init__(self, latent_dim, out_channels=3, resolution=256,
                 base_channels=32, max_channels=256):
        super().__init__()
        n_up = int(math.log2(resolution // 8))
        channels = [min(base_channels * 2 ** i, max_channels) for i in range(n_up)][::-1]

        self.start_ch = channels[0]
        self.fc = nn.Linear(latent_dim, self.start_ch * 8 * 8)

        blocks = []
        for i in range(n_up):
            out_ch = channels[min(i + 1, n_up - 1)]
            blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv_block(channels[i], out_ch),
            ))
            channels[min(i + 1, n_up - 1)] = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        h = self.fc(z).view(-1, self.start_ch, 8, 8)
        return self.out(self.blocks(h))


class RegimeHead(nn.Module):
    """Small MLP that reads the regime label (standardized log Re) from z_mu."""

    def __init__(self, latent_mu, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_mu, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )

    def forward(self, z_mu):
        return self.net(z_mu).squeeze(-1)


class SDFHead(nn.Module):
    """Small decoder that reconstructs a low-resolution SDF from z_g alone."""

    def __init__(self, latent_g, resolution=64, base_channels=32):
        super().__init__()
        n_up = int(math.log2(resolution // 8))
        self.fc = nn.Linear(latent_g, base_channels * 4 * 8 * 8)
        self.start_ch = base_channels * 4

        blocks, ch = [], self.start_ch
        for _ in range(n_up):
            blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv_block(ch, max(ch // 2, base_channels)),
            ))
            ch = max(ch // 2, base_channels)
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Conv2d(ch, 1, kernel_size=3, padding=1)
        self.resolution = resolution

    def forward(self, z_g):
        h = self.fc(z_g).view(-1, self.start_ch, 8, 8)
        return self.out(self.blocks(h))
