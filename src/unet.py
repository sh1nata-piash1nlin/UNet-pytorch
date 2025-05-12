from torch import nn
import torch
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.sequence = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.sequence(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.sequence = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.sequence(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        # input_dim = 256
        self.encoder = nn.ModuleList([
            DownConv(in_channels, 64),  # 128
            DownConv(64, 128),  # 64
            DownConv(128, 256),  # 32
            DownConv(256, 512)  # 16
        ])

        self.bottleneck = ConvBlock(512, 1024)

        # extra channels allow for concatenation of skip connections in upsampling block
        self.decoder = nn.ModuleList([
            UpConv(512 + 1024, 512),  # 32
            UpConv(256 + 512, 256),  # 64
            UpConv(128 + 256, 128),  # 128
            UpConv(64 + 128, 64)  # 256
        ])

        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        o = x
        for layer in self.encoder:
            o = layer(o)
            skips.append(o)

        o = self.bottleneck(o)

        for i, layer in enumerate(self.decoder):
            # print(o.size())
            o = torch.cat((skips[len(skips) - i - 1], o), dim=1)
            # print(o.size())
            o = layer(o)

        return self.output_conv(o)
