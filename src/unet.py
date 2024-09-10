import torch
import torch.nn as nn
from torchsummary import summary

class Unet(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()
        self.num_classes = num_classes
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.MaxPool2d(2, stride=2)
        self.blockDown1 = self.blockDown(3, 64)
        self.blockDown2 = self.blockDown(64, 128)
        self.blockDown3 = self.blockDown(128, 256)
        self.blockDown4 = self.blockDown(256, 512)
        self.blockNeck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.blockUp1 = self.blockUp(1536, 512)
        self.blockUp2 = self.blockUp(768, 256)
        self.blockUp3 = self.blockUp(384, 128)
        self.blockUp4 = self.blockUp(192, 64)
        self.final_conv = nn.Conv2d(64, self.num_classes, kernel_size=1)

    def blockDown(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def blockUp(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.blockDown1(x)
        x =  self.downsample(x1)
        x2 = self.blockDown2(x)
        x = self.downsample(x2)
        x3 = self.blockDown3(x)
        x = self.downsample(x3)
        x4 = self.blockDown4(x)
        x = self.downsample(x4)
        x = self.blockNeck(x)
        x = torch.cat([x4, self.upsample(x)], dim=1)
        x = self.blockUp1(x)
        x = torch.cat([x3, self.upsample(x)], dim=1)
        x = self.blockUp2(x)
        x = torch.cat([x2, self.upsample(x)], dim=1)
        x = self.blockUp3(x)
        x = torch.cat([x1, self.upsample(x)], dim=1)
        x = self.blockUp4(x)
        x = self.final_conv(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet()
    model.to(device)
    model.train()
    summary(model, (3, 384, 384))