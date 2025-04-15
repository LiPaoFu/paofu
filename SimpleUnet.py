import torch.nn as nn
import torch
from config import NUM_CLASSES


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 下采样部分添加池化层
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        # 上采样部分
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv1 = DoubleConv(128, 64)  # 输入通道数为64(up1输出)+64(down1的池化后通道)=128

        # 输出层
        self.out = nn.Conv2d(64, NUM_CLASSES, 1)  # 修改为输出类别数

    def forward(self, x):
        # 编码器
        x1 = self.down1(x)  # [B, 64, H, W]
        x1_pooled = self.pool1(x1)  # [B, 64, H/2, W/2]

        x2 = self.down2(x1_pooled)  # [B, 128, H/2, W/2]
        x2_pooled = self.pool2(x2)  # [B, 128, H/4, W/4]

        # 解码器
        x = self.up1(x2_pooled)  # [B, 64, H/2, W/2]
        x = torch.cat([x, x1_pooled], dim=1)  # 跳跃连接，拼接后[B, 128, H/2, W/2]
        x = self.up_conv1(x)  # [B, 64, H/2, W/2]

        # 上采样回原始尺寸（可选，根据标签尺寸调整）
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.out(x)  # 输出[B, 2, H, W]


model = SimpleUNet()
