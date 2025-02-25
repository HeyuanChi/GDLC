import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=None):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.conv1(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        return self.double_conv(x)
    

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=None, resize='padding'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, dropout)
        self.resize = resize

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        if self.resize == 'padding':
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            x1 = F.pad(x1, [(diffX + 1) // 2, diffX // 2, (diffY + 1) // 2, diffY // 2])
        
        elif self.resize == 'cropping':
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            x2 = x2[:, :, (diffY + 1) // 2 : - diffY // 2, (diffX + 1) // 2 : - diffX // 2]
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class FillUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64,   128)
        self.down2 = Down(128,  256)
        self.down3 = Down(256,  512)
        self.down4 = Down(512, 1024)

        # Decoder
        self.up1 = Up(1024 + 512, 512)
        self.up2 = Up(512  + 256, 256)
        self.up3 = Up(256  + 128, 128)
        self.up4 = Up(128  +  64,  64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)       # ~ [B,   64, 230, 230]
        x2 = self.down1(x1)    # ~ [B,  128, 115, 115]
        x3 = self.down2(x2)    # ~ [B,  256,  57,  57]
        x4 = self.down3(x3)    # ~ [B,  512,  28,  28]
        x5 = self.down4(x4)    # ~ [B, 1024,  14,  14]

        x = self.up1(x5, x4)   # ~ [B,  512,  28,  28]
        x = self.up2(x, x3)    # ~ [B,  256,  57,  57]
        x = self.up3(x, x2)    # ~ [B,  128, 115, 115]
        x = self.up4(x, x1)    # ~ [B,   64, 230, 230]

        logits = self.outc(x)  # ~ [B,    1, 230, 230]
        return logits