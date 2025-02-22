import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class FillUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_c=64,
        depth=5,
    ):
        super().__init__()
        self.depth = depth

        # ---- Encoder ----
        self.inc = DoubleConv(in_channels, base_c)

        # Down blocks
        self.downs = nn.ModuleList()
        in_ch = base_c
        for _ in range(depth - 1):
            out_ch = in_ch * 2
            self.downs.append(Down(in_ch, out_ch))
            in_ch = out_ch

        # ---- Decoder ----
        # Up blocks
        self.ups = nn.ModuleList()
        for _ in range(depth - 1):
            out_ch = in_ch // 2
            self.ups.append(Up(in_ch + out_ch, out_ch))
            in_ch = out_ch

        self.outc = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        xs = [x1]
        for down in self.downs:
            xs.append(down(xs[-1]))

        # Decoder
        x_up = xs[-1]
        for i in range(self.depth - 1):
            x_up = self.ups[i](x_up, xs[-(i + 2)])

        # Output
        logits = self.outc(x_up)
        return logits