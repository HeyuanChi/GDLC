import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, t):
        device = t.device
        half_dim = self.time_emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        emb = self.mlp(emb)
        return emb


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0, time_emb_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels)
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

        # Time Embedding
        self.emb_proj1 = nn.Linear(time_emb_dim * 4, out_channels)
        self.emb_proj2 = nn.Linear(time_emb_dim * 4, out_channels)

    def forward(self, x, time_emb):
        x1 = self.conv1(x)
        emb_out1 = self.emb_proj1(time_emb).unsqueeze(-1).unsqueeze(-1)
        x1 += emb_out1
        x1 = F.relu(x1, inplace=True)

        if self.dropout is not None:
            x1 = self.dropout(x1)

        x2 = self.conv2(x1)
        emb_out2 = self.emb_proj2(time_emb).unsqueeze(-1).unsqueeze(-1)
        x2 += emb_out2
        x2 = F.relu(x2, inplace=True)

        return x2


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0, time_emb_dim=256):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, dropout, time_emb_dim)

    def forward(self, x, time_emb):
        x = self.maxpool(x)
        x = self.conv(x, time_emb)
        return x


class AttentionGate(nn.Module):
    def __init__(self, F_l, F_g, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        out = x * psi
        return out


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0, resize='padding', time_emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, dropout, time_emb_dim)
        self.resize = resize

    def forward(self, x1, x2, time_emb):
        x1 = self.up(x1)

        if self.resize == 'padding':
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            if diffY > 0 or diffX > 0:
                x1 = F.pad(x1, [(diffX + 1) // 2, diffX // 2, (diffY + 1) // 2, diffY // 2])
        elif self.resize == 'cropping':
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            if diffY > 0 or diffX > 0:
                x2 = x2[:, :, (diffY + 1) // 2 : x2.size(2) - diffY // 2,
                            (diffX + 1) // 2 : x2.size(3) - diffX // 2]

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x, time_emb)
        return x


class UpAtt(nn.Module):
    def __init__(self, skip_channels, gating_channels, out_channels, kernel_size=3, dropout=0.0, resize='padding', time_emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att_gate = AttentionGate(F_l=skip_channels, F_g=gating_channels, F_int=out_channels // 2)
        self.conv = DoubleConv(skip_channels + gating_channels, out_channels, kernel_size, dropout, time_emb_dim)
        self.resize = resize

    def forward(self, x1, x2, time_emb):
        x1 = self.up(x1)

        if self.resize == 'padding':
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            if diffY > 0 or diffX > 0:
                x1 = F.pad(x1, [(diffX + 1) // 2, diffX // 2, (diffY + 1) // 2, diffY // 2])
        
        elif self.resize == 'cropping':
            diffY = x2.size(2) - x1.size(2)
            diffX = x2.size(3) - x1.size(3)
            if diffY > 0 or diffX > 0:
                x2 = x2[:, :, (diffY + 1) // 2 : x2.size(2) - diffY // 2, (diffX + 1) // 2 : x2.size(3) - diffX // 2]

        x2 = self.att_gate(x2, x1)
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x, time_emb)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class FWIUNetAttTime(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,time_emb_dim=64):
        super().__init__()

        # 1) Time Embedding
        self.time_mlp = TimeEmbedding(time_emb_dim=time_emb_dim)

        # 2) Encoder
        self.inc   = DoubleConv(in_channels, 16, kernel_size=4, time_emb_dim=time_emb_dim)
        self.down1 = Down(16,   32, kernel_size=4, dropout=0.1, time_emb_dim=time_emb_dim)
        self.down2 = Down(32,   64, kernel_size=4, dropout=0.2, time_emb_dim=time_emb_dim)
        self.down3 = Down(64,  128, kernel_size=4, dropout=0.2, time_emb_dim=time_emb_dim)
        self.down4 = Down(128, 256, kernel_size=4, dropout=0.3, time_emb_dim=time_emb_dim)

        # 3) Decoder with Attention
        self.up1 = UpAtt(128, 256, 128, kernel_size=4, dropout=0.0, resize='cropping', time_emb_dim=time_emb_dim)
        self.up2 = UpAtt( 64, 128,  64, kernel_size=4, dropout=0.0, resize='cropping', time_emb_dim=time_emb_dim)
        self.up3 = UpAtt( 32,  64,  32, kernel_size=4, dropout=0.0, resize='cropping', time_emb_dim=time_emb_dim)
        self.up4 = UpAtt( 16,  32,  16, kernel_size=4, dropout=0.0, resize='cropping', time_emb_dim=time_emb_dim)
        self.outc = nn.Conv2d(16, out_channels, kernel_size=1)

        # 4) Init
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, t):
        # Time Embedding
        time_emb = self.time_mlp(t)

        # Encoder
        x1 = self.inc(x, time_emb)        
        x2 = self.down1(x1, time_emb)     
        x3 = self.down2(x2, time_emb)     
        x4 = self.down3(x3, time_emb)     
        x5 = self.down4(x4, time_emb)     

        # Decoder
        x = self.up1(x5, x4, time_emb)    
        x = self.up2(x, x3, time_emb)     
        x = self.up3(x, x2, time_emb)     
        x = self.up4(x, x1, time_emb)     

        logits = self.outc(x)  
        return logits
