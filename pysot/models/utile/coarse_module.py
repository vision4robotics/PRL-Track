import torch
import torch.nn as nn
import torch.nn.functional as F


def lip2d(x, logit, kernel=3, stride=2, padding=1):
    """LIP: Local Importance-Based Pooling"""
    weight = logit.exp()
    return F.avg_pool2d(x * weight, kernel, stride, padding) / (
        F.avg_pool2d(weight, kernel, stride, padding) + 1e-8
    )


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AR(nn.Module):
    def __init__(self, in_channels1, in_channels2):
        super(AR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels1, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels2, 1, kernel_size=1)
        self.conv_concat = nn.Conv2d(2, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)

        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(channel=192)
        self.conv = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, bias=False, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, g1, g2):
        g1_conv = self.conv1(g1)  # torch.Size([128, 1, 41, 41])
        g2_conv = self.conv2(g2)  # torch.Size([128, 1, 21, 21])
        g1_conv_norm = self.bn(g1_conv)
        g1_conv_downsampled = lip2d(g1_conv_norm, logit=g1_conv_norm)
        concatenated_features = torch.cat([g1_conv_downsampled, g2_conv], dim=1)

        g12 = self.conv_concat(concatenated_features)
        g12 = self.relu(g12)

        gating = torch.mul(x, g12)

        w = x + gating
        result = self.conv(w)
        result = self.se(result)

        return result


class SR(nn.Module):
    def __init__(self, x_channels1, t_channels):
        super(SR, self).__init__()
        self.conv1 = nn.Conv2d(t_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.se = SELayer(channel=192)

        self.conv2 = nn.Sequential(
            nn.Conv2d(x_channels1, 192, kernel_size=3, bias=False, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, t):
        t_upsampled = F.interpolate(t, size=x.size()[2:], mode="nearest")
        t = self.conv1(t_upsampled)
        special = torch.mul(x, t)

        w = x + special
        result = self.conv2(w)
        result = self.se(result)

        return result
