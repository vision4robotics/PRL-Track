import torch
import torch.nn as nn
import torch.nn.functional as F


def lip2d(x, logit, kernel=3, stride=2, padding=1):
    """LIP: Local Importance-Based Pooling
    """
    weight = logit.exp()
    # print(weight)
    return F.avg_pool2d(x * weight, kernel, stride, padding) / (F.avg_pool2d(weight, kernel, stride, padding) + 1e-8)


class AAGU(nn.Module):
    """Appearance-aware gating unit
    """

    def __init__(self, in_channels1, in_channels2):
        super(AAGU, self).__init__()

        self.conv_g1 = nn.Conv2d(in_channels1, 1, kernel_size=1)
        self.conv_g2 = nn.Conv2d(in_channels2, 1, kernel_size=1)
        self.conv_concat = nn.Conv2d(2, 1, kernel_size=1)
        self.bn_g1 = nn.BatchNorm2d(1)  


        self.relu1 = nn.ReLU(inplace=True) 

        self.conv = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3,
                      bias=False, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, g1, g2):
        # x: torch.Size([128, 384, 21, 21]) 
        # g1: torch.Size([128, 96, 41, 41])  门控
        # g2: torch.Size([128, 256, 21, 21]) 门控

        g1_conv = self.conv_g1(g1) # torch.Size([128, 1, 41, 41])
        g2_conv = self.conv_g2(g2) # torch.Size([128, 1, 21, 21])
        g1_conv_norm = self.bn_g1(g1_conv)
        # print(g1_conv_norm)

        g1_conv_downsampled = lip2d(g1_conv_norm, logit=g1_conv_norm)

        concatenated_features = torch.cat([g1_conv_downsampled, g2_conv], dim=1)

        g12 = self.conv_concat(concatenated_features)
        g12 = self.relu1(g12)

        gating = torch.mul(x, g12)

        w = x + gating
        result = self.conv(w)

        return result
    

class AGU(nn.Module):
    def __init__(self, x_channels1, t_channels):
        super(AGU, self).__init__()
        self.conv_a1 = nn.Conv2d(t_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Sequential(
            nn.Conv2d(x_channels1, 192, kernel_size=3,
                      bias=False, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, t):
        # x: torch.Size([128, 384, 21, 21])
        # t: torch.Size([128, 192, 11, 11])
        # t 指导 x
        t_upsampled = F.interpolate(t, size=x.size()[2:], mode='bilinear', align_corners=False)
        
        t = self.conv_a1(t_upsampled)

        special = torch.mul(x, t)

        w = x + special
        result = self.conv(w)

        return result


if __name__ == "__main__":
    # aagu = AAGU(96, 256)
    # x = torch.randn(128, 384, 21, 21)
    # g1 = torch.randn(128, 96, 41, 41)
    # g2 = torch.randn(128, 256, 21, 21)
    # output = aagu(x, g1, g2)

    agu = AGU(384, 192)
    x = torch.randn(128, 384, 21, 21)
    t = torch.randn(128, 192, 11, 11)
    output = agu(x, t)
    print("Output size:", output.size())