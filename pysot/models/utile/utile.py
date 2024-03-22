import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from pysot.models.utile.coarse_module import AAGU, AGU
from pysot.models.utile.tran import VisionTransformer


class PRL(nn.Module):

    def __init__(self, cfg):
        super(PRL, self).__init__()

        self.aagu = AAGU(96, 256)
        self.agu1 = AGU(384, 192)
        self.agu2 = AGU(256, 192)

        channel = 192

        self.convloc = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(cfg.TRAIN.groupchannel, channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(cfg.TRAIN.groupchannel, channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(cfg.TRAIN.groupchannel, channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 4, kernel_size=3, stride=1, padding=1),
        )

        self.convcls = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(cfg.TRAIN.groupchannel, channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(cfg.TRAIN.groupchannel, channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(cfg.TRAIN.groupchannel, channel),
            nn.ReLU(inplace=True),
        )

        self.row_embed = nn.Embedding(50, channel // 2)
        self.col_embed = nn.Embedding(50, channel // 2)
        self.reset_parameters()

        self.vit = VisionTransformer(
            img_size_s=11,
            img_size_t=11,
            patch_size=1,
            in_chans=192,
            embed_dim=192,
            depth=2,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_path_rate=0.1,
        )

        self.cls1 = nn.Conv2d(channel, 2, kernel_size=3, stride=1, padding=1)
        self.cls2 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1)

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def xcorr_depthwise(self, x, kernel):
        """depthwise cross correlation"""
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out

    def forward(self, x, z):
        # x 裁切
        # z 模板
        s1 = self.xcorr_depthwise(x[0], z[0])
        s2 = self.xcorr_depthwise(x[1], z[1])
        s3 = self.xcorr_depthwise(x[2], z[2])
        s4 = self.xcorr_depthwise(x[3], z[3])
        s5 = self.xcorr_depthwise(x[4], z[4])

        res1 = self.aagu(s3, s1, s2)
        res2 = self.agu1(s4, res1)
        res3 = self.agu2(s5, res2)

        res = self.vit(res1, res2, res3)

        loc = self.convloc(res)
        acls = self.convcls(res)

        cls1 = self.cls1(acls)
        cls2 = self.cls2(acls)

        return loc, cls1, cls2
