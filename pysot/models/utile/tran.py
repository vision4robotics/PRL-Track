import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer
from timm.models.layers import DropPath, Mlp

from pysot.models.utile.utils import to_2tuple
from pysot.models.utile.pos_utils import get_2d_sincos_pos_embed


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous()  # BCHW -> BNC
        x = self.norm(x)
        return x


def split_channels(tensor, num_heads, dim=3):
    # Assuming the channels are the last dimension (dim=3 by default)
    channel_size = tensor.shape[dim] // num_heads
    return [
        tensor[:, :, :, i * channel_size : (i + 1) * channel_size]
        for i in range(num_heads)
    ]


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_mem = None

    def forward(self, x, t_h, t_w, s_h, s_w):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        q3, q4, q5 = torch.split(
            q, [q.shape[3] // 3, q.shape[3] // 3, q.shape[3] // 3], dim=3
        )
        k3, k4, k5 = torch.split(
            k, [k.shape[3] // 3, k.shape[3] // 3, k.shape[3] // 3], dim=3
        )
        v3, v4, v5 = torch.split(
            v, [v.shape[3] // 3, v.shape[3] // 3, v.shape[3] // 3], dim=3
        )

        # 3 and 4
        attn = (q4 @ torch.cat([k3, k4], dim=2).transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x3 = (attn @ torch.cat([v3, v4], dim=2)).transpose(1, 2).reshape(B, N, C // 3)

        # 3 and 5
        attn = (q5 @ torch.cat([k3, k5], dim=2).transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x4 = (attn @ torch.cat([v3, v5], dim=2)).transpose(1, 2).reshape(B, N, C // 3)

        # 4 and 5
        attn = (q5 @ torch.cat([k4, k5], dim=2).transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x5 = (attn @ torch.cat([v4, v5], dim=2)).transpose(1, 2).reshape(B, N, C // 3)

        x = torch.cat([x3, x4, x5], dim=2)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim * 3)
        self.attn = Attention(
            dim * 3,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim * 3)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim * 3,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, t_h, t_w, s_h, s_w):
        x = x + self.drop_path1(
            self.attn(self.norm1(x), t_h, t_w, s_h, s_w)
        )  # 第一个残差：Linear Projection 输出结果 + 原始结果
        x = x + self.drop_path2(
            self.mlp(self.norm2(x))
        )  # 第二个残差：上一步结果 + 上一步结果经过 mlp 处理
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        img_size_s=256,
        img_size_t=128,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
    ):
        super(VisionTransformer, self).__init__(
            img_size=224,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        self.patch_embed = embed_layer(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.grid_size_s = img_size_s // patch_size
        self.grid_size_t = img_size_t // patch_size
        self.num_patches_s = self.grid_size_s**2
        self.num_patches_t = self.grid_size_t**2
        self.pos_embed_s = nn.Parameter(
            torch.zeros(1, self.num_patches_s, embed_dim), requires_grad=False
        )
        self.pos_embed_t = nn.Parameter(
            torch.zeros(1, self.num_patches_t, embed_dim), requires_grad=False
        )

        self.init_pos_embed()

        if weight_init != "skip":
            self.init_weights(weight_init)

    def init_pos_embed(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_t = get_2d_sincos_pos_embed(
            self.pos_embed_t.shape[-1], int(self.num_patches_t**0.5), cls_token=False
        )
        self.pos_embed_t.data.copy_(torch.from_numpy(pos_embed_t).float().unsqueeze(0))

        pos_embed_s = get_2d_sincos_pos_embed(
            self.pos_embed_s.shape[-1], int(self.num_patches_s**0.5), cls_token=False
        )
        self.pos_embed_s.data.copy_(torch.from_numpy(pos_embed_s).float().unsqueeze(0))

    def forward(self, x_t, x_ot, x_s):
        """
        :param x_t: (batch, 3, 128, 128)
        :param x_s: (batch, 3, 288, 288)
        b c h w: batch 192 11 11
        :return:
        """
        x_t = self.patch_embed(x_t)  # BCHW-->BNC
        x_ot = self.patch_embed(x_ot)
        x_s = self.patch_embed(x_s)
        B, C = x_t.size(0), x_t.size(-1)
        H_s = W_s = self.grid_size_s
        H_t = W_t = self.grid_size_t

        x_s = x_s + self.pos_embed_s
        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t
        x = torch.cat([x_t, x_ot, x_s], dim=2)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, H_t, W_t, H_s, W_s)  # block 对应文章里面的 slimming

        x_t, x_ot, x_s = torch.split(x, x.shape[2] // 3, dim=2)

        x_t_2d = x_t.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_ot_2d = x_ot.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_s_2d = x_s.transpose(1, 2).reshape(B, C, H_s, W_s)

        # return x_t_2d, x_ot_2d, x_s_2d
        return x_s_2d
