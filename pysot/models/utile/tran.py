import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer
from timm.models.layers import DropPath, Mlp

from pysot.models.utile.utils import to_2tuple
from pysot.models.utile.pos_utils import get_2d_sincos_pos_embed


class PatchEmbed(nn.Module):

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

    def forward(self, x):
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

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(
        self,
        img_size=256,
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
            img_size=256,
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

        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2
        self.pos_embed1 = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )

        self.pos_embed2 = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )

        self.pos_embed3 = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )

        self.init_pos_embed()

        if weight_init != "skip":
            self.init_weights(weight_init)

    def init_pos_embed(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed1 = get_2d_sincos_pos_embed(
            self.pos_embed1.shape[-1], int(self.num_patches**0.5), cls_token=False
        )
        self.pos_embed1.data.copy_(torch.from_numpy(pos_embed1).float().unsqueeze(0))

        pos_embed2 = get_2d_sincos_pos_embed(
            self.pos_embed2.shape[-1], int(self.num_patches**0.5), cls_token=False
        )
        self.pos_embed2.data.copy_(torch.from_numpy(pos_embed2).float().unsqueeze(0))

        pos_embed3 = get_2d_sincos_pos_embed(
            self.pos_embed3.shape[-1], int(self.num_patches**0.5), cls_token=False
        )
        self.pos_embed3.data.copy_(torch.from_numpy(pos_embed3).float().unsqueeze(0))

    def forward(self, x3, x4, x5):
        x3 = self.patch_embed(x3)  # BCHW-->BNC
        x4 = self.patch_embed(x4)
        x5 = self.patch_embed(x5)
        B, C = x3.size(0), x3.size(-1)
        H = W = self.grid_size

        x3 = x3 + self.pos_embed1
        x4 = x4 + self.pos_embed2
        x5 = x5 + self.pos_embed3
        x = torch.cat([x3, x4, x5], dim=2)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x3, x4, x5 = torch.split(x, x.shape[2] // 3, dim=2)

        x5_2d = x5.transpose(1, 2).reshape(B, C, H, W)

        return x5_2d
