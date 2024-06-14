import logging
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, StdConv2dSame, to_2tuple, trunc_normal_


class PatchEmbed(nn.Module):
    """image to patch embedding
    CNN_proj + Reshape  [B,C,H,W] -> [B,EmbDim,Patch_num_col,Patch_num_row] -> [B,Patch_num,EmbDim] -> [B,Patch_num+1,EmbDim]
    """

    def __init__(self, img_size=224, channel=3, emb_dim=768, patch_size=16):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_num_col = img_size // patch_size
        self.patch_num_row = img_size // patch_size
        self.patch_num = self.patch_num_col * self.patch_num_row
        self.proj = nn.Conv2d(
            channel, emb_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # [B,C,H,W] -> [B,EmbDim,Patch_num_col,Patch_num_row]
        x = self.proj(x)
        # [B,EmbDim,Patch_num_col,Patch_num_row] -> [B,Patch_num,EmbDim]
        x = x.flatten(2).transpose(1, 2)
        return x


class MHSA(nn.Module):
    """
    Multi-Head Self Attention
    """

    def __init__(
        self,
        emb_dim=768,
        num_heads=12,
        qkv_bias=False,
        qkv_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = emb_dim // num_heads
        self.scale = qkv_scale or head_dim**-0.5
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # X: [B,N,D]
        B, N, D = x.shape
        # Proj + head split [B,N,D] -> [B,N,3D] -> [B,N,3,H,D/H] -> [3,B,H,N,D/H]
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, D // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # q,k,v: [B,H,N,D/H]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # head fusion [B,H,N,D/H] @ [B,H,N,D/H] -> [B,H,N,D/H] -> [B,N,H,D/H] -> [B,N,D]
        attn = (attn @ v).transpose(1, 2).reshape(B, N, D)

        # proj
        x = self.proj(attn)
        x = self.proj_drop(x)

        return x


class FFN(nn.Module):
    """Feed Forward Network"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    Encoder Block
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qkv_scale=None,
        attn_drop=0.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        # prenorm
        self.norm1 = norm_layer(dim)
        # attention
        self.attn = MHSA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_scale=qkv_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path)

        # add and norm
        self.norm2 = norm_layer(dim)

        # FFN
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = FFN(
            in_features=dim, hidden_features=hidden_dim, drop=drop, act_layer=act_layer
        )

    def forward(self, x):
        # MHSA + Res
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # FFN + Res
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        channel=3,
        emb_dim=768,
        num_classes=1000,
        patch_size=16,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        attn_drop=0.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        embed_layer=PatchEmbed,
    ):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = emb_dim
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, channel=channel, emb_dim=emb_dim, patch_size=patch_size
        )
        num_patches = self.patch_embed.patch_num

        # embedding tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))
        self.pos_drop = nn.Dropout(p=drop)

        # Stochastic Depth Decay Rule
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]

        # Encoders
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=emb_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=dpr[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # last norm
        self.norm = norm_layer(emb_dim)

        self.pre_logits = nn.Identity()

        # classifier heads
        self.head = (
            nn.Linear(self.num_features, self.num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.head_dist = None

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pre_logits(x[:, 0])
        x = self.head(x)
        return x


if __name__ == '__main__':
    X = torch.randn(2, 3, 224, 224)
    model = VisionTransformer()
    Y = model(X)
    print(Y.shape)
