from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


def drop_path_f(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    '''
    [B,H,W,C] -> [B,H//ws,ws,W//ws,ws,C] -> [BHW//ws**2,ws,ws,C]
    '''
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, -1, window_size, window_size, C)
    return x


def window_reverse(windows, window_size: int, H: int, W: int):
    '''
    [B,H,W,C] <- [B,H//ws,ws,W//ws,ws,C] <- [BHW//ws**2,ws,ws,C]
    '''
    B = windows.shape[0] // (H * W // window_size**2)
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    """image to patch embedding
    CNN_proj + Reshape  [B,C,H,W] -> [B,EmbDim,Patch_num_col,Patch_num_row] -> [B,Patch_num,EmbDim] -> [B,Patch_num+1,EmbDim]
    """

    def __init__(self, img_size=224, channel=3, emb_dim=96, patch_size=4):
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
        _, _, H, W = x.shape
        # [B,EmbDim,Patch_num_col,Patch_num_row] -> [B,Patch_num,EmbDim]
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # sample each 2x2 patch and flatten
        x0 = x[:, 0::2, 0::2, :].contiguous().view(B, -1, C)
        x1 = x[:, 1::2, 0::2, :].contiguous().view(B, -1, C)
        x2 = x[:, 0::2, 1::2, :].contiguous().view(B, -1, C)
        x3 = x[:, 1::2, 1::2, :].contiguous().view(B, -1, C)

        x = torch.cat([x0, x1, x2, x3], dim=2)  # [B, H/2*W/2, 4*C]
        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]
        return x


class MLP(nn.Module):
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


class WindowAttention(nn.Module):
    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # define a parameter table of relative position bias
        # 创建偏置bias项矩阵
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # [2*Mh-1 * 2*Mw-1, nH]    其元素的个数===>>[(2*Mh-1) * (2*Mw-1)]
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(
            self.window_size[0]
        )  # 如果此处的self.window_size[0]为2的话，则生成的coords_h为[0,1]
        coords_w = torch.arange(self.window_size[1])  # 同理得
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += (
            self.window_size[0] - 1
        )  # shift to start from 0  行标+（M-1）
        relative_coords[:, :, 1] += self.window_size[1] - 1  # 列表标+（M-1）
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer(
            "relative_position_index", relative_position_index
        )  # 将relative_position_index放入到模型的缓存当中

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        # 进行mask，相同区域使用0表示；不同区域使用-100表示
        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)  # 先经过层归一化处理

        # WindowAttention即为：SW-MSA或者W-MSA模块
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        # 判断是进行SW-MSA或者是W-MSA模块
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )  # 进行数据移动操作
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        # 将窗口按照window_size的大小进行划分，得到一个个窗口
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        # 将数据进行展平操作
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C
        )  # [nW*B, Mh, Mw, C]
        # 将多窗口拼接回大的featureMap
        shifted_x = window_reverse(
            attn_windows, self.window_size, Hp, Wp
        )  # [B, H', W', C]

        # reverse cyclic shift
        # 将移位的数据进行还原
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = (
            window_size // 2
        )  # 表示向右和向下偏移的窗口大小   即窗口大小除以2，然后向下取整

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(
                        0 if (i % 2 == 0) else self.shift_size
                    ),  # 通过判断shift_size是否等于0，来决定是使用W-MSA与SW-MSA
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer    即：PatchMerging类
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # 将img_mask划分成一个一个窗口
        mask_windows = window_partition(
            img_mask, self.window_size
        )  # [nW, Mh, Mw, 1]           # 输出的是按照指定的window_size划分成一个一个窗口的数据
        mask_windows = mask_windows.view(
            -1, self.window_size * self.window_size
        )  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(
            2
        )  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]  使用了广播机制
        # [nW, Mh*Mw, Mh*Mw]
        # 因为需要求得的是自身注意力机制，所以，所以相同的区域使用0表示，；不同的区域不等于0，填入-100，这样，在求得
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )  # 即对于不等于0的位置，赋值为-100；否则为0
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]   # 制作mask蒙版
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W


class SwinTransformer(nn.Module):

    def __init__(
        self,
        patch_size=4,  # 表示通过Patch Partition层后，下采样几倍
        in_chans=3,  # 输入图像通道
        num_classes=1000,  # 类别数
        embed_dim=96,  # Patch partition层后的LinearEmbedding层映射后的维度，之后的几层都是该数的整数倍  分别是 C、2C、4C、8C
        depths=(2, 2, 6, 2),  # 表示每一个Stage模块内，Swin Transformer Block重复的次数
        num_heads=(
            3,
            6,
            12,
            24,
        ),  # 表示每一个Stage模块内，Swin Transformer Block中采用的Multi-Head self-Attention的head的个数
        window_size=7,  # 表示W-MSA与SW-MSA所采用的window的大小
        mlp_ratio=4.0,  # 表示MLP模块中，第一个全连接层增大的倍数
        qkv_bias=True,
        drop_rate=0.0,  # 对应的PatchEmbed层后面的
        attn_drop_rate=0.0,  # 对应于Multi-Head self-Attention模块中对应的dropRate
        drop_path_rate=0.1,  # 对应于每一个Swin-Transformer模块中采用的DropRate   其是慢慢的递增的，从0增长到drop_path_rate
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(
            depths
        )  # depths:表示重复的Swin Transoformer Block模块的次数  表示每一个Stage模块内，Swin Transformer Block重复的次数
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches   即将图片划分成一个个没有重叠的patch
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            channel=in_chans,
            emb_dim=embed_dim,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)  # PatchEmbed层后面的Dropout层

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layers = BasicLayer(
                dim=int(
                    embed_dim * 2**i_layer
                ),  # 传入特征矩阵的维度，即channel方向的深度
                depth=depths[
                    i_layer
                ],  # 表示当前stage中需要堆叠的多少Swin Transformer Block
                num_heads=num_heads[
                    i_layer
                ],  # 表示每一个Stage模块内，Swin Transformer Block中采用的Multi-Head self-Attention的head的个数
                window_size=window_size,  # 表示W-MSA与SW-MSA所采用的window的大小
                mlp_ratio=self.mlp_ratio,  # 表示MLP模块中，第一个全连接层增大的倍数
                qkv_bias=qkv_bias,
                drop=drop_rate,  # 对应的PatchEmbed层后面的
                attn_drop=attn_drop_rate,  # 对应于Multi-Head self-Attention模块中对应的dropRate
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],  # 对应于每一个Swin-Transformer模块中采用的DropRate   其是慢慢的递增的，从0增长到drop_path_rate
                norm_layer=norm_layer,
                downsample=(
                    PatchMerging if (i_layer < self.num_layers - 1) else None
                ),  # 判断是否是第四个，因为第四个Stage是没有PatchMerging层的
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # 自适应的全局平均池化
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, L, C]
        x, H, W = self.patch_embed(x)  # 对图像下采样4倍
        x = self.pos_drop(x)

        # 依次传入各个stage中
        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)  # [B, L, C]
        x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        x = torch.flatten(x, 1)
        x = self.head(x)  # 经过全连接层，得到输出
        return x


if __name__ == '__main__':
    X = torch.randn(2, 3, 224, 224)
    model = SwinTransformer()
    Y = model(X)
    print(Y.shape)
