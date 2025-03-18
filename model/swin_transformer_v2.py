import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义Patch Embedding模块
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# 定义注意力模块（Multi-Head Attention）
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# 定义Transformer Block模块
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


# 定义特征金字塔模块（Transformer-based Feature Pyramid）
class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2],
                                       embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer)
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer)
            for i in range(depths[1])])

        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer)
            for i in range(depths[2])])

        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer)
            for i in range(depths[3])])

        self.norm1 = norm_layer(embed_dims[0])
        self.norm2 = norm_layer(embed_dims[1])
        self.norm3 = norm_layer(embed_dims[2])
        self.norm4 = norm_layer(embed_dims[3])

        self.num_classes = num_classes
        self.sr_ratios = sr_ratios

    def forward(self, x):
        B = x.shape[0]

        x1 = self.patch_embed1(x)
        for blk in self.block1:
            x1 = blk(x1)
        x1 = self.norm1(x1)

        if len(x1.shape) == 2:
            # 如果x1是二维的，可能需要调整形状
            N = x1.shape[1]
            H1 = W1 = int(N ** 0.5)
            x1 = x1.unsqueeze(1).reshape(B, -1, H1, W1)
        else:
            N = x1.shape[1]
            H1 = W1 = int(N ** 0.5)
            x1 = x1.permute(0, 2, 1).reshape(B, -1, H1, W1)

        x2 = self.patch_embed2(x1)
        for blk in self.block2:
            x2 = blk(x2)
        x2 = self.norm2(x2)
        _, H2, W2 = int(x2.shape[1] ** 0.5), int(x2.shape[1] ** 0.5)
        x2 = x2.permute(0, 2, 1).reshape(B, -1, H2, W2)

        x3 = self.patch_embed3(x2)
        for blk in self.block3:
            x3 = blk(x3)
        x3 = self.norm3(x3)
        _, H3, W3 = int(x3.shape[1] ** 0.5), int(x3.shape[1] ** 0.5)
        x3 = x3.permute(0, 2, 1).reshape(B, -1, H3, W3)

        x4 = self.patch_embed4(x3)
        for blk in self.block4:
            x4 = blk(x4)
        x4 = self.norm4(x4)
        _, H4, W4 = int(x4.shape[1] ** 0.5), int(x4.shape[1] ** 0.5)
        x4 = x4.permute(0, 2, 1).reshape(B, -1, H4, W4)

        return x1, x2, x3, x4


# 定义多层感知机解码器模块
class SegformerHead(nn.Module):
    def __init__(self, in_channels, embedding_dim, num_classes):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channel, embedding_dim, kernel_size=1) for in_channel in in_channels
        ])
        self.bn = nn.BatchNorm2d(embedding_dim * len(in_channels))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(embedding_dim * len(in_channels), num_classes, kernel_size=1)

    def forward(self, x):
        x = [self.convs[i](x[i]) for i in range(len(x))]
        x = [F.interpolate(x[i], size=x[0].shape[2:], mode='bilinear', align_corners=False) for i in range(len(x))]
        x = torch.cat(x, dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        return x


# 定义完整的SegFormer模型
class SegFormer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=10, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], embedding_dim=256):
        super().__init__()
        self.backbone = MixVisionTransformer(img_size=img_size, patch_size=16, in_chans=in_chans,
                                             embed_dims=embed_dims, num_heads=num_heads, mlp_ratios=mlp_ratios,
                                             qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                             attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                             norm_layer=norm_layer, depths=depths, sr_ratios=sr_ratios)
        self.head = SegformerHead(in_channels=embed_dims, embedding_dim=embedding_dim, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = F.interpolate(x, size=x.shape[2:], mode='bilinear', align_corners=False)
        return x
