"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn

from segmenter.utils import init_weights, resize_pos_embed
from segmenter.blocks import Block

# from timm.models.layers import trunc_normal_
# from timm.models.vision_transformer import _load_weights



class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1] # (512/16)^2=2^10=1024
        self.patch_size = patch_size
        
        # 这是唯一的embed 一个大核卷积，kernel_size=patch_size=16
        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im):
        B, C, H, W = im.shape
        
        # 把一个多维的张量拉平
        # t.flatten(start_dim=1) # start_dim=1表示对第2个轴开始进行压缩，第一个轴不变
        # 这里表示对
        x = self.proj(im).flatten(2).transpose(1, 2)
        # print(x.shape) # torch.Size([1, 768, 32, 32]) 为什么是32：因为卷积核16 步长也16 512中一共就32个16
        # 为什么是768 因为论文中将其reshape后，768 = P^2*C = (16)^2*3
        return x



class VisionTransformer(nn.Module):
    '''超参，挺难为我的'''
    # d_model 是 patch embedding 的 embed_dim
    # d_ff 是 decoder 的 FeedForward 中的 hidden_dim 原文中使用了mlp_expansion_ratio * d_model 比例为4
    def __init__(
        self,
        image_size=(512, 512),
        patch_size=8,
        
        n_cls=2,
        n_heads=12,
        n_layers=12,
        d_model=768,
        d_ff=768*4,
        
        dropout=0.1,
        drop_path_rate=0.0,
        distilled=False,
        channels=3,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            d_model,
            channels,
        )
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls

        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # 这里加distilled有3个，不加只有最基本的
        self.distilled = distilled
        if self.distilled:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.num_patches + 2, d_model)
            )
            self.head_dist = nn.Linear(d_model, n_cls)
        else:
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.num_patches + 1, d_model)
            )

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        # output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_cls)

        # trunc_normal_(self.pos_embed, std=0.02)
        # trunc_normal_(self.cls_token, std=0.02)
        # if self.distilled:
            # trunc_normal_(self.dist_token, std=0.02)
        self.pre_logits = nn.Identity()

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    # @torch.jit.ignore()
    # def load_pretrained(self, checkpoint_path, prefix=""):
        # _load_weights(self, checkpoint_path, prefix)

    def forward(self, im, return_features=False):
        B, _, H, W = im.shape
        PS = self.patch_size
        # print('b h w ps:', B, H, W, PS)

        x = self.patch_embed(im)
        # print('patch_embed-shape:', x.shape) # torch.Size([1, 1024, 768])
        
        # expand就是将形状扩充为B个
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # print(cls_tokens.shape) # torch.Size([1, 1, 768])
        
        # 根据是否distill，cat不同的内容
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1) # 给x加上了cls_tokens，不懂

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + self.distilled # 为False时就是0，第一次见这种表达式
        # print(pos_embed.shape, num_extra_tokens) # torch.Size([1, 1025, 768]) 1
        # print(x.shape)
        
        # 之前都加上了，所以是相等的
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // PS, W // PS),
                num_extra_tokens,
            )
        
        x = x + pos_embed
        x = self.dropout(x)
        # print(x.shape) # torch.Size([1, 1025, 768]) 为什么直接加起来
        
        '''重点，这是layer层的transformer'''
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # print(x.shape) # torch.Size([1, 1025, 768]) 加了层Normalization 形状没有改变
        
        # 这里是返回值，因为有decoder，所以不直接返回结果，返回特征
        if return_features:
            return x

        if self.distilled:
            x, x_dist = x[:, 0], x[:, 1]
            x = self.head(x)
            x_dist = self.head_dist(x_dist)
            x = (x + x_dist) / 2
        else:
            x = x[:, 0]
            x = self.head(x)
        return x

    def get_attention_map(self, im, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        B, _, H, W = im.shape
        PS = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + self.distilled
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // PS, W // PS),
                num_extra_tokens,
            )
        x = x + pos_embed

        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

