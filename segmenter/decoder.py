import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

# from timm.models.layers import trunc_normal_

from segmenter.blocks import Block, FeedForward
from segmenter.utils import init_weights



class DecoderLinear(nn.Module):
    def __init__(self, n_cls=2, patch_size=8, d_encoder=768):
        super().__init__()

        self.d_encoder = d_encoder
        # d_encoder 为 encoder 的输出，就是下面forward的最后一维
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        # print(x.shape) # torch.Size([1, 1024, 768]) 这里已经在segmenter里面去掉了cls_token，所以变为1024
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        # print(x.shape) # ([1, 1024, 2]) 只是线性
        x = x.permute(0,2,1)
        x = x.reshape(1, self.n_cls, GS, -1)
        # x = rearrange(x, "b (h w) c -> b c h w", h=GS)

        return x



class MaskTransformer(nn.Module):
    # 这里的 n_layers 不需要很多 一般为1或者2
    # 这里的 d_model=768 也和之前的不一样
    def __init__(
        self,
        patch_size=8,
        d_encoder=768,
        
        n_cls=2,
        n_heads=12,
        n_layers=1,
        d_model=768,
        d_ff=768*4,
        
        drop_path_rate=0.0,
        dropout=0.1,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        # trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        
        masks = masks.permute(0,2,1)
        masks = masks.reshape(1, self.n_cls, GS, -1)
        # print(masks.shape) # ([1, 2, 32, 32])
        # masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))
        # einops 的 rearrange 函数是一个对张量进行操作的函数，比较优雅
        # print('rearrange-shape', masks.shape) # ([1, 2, 32, 32])

        return masks

    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)
