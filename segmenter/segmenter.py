import torch
import torch.nn as nn
import torch.nn.functional as F

from segmenter.utils import padding, unpadding
# from timm.models.layers import trunc_normal_


class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        # print(H_ori, W_ori)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)
        # print(H, W)
        # 都是512

        x = self.encoder(im, return_features=True)
        # print('encoder-shape:', x.shape) # ([1, 1025, 768])

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]
        # print(x.shape) # torch.Size([1, 1024, 768]) 去掉token

        masks = self.decoder(x, (H, W))
        # print('decoder-shape:', masks.shape) # ([1, 2, 32, 32])
        
        masks = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=True)
        '''本身只有32大小，直接进行线性上采样是否会丢失很多精度'''
        # 这是担心大小不同的问题，没有用到
        masks = unpadding(masks, (H_ori, W_ori))
        # print('finalout-shape:', masks.shape) # ([1, 2, 512, 512]) 上采样至原大小

        return masks

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
