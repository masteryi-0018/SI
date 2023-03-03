from segmenter.vit import VisionTransformer
from segmenter.decoder import DecoderLinear, MaskTransformer
from segmenter.segmenter import Segmenter



def create_segmenter(patch_size):
    # 这里裁剪尺寸越小效果应当越好，本地batch_size=1时也只能到16
    encoder = VisionTransformer(patch_size=patch_size)
    # decoder = DecoderLinear(patch_size=patch_size)
    decoder = MaskTransformer(patch_size=patch_size)
    model = Segmenter(encoder, decoder, n_cls=2)

    return model