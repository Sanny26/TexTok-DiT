"""This file contains the model definition of TA-TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""

import torch
from einops import rearrange

from modeling.titok import TiTok
from modeling.modules.blocks import TATiTokDecoder, TexTokEncoder
from modeling.quantizer.quantizer import VectorQuantizer, DiagonalGaussianDistribution
from omegaconf import OmegaConf

from huggingface_hub import PyTorchModelHubMixin


class Textok(TiTok):
    def __init__(self, config):
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__(config)
        self.encoder = TexTokEncoder(config)
        self.decoder = TATiTokDecoder(config)

        if self.quantize_mode == "vq":
            self.quantize = VectorQuantizer(
                codebook_size=config.model.vq_model.codebook_size,
                token_size=config.model.vq_model.token_size,
                commitment_cost=config.model.vq_model.commitment_cost,
                use_l2_norm=config.model.vq_model.use_l2_norm,
                clustering_vq=config.model.vq_model.clustering_vq)
        elif self.quantize_mode == "vae":
            self.quantize = DiagonalGaussianDistribution
        else:
            raise NotImplementedError

    def encode(self, x, text_guidance):
        if self.finetune_decoder:
            with torch.no_grad():
                self.encoder.eval()
                self.quantize.eval()
                z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens, text_guidance=text_guidance)
                z_quantized, result_dict = self.quantize(z)
                result_dict["quantizer_loss"] *= 0
                result_dict["commitment_loss"] *= 0
                result_dict["codebook_loss"] *= 0
        else:
            z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens, text_guidance=text_guidance)
            if self.quantize_mode == "vq":
                z_quantized, result_dict = self.quantize(z)
            elif self.quantize_mode == "vae":
                posteriors = self.quantize(z)
                z_quantized = posteriors.sample()
                result_dict = posteriors

        return z_quantized, result_dict
    def decode(self, z_quantized, text_guidance):
        decoded = self.decoder(z_quantized, text_guidance)
        return decoded
    
    def decode_tokens(self, tokens, text_guidance):
        if self.quantize_mode == "vq":
            tokens = tokens.squeeze(1)
            batch, seq_len = tokens.shape # B x N
            z_quantized = self.quantize.get_codebook_entry(
                tokens.reshape(-1)).reshape(batch, 1, seq_len, -1)
            z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        elif self.quantize_mode == "vae":
            z_quantized = tokens
        decoded = self.decode(z_quantized, text_guidance)
        return decoded
    
    def forward(self, x, text_guidance):
        z_quantized, result_dict = self.encode(x, text_guidance)
        decoded = self.decode(z_quantized, text_guidance)
        return decoded, result_dict


def create_clip_model():
    import open_clip
    clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
    del clip.visual
    tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
    clip.transformer.batch_first = False
    clip.eval()
    clip.requires_grad_(False)
    return clip, tokenizer

if __name__ == "__main__":
    config = OmegaConf.load("configs/training/TexTok/textok_b132_vq.yaml")
    model = Textok(config)
    x = torch.randn(1, 3, 256, 256)
    text = ["a photo of a cat"]

    clip_encoder, clip_tokenizer = create_clip_model()
    with torch.no_grad():
        text_guidance = clip_tokenizer(text).to(x.device)
        cast_dtype = clip_encoder.transformer.get_cast_dtype()
        text_guidance = clip_encoder.token_embedding(text_guidance).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        text_guidance = text_guidance + clip_encoder.positional_embedding.to(cast_dtype)
        text_guidance = text_guidance.permute(1, 0, 2)  # NLD -> LND
        text_guidance = clip_encoder.transformer(text_guidance, attn_mask=clip_encoder.attn_mask)
        text_guidance = text_guidance.permute(1, 0, 2)  # LND -> NLD
        text_guidance = clip_encoder.ln_final(text_guidance)  # [batch_size, n_ctx, transformer.width]
        print(text_guidance.shape)
    decoded, result_dict = model(x, text_guidance)
    print(decoded.shape, result_dict)