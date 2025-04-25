"""transformer based denoiser"""

import torch
from einops.layers.torch import Rearrange
from torch import nn

from tld.transformer_blocks import DecoderBlock, MLPSepConv, SinusoidalEmbedding, MLP
from configs import Denoiser1DConfig
import pdb

class DenoiserTransBlock1D(nn.Module):
    def __init__(
        self,
        patch_size: int,
        seq_len: int,
        embed_dim: int,
        dropout: float,
        n_layers: int,
        mlp_multiplier: int = 4,
        n_channels: int = 4,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.mlp_multiplier = mlp_multiplier

        assert patch_size == 1, 'LDM in token space, so patch size = 1'
        patch_dim = n_channels
        # Learnable patch embedding layer
        self.patch_embedding = nn.Linear(self.n_channels, self.embed_dim)

        self.pos_embed = nn.Embedding(seq_len, self.embed_dim)
        self.register_buffer("precomputed_pos_enc", torch.arange(0, seq_len).long())

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim=self.embed_dim,
                    mlp_multiplier=self.mlp_multiplier,
                    # note that this is a non-causal block since we are
                    # denoising the entire image no need for masking
                    is_causal=False,
                    dropout_level=self.dropout,
                    # mlp_class=MLPSepConv, uses spatial on tokens, for patches makes sense
                    mlp_class=MLP,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.out_proj = nn.Linear(self.embed_dim, patch_dim)

    def forward(self, x, cond):
        # Convert input to high-dimensional embedding
        x = self.patch_embedding(x)  # B x seq_len x embed_dim

        pos_enc = self.precomputed_pos_enc[: x.size(1)].expand(x.size(0), -1)
        x = x + self.pos_embed(pos_enc)

        for block in self.decoder_blocks:
            x = block(x, cond)

        return self.out_proj(x)


class Denoiser1D(nn.Module):
    def __init__(
        self,
        seq_len: int,
        noise_embed_dims: int,
        patch_size: int,
        embed_dim: int,
        dropout: float,
        n_layers: int,
        text_emb_size: int = 768,
        mlp_multiplier: int = 4,
        n_channels: int = 4,
        image_emb_size: int = 768,
        super_res: bool = False
    ):
        super().__init__()

        self.seq_len = seq_len
        self.noise_embed_dims = noise_embed_dims
        self.embed_dim = embed_dim
        self.n_channels = n_channels
        self.super_res = super_res

        self.fourier_feats = nn.Sequential(
            SinusoidalEmbedding(embedding_dims=noise_embed_dims),
            nn.Linear(noise_embed_dims, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.denoiser_trans_block = DenoiserTransBlock1D(patch_size, seq_len, embed_dim, dropout, n_layers, mlp_multiplier, n_channels)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.label_proj = nn.Linear(text_emb_size, self.embed_dim)
        self.image_proj = nn.Linear(2*image_emb_size, self.embed_dim)

    def forward(self, x, noise_level, label, image = None):

        x = x.permute(0, 2, 1)
        noise_level = self.fourier_feats(noise_level).unsqueeze(1)

        label = self.label_proj(label).unsqueeze(1)
        if image != None and self.super_res:
            lr_img = self.image_proj(image).unsqueeze(1)
            noise_label_emb = torch.cat([noise_level, label, lr_img], dim=1)  # bs, 2, d
        else:
            noise_label_emb = torch.cat([noise_level, label], dim=1)  # bs, 2, d

        noise_label_emb = self.norm(noise_label_emb)
        x = self.denoiser_trans_block(x, noise_label_emb) #x: bs, 

        x = x.permute(0, 2, 1)
        return x


class DenoiserTransBlock(nn.Module):
    def __init__(
        self,
        patch_size: int,
        img_size: int,
        embed_dim: int,
        dropout: float,
        n_layers: int,
        mlp_multiplier: int = 4,
        n_channels: int = 4,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.mlp_multiplier = mlp_multiplier

        seq_len = int((self.img_size / self.patch_size) * (self.img_size / self.patch_size))
        patch_dim = self.n_channels * self.patch_size * self.patch_size

        self.patchify_and_embed = nn.Sequential(
            nn.Conv2d(
                self.n_channels,
                patch_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),
            Rearrange("bs d h w -> bs (h w) d"),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.rearrange2 = Rearrange(
            "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
            h=int(self.img_size / self.patch_size),
            p1=self.patch_size,
            p2=self.patch_size,
        )

        self.pos_embed = nn.Embedding(seq_len, self.embed_dim)
        self.register_buffer("precomputed_pos_enc", torch.arange(0, seq_len).long())

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim=self.embed_dim,
                    mlp_multiplier=self.mlp_multiplier,
                    # note that this is a non-causal block since we are
                    # denoising the entire image no need for masking
                    is_causal=False,
                    dropout_level=self.dropout,
                    mlp_class=MLPSepConv,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.out_proj = nn.Sequential(nn.Linear(self.embed_dim, patch_dim), self.rearrange2)

    def forward(self, x, cond):
        x = self.patchify_and_embed(x)
        pos_enc = self.precomputed_pos_enc[: x.size(1)].expand(x.size(0), -1)
        x = x + self.pos_embed(pos_enc)

        for block in self.decoder_blocks:
            x = block(x, cond)

        return self.out_proj(x)

class Denoiser(nn.Module):
    def __init__(
        self,
        image_size: int,
        noise_embed_dims: int,
        patch_size: int,
        embed_dim: int,
        dropout: float,
        n_layers: int,
        text_emb_size: int = 768,
        mlp_multiplier: int = 4,
        n_channels: int = 4,
        image_emb_size: int = 768,
        super_res: bool = False
    ):
        super().__init__()

        self.image_size = image_size
        self.noise_embed_dims = noise_embed_dims
        self.embed_dim = embed_dim
        self.n_channels = n_channels
        self.super_res = super_res
        
        self.fourier_feats = nn.Sequential(
            SinusoidalEmbedding(embedding_dims=noise_embed_dims),
            nn.Linear(noise_embed_dims, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.denoiser_trans_block = DenoiserTransBlock(patch_size, image_size, embed_dim, dropout, n_layers, mlp_multiplier, n_channels)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.label_proj = nn.Linear(text_emb_size, self.embed_dim)
        self.image_proj = nn.Linear(2*image_emb_size, self.embed_dim)

    def forward(self, x, noise_level, label, image = None):
        noise_level = self.fourier_feats(noise_level).unsqueeze(1)

        label = self.label_proj(label).unsqueeze(1)
        if image != None and self.super_res:
            lr_img = self.image_proj(image).unsqueeze(1)
            noise_label_emb = torch.cat([noise_level, label, lr_img], dim=1)  # bs, 2, d
        else:
            noise_label_emb = torch.cat([noise_level, label], dim=1)  # bs, 2, d
        noise_label_emb = self.norm(noise_label_emb)

        x = self.denoiser_trans_block(x, noise_label_emb)

        return x


if __name__ == "__main__":
    # Load configuration
    cfg = Denoiser1DConfig()

    print(cfg)
    # Define dummy inputs
    batch_size = 3
    num_tokens = cfg.seq_len
    latent_dim = cfg.n_channels
    clip_embedding_size = cfg.text_emb_size
    noise_level_dim = 1

    # Create dummy inputs
    x = torch.randn(batch_size * 2, num_tokens, latent_dim)  # 2B x N x d
    label = torch.randn(batch_size * 2, clip_embedding_size)  # 2B x Clip embedding size
    noise_level = torch.randn(batch_size * 2, noise_level_dim)  # 2B x 1
    image = torch.randn(batch_size*2, cfg.image_emb_size)

    # Initialize the Denoiser model
    model = Denoiser1D(
        seq_len=num_tokens,
        noise_embed_dims=cfg.noise_embed_dims,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        dropout=cfg.dropout,
        n_layers=cfg.n_layers,
        text_emb_size=cfg.text_emb_size,
        mlp_multiplier=cfg.mlp_multiplier,
        n_channels=cfg.n_channels,
        image_emb_size = cfg.image_emb_size,
        super_res = cfg.super_res
    )

    # Forward pass
    output = model(x, noise_level, label, image)

    # Print output shape
    print(f"Output shape: {output.shape}")  # Expected: 2B x N x d

