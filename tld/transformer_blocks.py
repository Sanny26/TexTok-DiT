import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import math

class SinusoidalEmbedding(nn.Module):
    def __init__(self, emb_min_freq=1.0, emb_max_freq=1000.0, embedding_dims=32):
        super(SinusoidalEmbedding, self).__init__()

        frequencies = torch.exp(
            torch.linspace(np.log(emb_min_freq), np.log(emb_max_freq), embedding_dims // 2)
        )

        self.register_buffer("angular_speeds", 2.0 * torch.pi * frequencies)
        print('Angular speeds device', self.angular_speeds.device)

    def forward(self, x):
        embeddings = torch.cat(
            [torch.sin(self.angular_speeds * x), torch.cos(self.angular_speeds * x)], dim=-1
        )
        return embeddings


class MHAttention(nn.Module):
    def __init__(self, is_causal=False, dropout_level=0.0, n_heads=4):
        super().__init__()
        self.is_causal = is_causal
        self.dropout_level = dropout_level
        self.n_heads = n_heads

    def forward(self, q, k, v, attn_mask=None, vis=None, to_vis=False):
        assert q.size(-1) == k.size(-1)
        assert k.size(-2) == v.size(-2)

        q, k, v = [rearrange(x, "bs n (h d) -> bs h n d", h=self.n_heads) for x in [q, k, v]]

        dropout_p=self.dropout_level if self.training else 0
        
        out = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=self.is_causal,
            dropout_p=dropout_p,
        )

        if to_vis:
            L, S = q.size(-2), k.size(-2)
            scale_factor = 1 / math.sqrt(q.size(-1))
            attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
            if self.is_causal:
                assert attn_mask is None
                temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
                attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                attn_bias.to(q.dtype)

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                else:
                    attn_bias = attn_mask + attn_bias

            attn_weight = q @ k.transpose(-2, -1) * scale_factor
            #print('aw1', attn_weight.shape)
            attn_weight += attn_bias
            attn_weight = torch.softmax(attn_weight, dim=-1)
            #print('aw2', attn_weight.shape)
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            
            #print(vis)
            np.savez(vis + '.npz', attn_weight = attn_weight.detach().cpu())

        out = rearrange(out, "bs h n d -> bs n (h d)", h=self.n_heads)

        return out


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, layer_num, is_causal=False, dropout_level=0.0, n_heads=4):
        super().__init__()
        self.qkv_linear = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.mha = MHAttention(is_causal, dropout_level, n_heads)
        self.layer_num = layer_num

    def forward(self, x, to_vis=False):
        q, k, v = self.qkv_linear(x).chunk(3, dim=2)
        return self.mha(q, k, v, vis="selfattention_"+str(self.layer_num), to_vis=to_vis)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, layer_num, is_causal=False, dropout_level=0, n_heads=4):
        super().__init__()
        self.kv_linear = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mha = MHAttention(is_causal, dropout_level, n_heads)
        self.layer_num = layer_num

    def forward(self, x, y, to_vis=False):
        q = self.q_linear(x)
        k, v = self.kv_linear(y).chunk(2, dim=2)
        return self.mha(q, k, v, vis="crossattention_"+str(self.layer_num), to_vis=to_vis)


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_multiplier * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_multiplier * embed_dim, embed_dim),
            nn.Dropout(dropout_level),
        )

    def forward(self, x):
        return self.mlp(x)


class MLPSepConv(nn.Module):
    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        """see: https://github.com/ofsoundof/LocalViT"""
        super().__init__()
        self.mlp = nn.Sequential(
            # this Conv with kernel size 1 is equivalent to the Linear layer in a "regular" transformer MLP
            nn.Conv2d(embed_dim, mlp_multiplier * embed_dim, kernel_size=1, padding="same"),
            nn.Conv2d(
                mlp_multiplier * embed_dim,
                mlp_multiplier * embed_dim,
                kernel_size=3,
                padding="same",
                groups=mlp_multiplier * embed_dim,
            ),  # <- depthwise conv
            nn.GELU(),
            nn.Conv2d(mlp_multiplier * embed_dim, embed_dim, kernel_size=1, padding="same"),
            nn.Dropout(dropout_level),
        )

    def forward(self, x):
        #print(x.shape)
        w = h = int(np.sqrt(x.size(1)))  # only square images for now
        x = rearrange(x, "bs (h w) d -> bs d h w", h=h, w=w)
        x = self.mlp(x)
        x = rearrange(x, "bs d h w -> bs (h w) d")
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        is_causal: bool,
        mlp_multiplier: int,
        dropout_level: float,
        mlp_class: type[MLP] | type[MLPSepConv],
        layer_num: int,
    ):
        super().__init__()
        self.self_attention = SelfAttention(embed_dim, layer_num, is_causal, dropout_level, n_heads=embed_dim // 64)
        self.cross_attention = CrossAttention(
            embed_dim, layer_num, is_causal=False, dropout_level=0, n_heads=embed_dim // 64
        )
        self.mlp = mlp_class(embed_dim, mlp_multiplier, dropout_level)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor, to_vis=False) -> torch.Tensor:
        x = self.self_attention(self.norm1(x), to_vis=to_vis) + x
        x = self.cross_attention(self.norm2(x), y, to_vis=to_vis) + x
        x = self.mlp(self.norm3(x)) + x
        return x
