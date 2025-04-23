import torch
from torch import nn
# import timm
# from timm.models.vision_transformer import VisionTransformer
from functools import partial
import math

from torchvision.models.vision_transformer import VisionTransformer, Encoder
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, pack, unpack
from transformers import T5Tokenizer, T5EncoderModel
from tld.configs import TexTokConfig
import pdb

def divisible_by(num, den):
    return (num % den) == 0

def pack_square_height_width(t):
    assert t.ndim == 4
    return rearrange(t, 'b h w d -> b (h w) d')

def unpack_square_height_width(t):
    assert t.ndim == 3
    hw = int(math.sqrt(t.shape[1]))
    return rearrange(t, 'b (h w) d -> b h w d', h = hw, w = hw)


class TexTok(nn.Module):
    def __init__(self, config: TexTokConfig, device):
        super().__init__()
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_heads = config.ViT_number_of_heads
        self.depth = config.ViT_number_of_layers
        
        assert divisible_by(self.image_size, self.patch_size)

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_tokens = config.num_tokens
        self.num_text_tokens = 32
        self.text_token_dim = 512
        self.mlp_dim = 3072
        self.seq_length = self.num_patches + self.num_tokens + self.num_text_tokens  # 1184

        self.device = device

        self.image_to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (c p1 p2)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(3 * self.patch_size * self.patch_size, config.hidden_size)
        )
        # self.patch_embed = nn.Conv2d(in_chans, hidden_size, kernel_size=patch_size, stride=patch_size)

        # Learnable image tokens (N x D)
        self.image_tokens = nn.Parameter(torch.randn(self.num_tokens, config.hidden_size))

        # Linear projection for text tokens
        self.text_proj_enc = nn.Linear(self.text_token_dim, config.hidden_size) 

        # Positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(self.seq_length, config.hidden_size))

        # Tokenizer (Encoder) ViT
        # self.vit = VisionTransformer(img_size=image_size, patch_size=patch_size, embed_dim=hidden_size, depth=self.depth, num_heads=self.num_heads, num_classes=0)
        # vit_model_encoder = timm.create_model(model_name='vit_base_patch16_224', pretrained=False)
        self.encoder = Encoder(
            seq_length = self.seq_length,
            num_layers = self.depth,
            num_heads = self.num_heads,
            hidden_dim = config.hidden_size,
            mlp_dim = self.mlp_dim,
            dropout = 0.0,
            attention_dropout = 0.0,
            norm_layer = partial(nn.LayerNorm, eps=1e-6),
        )
        # self.encoder_norm = vit_model_encoder.norm
        # self.encoder_head = vit_model_encoder.head

        # Linear projection to output image tokens (N x d)
        self.token_out_proj = nn.Linear(config.hidden_size, config.latent_dim)

        # Detokenizer (Decoder) ViT
        # self.decoder = VisionTransformer(img_size=image_size, patch_size=patch_size, embed_dim=hidden_size, depth=self.depth, num_heads=self.num_heads, num_classes=0)
        # vit_model_decoder = timm.create_model(model_name='vit_base_patch16_224', pretrained=False)
        # self.decoder_transformer = vit_model_decoder.blocks
        # self.decoder_norm = vit_model_decoder.norm
        # self.decoder_head = vit_model_decoder.head
        self.decoder = Encoder(
            seq_length = self.seq_length,
            num_layers = self.depth,
            num_heads = self.num_heads,
            hidden_dim = config.hidden_size,
            mlp_dim = self.mlp_dim,
            dropout = 0.0,
            attention_dropout = 0.0,
            norm_layer = partial(nn.LayerNorm, eps=1e-6),
        )
        # Learnable patch tokens (hw x D)
        self.patch_tokens = nn.Parameter(torch.randn(self.num_patches, config.hidden_size))
        
        # Linear projections for detokenizer
        self.image_token_proj = nn.Linear(config.latent_dim, config.hidden_size)
        self.text_proj_dec = nn.Linear(self.text_token_dim, config.hidden_size)
        
        # Reconstruction head
        # self.reconstruction_head = nn.ConvTranspose2d(hidden_size, in_chans, kernel_size=patch_size, stride=patch_size)

        self.tokens_to_image = nn.Sequential(
            nn.Linear(config.hidden_size, 3 * self.patch_size * self.patch_size),
            Rearrange('b h w (c p1 p2) -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size)
        )

        nn.init.normal_(self.image_tokens, std = 0.02)
        nn.init.normal_(self.patch_tokens, std = 0.02)
        nn.init.normal_(self.pos_emb, std = 0.02)

        
    def text_embeder(self, text_caption, max_length=32, device = "cpu"):
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5EncoderModel.from_pretrained("t5-small").to(device)

        # enc = tokenizer(text_caption, return_tensors="pt")
        enc = tokenizer(text_caption, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)

        # forward pass through encoder only
        with torch.no_grad():
            encoded = model(**enc).last_hidden_state  # (B, Nt, D)

        return encoded


    def encode(self, image, text):
        # Convert image to patch tokens
        img_patches = self.image_to_tokens(image)  # B x h x w x D
        img_patches = pack_square_height_width(img_patches)  # B x hw x D

        # Expand learnable image tokens
        img_learnable = self.image_tokens.expand(image.size(0), -1, -1)  # B x N x D

        # Embed text and project it
        text_embd = self.text_embeder(text, max_length=self.num_text_tokens, device=self.device).to(self.device)
        text_proj_enc = self.text_proj_enc(text_embd)  # B x Nt x D

        # Concatenate image patches, learnable tokens, and text tokens
        tokenizer_input = torch.cat([img_patches, img_learnable, text_proj_enc], dim=1)  # B x (hw + N + Nt) x D

        # Add positional embeddings
        pos_emb = repeat(self.pos_emb, 'N D -> B N D', B=tokenizer_input.shape[0])
        tokenizer_input = tokenizer_input + pos_emb

        # Pass through the encoder
        tokenizer_output = self.encoder(tokenizer_input)

        # Extract and project image tokens
        image_tokens = self.token_out_proj(tokenizer_output[:, self.num_patches:self.num_patches + self.num_tokens, :])  # B x N x d

        return image_tokens

    def decode(self, image_tokens, text):
        # Expand learnable patch tokens
        patch_tokens = self.patch_tokens.expand(image_tokens.size(0), -1, -1)  # B x hw x D

        # Project image tokens and text tokens
        print(self.image_token_proj)
        print(image_tokens.shape)
        image_token_proj = self.image_token_proj(image_tokens)  # B x N x D
        text_embd = self.text_embeder(text, max_length=self.num_text_tokens, device=self.device).to(self.device)
        text_proj_dec = self.text_proj_dec(text_embd)  # B x Nt x D

        # Concatenate patch tokens, projected image tokens, and text tokens
        detokenizer_input = torch.cat([patch_tokens, image_token_proj, text_proj_dec], dim=1)  # B x (hw + N + Nt) x D

        # Pass through the decoder
        detokenizer_output = self.decoder(detokenizer_input)

        # Extract patch tokens and reconstruct the image
        reconstructed_patches = detokenizer_output[:, :self.num_patches, :]  # B x hw x D
        reconstructed_patches = unpack_square_height_width(reconstructed_patches)  # B x h x w x D
        reconstructed_img = self.tokens_to_image(reconstructed_patches)  # B x C x H x W

        return reconstructed_img

    def forward(self, input):
        # Tokenizer: Encode image and text to generate image tokens
        image_tokens = self.encode(input["image"], input["text"])

        # Detokenizer: Decode image tokens and text to reconstruct the image
        reconstructed_img = self.decode(image_tokens, input["text"])

        return image_tokens, reconstructed_img

if __name__ == "__main__":
    # Define dummy inputs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TexTokConfig()

    # Create a dummy image and text
    dummy_image = torch.randn(config.batch_size, 3, config.image_size, config.image_size).to(device)  # B x C x H x W
    dummy_text = ["This is a test caption."] * config.batch_size

    # Initialize the TexTok model
    model = TexTok(config=config, device=device).to(device)

    # Test encode function
    image_tokens = model.encode(dummy_image, dummy_text)
    print(f"Image tokens shape: {image_tokens.shape}")  # Expected: (B, num_tokens, latent_dim)

    # Test decode function
    reconstructed_image = model.decode(image_tokens, dummy_text)
    print(f"Reconstructed image shape: {reconstructed_image.shape}")  # Expected: (B, 3, H, W)

    # Test forward function
    image_tokens, reconstructed_image = model({"image": dummy_image, "text": dummy_text})
    print(f"Image tokens shape (forward): {image_tokens.shape}")  # Expected: (B, num_tokens, latent_dim)
    print(f"Reconstructed image shape (forward): {reconstructed_image.shape}")  # Expected: (B, 3, H, W)