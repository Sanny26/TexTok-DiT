# write a test for the textok model

import torch
from TitokTokenizer.modeling.textok import Textok, create_clip_model
from omegaconf import OmegaConf
import os
from PIL import Image
import os
import numpy as np
def load_test_image():
    """Load a test image from assets folder"""
    asset_dir = os.path.join(os.path.dirname(__file__), "assets")
    img_path = os.path.join(asset_dir, "/home/san/imtokenizer/TexTok-DiT/TitokTokenizer/assets/test_image.png")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Test image not found at {img_path}")
    return Image.open(img_path)


if __name__ == "__main__":
    config = OmegaConf.load("TitokTokenizer/configs/training/TexTok/textok_b32_vae.yaml")
    model = Textok(config)
    model.load_state_dict(torch.load("/home/san/imtokenizer/TexTok-DiT/checkpoints/290k_ckpt.pth"))
    model.eval()
    model.to("cuda")

    x = load_test_image()
    x = x.resize((256, 256))
    x = x.convert("RGB")
    x = torch.from_numpy(np.array(x)).permute(2, 0, 1)
    x = x.unsqueeze(0) / 255.0 # 0-1 range

    x = x.to("cuda").float()

    text = ["a photo of a bird"]

    clip_encoder, clip_tokenizer = create_clip_model()
    clip_encoder.eval()
    clip_encoder.to("cuda")
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
        print(decoded.shape)

    decoded = decoded.squeeze(0).cpu().permute(1, 2, 0) # Convert from CHW to HWC
    decoded = torch.clamp(decoded, 0, 1) * 255.0
    decoded = decoded.numpy().astype(np.uint8)
    decoded = Image.fromarray(decoded)
   
    # Convert x back to same format as decoded for concatenation
    x = x.squeeze(0).cpu().permute(1, 2, 0) # Convert from CHW to HWC
    x = torch.clamp(x, 0, 1) * 255.0
    x = x.numpy().astype(np.uint8)
    x = Image.fromarray(x)  
    
    # Concatenate horizontally using PIL
    combined = Image.new('RGB', (x.width + decoded.width, x.height))
    combined.paste(x, (0, 0))
    combined.paste(decoded, (x.width, 0))
    decoded = combined

    # Save the decoded image
    save_path = os.path.join("/home/san/imtokenizer/TexTok-DiT/TitokTokenizer/assets/textok_test.jpg")
    decoded.save(save_path)
    print(f"Saved decoded image to {save_path}")