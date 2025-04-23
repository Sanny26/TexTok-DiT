import os
import time
import sys
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoImageProcessor, AutoModel
import open_clip
from open_clip.transformer import text_global_pool
from termcolor import cprint

from TitokTokenizer.modeling.titok import TiTok
from TitokTokenizer.modeling.tatitok import TATiTok
from tld.diffusion import encode_text
from tld.sr_train import SR_COCODataset
from tld.configs import ModelConfig, DataConfig, TrainConfig
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from PIL import Image
from datasets import load_dataset
from PIL import ImageDraw, ImageFont

DEBUG = True
# DEBUG = False

dataconfig = DataConfig()
dataconfig.img_path = "/home/ubuntu/val2017"
dataconfig.img_ann_path = "/home/ubuntu/annotations/captions_val2017.json"

config = ModelConfig(
        data_config=dataconfig,
        train_config=TrainConfig(),
    )

train_config = config.train_config


def save_comparison_image(sample_img, decoded_image, sample_caption, output_path="test_comparison.png"):
    # Convert sample_img to PIL Image format
    sample_img_pil = Image.fromarray((sample_img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
    decoded_image_pil = Image.fromarray(decoded_image)
    
    # Create combined image
    combined_width = sample_img_pil.width + decoded_image_pil.width
    combined_height = max(sample_img_pil.height, decoded_image_pil.height)
    combined_img = Image.new('RGB', (combined_width, combined_height + 30))  # Extra 30px for title
    
    # Paste images side by side
    combined_img.paste(sample_img_pil, (0, 30))
    combined_img.paste(decoded_image_pil, (sample_img_pil.width, 30))
    
    draw = ImageDraw.Draw(combined_img)
    draw.text((10, 5), sample_caption[:100], fill='white')  # Truncate long captions
    
    # Save combined image
    combined_img.save(output_path)

def convert_laion12m_to_wds(output_dir, max_train_samples_per_shard):
    """Convert Laion12M dataset to WebDataset format with TaTiTok, CLIP, and DINO embeddings.
    
    Args:
        output_dir: Directory to save the WebDataset files
        max_train_samples_per_shard: Maximum number of samples per training shard
        max_val_samples_per_shard: Maximum number of samples per validation shard
    """
    global DEBUG
    # Initialize models
    tatitok = TATiTok.from_pretrained("turkeyju/tokenizer_tatitok_bl32_vae").to('cuda')
    tatitok.eval()
    tatitok.requires_grad_(False)
    
    # Initialize CLIP
    clip_encoder, _, _ = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
    del clip_encoder.visual
    clip_tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
    clip_encoder.transformer.batch_first = False
    clip_encoder.eval()
    clip_encoder.requires_grad_(False)
    clip_encoder.to('cuda')
    
    # Initialize DINO
    dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to('cuda')
    dino_model.eval()
    dino_model.requires_grad_(False)
    
    # Image transforms for HR
    transform_hr = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Image transforms for LR
    transform_lr = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    dataset = load_dataset(
        "laion/conceptual-captions-12m-webdataset",
        streaming=True,
        split="train",
        token=True
    )
    
    opat = os.path.join(output_dir, "laion12m-processed-%06d.npz")
    
    now = time.time()
    i = 0
    print("Processing examples...")
    
    x_list, y1_list, y77_list, z_list = [], [], [], []
    batch_size = 32  # Process in batches
    current_batch = {'images': [], 'captions': [], 'lr_images': []}
    
    iterator = iter(dataset)
    while True:
        example = next(iterator)
        if i % 1000 == 0:
            print(f"Processing example {i}", file=sys.stderr)
        
        # Get image and caption
        img = example["jpg"]
        caption = example.get("txt", "")
        
        if img is None:
            print(f"Skipping corrupt image")
            continue
            
        # skipping low res images in training set!
        if img.size[0] <= 256 or img.size[1] <= 256:
            continue
        # Convert PIL image to tensor
        img_tensor = transform_hr(img).unsqueeze(0).to('cuda')
        lr_img = transform_lr(img).unsqueeze(0).to('cuda')
        
        current_batch['images'].append(img_tensor)
        current_batch['captions'].append(caption)
        current_batch['lr_images'].append(lr_img)
        
        # Process when batch is full or at the end
        if len(current_batch['images']) == batch_size:
            # Process batch
            with torch.no_grad():
                # TaTiTok processing
                img_batch = torch.cat(current_batch['images'], dim=0)
                posteriors = tatitok.encode(img_batch)[1]
                encoded_tokens = posteriors.sample()
                x = encoded_tokens.squeeze(2)

                # CLIP text processing
                idxs = clip_tokenizer(current_batch['captions']).to('cuda')
                cast_dtype = clip_encoder.transformer.get_cast_dtype()
                text_guidance = clip_encoder.token_embedding(idxs).to(cast_dtype)
                text_guidance = text_guidance + clip_encoder.positional_embedding.to(cast_dtype)
                text_guidance = text_guidance.permute(1, 0, 2)
                text_guidance = clip_encoder.transformer(text_guidance, attn_mask=clip_encoder.attn_mask)
                text_guidance = text_guidance.permute(1, 0, 2)
                text_guidance = clip_encoder.ln_final(text_guidance)
                y_77 = text_guidance
                
                pooled_embed, _ = text_global_pool(text_guidance, idxs, clip_encoder.text_pool_type)
                pooled_embed = pooled_embed @ clip_encoder.text_projection
                y_1 = pooled_embed
                
                # DINO processing for LR images
                lr_batch = torch.cat(current_batch['lr_images'], dim=0)
                # lr_batch = (lr_batch - lr_batch.min()) / (lr_batch.max() - lr_batch.min())
                dino_inputs = dino_processor(images=lr_batch, return_tensors="pt", do_rescale=False).to('cuda')
                dino_outputs = dino_model(**dino_inputs)[0]
                
                cls = dino_outputs[:, 0]
                max_pooled = torch.max(dino_outputs[:, 1:], dim=1)[0]
                z = torch.cat([cls, max_pooled], dim=1)
            
                if DEBUG:
                    cprint(f'Tatitok input details{img_batch.shape}, {img_batch.min()}, {img_batch.max()}', 'green')
                    cprint(f'Tatitok output details {encoded_tokens.shape}, {encoded_tokens.min()}, {encoded_tokens.max()}', 'green')
                    
                    cprint(f'CLIP output details y_77 {y_77.shape}', 'red')
                    cprint(f'CLIP output details y_1 {y_1.shape}', 'red')
                    
                    cprint(f'DINO input details {lr_batch.shape}, {lr_batch.min()}, {lr_batch.max()}', 'blue')
                    cprint(f'DINO output details {z.shape}', 'blue')


                    # debugging to decode the image
                    sample_img, sample_caption = img_batch[0], current_batch['captions'][0]
                    sample_token, sample_text_guidance = encoded_tokens[0][None, ...], text_guidance[0][None, ...]
                    decoded_image = tatitok.decode(sample_token, sample_text_guidance)
                    decoded_image = torch.clamp(decoded_image, 0.0, 1.0)
                    decoded_image = (decoded_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
                    
                    save_comparison_image(sample_img, decoded_image, sample_caption)
                    DEBUG = False

                
            # Store processed tensors
            x_list.append(x.detach().cpu().numpy())
            y1_list.append(y_1.detach().cpu().numpy())
            y77_list.append(y_77.detach().cpu().numpy())
            z_list.append(z.detach().cpu().numpy())
            
            # Save every N batches
            if len(x_list) * batch_size >= max_train_samples_per_shard:
                # Use first batch as validation, rest as training (like sr_data.py)
                x_val, y1_val, y77_val, z_val = x_list[0], y1_list[0], y77_list[0], z_list[0]
                
                x_all = np.concatenate(x_list[1:], axis=0)
                y1_all = np.concatenate(y1_list[1:], axis=0)
                y77_all = np.concatenate(y77_list[1:], axis=0)
                z_all = np.concatenate(z_list[1:], axis=0)
                
                shard_num = i // max_train_samples_per_shard
                
                # Save each type of latent in separate files, matching sr_data.py format
                np.savez(
                    os.path.join(output_dir, f"latent-{shard_num}.npz"),
                    x_all=x_all,
                    x_val=x_val
                )
                
                np.savez(
                    os.path.join(output_dir, f"text_emb-{shard_num}.npz"),
                    y1_all=y1_all,
                    y1_val=y1_val
                )
                
                np.savez(
                    os.path.join(output_dir, f"detokenizer_text_emb-{shard_num}.npz"),
                    y77_all=y77_all,
                    y77_val=y77_val
                )
                
                np.savez(
                    os.path.join(output_dir, f"lr_latent-{shard_num}.npz"),
                    z_all=z_all,
                    z_val=z_val
                )
                
                x_list, y1_list, y77_list, z_list = [], [], [], []
                break
            
            # Clear batch
            current_batch = {'images': [], 'captions': [], 'lr_images': []}
        
        i += 1
    
    
    time_taken = time.time() - now
    print(f"Processed {i} examples in {time_taken // 3600} hours.")


if __name__ == "__main__":
    # convert_laion12m_to_wds(output_dir="laion12m-processed", max_train_samples_per_shard=1000000, max_val_samples_per_shard=1000000)
    convert_laion12m_to_wds(output_dir="/home/ubuntu/cc12m/", max_train_samples_per_shard=1000)