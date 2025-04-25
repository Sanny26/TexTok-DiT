import os
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoImageProcessor, AutoModel

from TitokTokenizer.modeling.titok import TiTok
from TitokTokenizer.modeling.tatitok import TATiTok
from tld.diffusion import encode_text
from tld.sr_train import SR_COCODataset
from tld.configs import ModelConfig, DataConfig, TrainConfig
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

data_config = DataConfig()

config = ModelConfig(
        data_config=data_config,
        train_config=TrainConfig(),
    )

train_config = config.train_config
dataconfig = config.data_config

if config.use_titok:
    titok = TiTok.from_pretrained("yucornetto/tokenizer_titok_l32_imagenet").to('cuda')
    titok.eval()
    titok.requires_grad_(False)
    print("Loaded Titok")

elif config.use_tatitok:
    tatitok = TATiTok.from_pretrained("turkeyju/tokenizer_tatitok_bl32_vae").to('cuda')
    tatitok.eval()
    tatitok.requires_grad_(False)
    print("Loaded TaTitok")

else:
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32)
    vae = vae.to('cuda')
    vae.eval()
    vae.requires_grad_(False)
    print("Loaded VAE")


if True: #not os.path.exists(config.latents_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
    ])
    
    train_dataset = SR_COCODataset(img_dir=dataconfig.img_path,
                        ann_file=dataconfig.img_ann_path)

    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=False)

    if config.use_tatitok:
        clip_encoder, _, _ = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
        del clip_encoder.visual
        clip_tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
        clip_encoder.transformer.batch_first = False
        clip_encoder.eval()
        clip_encoder.requires_grad_(False)
        clip_encoder.to('cuda')
        print("Loaded clip_model")
    else:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda')
        clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
        clip_model.requires_grad_(False)
        
    if config.denoiser_config.super_res:
        dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to('cuda')

        print("Loaded DINO")

    x_list, y_list, z_list = [], [], []
    
    print('processing data!')
    
    for x, y, z in tqdm(train_loader):

        if config.use_tatitok:
            # Process Text
            idxs = clip_tokenizer(y).to('cuda')
            cast_dtype = clip_encoder.transformer.get_cast_dtype()
            text_guidance = clip_encoder.token_embedding(idxs).to(cast_dtype)  # [batch_size, n_ctx, d_model]
            text_guidance = text_guidance + clip_encoder.positional_embedding.to(cast_dtype)
            text_guidance = text_guidance.permute(1, 0, 2)  # NLD -> LND
            text_guidance = clip_encoder.transformer(text_guidance, attn_mask=clip_encoder.attn_mask)
            text_guidance = text_guidance.permute(1, 0, 2)  # LND -> NLD
            text_guidance = clip_encoder.ln_final(text_guidance)  # [batch_size, n_ctx, transformer.width]
            y_77 = text_guidance

            pooled_embed, _ = text_global_pool(text_guidance, idxs, clip_encoder.text_pool_type)
            pooled_embed = pooled_embed @ clip_encoder.text_projection
            y_1 = pooled_embed

        else:
            y = clip_preprocess(text = y, return_tensors = "pt", padding = True).to('cuda')
            y = encode_text(y, clip_model)

        # Process LR Image
        z = z.to('cuda') 

        z -= torch.min(z)
        z /= torch.max(z)
        z = dino_processor(images=z, return_tensors="pt", do_rescale=False).to('cuda')
    
        z = dino_model(**z)
        z = z[0]
        
        cls = z[:, 0]
        max_pooled = torch.max(z[:, 1:], dim = 1)[0]
        pooled = torch.cat([cls, max_pooled], dim=1)


        #Process image
        x = x.to('cuda') 
        if config.use_titok:
            # Process Titok Tokens
            with torch.no_grad():
                x, _ = titok.encode(x)
            x = x.squeeze(2)
        elif config.use_tatitok:
            # Process Taitok Tokens
            posteriors = tatitok.encode(x.to('cuda'))[1]
            encoded_tokens = posteriors.sample()
            x = encoded_tokens
            x = x.squeeze(2)
        else:
            x = x.to('cuda')
            x -= torch.min(x)
            x /= torch.max(x)
            x = x * 2 - 1
            x = vae.encode(x, return_dict=False)[0].sample()
        

        x_list.append(x.detach().cpu().numpy().astype(np.float32))
        z_list.append(pooled.detach().cpu().numpy().astype(np.float32))

        if config.use_tatitok:
            y1_list.append(y_1.detach().cpu().numpy().astype(np.float32))
            y77_list.append(y_77.detach().cpu().numpy().astype(np.float32))
        else:
            y_list.append(y.detach().cpu().numpy().astype(np.float32))

        if len(x_list) == 20:
            break
    
    print("lengths of lists:",len(x_list), len(y_list), len(z_list))

    if config.use_tatitok:
        x_val, y1_val, y77_val, z_val =  x_list[0], y1_list[0], y77_list[0], z_list[0]

        x_all = np.concatenate(x_list[1:], axis=0)
        y1_all = np.concatenate(y1_list[1:], axis=0)
        y77_all = np.concatenate(y77_list[1:], axis=0)
        z_all = np.concatenate(z_list[1:], axis=0)

        np.savez(dataconfig.latent_path, x_all = x_all, x_val = x_val)
        np.savez(dataconfig.text_emb_path, y1_all = y1_all, y1_val = y1_val)
        np.savez(dataconfig.detokenizer_text_emb_path, y77_all = y77_all, y77_val = y77_val)
        np.savez(dataconfig.lr_latent_path, z_all = z_all, z_val = z_val)
    else:
        x_val, y_val, z_val = x_list[0], y_list[0], z_list[0]
        x_all = np.concatenate(x_list[1:], axis=0)
        y_all = np.concatenate(y_list[1:], axis=0)
        z_all = np.concatenate(z_list[1:], axis=0)

        np.savez(dataconfig.latent_path, x_all = x_all, x_val = x_val)
        np.savez(dataconfig.text_emb_path, y_all = y_all, y_val = y_val)
        np.savez(dataconfig.lr_latent_path, z_all = z_all, z_val = z_val)