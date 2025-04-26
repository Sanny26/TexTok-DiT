# write a test for the textok model

import torch
from TitokTokenizer.modeling.textok import Textok, create_clip_model
from TitokTokenizer.modeling.tatitok import TATiTok
from omegaconf import OmegaConf
import os
from PIL import Image
import os
import numpy as np
import tqdm
import PIL
import torch.nn.functional as F
from pycocotools.coco import COCO
from transformers import BlipProcessor, BlipForConditionalGeneration
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch
import open_clip
import demo_util
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def generate_caption(blip_model, blip_processor, image):
    inputs = blip_processor(images=image, return_tensors="pt", use_fast=True)
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption


def apply_mask(tokens, codebook_dim, mask_prob=None):
    # tokens: shape [batch_size, seq_len]
    if mask_prob:
        # Create a mask for which tokens to replace
        mask = torch.rand_like(tokens, dtype=torch.float) < mask_prob
        # Generate random indices from the codebook dimension
        random_indices = torch.randint(0, codebook_dim, tokens.shape, device=tokens.device)
        # Apply the mask: replace selected tokens with random indices
        masked_tokens = torch.where(mask, random_indices, tokens)
        return masked_tokens
    return tokens

def load_tatitok_model():
    # Load model config
    config = demo_util.get_config("/home/san/imtokenizer/TexTok-DiT/TitokTokenizer/configs/infer/TA-TiTok/tatitok_bl32_vae.yaml")
    
    # Initialize model
    model = TATiTok(config)
    model.eval()
    model.requires_grad_(False)
    
    # Download checkpoint (safetensors)
    checkpoint_path = hf_hub_download(
        repo_id="turkeyju/tokenizer_tatitok_bl32_vae",
        filename="model.safetensors",
        local_dir="./checkpoints"
    )
    
    # Load safely from safetensors
    state_dict = load_file(checkpoint_path, device="cpu")
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys when loading model: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys when loading model: {unexpected_keys}")
    
    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    clip_encoder, clip_tokenizer = create_clip_model()
    clip_encoder.eval()
    clip_encoder.to("cuda")
    return model, clip_encoder, clip_tokenizer

def unnormalize_image(decoded):
    decoded = decoded.squeeze(0).cpu().permute(1, 2, 0) # Convert from CHW to HWC
    decoded = torch.clamp(decoded, 0, 1) * 255.0
    decoded = decoded.to(torch.uint8)
    # decoded = decoded.numpy().astype(np.uint8)
    # decoded = Image.fromarray(decoded)
    return decoded

def preprocess(x):
    x = x.resize((256, 256))
    x = x.convert("RGB")
    x = torch.from_numpy(np.array(x)).permute(2, 0, 1)
    x = x.unsqueeze(0) / 255.0  # 0-1 range
    x = x.to("cuda").float()
    return x


def textok_forward(img, caption, mask_prob=None):
    x = preprocess(img)
    with torch.no_grad():
        text_guidance = clip_tokenizer(caption).to(x.device)
        cast_dtype = clip_encoder.transformer.get_cast_dtype()
        text_guidance = clip_encoder.token_embedding(text_guidance).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        text_guidance = text_guidance + clip_encoder.positional_embedding.to(cast_dtype)
        text_guidance = text_guidance.permute(1, 0, 2)  # NLD -> LND
        text_guidance = clip_encoder.transformer(text_guidance, attn_mask=clip_encoder.attn_mask)
        text_guidance = text_guidance.permute(1, 0, 2)  # LND -> NLD
        text_guidance = clip_encoder.ln_final(text_guidance)  # [batch_size, n_ctx, transformer.width]

        # decoded, _ = model(x, text_guidance[0].unsqueeze(0))
        encoded, _ = model.encode(x)
        decoded = model.decode(encoded, text_guidance[0].unsqueeze(0))
        # print(text_guidance.shape, encoded.shape, decoded.shape)
        # print(x.shape, decoded.shape)
        if x.shape==decoded.shape:
            loss = F.mse_loss(x, decoded).item()
        else: 
            x_shape = x.shape
            decoded_shape = decoded.shape
            min_dims = [min(x_shape[i], decoded_shape[i]) for i in range(len(x_shape))]
            x_resized = F.interpolate(x, size=min_dims[2:], mode='bicubic', align_corners=False)
            decoded_resized = F.interpolate(decoded, size=min_dims[2:], mode='bicubic', align_corners=False)
            loss = F.mse_loss(x_resized, decoded_resized).item()
    return x, decoded, loss



if __name__ == "__main__":
    model, clip_encoder, clip_tokenizer = load_tatitok_model()
    
    root_folder = "/home/san/imtokenizer/"
    mask_prob = 0
    # im_folder_path = f'{root_folder}/coco/val2017'
    # save_image = True
    # rec_folder_path = f'{root_folder}/coco/val2017_textok_{mask_prob}'
    # ann_file = f'{root_folder}/coco/annotations/captions_val2017.json'
    # os.makedirs(rec_folder_path, exist_ok=True)

    # loss = 0
    # files_saved = 0

    # coco = COCO(ann_file)
    # img_ids = list(coco.imgs.keys())

    # blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    # blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # for img_id in tqdm.tqdm(img_ids):
    #     ann_ids = coco.getAnnIds(imgIds=img_id)
    #     anns = coco.loadAnns(ann_ids)
    #     # import pdb; pdb.set_trace()
    #     try:
    #         caption = anns[0]['caption'] if anns else ""
    #     except:
    #         caption = generate_caption(blip_model, blip_processor, image)


    #     img_info = coco.loadImgs(img_id)[0]
    #     img_path = f"{im_folder_path}/{img_info['file_name']}"
    #     image = PIL.Image.open(img_path).convert('RGB')

    #     decoded, sample_loss = textok_forward(image, caption, rec_folder_path, img_info['file_name'], mask_prob=mask_prob)
    #     if save_image:
    #         decoded.save(os.path.join(rec_folder_path, img_info['file_name']))
    #         print(caption)
    #         import pdb; pdb.set_trace()
    #     loss += sample_loss
    #     files_saved += 1


    im_folder_path = '/home/san/imtokenizer/DALL-E/coco2017/test2017'
    save_image = False
    rec_folder_path = f'/home/san/imtokenizer/DALL-E/coco2017/test2017_textok_{mask_prob}'
    os.makedirs(rec_folder_path, exist_ok=True)

    loss = 0
    files_saved = 0


    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    

    fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=False) # expected rgb images 0-255, np.uint8
    fid.set_dtype(torch.float64)   
    fid.to("cuda") 

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0) # using gaussian kernel with sigma=1.5, kernel_Szie -11 3 # resizing/downsizing might be a issue here
    ssim.to("cuda")

    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True) # using vgg network, 0-1 range
    lpips.to("cuda")

    lpips_scores = []
    ssim_scores = []

    for file in tqdm.tqdm(os.listdir(im_folder_path)):
        imfile = os.path.join(im_folder_path, file)
        try:
            image = PIL.Image.open(imfile).convert("RGB")
        except:
            print('Unable to read as PIL image', imfile)
            continue
            
        caption = generate_caption(blip_model, blip_processor, image)


        input, decoded, sample_loss = textok_forward(image, caption, mask_prob=mask_prob)


        decoded = torch.clamp(decoded, 0, 1) 
        input = torch.clamp(input, 0, 1) 
        ssim_score = ssim(decoded, input)
        lpips_score = lpips(decoded, input)
        
        input = input * 255.0
        decoded = decoded * 255.0

        decoded = decoded.to(torch.uint8)
        input = input.to(torch.uint8)
        fid.update(decoded.to("cuda"), real=False)
        fid.update(input.to("cuda"), real=True)


        ssim_scores.append(ssim_score)
        lpips_scores.append(lpips_score)

        # input = input.cpu().numpy()
        if save_image:
            decoded = decoded.cpu().numpy()
            decoded = decoded.numpy().astype(np.uint8)
            decoded = Image.fromarray(decoded)
            decoded.save(os.path.join(rec_folder_path, file))
            # print(caption)
            # import pdb; pdb.set_trace()
        loss += sample_loss
        files_saved += 1

        if files_saved > 100:
            break



    print(f'Masking out {mask_prob*100}: Rloss: ', loss/files_saved)
    average_ssim = sum(ssim_scores) / len(ssim_scores)
    average_lpips = sum(lpips_scores) / len(lpips_scores)
    print(f'Average SSIM: {average_ssim}, Average LPIPS: {average_lpips}')
    
    fid_score = fid.compute()
    print(f"FID Score: {fid_score.item()}")
