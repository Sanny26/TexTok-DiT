import torch
import numpy as np
#from TitokTokenizer.modeling.titok import TiTok
import cv2
from diffusers import AutoencoderKL
from tld.denoiser import Denoiser1D, Denoiser
from tld.diffusion import DiffusionGenerator, DiffusionGenerator1D
from tld.configs import DenoiserConfig, Denoiser1DConfig
import copy
import PIL
from PIL.Image import Image
from dataclasses import asdict
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
import gc
from numba import cuda
import multiprocessing as mp

def memory_stats():
    print(torch.cuda.memory_allocated()/1024**2)
    print(torch.cuda.memory_cached()/1024**2)
to_pil = torchvision.transforms.ToPILImage()

def eval_gen(diffuser: DiffusionGenerator, labels, img_size, img_labels= None) -> Image:
    class_guidance = 4.5
    seed = 15

    #print('evalgen', labels.shape, img_labels.shape)

    pred = diffuser.generate(
        labels=labels,#torch.repeat_interleave(labels, 2, dim=0),
        num_imgs=labels.shape[0],
        class_guidance=class_guidance,
        seed=seed,
        n_iter=40,
        exponent=1,
        sharp_f=0.1,
        img_size=img_size,
        img_labels = img_labels
    )
    
    return pred

    out = to_pil((vutils.make_grid((out + 1) / 2, nrow=8, padding=4)).float().clip(0, 1))
    out.save(f"emb_val_cfg:{class_guidance}_seed:{seed}.png")

    return out

def eval_gen_1D(diffuser: DiffusionGenerator1D, labels, n_tokens, inp, img_labels = None) -> Image:
    class_guidance = 4.5
    seed = 10
    out, _ = diffuser.generate(
        labels=labels, #torch.repeat_interleave(labels, 2, dim=0),
        num_imgs=labels.shape[0],
        class_guidance=class_guidance,
        seed=seed,
        n_iter=40,
        exponent=1,
        sharp_f=0.1,
        n_tokens=n_tokens,
        img_labels=img_labels
    )

    out = to_pil((vutils.make_grid((out + 1) / 2, nrow=8, padding=4)).float().clip(0, 1))
    out.save(f"emb_val_cfg:{class_guidance}_seed:{seed}.png")

    return out

vae_name = "madebyollin/sdxl-vae-fp16-fix"
vae_scale_factor = 8
scale_factor = 8 
vae = AutoencoderKL.from_pretrained(vae_name, torch_dtype=torch.float32)
del vae.encoder
gc.collect()
torch.cuda.empty_cache()

vae = vae.to('cuda')

memory_stats()

text_emb_file = np.load("preprocess_txt.npz")
lr_latent_file = np.load("preprocess_lr.npz")
y_val = torch.from_numpy(text_emb_file['y_val'][:16])
z_val = torch.from_numpy(lr_latent_file['z_val'][:16])

del text_emb_file, lr_latent_file

print("loaded files")

denoiser_config = DenoiserConfig()
ema_model = Denoiser(**asdict(denoiser_config)).to('cuda')

full_state_dict = torch.load("checkpoints/vae-sr/checkpoint_24000.pt")
ema_model.load_state_dict(full_state_dict["model_ema"])

diffuser = DiffusionGenerator(ema_model, vae, 'cuda', torch.float32)

preds = []
for i in range(16):
    with torch.no_grad():
        x_pred = eval_gen(diffuser = diffuser, labels=y_val[i:i+1], img_size = 32, img_labels=z_val[i:i+1])
        preds.append(x_pred[0].to(torch.float32).to('cpu'))

preds = torch.stack(preds).numpy()
print(preds.shape)
np.savez("preds.npz", preds = preds)