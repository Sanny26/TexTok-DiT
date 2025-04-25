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

vae_name = "madebyollin/sdxl-vae-fp16-fix"
vae_scale_factor = 8
scale_factor = 8 
vae = AutoencoderKL.from_pretrained(vae_name, torch_dtype=torch.float32)
del vae.encoder
gc.collect()
torch.cuda.empty_cache()

vae = vae.to('cuda')

memory_stats()

img_latent_file = np.load("preds.npz")
outputs_list = []

for i in range(15):
    print(i)
    output = vae.decode(torch.tensor(img_latent_file['preds'][i:i+1]*scale_factor).to('cuda'))[0].cpu()
    outputs_list.append(output[0].to('cpu'))
    del output

output = torch.stack(outputs_list)
out = to_pil((vutils.make_grid((output + 1) / 2, nrow=8, padding=4)).float().clip(0, 1))
out.save(f"lr_test_outputs.png")
print('output')
del output, out, img_latent_file 