#!/usr/bin/env python3

import copy
from dataclasses import asdict
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import wandb
import cv2 as cv
from accelerate import Accelerator
from diffusers import AutoencoderKL
from transformers import CLIPProcessor, CLIPModel

import PIL
from PIL.Image import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm

from tld.denoiser import Denoiser1D, Denoiser
from bsrgan_utils import utils_blindsr as blindsr
from tld.tokenizer import TexTok
from TitokTokenizer.modeling.titok import TiTok
from TitokTokenizer.modeling.tatitok import TATiTok

from tld.diffusion import DiffusionGenerator, DiffusionGenerator1D, encode_text, download_file
from tld.configs import ModelConfig, DataConfig, TrainConfig, Denoiser1DConfig, DenoiserLoad, DenoiserConfig

from pycocotools.coco import COCO
from datetime import datetime
import os

from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

import pdb

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, hr_size=256, scale_factor=4, bsr_mode:bool = False):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
                transforms.Resize((hr_size//scale_factor, hr_size//scale_factor)),
            ])
        self.img_ids = list(self.coco.imgs.keys())
        self.hr_size = hr_size
        self.bsr_mode = bsr_mode
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.img_ids)

    def _process(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        caption = anns[0]['caption'] if anns else ""
        
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = f"{self.img_dir}/{img_info['file_name']}"
        image = PIL.Image.open(img_path).convert('RGB').resize((self.hr_size,self.hr_size))

        #bsr degradations
        if self.bsr_mode:
            image = np.array(image).astype(np.float32)/255 # hr image as numpy
            image_lr, image = blindsr.degradation_bsrgan_plus_nopatch(image, sf=self.scale_factor, shuffle_prob=0.1, use_sharp=True) 
            # convert to tensors
            image = torch.from_numpy(image).permute(2,0,1)
            image_lr = torch.from_numpy(image_lr).permute(2,0,1)
        else:
            image = torch.from_numpy(np.array(image).astype(np.float32)).permute(2, 0, 1) / 255.0 # the hr image
            image_lr = self.transform(image)
        
        return image, caption, image_lr

    def __getitem__(self, idx):
        image, caption = self._process(idx)
        
        return image, caption

class SR_COCODataset(COCODataset):
    def __init__(self, img_dir, ann_file, hr_size=256, scale_factor=4, bsr_mode:bool=False):
        super().__init__(img_dir, ann_file, hr_size, scale_factor, bsr_mode)

    def __getitem__(self, idx):
        image, caption,lr_image = self._process(idx)
        return image, caption, lr_image

def eval_gen(diffuser: DiffusionGenerator, labels: Tensor, img_size: int, img_labels = None) -> Image:
    class_guidance = 4.5
    seed = 10
    
    #print('evalgen', labels.shape, img_labels.shape)
    
    start = time.time()

    out, _ = diffuser.generate(
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

    end = time.time()

    print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', start-end)

    out = to_pil((vutils.make_grid((out + 1) / 2, nrow=8, padding=4)).float().clip(0, 1))
    out.save(f"emb_val_cfg:{class_guidance}_seed:{seed}.png")

    return out

def eval_gen_1D(diffuser: DiffusionGenerator1D, labels: Tensor, n_tokens: int, img_labels = None) -> Image:
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

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_per_layer(model: nn.Module):
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")


to_pil = torchvision.transforms.ToPILImage()


def update_ema(ema_model: nn.Module, model: nn.Module, alpha: float = 0.999):
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(model_param.data, alpha=1 - alpha)



def main(config: ModelConfig) -> None:
    """main train loop to be used with accelerate"""
    if config.use_tatitok:
        denoiser_config = config.denoiser_config
    elif config.use_titok or config.use_textok:
        denoiser_config = config.denoiser_config
    else:
        denoiser_config = config.denoiser_old_config
    train_config = config.train_config
    dataconfig = config.data_config
    
    print(config.use_titok)

    log_with="wandb" if train_config.use_wandb else None
    accelerator = Accelerator(mixed_precision="fp16", log_with=log_with)

    accelerator.print("Loading Data:")
    
    current_time = datetime.now()
    checkpoint_folder = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(f'checkpoints/{checkpoint_folder}', exist_ok=True)
   
    if config.use_textok:
        print('Using Textok!')
        textok = TexTok(config.textok_cfg, accelerator.device).to(accelerator.device)
    elif config.use_titok:
        print('Using Titok!')
        titok = TiTok.from_pretrained("yucornetto/tokenizer_titok_l32_imagenet")
        if accelerator.is_main_process:
            titok = titok.to(accelerator.device)
    elif config.use_tatitok:
        tatitok = TATiTok.from_pretrained("turkeyju/tokenizer_tatitok_bl32_vae").to('cuda')
        tatitok.eval()
        tatitok.requires_grad_(False)
        if accelerator.is_main_process:
            tatitok = tatitok.to(accelerator.device)
    else:
        vae = AutoencoderKL.from_pretrained(config.vae_cfg.vae_name, torch_dtype=config.vae_cfg.vae_dtype)
        if accelerator.is_main_process:
            vae = vae.to(accelerator.device)
    

    if config.use_image_data:
        
        if False and not os.path.exists(dataconfig.lr_latent_path):
        # if True:
            # transform = transforms.Compose([
            #     transforms.Resize((256, 256)),
            # ])
            
            train_dataset = SR_COCODataset(img_dir=dataconfig.img_path,
                                ann_file=dataconfig.img_ann_path,bsr_mode=True)

            train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)

            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(accelerator.device)
            clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(accelerator.device)

            x_list, y_list, z_list = [], [], []
            
            print('processing data!')
            
            for x, y, z in tqdm(train_loader):
                

                x = x.to('cuda') 
                z = z.to('cuda') 

                z -= torch.min(z)
                z /= torch.max(z)
                z = dino_processor(images=z, return_tensors="pt").to(accelerator.device)
            
                z = dino_model(**z)
                z = z[0]

                cls = z[:, 0]
                max_pooled = torch.max(z[:, 1:], dim = 1)[0]
                pooled = torch.cat([cls, max_pooled], dim=1)

                if config.use_titok:
                    with torch.no_grad():
                        x, _ = titok.encode(x)
                    x = x.squeeze(2)

                y = clip_preprocess(text = y, return_tensors = "pt", padding = True).to(accelerator.device)
                y = encode_text(y, clip_model)

                x_list.append(x.detach().cpu().numpy().astype(np.float32))
                y_list.append(y.detach().cpu().numpy().astype(np.float32))
                z_list.append(pooled.detach().cpu().numpy().astype(np.float32))
            x_val, y_val, z_val = x_list[0], y_list[0], z_list[0]

            x_all = np.concatenate(x_list[1:], axis=0)
            y_all = np.concatenate(y_list[1:], axis=0)
            z_all = np.concatenate(z_list[1:], axis=0)

            np.savez(dataconfig.latent_path, x_all = x_all, x_val = x_val)
            np.savez(dataconfig.text_emb_path, y_all = y_all, y_val = y_val)
            np.savez(dataconfig.lr_latent_path, z_all = z_all, z_val = z_val)

            # np.savez(dataconfig.lr_latent_path, x_all = x_all, y_all = y_all, z_all = z_all, x_val = x_val, y_val = y_val, z_val = z_val)

        img_latent_file = np.load(dataconfig.latent_path)
        text_emb_file = np.load(dataconfig.text_emb_path)
        lr_latent_file = np.load(dataconfig.lr_latent_path)

        # print([(i, latents_file[i].shape) for i in latents_file.files])
        x_val = torch.from_numpy(img_latent_file['x_val']).to('cuda')
        y_val = torch.from_numpy(text_emb_file['y_val']).to('cuda')
        z_val = torch.from_numpy(lr_latent_file['z_val']).to('cuda')
       
        x_all = torch.from_numpy(img_latent_file['x_all'])
        x_all = x_all[:-16]
        y_all = torch.from_numpy(text_emb_file['y_all'])
        z_all = torch.from_numpy(lr_latent_file['z_all'])
        
        #print(x_all.shape, y_all.shape, z_all.shape)

        dataset = TensorDataset(x_all, y_all, z_all)
        train_loader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True)
    
    else:
        latent_train_data = torch.tensor(np.load(dataconfig.latent_path), dtype=torch.float32)
        train_label_embeddings = torch.tensor(np.load(dataconfig.text_emb_path), dtype=torch.float32)
        train_image_embeddings = torch.tensor(np.load(dataconfig.image_emb_path), dtype=torch.float32)
        emb_val = torch.tensor(np.load(dataconfig.val_path), dtype=torch.float32)
        img_val = torch.tensor(np.load(dataconfig.val_img_path), dtype=torch.float32)
        dataset = TensorDataset(latent_train_data, train_label_embeddings, train_image_embeddings)
        train_loader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True)


    #load weights 
    if config.use_titok:
        model = Denoiser1D(**asdict(denoiser_config))
    elif config.use_textok:
        model = Denoiser1D(**asdict(denoiser_config))
    elif config.use_tatitok:
        model = Denoiser1D(**asdict(denoiser_config))
    else:
        model = Denoiser(**asdict(denoiser_config))
        #print(f"Downloading model from huggingface")
        #download_file(url='https://huggingface.co/apapiu/small_ldt/resolve/main/state_dict_378000.pth',
        #               filename='state_dict_378000.pth')
        #state_dict = torch.load('state_dict_378000.pth', map_location=torch.device("cuda"))
        #model.load_state_dict(state_dict, strict=False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    # if train_config.compile:
    #     accelerator.print("Compiling model:")
    #     model = torch.compile(model)

    if not train_config.from_scratch:
        accelerator.print("Loading Model:")
        #wandb.restore(
        #    train_config.model_name, run_path=f"kaustavmu/TiTok/runs/{train_config.run_id}", replace=True, root="/home/"
        #)
        full_state_dict = torch.load(train_config.model_name)
        model.load_state_dict(full_state_dict["model_ema"])
        optimizer.load_state_dict(full_state_dict["opt_state"])
        global_step = full_state_dict["global_step"]
    else:
        global_step = 0

    if accelerator.is_local_main_process:
        ema_model = copy.deepcopy(model).to(accelerator.device)
        if config.use_titok:
            diffuser = DiffusionGenerator1D(ema_model, titok, accelerator.device, torch.float32)
        elif config.use_tatitok:
            diffuser = DiffusionGenerator1D(ema_model, titok, accelerator.device, torch.float32)
        else:
            diffuser = DiffusionGenerator(ema_model, vae, accelerator.device, torch.float32)

    accelerator.print("model prep")
    model, train_loader, optimizer = accelerator.prepare(model, train_loader, optimizer)

    if train_config.use_wandb:
        accelerator.init_trackers(project_name="TiTok", config=asdict(config))

    accelerator.print(count_parameters(model))
    accelerator.print(count_parameters_per_layer(model))
    
    save_model_cnt = 0

    ### Train:
    for i in range(1, train_config.n_epoch + 1):
        accelerator.print(f"epoch: {i}")

        for x, y, z in tqdm(train_loader):
            
            x, y, z = x.to('cuda'), y.to('cuda'), z.to('cuda')
            '''
            if config.use_image_data:
                
                #print(x.shape)
                #image = transforms.ToPILImage()(x)
                z -= torch.min(z)
                z /= torch.max(z)
                z = dino_processor(images=z, return_tensors="pt")
                z = dino_model(**z)
                z = z[0] 
                
                cls = z[:, 0]
                max_pooled = torch.max(z[:, 1:], dim=1)[0]
                pooled = torch.cat([cls, max_pooled], dim=1)

                #print('eeeeyahhhhh', pooled.shape)

                if config.use_titok:
                    with torch.no_grad():
                        x, _ = titok.encode(x) # encode return z_quantized, result_dict #using vq mode
                    x = x.squeeze(2)
                y = clip_preprocess(text=y, return_tensors="pt", padding=True).to(accelerator.device)
                y = encode_text(y, clip_model)
                

            else:
                x = x / config.vae_cfg.vae_scale_factor
            '''
            #print(x.shape, y.shape, z.shape) 

            noise_level = torch.tensor(
                np.random.beta(train_config.beta_a, train_config.beta_b, len(x)), device=accelerator.device
            )
            signal_level = 1 - noise_level
            noise = torch.randn_like(x)
            
            if config.use_titok or config.use_tatitok:
                x_noisy = noise_level.view(-1, 1, 1) * noise + signal_level.view(-1, 1, 1) * x
            else:
                x = x / config.vae_cfg.vae_scale_factor
                x_noisy = noise_level.view(-1, 1, 1, 1) * noise + signal_level.view(-1, 1, 1, 1) * x
            
            x_noisy = x_noisy.float()
            noise_level = noise_level.float()
            label = y
            img_label = z

            #print('cuhs', x_noisy.shape, noise_level.shape, label.shape, img_label.shape)

            prob = 0.15 # classifier free guidance
            mask_txt = torch.rand(y.size(0), device=accelerator.device) < prob
            mask_img = torch.rand(z.size(0), device=accelerator.device) < prob
            label[mask_txt] = 0  # OR replacement_vector
            img_label[mask_img] = 0

            if global_step % train_config.save_and_eval_every_iters == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ##eval and saving:
                    if config.use_titok or config.use_tatitok:
                        out = eval_gen_1D(diffuser=diffuser, labels=y_val, n_tokens=denoiser_config.seq_len, img_labels = z_val)
                    else:
                        for _ in range(10):
                            out = eval_gen(diffuser=diffuser, labels=y_val, img_size=denoiser_config.image_size, img_labels = z_val)
                    out.save("img.jpg")
                    if train_config.use_wandb:
                        print(global_step)
                        # accelerator.log({f"step: {global_step}": wandb.Image("img.jpg")})
                        accelerator.log({"eval_img": wandb.Image(out)}, step=global_step)

                    opt_unwrapped = accelerator.unwrap_model(optimizer)
                    full_state_dict = {
                        "model_ema": ema_model.state_dict(),
                        "opt_state": opt_unwrapped.state_dict(),
                        "global_step": global_step,
                    }
                    if train_config.save_model:
                        save_model_cnt += 1
                        if save_model_cnt%25 == 0:
                            print("saving model at ", )
                            accelerator.save(full_state_dict, f'checkpoints/{checkpoint_folder}/checkpoint_{global_step}.pt')
                            if train_config.use_wandb:
                                wandb.save(train_config.model_name)

            model.train()

            with accelerator.accumulate():
                ###train loop:
                optimizer.zero_grad()
               
                #print('srtrain', x_noisy.shape, noise_level.view(-1, 1).shape, label.shape, img_label.shape)\
                pred = model(x_noisy, noise_level.view(-1, 1), label, img_label)
                loss = loss_fn(pred, x)

                accelerator.log({"train_loss": loss.item()}, step=global_step)
                accelerator.backward(loss)
                optimizer.step()
                
                if global_step % train_config.save_and_eval_every_iters == 0:
                    if train_config.use_wandb:
                        if accelerator.is_main_process:
                            if config.use_titok or config.use_tatitok:
                                train_img = eval_gen_1D(diffuser=diffuser, labels=y[0].unsqueeze(0), n_tokens=denoiser_config.seq_len)
                            else:
                                train_img = eval_gen(diffuser = diffuser, labels=y[0].unsqueeze(0), img_size=denoiser_config.image_size, img_labels = z[0].unsqueeze(0))
                            accelerator.log({"train_img": wandb.Image(train_img)}, step=global_step)
                
                if accelerator.is_main_process:
                    update_ema(ema_model, model, alpha=train_config.alpha)

            global_step += 1
    accelerator.end_training()


# args = (config, data_path, val_path)
# notebook_launcher(training_loop)
if __name__ == "__main__":
    
    data_config = DataConfig()
    denoiser_config = Denoiser1DConfig(super_res=True)
    denoiser_old_config = DenoiserConfig()

    model_cfg = ModelConfig(
        data_config=data_config,
        denoiser_config=denoiser_config,
        denoiser_old_config=denoiser_old_config,
        train_config=TrainConfig(batch_size=128),
    )
    
    main(model_cfg)
