#!/usr/bin/env python3

import copy
from dataclasses import asdict

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
from omegaconf import OmegaConf

from tld.denoiser import Denoiser1D, Denoiser
from bsrgan_utils import utils_blindsr as blindsr
from tld.tokenizer import TexTok
from TitokTokenizer.modeling.titok import TiTok
from TitokTokenizer.modeling.tatitok import TATiTok
from TitokTokenizer.modeling.textok import Textok

from tld.diffusion import DiffusionGenerator, DiffusionGenerator1D, encode_text, download_file
from tld.configs import ModelConfig, DataConfig, TrainConfig, Denoiser1DConfig, DenoiserLoad, DenoiserConfig
from tld.sr_train_tatitok import COCODataset, SR_COCODataset, count_parameters, count_parameters_per_layer, update_ema

from pycocotools.coco import COCO
from datetime import datetime
import os

from transformers import AutoImageProcessor, AutoModel
from PIL import Image, ImageDraw, ImageFont
import requests

to_pil = torchvision.transforms.ToPILImage()

def eval_gen_1D_extensive(diffuser: DiffusionGenerator1D, gt_img: Tensor, labels: Tensor, n_tokens: int, labels_detokenizer = None, img_labels = None, class_guidance = [4.5, 4, 6, 8, 10]) -> Image:
    seed = 10
    batch_size = labels.shape[0]
    text_caption = " GT_truth  |  Pred with guidance : "
    # Create list to store all generated images
    generated_imgs = []
    
    # Add ground truth image first
    # gt_img = (torch.clamp(gt_img, 0.0, 1.0) * 255.0).to(dtype=torch.uint8)
    generated_imgs.append(gt_img)
    
    # Generate images for each guidance scale
    for guidance in class_guidance:
        out, _ = diffuser.generate(
            labels=labels,
            labels_detokenizer=labels_detokenizer,
            num_imgs=batch_size,
            class_guidance=guidance,
            seed=seed,
            n_iter=40,
            exponent=1,
            sharp_f=0.1,
            n_tokens=n_tokens,
            img_labels=img_labels
        )
        text_caption += f" {guidance}  |  "
        generated_imgs.append(out)

    # Stack all images horizontally for each sample in batch
    # Shape becomes [batch_size, num_versions, C, H, W]
    all_imgs = torch.stack(generated_imgs, dim=1)
    all_imgs = all_imgs.view(-1, *all_imgs.shape[2:])
    merged_imgs = vutils.make_grid(all_imgs, nrow=len(class_guidance)+1, padding=4)
    merged_imgs = to_pil(merged_imgs)
    return merged_imgs


def main(config: ModelConfig) -> None:
    """main train loop to be used with accelerate"""
    denoiser_config = config.denoiser_config
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
        tok_model = Textok(OmegaConf.load(config.textok_cfg))
        tok_model.load_state_dict(torch.load(config.textok_ckpt))
    elif config.use_tatitok:
        tok_model = TATiTok.from_pretrained("turkeyju/tokenizer_tatitok_bl32_vae").to('cuda')
    else:
        raise ValueError("No model selected")
    
    tok_model.eval()
    tok_model.requires_grad_(False)
    if accelerator.is_main_process:
        tok_model = tok_model.to(accelerator.device)

    if config.use_image_data:
        img_latent_file = np.load(dataconfig.latent_path)
        text_emb_file = np.load(dataconfig.text_emb_path)
        lr_latent_file = np.load(dataconfig.lr_latent_path)
        detokenizer_text_emb_file = np.load(dataconfig.detokenizer_text_emb_path)

        x_val = torch.from_numpy(img_latent_file['x_val']).to('cuda')
        z_val = torch.from_numpy(lr_latent_file['z_val']).to('cuda')
        y1_val = torch.from_numpy(text_emb_file['y1_val']).to('cuda')
        y77_val = torch.from_numpy(detokenizer_text_emb_file['y77_val']).to('cuda')
        
        x_all = torch.from_numpy(img_latent_file['x_all'])
        z_all = torch.from_numpy(lr_latent_file['z_all'])
        y1_all = torch.from_numpy(text_emb_file['y1_all'])
        y77_all = torch.from_numpy(detokenizer_text_emb_file['y77_all'])
        dataset = TensorDataset(x_all, y1_all, y77_all, z_all)
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
    model = Denoiser1D(**asdict(denoiser_config))
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    
    if not train_config.from_scratch:
        accelerator.print("Loading Model:")
        
        # wandb.restore(
        #    train_config.model_name, run_path=f"tchoudha-carnegie-mellon-university/TiTok/runs/{train_config.run_id}", replace=True #, root="/home/"
        # )
        full_state_dict = torch.load(train_config.model_name)
        model.load_state_dict(full_state_dict["model_ema"])
        optimizer.load_state_dict(full_state_dict["opt_state"])
        global_step = full_state_dict["global_step"]

        # Resume the wandb run
        accelerator.init_trackers(
            project_name="TiTok",
            config=vars(train_config),
            init_kwargs={
                "wandb": {
                    "id": train_config.run_id,
                    "resume": "allow",
                    "name": train_config.run_name,  # optional
                }
            }
        )
    else:
        global_step = 0
        accelerator.init_trackers(project_name="TiTok", config=asdict(config))

    if accelerator.is_local_main_process:
        ema_model = copy.deepcopy(model).to(accelerator.device)
        diffuser = DiffusionGenerator1D(ema_model, tok_model, accelerator.device, torch.float32)

    accelerator.print("model prep")
    model, train_loader, optimizer = accelerator.prepare(model, train_loader, optimizer)

    accelerator.print(count_parameters(model))
    accelerator.print(count_parameters_per_layer(model))
    
    save_model_cnt = 0

    ### Train:
    for i in range(1, train_config.n_epoch + 1):
        accelerator.print(f"epoch: {i}")

        for x, y1, y77, z in tqdm(train_loader):
            
            x, y1, y77, z = x.to('cuda'), y1.to('cuda'), y77.to('cuda'), z.to('cuda')

            noise_level = torch.tensor(
                np.random.beta(train_config.beta_a, train_config.beta_b, len(x)), device=accelerator.device
            )
            signal_level = 1 - noise_level
            noise = torch.randn_like(x)
            
            x_noisy = noise_level.view(-1, 1, 1) * noise + signal_level.view(-1, 1, 1) * x
            x_noisy = x_noisy.float()
            noise_level = noise_level.float()
            label = y1
            img_label = z

            
            prob = 0.15 # classifier free guidance
            mask_txt = torch.rand(y1.size(0), device=accelerator.device) < prob
            # mask_img = torch.rand(z.size(0), device=accelerator.device) < prob
            label[mask_txt] = 0  # OR replacement_vector
            # img_label[mask_img] = 0 #remove classifier free guidance for lr images

            if global_step % train_config.save_and_eval_every_iters == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:

                    vis_images = 8
                    with torch.no_grad():
                        # import pdb; pdb.set_trace()
                        gt_img = tok_model.decode(x_val.unsqueeze(2), y77_val)
                        gt_img = torch.clamp(gt_img, 0.0, 1.0)
                        gt_img = (gt_img * 255.0).to(dtype=torch.uint8).to("cpu")
                    
                    out = eval_gen_1D_extensive(diffuser=diffuser, gt_img=gt_img[0:vis_images], labels=y1_val[0:vis_images], labels_detokenizer = y77_val[0:vis_images], n_tokens=denoiser_config.seq_len, img_labels = z_val[0:vis_images])
                    
                    # out.save("img.jpg")
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
                            vis_images = 8
                            with torch.no_grad():
                                # import pdb; pdb.set_trace()
                                gt_img = tok_model.decode(x.unsqueeze(2), y77)
                                gt_img = torch.clamp(gt_img, 0.0, 1.0)
                                gt_img = (gt_img * 255.0).to(dtype=torch.uint8).to("cpu")
                            
                            ##eval and saving:
                            out = eval_gen_1D_extensive(diffuser=diffuser, gt_img=gt_img[0:vis_images], labels=y1[0:vis_images], labels_detokenizer = y77[0:vis_images], n_tokens=denoiser_config.seq_len, img_labels = z[0:vis_images])
                            accelerator.log({"train_img": wandb.Image(out)}, step=global_step)
                
                if accelerator.is_main_process:
                    update_ema(ema_model, model, alpha=train_config.alpha)

            global_step += 1
    accelerator.end_training()


# args = (config, data_path, val_path)
# notebook_launcher(training_loop)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Textok model')
    parser.add_argument('--config', type=str, default='configs_cc12m',
                        help='Name of config file to use (default: configs_cc12m)')
    args = parser.parse_args()

    # Dynamically import the specified config file
    print('Using config:', args.config)
    config_module = __import__(f'tld.{args.config}', fromlist=['DataConfig', 'Denoiser1DConfig', 'DenoiserConfig', 'TrainConfig'])
    DataConfig = config_module.DataConfig
    Denoiser1DConfig = config_module.Denoiser1DConfig
    DenoiserConfig = config_module.DenoiserConfig
    TrainConfig = config_module.TrainConfig


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
