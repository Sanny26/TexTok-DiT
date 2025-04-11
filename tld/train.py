#!/usr/bin/env python3

import copy
from dataclasses import asdict

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import wandb
from accelerate import Accelerator
from diffusers import AutoencoderKL
from transformers import CLIPProcessor, CLIPModel

import PIL
from PIL.Image import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm

from tld.denoiser import Denoiser

from tld.tokenizer import TexTok
from TitokTokenizer.modeling.titok import TiTok

from tld.diffusion import DiffusionGenerator, DiffusionGenerator1D
from tld.configs import ModelConfig, DataConfig, TrainConfig

from pycocotools.coco import COCO
import pdb
class COCODataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transform = transform
        self.img_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        caption = anns[0]['caption'] if anns else ""
        
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = f"{self.img_dir}/{img_info['file_name']}"
        image = PIL.Image.open(img_path).convert('RGB')
        image = torch.from_numpy(np.array(image).astype(np.float32)).permute(2, 0, 1) / 255.0
        if self.transform:
            image = self.transform(image)
        
        return image, caption

def eval_gen(diffuser: DiffusionGenerator, labels: Tensor, img_size: int) -> Image:
    class_guidance = 4.5
    seed = 10
    out, _ = diffuser.generate(
        labels=torch.repeat_interleave(labels, 2, dim=0),
        num_imgs=16,
        class_guidance=class_guidance,
        seed=seed,
        n_iter=40,
        exponent=1,
        sharp_f=0.1,
        img_size=img_size
    )

    out = to_pil((vutils.make_grid((out + 1) / 2, nrow=8, padding=4)).float().clip(0, 1))
    out.save(f"emb_val_cfg:{class_guidance}_seed:{seed}.png")

    return out

def eval_gen_1D(diffuser: DiffusionGenerator1D, labels: Tensor, n_tokens: int) -> Image:
    class_guidance = 4.5
    seed = 10
    out, _ = diffuser.generate(
        # labels=torch.repeat_interleave(labels, 2, dim=0),
        labels=labels, (num_imgs x 768)
        num_imgs=16,
        class_guidance=class_guidance,
        seed=seed,
        n_iter=40,
        exponent=1,
        sharp_f=0.1,
        n_tokens=n_tokens
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
    denoiser_config = config.denoiser_config
    train_config = config.train_config
    dataconfig = config.data_config

    log_with="wandb" if train_config.use_wandb else None
    accelerator = Accelerator(mixed_precision="fp16", log_with=log_with)

    accelerator.print("Loading Data:")
    if config.use_image_data:
        # train_transforms = transforms.Compose([
        #     transforms.RandomResizedCrop(224),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

        # val_transforms = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

        # train_dataset = torchvision.datasets.ImageNet(
        #     root=dataconfig.train_path, split='train', transform=train_transforms)
        # val_dataset = torchvision.datasets.ImageNet(
        #     root=dataconfig.val_path, split='val', transform=val_transforms)
        # torch.from_numpy(np.array(Image.open(img_path)).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.ToTensor()
        ])
        # train_dataset = COCODataset(img_dir="/home/tchoudha/coco2017/train2017",
        #                     ann_file="/home/tchoudha/coco2017/annotations/captions_train2017.json", 
        #                     transform=transform)
        
        # val_dataset = COCODataset(img_dir="/home/tchoudha/coco2017/val2017",
        #                     ann_file="/home/tchoudha/coco2017/annotations/captions_val2017.json", 
        #                     transform=transform)

        train_dataset = COCODataset(img_dir="../coco/val2017",
                            ann_file="../coco/annotations/captions_val2017.json", 
                            transform=transform)
        
        val_dataset = COCODataset(img_dir="../coco/val2017",
                            ann_file="../coco/annotations/captions_val2017.json", 
                            transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
        train_loader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=True)

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(accelerator.device)
        clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        txt_emb_projector = torch.nn.Linear(clip_model.text_model.config.hidden_size, denoiser_config.text_emb_size).to(accelerator.device)
        
        txt_inputs_val = clip_preprocess(text=("a cute grey great owl"), return_tensors="pt", padding=True).to(accelerator.device)
        emb_val = clip_model.get_text_features(**txt_inputs_val)
        emb_val = torch.nn.functional.normalize(emb_val, dim=-1)
        emb_val = txt_emb_projector(emb_val) # B x 768
        # import pdb; pdb.set_trace()
    
    else:
        latent_train_data = torch.tensor(np.load(dataconfig.latent_path), dtype=torch.float32)
        train_label_embeddings = torch.tensor(np.load(dataconfig.text_emb_path), dtype=torch.float32)
        emb_val = torch.tensor(np.load(dataconfig.val_path), dtype=torch.float32)
        dataset = TensorDataset(latent_train_data, train_label_embeddings)
        train_loader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True)

    if config.use_textok:
        print('Using Textok!')
        textok = TexTok(config.textok_cfg, accelerator.device).to(accelerator.device)
    elif config.use_titok:
        #TODO: freeze tokenizer 
        print('Using Titok!')
        titok = TiTok.from_pretrained("yucornetto/tokenizer_titok_l32_imagenet")
        if accelerator.is_main_process:
            titok = titok.to(accelerator.device)
    else:
        vae = AutoencoderKL.from_pretrained(config.vae_cfg.vae_name, torch_dtype=config.vae_cfg.vae_dtype)
        if accelerator.is_main_process:
            vae = vae.to(accelerator.device)

    model = Denoiser(**asdict(denoiser_config))

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    if train_config.compile:
        accelerator.print("Compiling model:")
        model = torch.compile(model)

    if not train_config.from_scratch:
        accelerator.print("Loading Model:")
        wandb.restore(
            train_config.model_name, run_path=f"apapiu/cifar_diffusion/runs/{train_config.run_id}", replace=True
        )
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
        else:
            diffuser = DiffusionGenerator(ema_model, vae, accelerator.device, torch.float32)

    accelerator.print("model prep")
    model, train_loader, optimizer = accelerator.prepare(model, train_loader, optimizer)

    if train_config.use_wandb:
        accelerator.init_trackers(project_name="cifar_diffusion", config=asdict(config))

    accelerator.print(count_parameters(model))
    accelerator.print(count_parameters_per_layer(model))

    ### Train:
    for i in range(1, train_config.n_epoch + 1):
        accelerator.print(f"epoch: {i}")

        for x, y in tqdm(train_loader):
            if config.use_image_data:
                if config.use_titok:
                    x = titok.encode(x)[1]["min_encoding_indices"].float() #using vq mode
                y = clip_preprocess(text=y, return_tensors="pt", padding=True).to(accelerator.device)
                y = clip_model.get_text_features(**y)
                y = torch.nn.functional.normalize(y, dim=-1)
                y = txt_emb_projector(y)
            else:
                x = x / config.vae_cfg.vae_scale_factor

            noise_level = torch.tensor(
                np.random.beta(train_config.beta_a, train_config.beta_b, len(x)), device=accelerator.device
            )
            signal_level = 1 - noise_level
            noise = torch.randn_like(x)

            if config.use_titok:
                x_noisy = noise_level.view(-1, 1, 1) * noise + signal_level.view(-1, 1, 1) * x
            else:
                x_noisy = noise_level.view(-1, 1, 1, 1) * noise + signal_level.view(-1, 1, 1, 1) * x

            x_noisy = x_noisy.float()
            noise_level = noise_level.float()
            label = y
            
            prob = 0.15 # classifier free guidance
            mask = torch.rand(y.size(0), device=accelerator.device) < prob
            label[mask] = 0  # OR replacement_vector

            if global_step % train_config.save_and_eval_every_iters == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ##eval and saving:
                    if config.use_titok:
                        out = eval_gen_1D(diffuser=diffuser, labels=emb_val, n_tokens=denoiser_config.seq_len)
                    else:
                        out = eval_gen(diffuser=diffuser, labels=emb_val, img_size=denoiser_config.image_size)
                    out.save("img.jpg")
                    if train_config.use_wandb:
                        accelerator.log({f"step: {global_step}": wandb.Image("img.jpg")})

                    opt_unwrapped = accelerator.unwrap_model(optimizer)
                    full_state_dict = {
                        "model_ema": ema_model.state_dict(),
                        "opt_state": opt_unwrapped.state_dict(),
                        "global_step": global_step,
                    }
                    if train_config.save_model:
                        accelerator.save(full_state_dict, train_config.model_name)
                        if train_config.use_wandb:
                            wandb.save(train_config.model_name)

            model.train()

            with accelerator.accumulate():
                ###train loop:
                optimizer.zero_grad()

                pred = model(x_noisy, noise_level.view(-1, 1), label)
                loss = loss_fn(pred, x)
                accelerator.log({"train_loss": loss.item()}, step=global_step)
                accelerator.backward(loss)
                optimizer.step()

                if accelerator.is_main_process:
                    update_ema(ema_model, model, alpha=train_config.alpha)

            global_step += 1
    accelerator.end_training()


# args = (config, data_path, val_path)
# notebook_launcher(training_loop)
if __name__ == "__main__":

    data_config = DataConfig(
        latent_path="latents.npy", text_emb_path="text_emb.npy", val_path="val_emb.npy"
    )

    model_cfg = ModelConfig(
        data_config=data_config,
        train_config=TrainConfig(n_epoch=100, save_model=False, compile=False, use_wandb=False),
    )

    main(model_cfg)
