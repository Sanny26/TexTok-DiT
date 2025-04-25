from dataclasses import dataclass, asdict

# import clip
from transformers import CLIPProcessor, CLIPModel

import numpy as np
import requests
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from diffusers import AutoencoderKL
from torch import Tensor
from tqdm import tqdm

from tld.denoiser import Denoiser1D, Denoiser
from tld.tokenizer import TexTok
from TitokTokenizer.modeling.titok import TiTok

from tld.configs import LTDConfig


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
to_pil = transforms.ToPILImage()


@dataclass
class DiffusionGenerator:
    model: Denoiser
    vae: AutoencoderKL
    device: torch.device
    model_dtype: torch.dtype = torch.float32

    @torch.no_grad()
    def generate(
        self,
        labels: Tensor,  # embeddings to condition on
        n_iter: int = 30,
        num_imgs: int = 32,
        class_guidance: float = 3,
        seed: int = 10,
        scale_factor: int = 8,  # latent scaling before decoding - should be ~ std of latent space
        img_size: int = 32,  # height, width of latent
        sharp_f: float = 0.1,
        bright_f: float = 0.1,
        exponent: float = 1,
        seeds: Tensor | None = None,
        noise_levels=None,
        use_ddpm_plus: bool = True,
        img_labels = None,
    ):
        """Generate images via reverse diffusion.
        if use_ddpm_plus=True uses Algorithm 2 DPM-Solver++(2M) here: https://arxiv.org/pdf/2211.01095.pdf
        else use ddim with alpha = 1-sigma
        """
        if noise_levels is None:
            noise_levels = (1 - torch.pow(torch.arange(0, 1, 1 / n_iter), exponent)).tolist()
        noise_levels[0] = 0.99

        if use_ddpm_plus:
            lambdas = [np.log((1 - sigma) / sigma) for sigma in noise_levels]  # log snr
            hs = [lambdas[i] - lambdas[i - 1] for i in range(1, len(lambdas))]
            rs = [hs[i - 1] / hs[i] for i in range(1, len(hs))]

        x_t = self.initialize_image(seeds, num_imgs, img_size, seed)

        labels = torch.cat([labels, torch.zeros_like(labels)])
        if img_labels is not None:
            img_labels = torch.cat([img_labels, torch.zeros_like(img_labels)]).to(self.device, self.model_dtype)

        self.model.eval()

        x0_pred_prev = None

        for i in tqdm(range(len(noise_levels) - 1)):
            curr_noise, next_noise = noise_levels[i], noise_levels[i + 1]
            
            x0_pred = self.pred_image(x_t, labels, curr_noise, class_guidance, img_labels=img_labels)

            if x0_pred_prev is None:
                x_t = ((curr_noise - next_noise) * x0_pred + next_noise * x_t) / curr_noise
            else:
                if use_ddpm_plus:
                    # x0_pred is a combination of the two previous x0_pred:
                    D = (1 + 1 / (2 * rs[i - 1])) * x0_pred - (1 / (2 * rs[i - 1])) * x0_pred_prev
                else:
                    # ddim:
                    D = x0_pred

                x_t = ((curr_noise - next_noise) * D + next_noise * x_t) / curr_noise

            x0_pred_prev = x0_pred

        x0_pred = self.pred_image(x_t, labels, next_noise, class_guidance, img_labels=img_labels)

        # shifting latents works a bit like an image editor:
        x0_pred[:, 3, :, :] += sharp_f
        x0_pred[:, 0, :, :] += bright_f

        x0_pred_img = self.vae.decode((x0_pred * scale_factor).to(self.model_dtype))[0].cpu()
        return x0_pred_img, x0_pred

    def pred_image(self, noisy_image, labels, noise_level, class_guidance, img_labels):
        num_imgs = noisy_image.size(0)
        noises = torch.full((2 * num_imgs, 1), noise_level)
        x0_pred = self.model(
            torch.cat([noisy_image, noisy_image]),
            noises.to(self.device, self.model_dtype),
            labels.to(self.device, self.model_dtype),
            img_labels
        )
        x0_pred = self.apply_classifier_free_guidance(x0_pred, num_imgs, class_guidance)
        return x0_pred

    def initialize_image(self, seeds, num_imgs, img_size, seed):
        """Initialize the seed tensor."""
        if seeds is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            return torch.randn(
                num_imgs,
                self.model.n_channels,
                img_size,
                img_size,
                dtype=self.model_dtype,
                device=self.device,
                generator=generator,
            )
        else:
            return seeds.to(self.device, self.model_dtype)

    def apply_classifier_free_guidance(self, x0_pred, num_imgs, class_guidance):
        """Apply classifier-free guidance to the predictions."""
        x0_pred_label, x0_pred_no_label = x0_pred[:num_imgs], x0_pred[num_imgs:]
        return class_guidance * x0_pred_label + (1 - class_guidance) * x0_pred_no_label



@dataclass
class DiffusionGenerator1D:
    model: Denoiser1D
    tokenizer: TexTok
    device: torch.device
    model_dtype: torch.dtype = torch.float32

    @torch.no_grad()
    def generate(
        self,
        labels: Tensor,  # embeddings to condition on
        prompts: str = "", #for titok
        n_iter: int = 30,
        num_imgs: int = 16,
        class_guidance: float = 3,
        seed: int = 10,
        scale_factor: int = 8,  # latent scaling before decoding - should be ~ std of latent space
        n_tokens: int = 32,  # N_tokens
        sharp_f: float = 0.1,
        bright_f: float = 0.1,
        exponent: float = 1,
        seeds: Tensor | None = None,
        noise_levels=None,
        use_ddpm_plus: bool = True,
        img_labels = None,
    ):
        """Generate images via reverse diffusion.
        if use_ddpm_plus=True uses Algorithm 2 DPM-Solver++(2M) here: https://arxiv.org/pdf/2211.01095.pdf
        else use ddim with alpha = 1-sigma
        """
        if noise_levels is None:
            noise_levels = (1 - torch.pow(torch.arange(0, 1, 1 / n_iter), exponent)).tolist()
        noise_levels[0] = 0.99

        if use_ddpm_plus:
            lambdas = [np.log((1 - sigma) / sigma) for sigma in noise_levels]  # log snr
            hs = [lambdas[i] - lambdas[i - 1] for i in range(1, len(lambdas))]
            rs = [hs[i - 1] / hs[i] for i in range(1, len(hs))]

        x_t = self.initialize_image(seeds, num_imgs, n_tokens, seed) #change to init tokens?
        # print(f'generate func - {x_t.shape}, {seeds}, {labels.shape}') #should be of the shape B x 1 x 32 ?

        labels = torch.cat([labels, torch.zeros_like(labels)])
        if img_labels is not None:
            img_labels = torch.cat([img_labels, torch.zeros_like(img_labels)]).to(self.device, self.model_dtype)
        self.model.eval()

        x0_pred_prev = None

        for i in tqdm(range(len(noise_levels) - 1)):
            curr_noise, next_noise = noise_levels[i], noise_levels[i + 1]

            x0_pred = self.pred_image(x_t, labels, curr_noise, class_guidance, img_labels=img_labels)

            if x0_pred_prev is None:
                x_t = ((curr_noise - next_noise) * x0_pred + next_noise * x_t) / curr_noise
            else:
                if use_ddpm_plus:
                    # x0_pred is a combination of the two previous x0_pred:
                    D = (1 + 1 / (2 * rs[i - 1])) * x0_pred - (1 / (2 * rs[i - 1])) * x0_pred_prev
                else:
                    # ddim:
                    D = x0_pred

                x_t = ((curr_noise - next_noise) * D + next_noise * x_t) / curr_noise

            x0_pred_prev = x0_pred

        x0_pred = self.pred_image(x_t, labels, next_noise, class_guidance, img_labels = img_labels)

        # shifting latents works a bit like an image editor:
        x0_pred[:, 3, : ] += sharp_f
        x0_pred[:, 0, : ] += bright_f

        pred_img_tokens = (x0_pred * scale_factor).to(self.model_dtype)
        pred_img_tokens = pred_img_tokens.permute(0, 2, 1) # changing it back to BND format
        
        if LTDConfig.use_titok:
            pred_img_tokens = pred_img_tokens.permute(0,2,1)
            x0_pred_img = self.tokenizer.decode(pred_img_tokens.unsqueeze(2)).cpu()
        else:
            x0_pred_img = self.tokenizer.decode(pred_img_tokens, prompts).cpu()
        return x0_pred_img, x0_pred

    def pred_image(self, noisy_latent, labels, noise_level, class_guidance, img_labels = None):
        num_imgs = noisy_latent.size(0)
        noises = torch.full((2 * num_imgs, 1), noise_level) # noiselevel - timestep??
        x0_pred = self.model(
            torch.cat([noisy_latent, noisy_latent]),
            noises.to(self.device, self.model_dtype),
            labels.to(self.device, self.model_dtype),
            img_labels
        )
        x0_pred = self.apply_classifier_free_guidance(x0_pred, num_imgs, class_guidance)
        return x0_pred

    def initialize_image(self, seeds, num_imgs, n_tokens, seed):
        """Initialize the seed tensor."""
        if seeds is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            return torch.randn(
                num_imgs,
                self.model.n_channels,
                n_tokens,
                dtype=self.model_dtype,
                device=self.device,
                generator=generator,
            )
        else:
            return seeds.to(self.device, self.model_dtype)

    def apply_classifier_free_guidance(self, x0_pred, num_imgs, class_guidance):
        """Apply classifier-free guidance to the predictions."""
        x0_pred_label, x0_pred_no_label = x0_pred[:num_imgs], x0_pred[num_imgs:]
        return class_guidance * x0_pred_label + (1 - class_guidance) * x0_pred_no_label

def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


@torch.no_grad()
def encode_text(label, model):
    # TODO: Switchout to T5 encoder?
    # text_tokens = clip.tokenize(label, truncate=True).to(device)
    # text_encoding = model.encode_text(text_tokens)
    # return text_encoding.cpu()
    embedding = model.get_text_features(**label)
    embedding = torch.nn.functional.normalize(embedding, dim=-1)
    return embedding


class DiffusionTransformer:
    # NOTE: Used only for eval!
    def __init__(self, cfg: LTDConfig):
        denoiser = Denoiser1D(**asdict(cfg.denoiser_cfg))
        denoiser = denoiser.to(cfg.denoiser_load.dtype)

        if cfg.denoiser_load.file_url is not None:
            if cfg.denoiser_load.local_filename is not None:
                print(f"Downloading model from {cfg.denoiser_load.file_url}")
                download_file(cfg.denoiser_load.file_url, cfg.denoiser_load.local_filename)
                state_dict = torch.load(cfg.denoiser_load.local_filename, map_location=torch.device("cuda"))
                denoiser.load_state_dict(state_dict)

        denoiser = denoiser.to(device)

        # self.clip_model, preprocess = clip.load(cfg.clip_cfg.clip_model_name)
        # self.clip_model = self.clip_model.to(device)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        if cfg.use_textok:
            print('Using Textok!')
            textok = TexTok(cfg.textok_cfg, device).to(device)
            self.diffuser = DiffusionGenerator1D(denoiser,
                                                 textok,
                                                 device,
                                                 cfg.denoiser_load.dtype)
        elif cfg.use_titok:
            print('Using Titok!')
            titok = TiTok.from_pretrained("yucornetto/tokenizer_titok_l32_imagenet").to(device)
            self.diffuser = DiffusionGenerator1D(denoiser,
                                                 titok,
                                                 device,
                                                 cfg.denoiser_load.dtype)
        else:
            vae = AutoencoderKL.from_pretrained(cfg.vae_cfg.vae_name, 
            torch_dtype=cfg.vae_cfg.vae_dtype).to(device)
            self.diffuser = DiffusionGenerator(denoiser, vae, device, cfg.denoiser_load.dtype)

    def generate_image_from_text(
        self, prompt: str, class_guidance=6, seed=11, num_imgs=1, n_iter=15
    ):
        nrow = int(np.sqrt(num_imgs))

        cur_prompts = [prompt] * num_imgs
        labels = encode_text(cur_prompts, self.clip_model)
        
        out, out_latent = self.diffuser.generate(
            prompts=cur_prompts,
            labels=labels,
            num_imgs=num_imgs,
            n_tokens=self.diffuser.model.seq_len,
            class_guidance=class_guidance,
            seed=seed,
            n_iter=n_iter,
            exponent=1,
            scale_factor=8,
            sharp_f=0,
            bright_f=0,
        )

        out = to_pil((vutils.make_grid((out + 1) / 2, nrow=nrow, padding=4)).float().clip(0, 1))
        return out


if __name__ == "__main__":
    # Load configuration
    cfg = LTDConfig()

    # Initialize the diffusion transformer
    diffusion_transformer = DiffusionTransformer(cfg)

    # Test text-to-image generation
    prompt = "A beautiful sunset over the mountains"
    generated_image = diffusion_transformer.generate_image_from_text(
        prompt=prompt,
        class_guidance=6,
        seed=42,
        num_imgs=3,
        n_iter=15,
    )

    # Save or display the generated image
    generated_image.save("generated_image.png")
    print("Generated image saved as 'generated_image.png'")