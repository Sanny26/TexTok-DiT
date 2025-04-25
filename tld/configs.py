from dataclasses import dataclass, field
import torch

@dataclass
class DataDownloadConfig:
    """config for downloading and processing latents"""
    data_link: str
    caption_col: str = "caption"
    url_col: str = "url"
    latent_save_path: str = "latents_folder"
    raw_imgs_save_path: str = "raw_imgs_folder"
    use_drive: bool = False
    initial_csv_path: str = "imgs.csv"
    number_sample_per_shard: int = 10000
    image_size: int = 256
    batch_size: int = 64
    download_data: bool = True
    first_n_rows: int = 1000000
    use_wandb: bool = False

@dataclass
class DenoiserConfig:
    image_size: int = 32
    noise_embed_dims: int = 256
    patch_size: int = 2
    embed_dim: int = 768
    dropout: float = 0
    n_layers: int = 12
    text_emb_size: int = 512
    image_emb_size: int = 768
    n_channels: int = 4 
    mlp_multiplier: int = 4 
    super_res: bool = True

@dataclass
class Denoiser1DConfig:
    seq_len: int = 32  # number of tokens
    noise_embed_dims: int = 256
    patch_size: int = 1  # diffusion patch size
    embed_dim: int = 768  # DiT hidden dimension 
    dropout: float = 0
    n_layers: int = 12  # DiT layer count
    text_emb_size: int = 512  # CLIP embedding size
    n_channels: int = 12  # Latent token dimension
    mlp_multiplier: int = 4
    image_emb_size: int | None = 768
    super_res: bool = True

@dataclass
class DenoiserLoad:
    dtype: torch.dtype = torch.float32
    file_url: str | None = None
    local_filename: str | None = None

@dataclass
class VaeConfig:
    vae_scale_factor: float = 8
    vae_name: str = "madebyollin/sdxl-vae-fp16-fix"
    vae_dtype: torch.dtype = torch.float32

@dataclass
class TexTokConfig:
    batch_size: int = 2
    image_size: int = 256
    patch_size: int = 8
    hidden_size: int = 768
    latent_dim: int = 4
    num_tokens: int = 64
    ViT_number_of_heads: int = 12
    ViT_number_of_layers: int = 12
    textok_dtype: torch.dtype = torch.float32

@dataclass
class ClipConfig:
    clip_model_name: str = "ViT-L/14"
    clip_dtype: torch.dtype = torch.float16

@dataclass
class DataConfig:
    """where is the latent data stored"""
    latent_path: str = "preprocess_img_vae.npz"
    text_emb_path: str = "preprocess_txt_vae.npz"
    lr_latent_path: str = "preprocess_lr_vae.npz"
    val_path: str = ""
    img_path: str = "/home/tchoudha/coco/train2017"
    img_ann_path: str = "/home/tchoudha/coco/annotations/captions_train2017.json"

@dataclass
class TrainConfig:
    batch_size: int = 8
    lr: float = 3e-4
    n_epoch: int = 250
    alpha: float = 0.999
    from_scratch: bool = True
    ##betas determine the distribution of noise seen during training
    beta_a: float = 1  
    beta_b: float = 2.5
    save_and_eval_every_iters: int = 1000
    run_id: str = "sr-run"
    model_name: str = "checkpoints.pt"
    compile: bool = True
    save_model: bool = True
    use_wandb: bool = True


@dataclass
class LTDConfig:
    """main config for inference"""
    denoiser_cfg: Denoiser1DConfig = field(default_factory=Denoiser1DConfig)
    denoiser_old_cfg: DenoiserConfig = field(default_factory=DenoiserConfig)
    denoiser_load: DenoiserLoad = field(default_factory=DenoiserLoad)
    vae_cfg: VaeConfig = field(default_factory=VaeConfig)
    clip_cfg: ClipConfig = field(default_factory=ClipConfig)
    textok_cfg: TexTokConfig = field(default_factory=TexTokConfig)
    use_textok: bool = False
    use_titok: bool = False


@dataclass
class ModelConfig:
    """main config for getting data, training and inference"""
    data_config: DataConfig 
    download_config: DataDownloadConfig | None = None
    denoiser_config: Denoiser1DConfig = field(default_factory=Denoiser1DConfig)
    denoiser_old_config: DenoiserConfig = field(default_factory=DenoiserConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    vae_cfg: VaeConfig = field(default_factory=VaeConfig)
    clip_cfg: ClipConfig = field(default_factory=ClipConfig)
    textok_cfg: TexTokConfig = field(default_factory=TexTokConfig)
    use_textok: bool = False
    use_titok: bool = False
    use_tatitok: bool = False
    use_image_data: bool = True

if __name__=='__main__':
    cfg = Denoiser1DConfig()
    print(cfg)