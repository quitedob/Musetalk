import os
import json
import logging
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import WhisperModel
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from einops import rearrange

from musetalk.models.syncnet import SyncNet
from musetalk.models.adapter import BidirectionalAdapter
from musetalk.models.engram import EngramModule
from musetalk.utils.unet_injection import inject_attention_processors, inject_mhc_mixers
from musetalk.loss.discriminator import MultiScaleDiscriminator, DiscriminatorFullModel
from musetalk.loss.basic_loss import Interpolate
import musetalk.loss.vgg_face as vgg_face
from musetalk.data.dataset import PortraitDataset
from musetalk.utils.utils import (
    get_image_pred,
    process_audio_features,
    process_and_save_images,
    reduce_audio_tokens,
)

class Net(nn.Module):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        adapter: nn.Module = None,
        engram: nn.Module = None,
    ):
        super().__init__()
        self.unet = unet
        self.adapter = adapter
        self.engram = engram

    def forward(
        self,
        input_latents,
        timesteps,
        audio_prompts,
    ):
        if self.adapter is not None:
            expected_channels = int(getattr(self.adapter, "expected_input_channels", -1))
            if expected_channels > 0 and input_latents.shape[1] == expected_channels:
                masked_latents, ref_latents = torch.chunk(input_latents, chunks=2, dim=1)
                input_latents = self.adapter.adapt_input(masked_latents, ref_latents)

        model_pred = self.unet(
            input_latents,
            timesteps,
            encoder_hidden_states=audio_prompts
        ).sample

        if self.adapter is not None:
            model_pred = self.adapter.adapt_output(model_pred)

        if self.engram is not None:
            model_pred = self.engram(model_pred, audio_prompts, use_engram=True)

        return model_pred

logger = logging.getLogger(__name__)

try:
    from diffusers import AutoencoderKLFlux2
except Exception:
    AutoencoderKLFlux2 = None


def _cfg_get(section, key: str, default=None):
    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


def _resolve_unet_checkpoint_path(cfg, default_path: str) -> str:
    """
    Resolve which UNet checkpoint to load for warm-start.

    Priority:
    1) Explicit file path in cfg.stage1_checkpoint
    2) Directory in cfg.stage1_checkpoint:
       - best_unet.pt
       - latest unet-*.pth
    3) Fallback to default pretrained path
    """
    stage1_ckpt = getattr(cfg, "stage1_checkpoint", None)
    if not stage1_ckpt:
        return default_path

    if os.path.isfile(stage1_ckpt):
        return stage1_ckpt

    if os.path.isdir(stage1_ckpt):
        best_path = os.path.join(stage1_ckpt, "best_unet.pt")
        if os.path.isfile(best_path):
            return best_path

        staged_ckpts = sorted(
            glob.glob(os.path.join(stage1_ckpt, "unet-*.pth")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("-")[-1]),
        )
        if staged_ckpts:
            return staged_ckpts[-1]

    logger.warning(
        f"stage1_checkpoint='{stage1_ckpt}' not found/invalid, fallback to {default_path}"
    )
    return default_path


def _normalize_unet_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _select_vae_loader(vae_path: str, vae_type: str, load_kwargs: dict):
    if vae_type != "flux-vae":
        return AutoencoderKL
    subfolder = load_kwargs.get("subfolder", "")
    config_path = os.path.join(vae_path, subfolder, "config.json") if subfolder else os.path.join(vae_path, "config.json")
    if not os.path.exists(config_path):
        return AutoencoderKL
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            class_name = json.load(f).get("_class_name", "")
    except Exception:
        return AutoencoderKL
    if class_name == "AutoencoderKLFlux2" and AutoencoderKLFlux2 is not None:
        return AutoencoderKLFlux2
    return AutoencoderKL


def _compute_binary_prf1(pred_images, target_images, region_mask, threshold=0.5):
    """计算二值化 Precision/Recall/F1 指标。"""
    # 将输入从 [-1,1] 映射到 [0,1]。  
    pred = (pred_images.clamp(-1, 1) + 1.0) / 2.0
    target = (target_images.clamp(-1, 1) + 1.0) / 2.0
    # 转为灰度用于阈值分割。  
    pred_gray = pred.mean(dim=1, keepdim=True)
    target_gray = target.mean(dim=1, keepdim=True)
    # 仅在口型区域统计。  
    mask = (region_mask > 0.5).float()
    pred_bin = (pred_gray > threshold).float() * mask
    target_bin = (target_gray > threshold).float() * mask
    # 计算 TP/FP/FN。  
    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1.0 - target_bin)).sum()
    fn = ((1.0 - pred_bin) * target_bin).sum()
    # 计算三项指标。  
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return float(precision.item()), float(recall.item()), float(f1.item())

def initialize_models_and_optimizers(cfg, accelerator, weight_dtype):
    """Initialize models and optimizers"""
    model_dict = {
        'vae': None,
        'unet': None,
        'adapter': None,
        'engram': None,
        'net': None,
        'wav2vec': None,
        'optimizer': None,
        'lr_scheduler': None,
        'scheduler_max_steps': None,
        'trainable_params': None
    }
    
    vae_type = str(getattr(cfg, "vae_type", "sd-vae"))
    vae_path = getattr(cfg, "vae_path", None)

    if vae_path:
        if not os.path.exists(vae_path):
            raise FileNotFoundError(
                f"Configured vae_path does not exist: {vae_path}. "
                "Please download or place Flux/SD VAE weights at this path."
            )
        vae_load_kwargs = {}
        if vae_type == "flux-vae" and os.path.isdir(os.path.join(vae_path, "vae")):
            vae_load_kwargs["subfolder"] = "vae"
        vae_loader_cls = _select_vae_loader(vae_path, vae_type, vae_load_kwargs)
        model_dict['vae'] = vae_loader_cls.from_pretrained(vae_path, **vae_load_kwargs)
    else:
        model_dict['vae'] = AutoencoderKL.from_pretrained(
            cfg.pretrained_model_name_or_path,
            subfolder=vae_type,
        )

    if hasattr(cfg, "scale_factor") and cfg.scale_factor is not None:
        model_dict['vae'].config.scaling_factor = float(cfg.scale_factor)
    elif vae_type == "flux-vae":
        raw_scale = getattr(model_dict['vae'].config, "scaling_factor", None)
        try:
            raw_scale = float(raw_scale)
        except (TypeError, ValueError):
            raw_scale = None
        if raw_scale is None or abs(raw_scale - 0.18215) < 1e-6:
            model_dict['vae'].config.scaling_factor = 0.3611
    if hasattr(cfg, "shift_factor") and cfg.shift_factor is not None:
        model_dict['vae'].config.shift_factor = float(cfg.shift_factor)
    elif vae_type == "flux-vae" and getattr(model_dict['vae'].config, "shift_factor", None) is None:
        model_dict['vae'].config.shift_factor = 0.1159

    unet_config_file = os.path.join(
        cfg.pretrained_model_name_or_path, 
        cfg.unet_sub_folder + "/musetalk.json"
    )
    
    with open(unet_config_file, 'r') as f:
        unet_config = json.load(f)
    model_dict['unet'] = UNet2DConditionModel(**unet_config)

    if not cfg.random_init_unet:
        default_pretrained_unet_path = os.path.join(
            cfg.pretrained_model_name_or_path, cfg.unet_sub_folder, "pytorch_model.bin"
        )
        pretrained_unet_path = _resolve_unet_checkpoint_path(cfg, default_pretrained_unet_path)
        print(f"### Loading existing unet weights from {pretrained_unet_path}. ###")
        checkpoint = torch.load(pretrained_unet_path, map_location=accelerator.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            checkpoint = checkpoint["state_dict"]
        checkpoint = _normalize_unet_state_dict(checkpoint)
        model_dict['unet'].load_state_dict(checkpoint, strict=True)

    # Inject after base weights are loaded so extra params (e.g., gate_proj) do not
    # break strict checkpoint loading in stage2 warm-start.
    attn_inj = inject_attention_processors(
        model_dict['unet'],
        use_gated_attn=bool(getattr(cfg, "use_gated_attn", False)),
        use_dsa=bool(getattr(cfg, "use_dsa", False)),
        dsa_topk=int(_cfg_get(getattr(cfg, "dsa", None), "topk", 2048)),
    )
    if attn_inj["enabled"]:
        logger.info(
            f"Injected attention processors: cross_attn_layers={attn_inj['num_injected']}, "
            f"use_gated_attn={bool(getattr(cfg, 'use_gated_attn', False))}, "
            f"use_dsa={bool(getattr(cfg, 'use_dsa', False))}"
        )

    mhc_inj = inject_mhc_mixers(
        model_dict['unet'],
        use_mhc=bool(getattr(cfg, "use_mhc", False)),
        num_streams=int(_cfg_get(getattr(cfg, "mhc", None), "num_streams", 2)),
        sinkhorn_iters=int(_cfg_get(getattr(cfg, "mhc", None), "sinkhorn_iters", 10)),
    )
    if mhc_inj["enabled"]:
        logger.info(f"Injected mHC mixers: resnet_blocks={mhc_inj['num_injected']}")
      
    unet_params = [p.numel() for n, p in model_dict['unet'].named_parameters()]
    logger.info(f"unet {sum(unet_params) / 1e6}M-parameter")
    
    model_dict['vae'].requires_grad_(False)
    model_dict['unet'].requires_grad_(True)

    model_dict['vae'].to(accelerator.device, dtype=weight_dtype)

    vae_latent_channels = int(getattr(model_dict['vae'].config, "latent_channels", 4))
    use_adapter = bool(getattr(cfg, "use_adapter", False))
    if use_adapter and vae_type == "flux-vae":
        model_dict['adapter'] = BidirectionalAdapter(
            use_flux_vae=True,
            flux_latent_channels=vae_latent_channels,
            unet_in_channels=int(unet_config.get("in_channels", 8)),
            unet_out_channels=int(unet_config.get("out_channels", 4)),
        )
        logger.info(
            "Flux adapter enabled: "
            f"{vae_latent_channels * 2}ch input -> {int(unet_config.get('in_channels', 8))}ch UNet -> "
            f"{vae_latent_channels}ch output"
        )
    elif use_adapter:
        logger.info("use_adapter=true but vae_type is not flux-vae, adapter disabled")

    use_engram = bool(getattr(cfg, "use_engram", False))
    if use_engram:
        engram_cfg = getattr(cfg, "engram", {})
        output_channels = int(getattr(cfg, "latent_channels", unet_config.get("out_channels", 4)))
        if use_adapter and vae_type == "flux-vae":
            output_channels = vae_latent_channels
        else:
            output_channels = int(unet_config.get("out_channels", 4))
        latent_size = int(getattr(cfg.data, "image_size", 256) // 8)
        model_dict['engram'] = EngramModule(
            audio_dim=int(_cfg_get(engram_cfg, "audio_dim", 384)),
            unet_dim=output_channels,
            latent_channels=output_channels,
            latent_size=latent_size,
            n_gram=int(_cfg_get(engram_cfg, "n_gram", 3)),
            num_hashes=int(_cfg_get(engram_cfg, "num_hashes", 8)),
            memory_size=int(_cfg_get(engram_cfg, "memory_size", 10000)),
            enable_prefetch=False,
        )
        memory_path = _cfg_get(engram_cfg, "memory_path", None)
        if memory_path and os.path.exists(memory_path):
            loaded = model_dict['engram'].load_memory(memory_path)
            logger.info(f"Engram memory loaded={loaded} path={memory_path}")
        logger.info(
            f"Engram enabled: audio_dim={_cfg_get(engram_cfg, 'audio_dim', 384)}, "
            f"unet_dim={output_channels}, latent_size={latent_size}"
        )

    model_dict['net'] = Net(model_dict['unet'], model_dict['adapter'], model_dict['engram'])

    # 根据配置选择训练音频后端。  
    audio_backend = getattr(cfg, "audio_backend", "whisper")
    if audio_backend == "whisper":
        model_dict['wav2vec'] = WhisperModel.from_pretrained(cfg.whisper_path).to(
            device=accelerator.device, dtype=weight_dtype).eval()
        model_dict['wav2vec'].requires_grad_(False)
    else:
        # 非 whisper 后端使用通用 mel 路径，不加载 Whisper 编码器。  
        model_dict['wav2vec'] = None
        logger.info(f"audio_backend={audio_backend}, using generic mel training path without WhisperModel")

    if cfg.solver.gradient_checkpointing:
        model_dict['unet'].enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    model_dict['trainable_params'] = list(filter(lambda p: p.requires_grad, model_dict['net'].parameters()))
    if accelerator.is_main_process:
        print('trainable params')
        for n, p in model_dict['net'].named_parameters():
            if p.requires_grad:
                print(n)

    model_dict['optimizer'] = optimizer_cls(
        model_dict['trainable_params'],
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    model_dict['scheduler_max_steps'] = cfg.solver.max_train_steps * cfg.solver.gradient_accumulation_steps
    model_dict['lr_scheduler'] = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=model_dict['optimizer'],
        num_warmup_steps=cfg.solver.lr_warmup_steps * cfg.solver.gradient_accumulation_steps,
        num_training_steps=model_dict['scheduler_max_steps'],
    )

    return model_dict

def initialize_dataloaders(cfg):
    """Initialize training and validation dataloaders"""
    dataloader_dict = {
        'train_dataset': None,
        'val_dataset': None,
        'train_dataloader': None,
        'val_dataloader': None
    }
    
    # 统一数据集特征提取路径配置。  
    feature_extractor_path = getattr(cfg, "feature_extractor_path", getattr(cfg, "whisper_path", "openai/whisper-tiny"))
    num_workers = int(cfg.data.num_workers)
    if os.name == "nt" and num_workers > 0:
        logger.warning(
            f"Windows detected: forcing DataLoader num_workers from {num_workers} to 0 for stability."
        )
        num_workers = 0
    val_num_workers = 0 if os.name == "nt" else 1
    dataloader_dict['train_dataset'] = PortraitDataset(cfg={
        'image_size': cfg.data.image_size,
        'T': cfg.data.n_sample_frames,
        "sample_method": cfg.data.sample_method,
        'top_k_ratio': cfg.data.top_k_ratio,
        "contorl_face_min_size": cfg.data.contorl_face_min_size,
        "dataset_key": cfg.data.dataset_key,
        "padding_pixel_mouth": cfg.padding_pixel_mouth,
        "feature_extractor_path": feature_extractor_path,
        "whisper_path": feature_extractor_path,  # 兼容旧字段。  
        "min_face_size": cfg.data.min_face_size,
        "cropping_jaw2edge_margin_mean": cfg.cropping_jaw2edge_margin_mean,
        "cropping_jaw2edge_margin_std": cfg.cropping_jaw2edge_margin_std,
        "crop_type": cfg.crop_type,
        "random_margin_method": cfg.random_margin_method,
    })

    dataloader_dict['train_dataloader'] = torch.utils.data.DataLoader(
        dataloader_dict['train_dataset'],
        batch_size=cfg.data.train_bs,
        shuffle=True,
        num_workers=num_workers,
    )
    
    dataloader_dict['val_dataset'] = PortraitDataset(cfg={
        'image_size': cfg.data.image_size,
        'T': cfg.data.n_sample_frames,
        "sample_method": cfg.data.sample_method,
        'top_k_ratio': cfg.data.top_k_ratio,
        "contorl_face_min_size": cfg.data.contorl_face_min_size,
        "dataset_key": cfg.data.dataset_key,
        "padding_pixel_mouth": cfg.padding_pixel_mouth,
        "feature_extractor_path": feature_extractor_path,
        "whisper_path": feature_extractor_path,  # 兼容旧字段。  
        "min_face_size": cfg.data.min_face_size,
        "cropping_jaw2edge_margin_mean": cfg.cropping_jaw2edge_margin_mean,
        "cropping_jaw2edge_margin_std": cfg.cropping_jaw2edge_margin_std,
        "crop_type": cfg.crop_type,
        "random_margin_method": cfg.random_margin_method,
    })

    dataloader_dict['val_dataloader'] = torch.utils.data.DataLoader(
        dataloader_dict['val_dataset'],
        batch_size=cfg.data.train_bs,
        shuffle=True,
        num_workers=val_num_workers,
    )
    
    return dataloader_dict

def initialize_loss_functions(cfg, accelerator, scheduler_max_steps):
    """Initialize loss functions and discriminators"""
    loss_dict = {
        'L1_loss': nn.L1Loss(reduction='mean'),
        'discriminator': None,
        'mouth_discriminator': None,
        'optimizer_D': None,
        'mouth_optimizer_D': None,
        'scheduler_D': None,
        'mouth_scheduler_D': None,
        'disc_scales': None,
        'discriminator_full': None,
        'mouth_discriminator_full': None
    }
    
    if cfg.loss_params.gan_loss > 0:
        loss_dict['discriminator'] = MultiScaleDiscriminator(
            **cfg.model_params.discriminator_params).to(accelerator.device)
        loss_dict['discriminator_full'] = DiscriminatorFullModel(loss_dict['discriminator'])
        loss_dict['disc_scales'] = cfg.model_params.discriminator_params.scales
        loss_dict['optimizer_D'] = optim.AdamW(
            loss_dict['discriminator'].parameters(),
            lr=cfg.discriminator_train_params.lr,
            weight_decay=cfg.discriminator_train_params.weight_decay,
            betas=cfg.discriminator_train_params.betas,
            eps=cfg.discriminator_train_params.eps)
        loss_dict['scheduler_D'] = CosineAnnealingLR(
            loss_dict['optimizer_D'],
            T_max=scheduler_max_steps,
            eta_min=1e-6
        )

    if cfg.loss_params.mouth_gan_loss > 0:
        loss_dict['mouth_discriminator'] = MultiScaleDiscriminator(
            **cfg.model_params.discriminator_params).to(accelerator.device)
        loss_dict['mouth_discriminator_full'] = DiscriminatorFullModel(loss_dict['mouth_discriminator'])
        loss_dict['mouth_optimizer_D'] = optim.AdamW(
            loss_dict['mouth_discriminator'].parameters(),
            lr=cfg.discriminator_train_params.lr,
            weight_decay=cfg.discriminator_train_params.weight_decay,
            betas=cfg.discriminator_train_params.betas,
            eps=cfg.discriminator_train_params.eps)
        loss_dict['mouth_scheduler_D'] = CosineAnnealingLR(
            loss_dict['mouth_optimizer_D'],
            T_max=scheduler_max_steps,
            eta_min=1e-6
        )
        
    return loss_dict

def initialize_syncnet(cfg, accelerator, weight_dtype):
    """Initialize SyncNet model"""
    if cfg.loss_params.sync_loss > 0 or cfg.use_adapted_weight:
        if cfg.data.n_sample_frames != 16:
            raise ValueError(
                f"Invalid n_sample_frames {cfg.data.n_sample_frames} for sync_loss, it should be 16."
            )
        syncnet_config = OmegaConf.load(cfg.syncnet_config_path)
        syncnet = SyncNet(OmegaConf.to_container(
            syncnet_config.model)).to(accelerator.device)
        print(
            f"Load SyncNet checkpoint from: {syncnet_config.ckpt.inference_ckpt_path}")
        checkpoint = torch.load(
            syncnet_config.ckpt.inference_ckpt_path, map_location=accelerator.device)
        syncnet.load_state_dict(checkpoint["state_dict"])
        syncnet.to(dtype=weight_dtype)
        syncnet.requires_grad_(False)
        syncnet.eval()
        return syncnet
    return None

def initialize_vgg(cfg, accelerator):
    """Initialize VGG model"""
    if cfg.loss_params.vgg_loss > 0:
        vgg_IN = vgg_face.Vgg19().to(accelerator.device,)
        pyramid = vgg_face.ImagePyramide(
            cfg.loss_params.pyramid_scale, 3).to(accelerator.device)
        vgg_IN.eval()
        downsampler = Interpolate(
            size=(224, 224), mode='bilinear', align_corners=False).to(accelerator.device)
        return vgg_IN, pyramid, downsampler
    return None, None, None

def validation(
    cfg,
    val_dataloader,
    net,
    vae,
    wav2vec,
    accelerator,
    save_dir,
    global_step,
    weight_dtype,
    syncnet_score=1,
    return_metrics=False,
):
    """Validation function for model evaluation"""
    # 设置模型为验证模式。  
    net.eval()
    # 初始化验证指标字典。  
    metrics = {
        "val_l1_train": None,
        "val_l1_infer": None,
        "val_precision": None,
        "val_recall": None,
        "val_f1": None,
    }
    for batch in val_dataloader:
        # The same ref_latents
        ref_pixel_values = batch["pixel_values_ref_img"].to(weight_dtype).to(
            accelerator.device, non_blocking=True
        )
        pixel_values = batch["pixel_values_vid"].to(weight_dtype).to(
            accelerator.device, non_blocking=True
        )
        bsz, num_frames, c, h, w = ref_pixel_values.shape

        audio_prompts = process_audio_features(cfg, batch, wav2vec, bsz, num_frames, weight_dtype)
        # audio feature for unet
        audio_prompts = rearrange(
            audio_prompts, 
            'b f c h w-> (b f) c h w'
        )
        audio_prompts = rearrange(
            audio_prompts, 
            '(b f) c h w -> (b f) (c h) w', 
            b=bsz
        )
        audio_token_keep = int(getattr(cfg, "audio_token_keep", 0))
        if audio_token_keep > 0:
            audio_prompts = reduce_audio_tokens(audio_prompts, audio_token_keep)
        # different masked_latents
        image_pred_train = get_image_pred(
            pixel_values, ref_pixel_values, audio_prompts, vae, net, weight_dtype)
        image_pred_infer = get_image_pred(
            ref_pixel_values, ref_pixel_values, audio_prompts, vae, net, weight_dtype)

        # 计算验证损失指标（使用单批次快速验证）。  
        gt_frames = rearrange(pixel_values, 'b f c h w -> (b f) c h w').float()
        infer_frames = rearrange(ref_pixel_values, 'b f c h w -> (b f) c h w').float()
        metrics["val_l1_train"] = F.l1_loss(image_pred_train.float(), gt_frames).item()
        metrics["val_l1_infer"] = F.l1_loss(image_pred_infer.float(), infer_frames).item()
        # 计算口型区域二值分类指标。  
        mouth_mask = rearrange(batch["pixel_values_face_mask"], "b f c h w -> (b f) c h w").to(gt_frames.device).float()
        precision, recall, f1 = _compute_binary_prf1(image_pred_train.float(), gt_frames, mouth_mask, threshold=0.5)
        metrics["val_precision"] = precision
        metrics["val_recall"] = recall
        metrics["val_f1"] = f1

        process_and_save_images(
            batch,
            image_pred_train,
            image_pred_infer,
            save_dir,
            global_step,
            accelerator,
            cfg.num_images_to_keep,
            syncnet_score
        )
        # only infer 1 image in validation
        break
    # 恢复训练模式。  
    net.train()
    # 根据开关返回验证指标。  
    if return_metrics:
        return metrics
