import argparse
import diffusers
import logging
import math
import os
import time
import csv
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import warnings
import random

from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate import InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from datetime import datetime
from datetime import timedelta

from diffusers.utils import check_min_version
from einops import rearrange
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from musetalk.utils.utils import (
    delete_additional_ckpt, 
    seed_everything, 
    get_mouth_region,
    process_audio_features,
    save_models,
    reduce_audio_tokens,
)
from musetalk.loss.basic_loss import set_requires_grad
from musetalk.loss.pmf_loss import PixelSpacePMFLoss
from musetalk.loss.syncnet import get_sync_loss
from musetalk.utils.training_utils import (
    initialize_models_and_optimizers,
    initialize_dataloaders,
    initialize_loss_functions,
    initialize_syncnet,
    initialize_vgg,
    validation
)

logger = get_logger(__name__, log_level="INFO")
warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")


def _plot_results_curve(results_csv_path, output_png_path):
    """从 results.csv 绘制训练曲线图。"""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    # 读取 CSV 内容。  
    rows = []
    with open(results_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if len(rows) == 0:
        return
    # 提取主曲线。  
    steps = [int(r["global_step"]) for r in rows]
    train_loss = [float(r["epoch_avg_total_loss"]) for r in rows]
    val_l1 = [float(r["val_l1_train"]) if r["val_l1_train"] not in ("", "None", "nan") else float("nan") for r in rows]
    val_f1 = [float(r["val_f1"]) if r["val_f1"] not in ("", "None", "nan") else float("nan") for r in rows]
    # 绘图并保存。  
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps, train_loss, label="epoch_avg_total_loss")
    axes[0].plot(steps, val_l1, label="val_l1_train")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("global_step")
    axes[0].legend()
    axes[1].plot(steps, val_f1, label="val_f1")
    axes[1].set_title("F1 Curve")
    axes[1].set_xlabel("global_step")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main(cfg):
    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)

    kwargs = DistributedDataParallelKwargs()
    process_group_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=5400))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        log_with=["tensorboard", LoggerType.TENSORBOARD],
        project_dir=os.path.join(save_dir, "./tensorboard"),
        kwargs_handlers=[kwargs, process_group_kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        print('cfg.seed', cfg.seed, accelerator.process_index)
        seed_everything(cfg.seed + accelerator.process_index)

    weight_dtype = torch.float32

    model_dict = initialize_models_and_optimizers(cfg, accelerator, weight_dtype)
    dataloader_dict = initialize_dataloaders(cfg)
    loss_dict = initialize_loss_functions(cfg, accelerator, model_dict['scheduler_max_steps'])
    syncnet = initialize_syncnet(cfg, accelerator, weight_dtype)
    vgg_IN, pyramid, downsampler = initialize_vgg(cfg, accelerator)

    # Prepare everything with our `accelerator`.
    model_dict['net'], model_dict['optimizer'], model_dict['lr_scheduler'], dataloader_dict['train_dataloader'], dataloader_dict['val_dataloader'] = accelerator.prepare(
        model_dict['net'], model_dict['optimizer'], model_dict['lr_scheduler'], dataloader_dict['train_dataloader'], dataloader_dict['val_dataloader']
    )
    print("length train/val", len(dataloader_dict['train_dataloader']), len(dataloader_dict['val_dataloader']))
    
    # Calculate training steps and epochs
    num_update_steps_per_epoch = math.ceil(
        len(dataloader_dict['train_dataloader']) / cfg.solver.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # Initialize trackers on the main process
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            cfg.exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )

    # 初始化结果记录文件路径。  
    results_csv_path = os.path.join(save_dir, "results.csv")
    best_model_path = os.path.join(save_dir, "best_unet.pt")
    best_meta_path = os.path.join(save_dir, "best_metrics.json")
    latest_metrics_path = os.path.join(save_dir, "latest_metrics.json")
    results_curve_path = os.path.join(save_dir, "results_curve.png")
    # 初始化最佳验证指标。  
    best_val_l1 = float("inf")
    # 记录最新验证指标。  
    latest_val_metrics = {
        "val_l1_train": None,
        "val_l1_infer": None,
        "val_precision": None,
        "val_recall": None,
        "val_f1": None,
    }
    # 初始化早停参数。  
    early_stop_cfg = getattr(cfg, "early_stopping", None)
    early_stop_enabled = bool(getattr(early_stop_cfg, "enabled", False)) if early_stop_cfg is not None else False
    early_stop_metric = str(getattr(early_stop_cfg, "metric", "val_l1_train")) if early_stop_cfg is not None else "val_l1_train"
    early_stop_mode = str(getattr(early_stop_cfg, "mode", "min")) if early_stop_cfg is not None else "min"
    early_stop_patience = int(getattr(early_stop_cfg, "patience", 10)) if early_stop_cfg is not None else 10
    early_stop_min_delta = float(getattr(early_stop_cfg, "min_delta", 0.0)) if early_stop_cfg is not None else 0.0
    early_stop_bad_count = 0
    early_stop_best = float("inf") if early_stop_mode == "min" else -float("inf")
    should_stop_early = False

    # 初始化 CSV 表头。  
    if accelerator.is_main_process and not os.path.exists(results_csv_path):
        with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "global_step",
                    "epoch_avg_total_loss",
                    "epoch_avg_l1_loss",
                    "epoch_avg_vgg_loss",
                    "epoch_avg_sync_loss",
                    "epoch_avg_mse_loss",
                    "epoch_avg_pmf_loss",
                    "val_l1_train",
                    "val_l1_infer",
                    "val_precision",
                    "val_recall",
                    "val_f1",
                    "best_val_l1_so_far",
                ],
            )
            writer.writeheader()

    # Calculate total batch size
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    # Log training information
    logger.info("***** Running training *****")
    logger.info(f"Num Epochs = {num_train_epochs}")
    logger.info(f"Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps = {cfg.solver.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Load checkpoint if resuming training
    if cfg.resume_from_checkpoint:
        resume_dir = save_dir
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        if len(dirs) > 0:
            path = dirs[-1]
            accelerator.load_state(os.path.join(resume_dir, path))
            accelerator.print(f"Resuming from checkpoint {path}")
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch

    # Initialize progress bar
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    
    # Log model types
    print("log type of models")
    print("unet", model_dict['unet'].dtype)
    print("vae", model_dict['vae'].dtype)
    if model_dict['wav2vec'] is not None:
        print("wav2vec", model_dict['wav2vec'].dtype)
    else:
        print("wav2vec", "None (generic mel backend)")
    # 初始化可选 pMF 损失。  
    pmf_weight = float(getattr(cfg.loss_params, "pmf_loss", 0.0))
    pmf_loss_fn = PixelSpacePMFLoss().to(accelerator.device) if pmf_weight > 0 else None
    # 初始化可选 MSE 损失。  
    mse_weight = float(getattr(cfg.loss_params, "mse_loss", 0.0))
    mse_loss_fn = nn.MSELoss(reduction="mean") if mse_weight > 0 else None

    def get_ganloss_weight(step):
        """Calculate GAN loss weight based on training step"""
        if step < cfg.discriminator_train_params.start_gan:
            return 0.0
        else:
            return 1.0
        
    # Training loop
    for epoch in range(first_epoch, num_train_epochs):
        # Set models to training mode
        model_dict['unet'].train()
        if cfg.loss_params.gan_loss > 0:
            loss_dict['discriminator'].train()
        if cfg.loss_params.mouth_gan_loss > 0:
            loss_dict['mouth_discriminator'].train()

        # Initialize loss accumulators
        train_loss = 0.0
        train_loss_D = 0.0
        train_loss_D_mouth = 0.0
        l1_loss_accum = 0.0
        vgg_loss_accum = 0.0
        gan_loss_accum = 0.0
        gan_loss_accum_mouth = 0.0
        fm_loss_accum = 0.0
        sync_loss_accum = 0.0
        mse_loss_accum = 0.0
        pmf_loss_accum = 0.0
        adapted_weight_accum = 0.0
        # 记录 epoch 级平均指标。  
        epoch_total_loss_sum = 0.0
        epoch_l1_loss_sum = 0.0
        epoch_vgg_loss_sum = 0.0
        epoch_sync_loss_sum = 0.0
        epoch_mse_loss_sum = 0.0
        epoch_pmf_loss_sum = 0.0
        epoch_step_count = 0

        t_data_start = time.time()
        for step, batch in enumerate(dataloader_dict['train_dataloader']):
            t_data = time.time() - t_data_start
            t_model_start = time.time()

            with torch.no_grad():
                # Process input data
                pixel_values = batch["pixel_values_vid"].to(weight_dtype).to(
                    accelerator.device, 
                    non_blocking=True
                )
                bsz, num_frames, c, h, w = pixel_values.shape
                
                # Process reference images
                ref_pixel_values = batch["pixel_values_ref_img"].to(weight_dtype).to(
                    accelerator.device, 
                    non_blocking=True
                )
                
                # Get face mask for GAN
                pixel_values_face_mask = batch['pixel_values_face_mask']
                
                # Process audio features
                audio_prompts = process_audio_features(cfg, batch, model_dict['wav2vec'], bsz, num_frames, weight_dtype)
                
                # Initialize adapted weight
                adapted_weight = 1
                
                # Process sync loss if enabled
                if cfg.loss_params.sync_loss > 0:
                    mels = batch['mel']
                    # Prepare frames for latentsync (combine channels and frames)
                    gt_frames = rearrange(pixel_values, 'b f c h w-> b (f c) h w')
                    # Use lower half of face for latentsync
                    height = gt_frames.shape[2]
                    gt_frames = gt_frames[:, :, height // 2:, :]
                    
                    # Get audio embeddings
                    audio_embed = syncnet.get_audio_embed(mels)
                    
                    # Calculate adapted weight based on audio-visual similarity
                    if cfg.use_adapted_weight:
                        vision_embed_gt = syncnet.get_vision_embed(gt_frames)
                        image_audio_sim_gt = F.cosine_similarity(
                            audio_embed, 
                            vision_embed_gt, 
                            dim=1
                        )[0]
                        
                        if image_audio_sim_gt < 0.05 or image_audio_sim_gt > 0.65:
                            if cfg.adapted_weight_type == "cut_off":
                                adapted_weight = 0.0  # Skip this batch
                                print(
                                    f"\nThe i-a similarity in step {global_step} is {image_audio_sim_gt}, set adapted_weight to {adapted_weight}.")
                            elif cfg.adapted_weight_type == "linear":
                                adapted_weight = image_audio_sim_gt
                            else:
                                print(f"unknown adapted_weight_type: {cfg.adapted_weight_type}")
                                adapted_weight = 1
                    
                    # Random frame selection for memory efficiency
                    max_start = 16 - cfg.num_backward_frames
                    frames_left_index = random.randint(0, max_start) if max_start > 0 else 0
                    frames_right_index = frames_left_index + cfg.num_backward_frames         
                else:
                    frames_left_index = 0
                    frames_right_index = cfg.data.n_sample_frames

                # Extract frames for backward pass
                pixel_values_backward = pixel_values[:, frames_left_index:frames_right_index, ...]
                ref_pixel_values_backward = ref_pixel_values[:, frames_left_index:frames_right_index, ...]
                pixel_values_face_mask_backward = pixel_values_face_mask[:, frames_left_index:frames_right_index, ...]
                audio_prompts_backward = audio_prompts[:, frames_left_index:frames_right_index, ...]
                
                # Encode target images
                frames = rearrange(pixel_values_backward, 'b f c h w-> (b f) c h w')
                vae_scaling_raw = getattr(model_dict['vae'].config, "scaling_factor", None)
                vae_shift_raw = getattr(model_dict['vae'].config, "shift_factor", None)
                if vae_scaling_raw is None:
                    vae_scaling_raw = 0.3611 if getattr(cfg, "vae_type", "sd-vae") == "flux-vae" else 0.18215
                if vae_shift_raw is None:
                    vae_shift_raw = 0.1159 if getattr(cfg, "vae_type", "sd-vae") == "flux-vae" else 0.0
                vae_scaling_factor = float(vae_scaling_raw)
                vae_shift_factor = float(vae_shift_raw)
                latents = model_dict['vae'].encode(frames).latent_dist.mode()
                latents = (latents - vae_shift_factor) * vae_scaling_factor
                latents = latents.float()

                # Create masked images
                masked_pixel_values = pixel_values_backward.clone()
                masked_pixel_values[:, :, :, h//2:, :] = -1
                masked_frames = rearrange(masked_pixel_values, 'b f c h w -> (b f) c h w')
                masked_latents = model_dict['vae'].encode(masked_frames).latent_dist.mode()
                masked_latents = (masked_latents - vae_shift_factor) * vae_scaling_factor
                masked_latents = masked_latents.float()

                # Encode reference images
                ref_frames = rearrange(ref_pixel_values_backward, 'b f c h w-> (b f) c h w')
                ref_latents = model_dict['vae'].encode(ref_frames).latent_dist.mode()
                ref_latents = (ref_latents - vae_shift_factor) * vae_scaling_factor
                ref_latents = ref_latents.float()

                # Prepare face mask and audio features
                pixel_values_face_mask_backward = rearrange(
                    pixel_values_face_mask_backward, 
                    "b f c h w -> (b f) c h w"
                )
                audio_prompts_backward = rearrange(
                    audio_prompts_backward, 
                    'b f c h w-> (b f) c h w'
                )
                audio_prompts_backward = rearrange(
                    audio_prompts_backward, 
                    '(b f) c h w -> (b f) (c h) w', 
                    b=bsz
                )
                audio_token_keep = int(getattr(cfg, "audio_token_keep", 0))
                if audio_token_keep > 0:
                    audio_prompts_backward = reduce_audio_tokens(audio_prompts_backward, audio_token_keep)

            # Apply reference dropout (currently inactive)
            dropout = nn.Dropout(p=cfg.ref_dropout_rate)
            ref_latents = dropout(ref_latents)

            # Prepare model inputs
            input_latents = torch.cat([masked_latents, ref_latents], dim=1)
            input_latents = input_latents.to(weight_dtype)
            timesteps = torch.tensor([0], device=input_latents.device)

            # Forward pass
            latents_pred = model_dict['net'](
                input_latents,
                timesteps,
                audio_prompts_backward,
            )
            latents_pred = latents_pred / vae_scaling_factor + vae_shift_factor
            image_pred = model_dict['vae'].decode(latents_pred).sample
            
            # Convert to float
            image_pred = image_pred.float()
            frames = frames.float()
            
            # Calculate L1 loss
            l1_loss = loss_dict['L1_loss'](frames, image_pred)
            l1_loss_accum += l1_loss.item()
            loss = cfg.loss_params.l1_loss * l1_loss * adapted_weight
            # 计算可选 MSE 损失。  
            if mse_loss_fn is not None and mse_weight > 0:
                mse_loss = mse_loss_fn(image_pred, frames)
                mse_loss_accum += mse_loss.item()
                epoch_mse_loss_sum += mse_loss.item()
                loss += mse_weight * mse_loss * adapted_weight
            # 计算可选 pMF 损失（像素空间）。  
            if pmf_loss_fn is not None and pmf_weight > 0:
                pmf_loss = pmf_loss_fn(image_pred, frames)
                pmf_loss_accum += pmf_loss.item()
                loss += pmf_weight * pmf_loss * adapted_weight
                epoch_pmf_loss_sum += pmf_loss.item()

            # Process mouth GAN loss if enabled
            if cfg.loss_params.mouth_gan_loss > 0:
                frames_mouth, image_pred_mouth = get_mouth_region(
                    frames, 
                    image_pred, 
                    pixel_values_face_mask_backward
                )
                pyramide_real_mouth = pyramid(downsampler(frames_mouth))
                pyramide_generated_mouth = pyramid(downsampler(image_pred_mouth))

            # Process VGG loss if enabled
            if cfg.loss_params.vgg_loss > 0:
                pyramide_real = pyramid(downsampler(frames))
                pyramide_generated = pyramid(downsampler(image_pred))

                loss_IN = 0
                for scale in cfg.loss_params.pyramid_scale:
                    x_vgg = vgg_IN(pyramide_generated['prediction_' + str(scale)])
                    y_vgg = vgg_IN(pyramide_real['prediction_' + str(scale)])
                    for i, weight in enumerate(cfg.loss_params.vgg_layer_weight):
                        value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                        loss_IN += weight * value
                loss_IN /= sum(cfg.loss_params.vgg_layer_weight)
                loss += loss_IN * cfg.loss_params.vgg_loss * adapted_weight
                vgg_loss_accum += loss_IN.item()

            # Process GAN loss if enabled
            if cfg.loss_params.gan_loss > 0:
                set_requires_grad(loss_dict['discriminator'], False)
                loss_G = 0.
                discriminator_maps_generated = loss_dict['discriminator'](pyramide_generated)
                discriminator_maps_real = loss_dict['discriminator'](pyramide_real)

                for scale in loss_dict['disc_scales']:
                    key = 'prediction_map_%s' % scale
                    value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                    loss_G += value
                gan_loss_accum += loss_G.item()

                loss += loss_G * cfg.loss_params.gan_loss * get_ganloss_weight(global_step) * adapted_weight

                # Process feature matching loss if enabled
                if cfg.loss_params.fm_loss[0] > 0:
                    L_feature_matching = 0.
                    for scale in loss_dict['disc_scales']:
                        key = 'feature_maps_%s' % scale
                        for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                            value = torch.abs(a - b).mean()
                            L_feature_matching += value * cfg.loss_params.fm_loss[i]
                    loss += L_feature_matching * adapted_weight
                    fm_loss_accum += L_feature_matching.item()

            # Process mouth GAN loss if enabled
            if cfg.loss_params.mouth_gan_loss > 0:
                set_requires_grad(loss_dict['mouth_discriminator'], False)
                loss_G = 0.
                mouth_discriminator_maps_generated = loss_dict['mouth_discriminator'](pyramide_generated_mouth)
                mouth_discriminator_maps_real = loss_dict['mouth_discriminator'](pyramide_real_mouth)

                for scale in loss_dict['disc_scales']:
                    key = 'prediction_map_%s' % scale
                    value = ((1 - mouth_discriminator_maps_generated[key]) ** 2).mean()
                    loss_G += value
                gan_loss_accum_mouth += loss_G.item()

                loss += loss_G * cfg.loss_params.mouth_gan_loss * get_ganloss_weight(global_step) * adapted_weight

                # Process feature matching loss for mouth if enabled
                if cfg.loss_params.fm_loss[0] > 0:
                    L_feature_matching = 0.
                    for scale in loss_dict['disc_scales']:
                        key = 'feature_maps_%s' % scale
                        for i, (a, b) in enumerate(zip(mouth_discriminator_maps_real[key], mouth_discriminator_maps_generated[key])):
                            value = torch.abs(a - b).mean()
                            L_feature_matching += value * cfg.loss_params.fm_loss[i]
                    loss += L_feature_matching * adapted_weight
                    fm_loss_accum += L_feature_matching.item()
        
            # Process sync loss if enabled
            if cfg.loss_params.sync_loss > 0:
                pred_frames = rearrange(
                    image_pred, '(b f) c h w-> b (f c) h w', f=pixel_values_backward.shape[1])
                pred_frames = pred_frames[:, :, height // 2 :, :]
                sync_loss, image_audio_sim_pred = get_sync_loss(
                    audio_embed, 
                    gt_frames, 
                    pred_frames, 
                    syncnet, 
                    adapted_weight,
                    frames_left_index=frames_left_index,
                    frames_right_index=frames_right_index,
                )
                sync_loss_accum += sync_loss.item()
                loss += sync_loss * cfg.loss_params.sync_loss * adapted_weight

            # Backward pass
            avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
            train_loss += avg_loss.item()
            epoch_total_loss_sum += avg_loss.item()
            epoch_l1_loss_sum += l1_loss.item()
            if cfg.loss_params.vgg_loss > 0:
                epoch_vgg_loss_sum += loss_IN.item()
            if cfg.loss_params.sync_loss > 0:
                epoch_sync_loss_sum += sync_loss.item()
            epoch_step_count += 1
            accelerator.backward(loss)

            # Train discriminator if GAN loss is enabled
            if cfg.loss_params.gan_loss > 0:
                set_requires_grad(loss_dict['discriminator'], True)
                loss_D = loss_dict['discriminator_full'](frames, image_pred.detach())
                avg_loss_D = accelerator.gather(loss_D.repeat(cfg.data.train_bs)).mean()
                train_loss_D += avg_loss_D.item() / 1
                loss_D = loss_D * get_ganloss_weight(global_step) * adapted_weight
                accelerator.backward(loss_D)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        loss_dict['discriminator'].parameters(), cfg.solver.max_grad_norm)
                if (global_step + 1) % cfg.solver.gradient_accumulation_steps == 0:
                    loss_dict['optimizer_D'].step()
                    loss_dict['scheduler_D'].step()
                    loss_dict['optimizer_D'].zero_grad()

            # Train mouth discriminator if mouth GAN loss is enabled
            if cfg.loss_params.mouth_gan_loss > 0:
                set_requires_grad(loss_dict['mouth_discriminator'], True)
                mouth_loss_D = loss_dict['mouth_discriminator_full'](
                    frames_mouth, image_pred_mouth.detach())
                avg_mouth_loss_D = accelerator.gather(
                    mouth_loss_D.repeat(cfg.data.train_bs)).mean()
                train_loss_D_mouth += avg_mouth_loss_D.item() / 1
                mouth_loss_D = mouth_loss_D * get_ganloss_weight(global_step) * adapted_weight
                accelerator.backward(mouth_loss_D)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        loss_dict['mouth_discriminator'].parameters(), cfg.solver.max_grad_norm)
                if (global_step + 1) % cfg.solver.gradient_accumulation_steps == 0:
                    loss_dict['mouth_optimizer_D'].step()
                    loss_dict['mouth_scheduler_D'].step()
                    loss_dict['mouth_optimizer_D'].zero_grad()

            # Update main model
            if (global_step + 1) % cfg.solver.gradient_accumulation_steps == 0:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model_dict['trainable_params'],
                        cfg.solver.max_grad_norm,
                    )
                model_dict['optimizer'].step()
                model_dict['lr_scheduler'].step()
                model_dict['optimizer'].zero_grad()

            # Update progress and log metrics
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({
                    "train_loss": train_loss,
                    "train_loss_D": train_loss_D,
                    "train_loss_D_mouth": train_loss_D_mouth,
                    "l1_loss": l1_loss_accum,
                    "vgg_loss": vgg_loss_accum,
                    "gan_loss": gan_loss_accum,
                    "fm_loss": fm_loss_accum,
                    "sync_loss": sync_loss_accum,
                    "mse_loss": mse_loss_accum,
                    "pmf_loss": pmf_loss_accum,
                    "adapted_weight": adapted_weight_accum,
                    "lr": model_dict['lr_scheduler'].get_last_lr()[0],
                }, step=global_step)

                # Reset loss accumulators
                train_loss = 0.0
                l1_loss_accum = 0.0
                vgg_loss_accum = 0.0
                gan_loss_accum = 0.0
                fm_loss_accum = 0.0
                sync_loss_accum = 0.0
                mse_loss_accum = 0.0
                pmf_loss_accum = 0.0
                adapted_weight_accum = 0.0
                train_loss_D = 0.0
                train_loss_D_mouth = 0.0

                # Run validation if needed
                if global_step % cfg.val_freq == 0 or global_step == 10:
                    try:
                        val_metrics = validation(
                            cfg,
                            dataloader_dict['val_dataloader'],
                            model_dict['net'],
                            model_dict['vae'],
                            model_dict['wav2vec'],
                            accelerator,
                            save_dir,
                            global_step,
                            weight_dtype,
                            syncnet_score=adapted_weight,
                            return_metrics=True,
                        )
                        # 记录验证指标并写入日志。  
                        if isinstance(val_metrics, dict):
                            latest_val_metrics = val_metrics
                            accelerator.log(
                                {
                                    "val_l1_train": val_metrics.get("val_l1_train", 0.0),
                                    "val_l1_infer": val_metrics.get("val_l1_infer", 0.0),
                                    "val_precision": val_metrics.get("val_precision", 0.0),
                                    "val_recall": val_metrics.get("val_recall", 0.0),
                                    "val_f1": val_metrics.get("val_f1", 0.0),
                                },
                                step=global_step,
                            )
                            # 每次验证都写最新指标文件。  
                            if accelerator.is_main_process:
                                with open(latest_metrics_path, "w", encoding="utf-8") as f:
                                    json.dump(
                                        {
                                            "global_step": int(global_step),
                                            "epoch": int(epoch),
                                            "metrics": latest_val_metrics,
                                        },
                                        f,
                                        ensure_ascii=False,
                                        indent=2,
                                    )
                            # 主进程保存当前最佳模型。  
                            cur_val_l1 = val_metrics.get("val_l1_train", None)
                            if (
                                accelerator.is_main_process
                                and cur_val_l1 is not None
                                and cur_val_l1 < best_val_l1
                            ):
                                best_val_l1 = cur_val_l1
                                unwarp_net = accelerator.unwrap_model(model_dict['net'])
                                torch.save(unwarp_net.unet.state_dict(), best_model_path)
                                with open(best_meta_path, "w", encoding="utf-8") as f:
                                    json.dump(
                                        {
                                            "best_val_l1_train": float(best_val_l1),
                                            "global_step": int(global_step),
                                            "epoch": int(epoch),
                                            "val_metrics": latest_val_metrics,
                                        },
                                        f,
                                        ensure_ascii=False,
                                        indent=2,
                                    )
                            # 早停判定。  
                            monitored_val = val_metrics.get(early_stop_metric, None)
                            if early_stop_enabled and monitored_val is not None:
                                if early_stop_mode == "min":
                                    improved = monitored_val < (early_stop_best - early_stop_min_delta)
                                else:
                                    improved = monitored_val > (early_stop_best + early_stop_min_delta)
                                if improved:
                                    early_stop_best = monitored_val
                                    early_stop_bad_count = 0
                                else:
                                    early_stop_bad_count += 1
                                    if early_stop_bad_count >= early_stop_patience:
                                        should_stop_early = True
                                        logger.info(
                                            f"Early stopping triggered: metric={early_stop_metric}, value={monitored_val}, bad_count={early_stop_bad_count}"
                                        )
                    except Exception as e:
                        print(f"An error occurred during validation: {e}")

                # Save checkpoint if needed
                if global_step % cfg.checkpointing_steps == 0:
                    save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                    try:
                        start_time = time.time()
                        if accelerator.is_main_process:
                            save_models(
                                accelerator, 
                                model_dict['net'],
                                save_dir, 
                                global_step, 
                                cfg, 
                                logger=logger
                            )
                            delete_additional_ckpt(save_dir, cfg.total_limit)
                        elapsed_time = time.time() - start_time
                        if elapsed_time > 300:
                            print(f"Skipping storage as it took too long in step {global_step}.")
                        else:
                            print(f"Resume states saved at {save_dir} successfully in {elapsed_time}s.")
                    except Exception as e:
                        print(f"Error when saving model in step {global_step}:", e)

            # Update progress bar
            t_model = time.time() - t_model_start
            logs = {
                "step_loss": loss.detach().item(),
                "lr": model_dict['lr_scheduler'].get_last_lr()[0],
                "td": f"{t_data:.2f}s",
                "tm": f"{t_model:.2f}s",
            }
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break
            if should_stop_early:
                break

        # Save model after each epoch
        if (epoch + 1) % cfg.save_model_epoch_interval == 0:
            try:
                start_time = time.time()
                if accelerator.is_main_process:
                    save_models(accelerator, model_dict['net'], save_dir, global_step, cfg)
                    accelerator.save_state(save_path)
                elapsed_time = time.time() - start_time
                if elapsed_time > 120:
                    print(f"Skipping storage as it took too long in step {global_step}.")
                else:
                    print(f"Model saved successfully in {elapsed_time}s.")
            except Exception as e:
                print(f"Error when saving model in step {global_step}:", e)
        accelerator.wait_for_everyone()

        # 每个 epoch 结束后写入结果 CSV。  
        if accelerator.is_main_process:
            divisor = max(1, epoch_step_count)
            with open(results_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch",
                        "global_step",
                        "epoch_avg_total_loss",
                        "epoch_avg_l1_loss",
                        "epoch_avg_vgg_loss",
                        "epoch_avg_sync_loss",
                        "epoch_avg_mse_loss",
                        "epoch_avg_pmf_loss",
                        "val_l1_train",
                        "val_l1_infer",
                        "val_precision",
                        "val_recall",
                        "val_f1",
                        "best_val_l1_so_far",
                    ],
                )
                writer.writerow(
                    {
                        "epoch": int(epoch),
                        "global_step": int(global_step),
                        "epoch_avg_total_loss": float(epoch_total_loss_sum / divisor),
                        "epoch_avg_l1_loss": float(epoch_l1_loss_sum / divisor),
                        "epoch_avg_vgg_loss": float(epoch_vgg_loss_sum / divisor),
                        "epoch_avg_sync_loss": float(epoch_sync_loss_sum / divisor),
                        "epoch_avg_mse_loss": float(epoch_mse_loss_sum / divisor),
                        "epoch_avg_pmf_loss": float(epoch_pmf_loss_sum / divisor),
                        "val_l1_train": latest_val_metrics.get("val_l1_train", None),
                        "val_l1_infer": latest_val_metrics.get("val_l1_infer", None),
                        "val_precision": latest_val_metrics.get("val_precision", None),
                        "val_recall": latest_val_metrics.get("val_recall", None),
                        "val_f1": latest_val_metrics.get("val_f1", None),
                        "best_val_l1_so_far": float(best_val_l1) if best_val_l1 < float("inf") else None,
                    }
                )
        # 每个 epoch 后更新曲线图。  
        if accelerator.is_main_process:
            _plot_results_curve(results_csv_path, results_curve_path)
        # 触发早停则结束训练。  
        if should_stop_early:
            break

    # End training
    if accelerator.is_main_process:
        _plot_results_curve(results_csv_path, results_curve_path)
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/training/stage2.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)
