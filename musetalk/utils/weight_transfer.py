"""
Weight Transfer Utility for MuseTalk v1.5 → v2.0 Migration

Handles:
- Channel adapter initialization (4ch → 16ch VAE)
- UNet weight transfer with dimension matching
- Gradual unfreezing strategy support
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import os


def transfer_unet_weights(unet_v15: nn.Module, unet_v20: nn.Module,
                          skip_layers: List[str] = None,
                          verbose: bool = True) -> nn.Module:
    """
    Transfer weights from MuseTalk 1.5 UNet to 2.0 UNet.
    
    Strategy:
    1. Copy matching layers (conv, norm, attention)
    2. Skip input/output projection layers (dimensions changed)
    3. Initialize new layers (mHC, Engram, Gated Attention)
    
    Args:
        unet_v15: Source UNet (v1.5)
        unet_v20: Target UNet (v2.0)
        skip_layers: Layer name patterns to skip
        verbose: Print transfer progress
        
    Returns:
        unet_v20 with transferred weights
    """
    skip_layers = skip_layers or [
        'conv_in', 'conv_out',  # Input/output changed for 16ch VAE
        'adapter', 'engram', 'gate', 'mhc'  # New v2.0 components
    ]
    
    state_dict_v15 = unet_v15.state_dict()
    state_dict_v20 = unet_v20.state_dict()
    
    transferred = []
    skipped = []
    initialized = []
    
    for name_v20, param_v20 in state_dict_v20.items():
        # Check if should skip
        should_skip = any(skip in name_v20 for skip in skip_layers)
        
        if should_skip:
            skipped.append(name_v20)
            continue
        
        # Try to find matching parameter in v1.5
        # Handle wrapper prefixes
        name_v15_candidates = [
            name_v20,
            name_v20.replace('base_unet.', ''),
            name_v20.replace('model.', ''),
            'model.' + name_v20,
        ]
        
        matched = False
        for name_v15 in name_v15_candidates:
            if name_v15 in state_dict_v15:
                param_v15 = state_dict_v15[name_v15]
                
                if param_v15.shape == param_v20.shape:
                    param_v20.data.copy_(param_v15.data)
                    transferred.append(name_v20)
                    matched = True
                    break
                else:
                    # Dimension mismatch - try partial transfer
                    if _try_partial_transfer(param_v15, param_v20):
                        transferred.append(f"{name_v20} (partial)")
                        matched = True
                        break
        
        if not matched:
            initialized.append(name_v20)
    
    if verbose:
        print(f"Weight Transfer Summary:")
        print(f"  Transferred: {len(transferred)} parameters")
        print(f"  Skipped: {len(skipped)} parameters")
        print(f"  Initialized (new): {len(initialized)} parameters")
    
    return unet_v20


def _try_partial_transfer(src: torch.Tensor, dst: torch.Tensor) -> bool:
    """
    Try to partially transfer weights when dimensions don't match exactly.
    
    Handles cases like:
    - Expanding channels (4 → 16)
    - Truncating channels (16 → 4)
    """
    try:
        if len(src.shape) != len(dst.shape):
            return False
        
        # For conv weights: (out_ch, in_ch, kH, kW)
        if len(src.shape) == 4:
            min_out = min(src.shape[0], dst.shape[0])
            min_in = min(src.shape[1], dst.shape[1])
            
            dst.data[:min_out, :min_in, :, :] = src.data[:min_out, :min_in, :, :]
            return True
        
        # For linear weights: (out, in)
        if len(src.shape) == 2:
            min_out = min(src.shape[0], dst.shape[0])
            min_in = min(src.shape[1], dst.shape[1])
            
            dst.data[:min_out, :min_in] = src.data[:min_out, :min_in]
            return True
        
        # For bias: (out,)
        if len(src.shape) == 1:
            min_size = min(src.shape[0], dst.shape[0])
            dst.data[:min_size] = src.data[:min_size]
            return True
        
        return False
        
    except Exception:
        return False


def initialize_adapter_from_pca(adapter: nn.Module,
                                sample_latents_16ch: torch.Tensor,
                                sample_latents_4ch: torch.Tensor,
                                verbose: bool = True):
    """
    Initialize channel adapter using PCA analysis.
    
    Projects 16-channel VAE latents to preserve maximum variance
    from 4-channel training distribution.
    
    Args:
        adapter: LatentChannelAdapter module
        sample_latents_16ch: Sample 16ch latents (N, 16, H, W)
        sample_latents_4ch: Corresponding 4ch latents (N, 4, H, W)
        verbose: Print progress
    """
    if verbose:
        print("Initializing adapter with PCA projection...")
    
    N, C16, H, W = sample_latents_16ch.shape
    _, C4, _, _ = sample_latents_4ch.shape
    
    # Flatten spatial dimensions
    flat_16 = sample_latents_16ch.permute(0, 2, 3, 1).reshape(-1, C16)  # (N*H*W, 16)
    flat_4 = sample_latents_4ch.permute(0, 2, 3, 1).reshape(-1, C4)    # (N*H*W, 4)
    
    # Compute pseudo-inverse projection: W such that flat_4 ≈ flat_16 @ W
    # Using least squares: W = (X^T X)^-1 X^T Y
    try:
        W = torch.linalg.lstsq(flat_16, flat_4).solution  # (16, 4)
        
        # Initialize first conv layer of adapter
        # adapter expects 32ch input (16 masked + 16 ref)
        # We'll initialize the first layer to use this projection
        with torch.no_grad():
            first_conv = adapter.adapter[0]
            if hasattr(first_conv, 'weight'):
                # Initialize to approximate identity for first 16ch, zeros for rest
                first_conv.weight.data.zero_()
                # Copy PCA weights for first 16 channels
                min_out = min(first_conv.weight.shape[0], W.shape[1])
                min_in = min(first_conv.weight.shape[1], W.shape[0])
                first_conv.weight.data[:min_out, :min_in, 0, 0] = W[:min_in, :min_out].T
        
        if verbose:
            print("  Adapter initialized with PCA projection")
            
    except Exception as e:
        if verbose:
            print(f"  PCA initialization failed: {e}")
            print("  Using default Xavier initialization")


def create_gradual_unfreeze_schedule(model: nn.Module,
                                     total_steps: int,
                                     num_phases: int = 3) -> List[Dict]:
    """
    Create gradual unfreezing schedule for training.
    
    Phase 1: Only adapter layers
    Phase 2: Encoder/Decoder + adapter
    Phase 3: Full network
    
    Args:
        model: Model to create schedule for
        total_steps: Total training steps
        num_phases: Number of unfreezing phases
        
    Returns:
        List of phase configurations
    """
    steps_per_phase = total_steps // num_phases
    
    schedule = [
        {
            'phase': 'adapter_only',
            'start_step': 0,
            'end_step': steps_per_phase,
            'freeze_patterns': ['unet', 'vae'],
            'train_patterns': ['adapter'],
            'lr_multiplier': 10.0,  # Higher LR for adapter
        },
        {
            'phase': 'encoder_decoder',
            'start_step': steps_per_phase,
            'end_step': steps_per_phase * 2,
            'freeze_patterns': ['mid_block'],
            'train_patterns': ['adapter', 'down_blocks', 'up_blocks'],
            'lr_multiplier': 1.0,
        },
        {
            'phase': 'full_network',
            'start_step': steps_per_phase * 2,
            'end_step': total_steps,
            'freeze_patterns': [],
            'train_patterns': ['.*'],  # All parameters
            'lr_multiplier': 0.5,  # Lower LR for fine-tuning
        },
    ]
    
    return schedule


def apply_freeze_schedule(model: nn.Module, phase_config: Dict):
    """
    Apply freezing configuration to model.
    
    Args:
        model: Model to configure
        phase_config: Phase configuration from schedule
    """
    import re
    
    freeze_patterns = phase_config.get('freeze_patterns', [])
    train_patterns = phase_config.get('train_patterns', [])
    
    for name, param in model.named_parameters():
        # Check if should freeze
        should_freeze = any(
            re.search(pattern, name) for pattern in freeze_patterns
        )
        
        # Check if should train
        should_train = any(
            re.search(pattern, name) for pattern in train_patterns
        )
        
        # Train takes precedence over freeze
        if should_train:
            param.requires_grad = True
        elif should_freeze:
            param.requires_grad = False
        else:
            # Default: trainable
            param.requires_grad = True


def save_checkpoint_v2(model_dict: Dict, save_path: str,
                       global_step: int, config: Dict):
    """
    Save v2.0 checkpoint with all components.
    
    Args:
        model_dict: Dictionary containing all model components
        save_path: Path to save checkpoint
        global_step: Current training step
        config: Training configuration
    """
    checkpoint = {
        'global_step': global_step,
        'config': config,
    }
    
    # Save each component
    for name, component in model_dict.items():
        if hasattr(component, 'state_dict'):
            checkpoint[f'{name}_state_dict'] = component.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint_v2(checkpoint_path: str, model_dict: Dict,
                       strict: bool = False) -> int:
    """
    Load v2.0 checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model_dict: Dictionary of model components to load into
        strict: Whether to require exact key matching
        
    Returns:
        Global step from checkpoint
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    for name, component in model_dict.items():
        key = f'{name}_state_dict'
        if key in checkpoint and hasattr(component, 'load_state_dict'):
            try:
                component.load_state_dict(checkpoint[key], strict=strict)
                print(f"  Loaded: {name}")
            except Exception as e:
                print(f"  Failed to load {name}: {e}")
    
    return checkpoint.get('global_step', 0)
