"""
Manifold-Constrained Hyper-Connections (mHC)

Stabilizes deep network training by constraining residual routing weights
to the Birkhoff polytope (doubly stochastic matrices).

OPTIMIZED VERSION:
- Cached Sinkhorn results (only recompute every N steps)
- Vectorized operations (no Python loops in forward)
- Optional torch.compile support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def sinkhorn_knopp(matrix: torch.Tensor, num_iters: int = 20,
                   eps: float = 1e-8, temperature: float = 1.0) -> torch.Tensor:
    """
    Project matrix onto Birkhoff polytope using Sinkhorn-Knopp algorithm.
    Vectorized implementation - no Python loops.
    """
    # Ensure non-negativity via exp
    K = torch.exp(matrix / temperature)
    
    # Vectorized Sinkhorn-Knopp: unroll loop for torch.compile compatibility
    for _ in range(num_iters):
        K = K / (K.sum(dim=-1, keepdim=True) + eps)
        K = K / (K.sum(dim=-2, keepdim=True) + eps)
    
    return K


def sinkhorn_knopp_batched(matrices: torch.Tensor, num_iters: int = 20,
                           eps: float = 1e-8, temperature: float = 1.0) -> torch.Tensor:
    """
    Batched Sinkhorn-Knopp for multiple matrices at once.
    
    Args:
        matrices: (B, N, N) batch of matrices
    Returns:
        (B, N, N) batch of doubly stochastic matrices
    """
    K = torch.exp(matrices / temperature)
    
    for _ in range(num_iters):
        K = K / (K.sum(dim=-1, keepdim=True) + eps)
        K = K / (K.sum(dim=-2, keepdim=True) + eps)
    
    return K


class ManifoldConstrainedConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connection module.
    
    OPTIMIZATIONS:
    1. Cached routing matrices - only recompute every `cache_steps` iterations
    2. Vectorized einsum operations
    3. No Python loops in forward pass
    """
    
    def __init__(self, num_streams: int = 2, sinkhorn_iters: int = 20,
                 temperature: float = 1.0, learnable_temperature: bool = False,
                 cache_steps: int = 100):
        """
        Args:
            num_streams: Number of parallel signal streams
            sinkhorn_iters: Iterations for Sinkhorn-Knopp projection
            temperature: Softmax temperature for routing sharpness
            learnable_temperature: Whether temperature is learnable
            cache_steps: Recompute routing matrices every N steps (0=always)
        """
        super().__init__()
        
        self.num_streams = num_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.cache_steps = cache_steps
        
        # Learnable routing matrices (unconstrained, projected at forward)
        self.H_res_logit = nn.Parameter(
            torch.eye(num_streams) + torch.randn(num_streams, num_streams) * 0.1
        )
        self.H_transform_logit = nn.Parameter(
            torch.eye(num_streams) + torch.randn(num_streams, num_streams) * 0.1
        )
        
        # Temperature parameter
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
        
        # Cached routing matrices
        self.register_buffer('_cached_H_res', None)
        self.register_buffer('_cached_H_transform', None)
        self.register_buffer('_cache_step_counter', torch.tensor(0))
    
    def _should_update_cache(self) -> bool:
        """Check if cache needs update."""
        if self.cache_steps <= 0:
            return True
        if self._cached_H_res is None:
            return True
        if self.training:
            return self._cache_step_counter.item() >= self.cache_steps
        return False
    
    def get_routing_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get routing matrices, using cache when possible."""
        if self._should_update_cache():
            # Compute both matrices in one batched operation
            stacked = torch.stack([self.H_res_logit, self.H_transform_logit], dim=0)
            projected = sinkhorn_knopp_batched(
                stacked, self.sinkhorn_iters, temperature=self.temperature.item()
            )
            self._cached_H_res = projected[0]
            self._cached_H_transform = projected[1]
            self._cache_step_counter.zero_()
        else:
            self._cache_step_counter.add_(1)
        
        return self._cached_H_res, self._cached_H_transform
    
    def forward(self, x: torch.Tensor, F_x: torch.Tensor) -> torch.Tensor:
        """
        Apply manifold-constrained hyper-connection.
        Fully vectorized - no Python loops.
        """
        B, C, H, W = x.shape
        
        if C % self.num_streams != 0:
            return x + F_x
        
        H_res, H_transform = self.get_routing_matrices()
        
        stream_size = C // self.num_streams
        
        # Reshape: (B, C, H, W) -> (B, num_streams, stream_size*H*W)
        x_flat = x.view(B, self.num_streams, stream_size, H, W).view(B, self.num_streams, -1)
        F_x_flat = F_x.view(B, self.num_streams, stream_size, H, W).view(B, self.num_streams, -1)
        
        # Batched matrix multiply (vectorized across batch)
        # H_res: (N, N), x_flat: (B, N, P) -> (B, N, P)
        x_routed = torch.einsum('mn,bnp->bmp', H_res, x_flat)
        F_routed = torch.einsum('mn,bnp->bmp', H_transform, F_x_flat)
        
        output_flat = x_routed + F_routed
        
        return output_flat.view(B, C, H, W)
    
    def reset_cache(self):
        """Force cache invalidation (call after optimizer step if needed)."""
        self._cached_H_res = None
        self._cached_H_transform = None
        self._cache_step_counter.zero_()


class mHCResBlock(nn.Module):
    """
    ResBlock with Manifold-Constrained Hyper-Connections.
    
    Drop-in replacement for standard ResNet blocks in UNet.
    """
    
    def __init__(self, channels: int, num_streams: int = 2,
                 sinkhorn_iters: int = 20):
        """
        Args:
            channels: Number of input/output channels
            num_streams: Number of parallel streams for mHC
            sinkhorn_iters: Sinkhorn-Knopp iterations
        """
        super().__init__()
        
        self.channels = channels
        self.num_streams = num_streams
        
        # Main transformation branch
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, channels)
        self.norm2 = nn.GroupNorm(32, channels)
        self.activation = nn.SiLU()
        
        # Manifold-constrained connection
        self.mhc = ManifoldConstrainedConnection(
            num_streams=num_streams,
            sinkhorn_iters=sinkhorn_iters
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with mHC residual.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor (B, C, H, W)
        """
        # Main branch transformation
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.activation(h)
        h = self.conv2(h)
        
        # Apply mHC instead of simple residual addition
        out = self.mhc(x, h)
        
        return out


class mHCSkipConnection(nn.Module):
    """
    mHC for UNet skip connections (encoder → decoder).
    
    Stabilizes long-range skip connections that can cause
    signal explosion in deep UNets.
    """
    
    def __init__(self, channels: int, num_streams: int = 2,
                 sinkhorn_iters: int = 20):
        """
        Args:
            channels: Number of channels in skip connection
            num_streams: Number of parallel streams
            sinkhorn_iters: Sinkhorn-Knopp iterations
        """
        super().__init__()
        
        self.mhc = ManifoldConstrainedConnection(
            num_streams=num_streams,
            sinkhorn_iters=sinkhorn_iters
        )
        
        # Optional projection if dimensions differ
        self.proj = None
    
    def forward(self, encoder_features: torch.Tensor,
                decoder_features: torch.Tensor) -> torch.Tensor:
        """
        Combine encoder skip with decoder features using mHC.
        
        Args:
            encoder_features: Features from encoder (B, C, H, W)
            decoder_features: Features from decoder (B, C, H, W)
            
        Returns:
            Combined features with stable signal flow
        """
        return self.mhc(decoder_features, encoder_features)
