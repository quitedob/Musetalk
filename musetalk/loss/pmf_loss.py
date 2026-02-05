"""
Pixel MeanFlow (pMF) Loss

Replaces unstable GAN Loss with deterministic flow matching for stable training.
Based on "One-step Latent-free Image Generation with Pixel Mean Flows" (MIT/CMU 2026)

Key insights:
- Traditional flow matching predicts velocity v (high-dimensional, noisy)
- pMF predicts x (denoised image on data manifold) - much easier to learn
- Loss is still computed in velocity space for correct dynamics
- Enables direct application of perceptual losses (LPIPS) - "WYSIWYG"

Mathematical formulation:
- Flow: z_t = (1-t)x_0 + t*ε  (linear interpolation)
- Velocity: v = ε - x_0
- x-prediction: x(z_t, r, t) ≜ z_t - t·u(z_t, r, t)
- Network outputs x, loss computed on derived velocity

Manifold Hypothesis:
- Real images lie on low-dimensional manifold in pixel space
- Predicting x (on manifold) is easier than predicting v (off manifold)
- This is why x-prediction works better than v-prediction

Key advantages over GAN:
- No mode collapse (single network, no adversarial dynamics)
- Deterministic regression training
- Smooth, L2-based gradients
- Predictable convergence
- Direct perceptual loss application
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PixelMeanFlowLoss(nn.Module):
    """
    Pixel MeanFlow Loss for stable one-step generation.
    
    Core mechanism (x-prediction with v-space loss):
    1. Sample timestep t and noise ε
    2. Create noisy sample: z_t = (1-t)x_0 + t*ε
    3. Network predicts clean image: x_pred = Network(z_t, t)
    4. Convert to velocity: u = (z_t - x_pred) / t
    5. Compute loss: ||u - v_target||² where v_target = ε - x_0
    
    IMPORTANT: This loss expects the network to receive z_t as input
    and output x_pred (the denoised prediction).
    """
    
    def __init__(self, eps: float = 1e-5, min_t: float = 0.001, max_t: float = 0.999):
        """
        Args:
            eps: Small value for numerical stability
            min_t: Minimum timestep (avoid division by zero)
            max_t: Maximum timestep (avoid pure noise)
        """
        super().__init__()
        self.eps = eps
        self.min_t = min_t
        self.max_t = max_t
    
    def sample_timestep(self, batch_size: int, device: torch.device,
                        dtype: torch.dtype) -> torch.Tensor:
        """Sample timesteps with logit-normal distribution (better coverage)."""
        # Logit-normal sampling for better coverage of t ∈ (0, 1)
        u = torch.randn(batch_size, device=device, dtype=dtype)
        t = torch.sigmoid(u)  # Logit-normal
        # Clamp to valid range
        t = t * (self.max_t - self.min_t) + self.min_t
        return t
    
    def compute_noisy_sample(self, x_0: torch.Tensor, eps: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
        """
        Compute noisy sample z_t via linear interpolation.
        
        Flow: z_t = (1-t)x_0 + t*ε
        
        Args:
            x_0: Clean data (B, C, H, W)
            eps: Noise sample (B, C, H, W)
            t: Timestep (B,)
            
        Returns:
            z_t: Noisy sample at time t
        """
        t_expanded = t.view(-1, *([1] * (x_0.ndim - 1)))
        z_t = (1 - t_expanded) * x_0 + t_expanded * eps
        return z_t
    
    def x_to_velocity(self, z_t: torch.Tensor, x_pred: torch.Tensor,
                      t: torch.Tensor) -> torch.Tensor:
        """
        Convert x-prediction to velocity.
        
        Formula: u = (z_t - x_pred) / t
        
        This is the key insight: network predicts x (on manifold),
        but we compute loss on derived velocity u.
        """
        t_expanded = t.view(-1, *([1] * (z_t.ndim - 1)))
        t_safe = torch.clamp(t_expanded, min=self.eps)
        u = (z_t - x_pred) / t_safe
        return u
    
    def forward(self, x_pred: torch.Tensor, x_0: torch.Tensor,
                z_t: torch.Tensor, t: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:
        """
        Compute pMF loss with x-prediction paradigm.
        
        Args:
            x_pred: Network predicted clean image (output of network given z_t)
            x_0: Ground truth clean data
            z_t: Noisy input that was fed to network
            t: Timestep used to create z_t
            return_components: Return loss components for logging
            
        Returns:
            pMF loss scalar (or dict if return_components=True)
        """
        # Target velocity: v = ε - x_0
        # Since z_t = (1-t)x_0 + t*ε, we have ε = (z_t - (1-t)x_0) / t
        t_expanded = t.view(-1, *([1] * (x_0.ndim - 1)))
        t_safe = torch.clamp(t_expanded, min=self.eps)
        eps_reconstructed = (z_t - (1 - t_expanded) * x_0) / t_safe
        v_target = eps_reconstructed - x_0
        
        # Convert x-prediction to velocity: u = (z_t - x_pred) / t
        u_pred = self.x_to_velocity(z_t, x_pred, t)
        
        # pMF loss: ||u_pred - v_target||²
        loss = F.mse_loss(u_pred, v_target)
        
        if return_components:
            return {
                'loss': loss,
                'u_pred_norm': u_pred.norm().item(),
                'v_target_norm': v_target.norm().item(),
                't_mean': t.mean().item(),
            }
        
        return loss
    
    def prepare_training_batch(self, x_0: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare a training batch: sample t, ε, and compute z_t.
        
        Args:
            x_0: Clean data batch (B, C, H, W)
            
        Returns:
            (z_t, t, eps) tuple for network input and loss computation
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        dtype = x_0.dtype
        
        # Sample timesteps
        t = self.sample_timestep(batch_size, device, dtype)
        
        # Sample noise
        eps = torch.randn_like(x_0)
        
        # Compute noisy sample
        z_t = self.compute_noisy_sample(x_0, eps, t)
        
        return z_t, t, eps


class PMFLossWrapper(nn.Module):
    """
    Wrapper combining pMF loss with L1 and perceptual losses.
    
    Since pMF uses x-prediction, we can directly apply perceptual
    losses (LPIPS) to the network output - "What You See Is What You Get".
    
    Usage:
        loss_fn = PMFLossWrapper(pmf_weight=0.1)
        
        # In training loop:
        z_t, t, eps = loss_fn.pmf_loss.prepare_training_batch(x_0)
        x_pred = network(z_t, t)  # Network predicts clean image
        total_loss, l1, pmf = loss_fn(x_pred, x_0, z_t, t)
    """
    
    def __init__(self, pmf_weight: float = 0.1, l1_weight: float = 1.0,
                 use_perceptual: bool = False):
        """
        Args:
            pmf_weight: Weight for pMF loss
            l1_weight: Weight for L1 reconstruction loss
            use_perceptual: Whether to include perceptual loss (requires LPIPS)
        """
        super().__init__()
        
        self.pmf_loss = PixelMeanFlowLoss()
        self.pmf_weight = pmf_weight
        self.l1_weight = l1_weight
        self.use_perceptual = use_perceptual
        
        # Optional perceptual loss (LPIPS)
        self.lpips = None
        if use_perceptual:
            try:
                import lpips
                self.lpips = lpips.LPIPS(net='vgg')
                self.lpips.eval()
                for p in self.lpips.parameters():
                    p.requires_grad = False
            except ImportError:
                print("LPIPS not available, disabling perceptual loss")
                self.use_perceptual = False
    
    def forward(self, x_pred: torch.Tensor, x_0: torch.Tensor,
                z_t: torch.Tensor, t: torch.Tensor,
                pred_images: Optional[torch.Tensor] = None,
                target_images: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            x_pred: Network prediction (what network outputs given z_t)
            x_0: Ground truth clean data
            z_t: Noisy input fed to network
            t: Timestep
            pred_images: Decoded prediction (for perceptual loss)
            target_images: Decoded target (for perceptual loss)
            
        Returns:
            (total_loss, l1_loss, pmf_loss) tuple
        """
        # L1 reconstruction loss (direct comparison)
        l1_loss = F.l1_loss(x_pred, x_0)
        
        # pMF loss (velocity space)
        pmf_loss = self.pmf_loss(x_pred, x_0, z_t, t)
        
        # Combined loss
        total_loss = self.l1_weight * l1_loss + self.pmf_weight * pmf_loss
        
        # Optional perceptual loss (WYSIWYG benefit of x-prediction)
        if self.use_perceptual and self.lpips is not None:
            if pred_images is not None and target_images is not None:
                perceptual_loss = self.lpips(pred_images, target_images).mean()
                total_loss = total_loss + 0.1 * perceptual_loss
        
        return total_loss, l1_loss, pmf_loss


class PixelSpacePMFLoss(nn.Module):
    """
    pMF loss computed directly in pixel space (after VAE decode).
    
    Useful when you want to apply perceptual losses alongside pMF.
    """
    
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred_images: torch.Tensor, target_images: torch.Tensor,
                t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute pMF loss in pixel space.
        
        Args:
            pred_images: Predicted images (B, 3, H, W)
            target_images: Target images (B, 3, H, W)
            t: Timestep (optional)
            
        Returns:
            pMF loss
        """
        batch_size = pred_images.shape[0]
        device = pred_images.device
        dtype = pred_images.dtype
        
        if t is None:
            t = torch.rand(batch_size, device=device, dtype=dtype)
        
        # Sample noise in pixel space
        eps = torch.randn_like(target_images)
        
        # Compute noisy image
        t_expanded = t.view(-1, 1, 1, 1)
        z_t = (1 - t_expanded) * target_images + t_expanded * eps
        
        # Target velocity
        V_target = eps - target_images
        
        # Predicted velocity
        t_safe = torch.clamp(t_expanded, min=self.eps)
        u_theta = (z_t - pred_images) / t_safe
        
        return F.mse_loss(u_theta, V_target)
