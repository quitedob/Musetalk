"""
Gated Attention (Qwen-style)

Eliminates Attention Sink phenomenon causing background flickering in long videos.

OPTIMIZED VERSION:
- Uses PyTorch SDPA (auto-selects Flash/Efficient/Math backend)
- Optional SageAttention support (2.1x faster than Flash)
- Fused gate computation
- No Python loops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# Try to import SageAttention for maximum performance
_SAGE_AVAILABLE = False
try:
    from sageattention import sageattn
    _SAGE_AVAILABLE = True
except ImportError:
    pass


def get_attention_backend() -> str:
    """Return available attention backend."""
    if _SAGE_AVAILABLE:
        return "sage"
    return "sdpa"


class GatedAttention(nn.Module):
    """
    Gated Attention mechanism (Qwen-style) with G1 gate position.
    
    OPTIMIZATIONS:
    1. Uses F.scaled_dot_product_attention (auto Flash/Efficient)
    2. Optional SageAttention (2.1x faster than Flash)
    3. Fused gate projection
    4. Memory-efficient reshape operations
    """
    
    def __init__(self, embed_dim: int, num_heads: int,
                 gate_hidden_dim: Optional[int] = None,
                 dropout: float = 0.0,
                 gate_init_bias: float = 0.0,
                 use_sage: bool = True):
        """
        Args:
            embed_dim: Dimension of attention embeddings
            num_heads: Number of attention heads
            gate_hidden_dim: Hidden dimension for gate network
            dropout: Attention dropout rate
            gate_init_bias: Initial bias for gate
            use_sage: Use SageAttention if available
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.use_sage = use_sage and _SAGE_AVAILABLE
        
        gate_hidden_dim = gate_hidden_dim or embed_dim // 4
        
        # Fused QKV projection (more efficient than separate)
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Simplified gate: single linear + sigmoid (fused)
        self.gate_proj = nn.Linear(embed_dim, embed_dim)
        self.gate_init_bias = gate_init_bias
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        
        # Gate: small weights + bias for mostly-open initialization
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)
        nn.init.constant_(self.gate_proj.bias, self.gate_init_bias)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Gated attention forward pass - optimized.
        
        Args:
            query: (B, T_q, C) query tokens
            key: (B, T_k, C) key tokens  
            value: (B, T_k, C) value tokens
            attention_mask: Optional mask
            
        Returns:
            (B, T_q, C) output tokens
        """
        B, T_q, C = query.shape
        _, T_k, _ = key.shape
        
        # For self-attention, use fused QKV
        if query.data_ptr() == key.data_ptr() == value.data_ptr():
            qkv = self.qkv_proj(query)
            qkv = qkv.view(B, T_q, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
            Q, K, V = qkv[0], qkv[1], qkv[2]
        else:
            # Cross-attention: separate projections
            qkv_q = self.qkv_proj(query)[:, :, :C]
            qkv_k = self.qkv_proj(key)[:, :, C:2*C]
            qkv_v = self.qkv_proj(value)[:, :, 2*C:]
            
            Q = qkv_q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
            K = qkv_k.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
            V = qkv_v.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        if self.use_sage:
            # SageAttention: expects (B, H, T, D) layout
            attn_output = sageattn(Q, K, V, tensor_layout="HND", is_causal=False)
        else:
            # PyTorch SDPA: auto-selects Flash/Efficient/Math
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )
        
        # Reshape: (B, H, T, D) -> (B, T, C)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, C)
        
        # G1 Gate: Y' = Y ⊙ σ(X W_θ)
        gate = torch.sigmoid(self.gate_proj(query))
        gated_output = attn_output * gate
        
        return self.out_proj(gated_output)


class GatedSelfAttention(nn.Module):
    """Gated Self-Attention - optimized wrapper."""
    
    def __init__(self, embed_dim: int, num_heads: int,
                 gate_hidden_dim: Optional[int] = None,
                 use_sage: bool = True):
        super().__init__()
        
        self.norm = nn.LayerNorm(embed_dim)
        self.gated_attn = GatedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            gate_hidden_dim=gate_hidden_dim,
            use_sage=use_sage
        )
    
    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.gated_attn(x, x, x, attention_mask)
        return residual + x


class GatedCrossAttention(nn.Module):
    """Gated Cross-Attention for audio-visual fusion - optimized."""
    
    def __init__(self, embed_dim: int, num_heads: int,
                 context_dim: int = 384,
                 gate_hidden_dim: Optional[int] = None,
                 use_sage: bool = True):
        super().__init__()
        
        self.norm = nn.LayerNorm(embed_dim)
        self.context_proj = nn.Linear(context_dim, embed_dim)
        self.gated_attn = GatedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            gate_hidden_dim=gate_hidden_dim,
            use_sage=use_sage
        )
    
    def forward(self, visual_features: torch.Tensor,
                audio_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = visual_features
        x = self.norm(visual_features)
        audio_proj = self.context_proj(audio_features)
        x = self.gated_attn(x, audio_proj, audio_proj, attention_mask)
        return residual + x


class GatedTemporalAttention(nn.Module):
    """
    Gated temporal attention for video - optimized.
    
    OPTIMIZATION: Batched reshape instead of per-pixel loop.
    """
    
    def __init__(self, channels: int, num_heads: int = 8,
                 gate_hidden_dim: Optional[int] = None,
                 use_sage: bool = True):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.gated_attn = GatedAttention(
            embed_dim=channels,
            num_heads=num_heads,
            gate_hidden_dim=gate_hidden_dim,
            use_sage=use_sage
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Temporal attention across frames - vectorized.
        
        Args:
            x: (B, T, C, H, W) video features
        Returns:
            (B, T, C, H, W) temporally attended features
        """
        B, T, C, H, W = x.shape
        
        # Efficient reshape: (B, T, C, H, W) -> (B*H*W, T, C)
        # Use contiguous() only if needed
        x_reshaped = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        
        # Normalize (apply GroupNorm in channel dimension)
        # GroupNorm expects (N, C, *), so we need to handle this
        x_for_norm = x_reshaped.transpose(1, 2)  # (B*H*W, C, T)
        x_normed = self.norm(x_for_norm).transpose(1, 2)  # (B*H*W, T, C)
        
        # Self-attention across time
        x_attn = self.gated_attn(x_normed, x_normed, x_normed)
        
        # Residual + reshape back
        x_out = x_reshaped + x_attn
        x_out = x_out.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)
        
        return x_out
