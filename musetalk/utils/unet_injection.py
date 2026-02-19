import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers.models.attention_processor import AttnProcessor
except Exception:
    class AttnProcessor(nn.Module):
        def __call__(
            self,
            attn,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            *args,
            **kwargs,
        ):
            residual = hidden_states
            if hidden_states.ndim == 4:
                b, c, h, w = hidden_states.shape
                hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
                is_4d = (b, c, h, w)
            else:
                is_4d = None

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states

            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            if is_4d is not None:
                b, c, h, w = is_4d
                hidden_states = hidden_states.transpose(1, 2).reshape(b, c, h, w)
            if getattr(attn, "residual_connection", False):
                hidden_states = hidden_states + residual
            if hasattr(attn, "rescale_output_factor"):
                hidden_states = hidden_states / attn.rescale_output_factor
            return hidden_states


def _sinkhorn_project(
    logits: torch.Tensor,
    num_iters: int = 10,
    eps: float = 1e-6,
    temperature: float = 1.0,
) -> torch.Tensor:
    mat = torch.exp(logits / max(temperature, eps))
    for _ in range(max(1, num_iters)):
        mat = mat / (mat.sum(dim=-1, keepdim=True) + eps)
        mat = mat / (mat.sum(dim=-2, keepdim=True) + eps)
    return mat


class MHCChannelMixer(nn.Module):
    """
    Lightweight mHC-style stream mixer for UNet ResNet blocks.
    """

    def __init__(self, channels: int, num_streams: int = 2, sinkhorn_iters: int = 10):
        super().__init__()
        self.channels = channels
        self.num_streams = max(1, int(num_streams))
        self.sinkhorn_iters = max(1, int(sinkhorn_iters))
        self.routing_logits = nn.Parameter(torch.eye(self.num_streams))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            return x
        b, c, h, w = x.shape
        if c % self.num_streams != 0:
            return x
        stream_size = c // self.num_streams
        routing = _sinkhorn_project(self.routing_logits, num_iters=self.sinkhorn_iters)
        x_flat = x.view(b, self.num_streams, stream_size, h, w).reshape(b, self.num_streams, -1)
        x_mix = torch.einsum("mn,bnp->bmp", routing, x_flat)
        return x_mix.reshape(b, c, h, w)


class HybridCrossAttnProcessor(nn.Module):
    """
    Cross-attention processor with optional gated output and sparse top-k KV.
    """

    def __init__(
        self,
        hidden_size: int,
        use_gated_attn: bool = False,
        use_dsa: bool = False,
        dsa_topk: int = 2048,
        gate_init_bias: float = -1.5,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.use_gated_attn = bool(use_gated_attn)
        self.use_dsa = bool(use_dsa)
        self.dsa_topk = max(1, int(dsa_topk))
        self.gate_proj = None
        if self.use_gated_attn:
            self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size)
            nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)
            nn.init.constant_(self.gate_proj.bias, gate_init_bias)

    def _to_sequence(self, hidden_states: torch.Tensor):
        if hidden_states.ndim == 4:
            b, c, h, w = hidden_states.shape
            return hidden_states.view(b, c, h * w).transpose(1, 2), (b, c, h, w)
        return hidden_states, None

    def _to_spatial(self, hidden_states: torch.Tensor, shape_4d):
        if shape_4d is None:
            return hidden_states
        b, c, h, w = shape_4d
        return hidden_states.transpose(1, 2).reshape(b, c, h, w)

    def _sparse_select(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # q: [B, H, Q, D], k/v: [B, H, K, D]
        k_len = k.shape[2]
        topk = min(self.dsa_topk, k_len)
        if topk >= k_len:
            return k, v

        q_summary = q.mean(dim=2)  # [B, H, D]
        score = torch.matmul(q_summary, k.transpose(-1, -2)) / math.sqrt(k.shape[-1])  # [B, H, K]
        idx = torch.topk(score, k=topk, dim=-1).indices  # [B, H, topk]
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
        k_sel = torch.gather(k, dim=2, index=gather_idx)
        v_sel = torch.gather(v, dim=2, index=gather_idx)
        return k_sel, v_sel

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states, shape_4d = self._to_sequence(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if getattr(attn, "spatial_norm", None) is not None and shape_4d is not None:
            hidden_states = attn.spatial_norm(self._to_spatial(hidden_states, shape_4d), temb)
            hidden_states, _ = self._to_sequence(hidden_states)

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        bsz, q_len, _ = query.shape
        head_dim = query.shape[-1] // attn.heads

        q = query.view(bsz, q_len, attn.heads, head_dim).transpose(1, 2)
        k = key.view(bsz, key.shape[1], attn.heads, head_dim).transpose(1, 2)
        v = value.view(bsz, value.shape[1], attn.heads, head_dim).transpose(1, 2)

        if self.use_dsa and encoder_hidden_states is not hidden_states:
            k, v = self._sparse_select(q, k, v)
            attention_mask = None

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, k.shape[2], bsz)
            while attention_mask.ndim < attn_scores.ndim:
                attention_mask = attention_mask.unsqueeze(1)
            attn_scores = attn_scores + attention_mask
        attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(attn_scores.dtype)
        hidden_states = torch.matmul(attn_probs, v)
        hidden_states = hidden_states.transpose(1, 2).reshape(bsz, q_len, attn.heads * head_dim)

        if self.gate_proj is not None:
            gate = torch.sigmoid(self.gate_proj(hidden_states))
            hidden_states = hidden_states * gate

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = self._to_spatial(hidden_states, shape_4d)

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual
        if hasattr(attn, "rescale_output_factor"):
            hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def inject_attention_processors(
    unet,
    use_gated_attn: bool = False,
    use_dsa: bool = False,
    dsa_topk: int = 2048,
) -> Dict[str, int]:
    result = {"enabled": 0, "num_injected": 0}
    if not (use_gated_attn or use_dsa):
        return result
    if not (hasattr(unet, "attn_processors") and hasattr(unet, "set_attn_processor")):
        return result

    processors = {}
    old_processors = unet.attn_processors
    block_channels = list(unet.config.block_out_channels)
    up_channels = list(reversed(block_channels))
    for name in unet.attn_processors.keys():
        if name.startswith("mid_block"):
            hidden_size = block_channels[-1]
        elif name.startswith("up_blocks."):
            block_id = int(name.split(".")[1])
            hidden_size = up_channels[block_id]
        elif name.startswith("down_blocks."):
            block_id = int(name.split(".")[1])
            hidden_size = block_channels[block_id]
        else:
            hidden_size = block_channels[0]

        is_cross = name.endswith("attn2.processor")
        if is_cross:
            processors[name] = HybridCrossAttnProcessor(
                hidden_size=hidden_size,
                use_gated_attn=use_gated_attn,
                use_dsa=use_dsa,
                dsa_topk=dsa_topk,
            )
            result["num_injected"] += 1
        else:
            processors[name] = old_processors.get(name, AttnProcessor())

    unet.set_attn_processor(processors)
    if result["num_injected"] > 0:
        result["enabled"] = 1
    return result


def inject_mhc_mixers(
    unet,
    use_mhc: bool = False,
    num_streams: int = 2,
    sinkhorn_iters: int = 10,
) -> Dict[str, int]:
    result = {"enabled": 0, "num_injected": 0}
    if not use_mhc:
        return result

    handles = getattr(unet, "_mhc_hook_handles", [])
    for handle in handles:
        try:
            handle.remove()
        except Exception:
            pass
    unet._mhc_hook_handles = []

    for _, module in unet.named_modules():
        if module.__class__.__name__ != "ResnetBlock2D":
            continue
        if hasattr(module, "_mhc_mixer"):
            continue

        out_channels = getattr(module, "out_channels", None)
        if out_channels is None and hasattr(module, "conv2"):
            out_channels = getattr(module.conv2, "out_channels", None)
        if out_channels is None or out_channels % max(1, num_streams) != 0:
            continue

        module.add_module(
            "_mhc_mixer",
            MHCChannelMixer(
                channels=int(out_channels),
                num_streams=num_streams,
                sinkhorn_iters=sinkhorn_iters,
            ),
        )

        def _hook(mod, _inputs, output):
            if isinstance(output, tuple):
                if not output:
                    return output
                first = mod._mhc_mixer(output[0])
                return (first, *output[1:])
            return mod._mhc_mixer(output)

        handle = module.register_forward_hook(_hook)
        unet._mhc_hook_handles.append(handle)
        result["num_injected"] += 1

    if result["num_injected"] > 0:
        result["enabled"] = 1
    return result
