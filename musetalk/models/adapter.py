"""Latent channel adapters between Flux VAEs and MuseTalk UNet."""

import torch
import torch.nn as nn


class LatentChannelAdapter(nn.Module):
    """Adapt concatenated Flux latents to MuseTalk UNet input channels."""

    def __init__(self, in_channels: int, out_channels: int = 8, hidden_channels: int = None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(out_channels * 3, min(64, in_channels))
        bottleneck_channels = max(out_channels * 2, min(32, hidden_channels))

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.adapter = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, bottleneck_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(bottleneck_channels, self.out_channels, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.xavier_uniform_(mod.weight, gain=0.01)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def forward(self, masked_latents: torch.Tensor, ref_latents: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([masked_latents, ref_latents], dim=1)
        return self.adapter(combined)


class LatentOutputAdapter(nn.Module):
    """Adapt MuseTalk UNet output channels back to Flux latent channels."""

    def __init__(self, in_channels: int = 4, out_channels: int = 16, hidden_channels: int = None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(8, min(64, out_channels))

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.adapter = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, self.out_channels, kernel_size=3, padding=1),
        )
        self._init_weights()

    def _init_weights(self):
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.xavier_uniform_(mod.weight, gain=0.1)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def forward(self, unet_output: torch.Tensor) -> torch.Tensor:
        return self.adapter(unet_output)


class BidirectionalAdapter(nn.Module):
    """Adapt both UNet input and output for Flux latent spaces."""

    def __init__(
        self,
        use_flux_vae: bool = True,
        flux_latent_channels: int = 16,
        unet_in_channels: int = 8,
        unet_out_channels: int = 4,
    ):
        super().__init__()
        self.use_flux_vae = bool(use_flux_vae)
        self.flux_latent_channels = int(flux_latent_channels)
        self.unet_in_channels = int(unet_in_channels)
        self.unet_out_channels = int(unet_out_channels)
        self.expected_input_channels = self.flux_latent_channels * 2

        if self.use_flux_vae:
            self.input_adapter = LatentChannelAdapter(
                in_channels=self.expected_input_channels,
                out_channels=self.unet_in_channels,
            )
            self.output_adapter = LatentOutputAdapter(
                in_channels=self.unet_out_channels,
                out_channels=self.flux_latent_channels,
            )
        else:
            self.input_adapter = None
            self.output_adapter = None

    def adapt_input(self, masked_latents: torch.Tensor, ref_latents: torch.Tensor) -> torch.Tensor:
        if self.use_flux_vae and self.input_adapter is not None:
            return self.input_adapter(masked_latents, ref_latents)
        return torch.cat([masked_latents, ref_latents], dim=1)

    def adapt_output(self, unet_output: torch.Tensor) -> torch.Tensor:
        if self.use_flux_vae and self.output_adapter is not None:
            return self.output_adapter(unet_output)
        return unet_output
