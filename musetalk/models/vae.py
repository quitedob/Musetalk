from diffusers import AutoencoderKL
import json
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os

try:
    from diffusers import AutoencoderKLFlux2
except Exception:
    AutoencoderKLFlux2 = None

try:
    from musetalk.models.adapter import BidirectionalAdapter
except Exception:
    BidirectionalAdapter = None


def _safe_float(value, default):
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _select_vae_loader(model_path, vae_type, load_kwargs):
    if vae_type != "flux-vae":
        return AutoencoderKL
    subfolder = load_kwargs.get("subfolder", "")
    config_path = os.path.join(model_path, subfolder, "config.json") if subfolder else os.path.join(model_path, "config.json")
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


class VAE:
    """VAE wrapper for MuseTalk inference supporting SD and Flux variants."""

    def __init__(
        self,
        model_path="./models/sd-vae-ft-mse/",
        resized_img=256,
        use_float16=False,
        vae_type="sd-vae",
        device=None,
        use_adapter=False,
        target_channels=8,
        unet_out_channels=4,
    ):
        self.model_path = model_path
        self.vae_type = vae_type
        self.use_adapter = bool(use_adapter)
        self.target_channels = int(target_channels)
        self.unet_out_channels = int(unet_out_channels)

        default_scaling_factor = 0.18215
        default_shift_factor = 0.0
        fallback_latent_channels = 4

        if vae_type == "flux-vae":
            default_scaling_factor = 0.3611
            default_shift_factor = 0.1159
            fallback_latent_channels = 16
            try:
                load_kwargs = {}
                if os.path.isdir(os.path.join(self.model_path, "vae")):
                    load_kwargs["subfolder"] = "vae"
                loader_cls = _select_vae_loader(self.model_path, vae_type, load_kwargs)
                self.vae = loader_cls.from_pretrained(self.model_path, **load_kwargs)
            except Exception as e:
                print(f"Failed to load Flux VAE: {e}")
                print("Falling back to SD-VAE")
                self.vae = AutoencoderKL.from_pretrained("./models/sd-vae-ft-mse/")
                self.vae_type = "sd-vae"
                default_scaling_factor = 0.18215
                default_shift_factor = 0.0
                fallback_latent_channels = 4
        else:
            self.vae = AutoencoderKL.from_pretrained(self.model_path)

        config_latent_channels = getattr(self.vae.config, "latent_channels", None)
        if config_latent_channels is None:
            self.latent_channels = fallback_latent_channels
        else:
            self.latent_channels = int(config_latent_channels)

        raw_scaling = getattr(self.vae.config, "scaling_factor", None)
        raw_shift = getattr(self.vae.config, "shift_factor", None)
        self.scaling_factor = _safe_float(raw_scaling, default_scaling_factor)
        self.shift_factor = _safe_float(raw_shift, default_shift_factor)

        # Older diffusers may parse Flux2 config through AutoencoderKL and inject SD defaults.
        if self.vae_type == "flux-vae" and abs(self.scaling_factor - 0.18215) < 1e-6:
            self.scaling_factor = default_scaling_factor
        if self.vae_type == "flux-vae" and raw_shift is None:
            self.shift_factor = default_shift_factor

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        self.vae.to(self.device)

        if use_float16:
            self.vae = self.vae.half()
            self._use_float16 = True
        else:
            self._use_float16 = False

        self.transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self._resized_img = resized_img
        self._mask_tensor = self.get_mask_tensor()

        self.adapter = None
        if self.use_adapter and self.vae_type == "flux-vae" and BidirectionalAdapter is not None:
            self.adapter = BidirectionalAdapter(
                use_flux_vae=True,
                flux_latent_channels=self.latent_channels,
                unet_in_channels=self.target_channels,
                unet_out_channels=self.unet_out_channels,
            )
            self.adapter.to(self.device)
            if self._use_float16:
                self.adapter = self.adapter.half()
        elif self.vae_type == "flux-vae" and (self.latent_channels * 2) != self.target_channels:
            print(
                "Warning: flux-vae is loaded without adapter and latent channels do not match "
                f"UNet input ({self.latent_channels * 2} vs {self.target_channels})."
            )

        print(
            f"VAE initialized: {self.vae_type}, latent_channels={self.latent_channels}, "
            f"scale={self.scaling_factor}, shift={self.shift_factor}, "
            f"adapter={'on' if self.adapter is not None else 'off'}"
        )

    def get_mask_tensor(self):
        mask_tensor = torch.zeros((self._resized_img, self._resized_img))
        mask_tensor[: self._resized_img // 2, :] = 1
        mask_tensor[mask_tensor < 0.5] = 0
        mask_tensor[mask_tensor >= 0.5] = 1
        return mask_tensor

    def preprocess_img(self, img_name, half_mask=False):
        window = []
        if isinstance(img_name, str):
            window_fnames = [img_name]
            for fname in window_fnames:
                img = cv2.imread(fname)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(
                    img,
                    (self._resized_img, self._resized_img),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                window.append(img)
        else:
            img = cv2.cvtColor(img_name, cv2.COLOR_BGR2RGB)
            window.append(img)

        x = np.asarray(window) / 255.0
        x = np.transpose(x, (3, 0, 1, 2))
        x = torch.squeeze(torch.FloatTensor(x))
        if half_mask:
            x = x * (self._mask_tensor > 0.5)
        x = self.transform(x)

        x = x.unsqueeze(0)
        x = x.to(self.vae.device)
        return x

    def encode_latents(self, image):
        with torch.no_grad():
            init_latent_dist = self.vae.encode(image.to(self.vae.dtype)).latent_dist
        init_latents = (init_latent_dist.sample() - self.shift_factor) * self.scaling_factor
        return init_latents

    def adapt_output_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.shape[1] == self.latent_channels:
            return latents
        if self.adapter is not None:
            return self.adapter.adapt_output(latents)
        raise ValueError(
            f"Latent channel mismatch: got {latents.shape[1]}, expected {self.latent_channels}. "
            "Enable VAE adapter or use a matching UNet/VAE pair."
        )

    def decode_latents(self, latents):
        latents = self.adapt_output_latents(latents)
        latents = latents / self.scaling_factor + self.shift_factor
        image = self.vae.decode(latents.to(self.vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        image = image[..., ::-1]
        return image

    def get_latents_for_unet(self, img):
        ref_image = self.preprocess_img(img, half_mask=True)
        masked_latents = self.encode_latents(ref_image)
        ref_image = self.preprocess_img(img, half_mask=False)
        ref_latents = self.encode_latents(ref_image)
        if self.adapter is not None and self.vae_type == "flux-vae":
            latent_model_input = self.adapter.adapt_input(masked_latents, ref_latents)
        else:
            latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)
        return latent_model_input

    def get_latent_channels(self):
        return self.latent_channels

    def is_flux_vae(self):
        return self.vae_type == "flux-vae"


if __name__ == "__main__":
    vae_mode_path = "./models/sd-vae-ft-mse/"
    vae = VAE(model_path=vae_mode_path, use_float16=False)
    img_path = "./results/sun001_crop/00000.png"

    crop_imgs_path = "./results/sun001_crop/"
    latents_out_path = "./results/latents/"
    if not os.path.exists(latents_out_path):
        os.mkdir(latents_out_path)

    files = os.listdir(crop_imgs_path)
    files.sort()
    files = [file for file in files if file.split(".")[-1] == "png"]

    for file in files:
        index = file.split(".")[0]
        img_path = crop_imgs_path + file
        latents = vae.get_latents_for_unet(img_path)
        print(img_path, "latents", latents.size())
