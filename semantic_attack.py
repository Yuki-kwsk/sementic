from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


@dataclass
class AttackConfig:
    steps: int = 150
    lr: float = 0.03
    eps: float = 10 / 255
    semantic_weight: float = 1.0
    preserve_weight: float = 45.0
    tv_weight: float = 0.03
    clip_input_size: int = 224
    log_every: int = 25
    clip_model_id: str = "openai/clip-vit-large-patch14"


def _to_tensor(img: Image.Image, size_hw: Tuple[int, int], device: torch.device) -> torch.Tensor:
    h, w = size_hw
    tfm = transforms.Compose(
        [
            transforms.Resize((h, w)),
            transforms.ToTensor(),
        ]
    )
    return tfm(img.convert("RGB")).unsqueeze(0).to(device)


def _to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().clamp(0.0, 1.0).squeeze(0).cpu()
    return transforms.ToPILImage()(x)


class CLIPFeatureExtractor(torch.nn.Module):
    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def encode(self, x_01: torch.Tensor, input_size: int) -> torch.Tensor:
        x = F.interpolate(
            x_01,
            size=(input_size, input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        mean = CLIP_MEAN.to(x.device)
        std = CLIP_STD.to(x.device)
        x = (x - mean) / std
        feats = self.model.get_image_features(pixel_values=x)
        feats = F.normalize(feats, dim=-1)
        return feats


def semantic_attack(
    semantic_imgs: List[Image.Image],
    victim_img: Image.Image,
    config: AttackConfig,
    progress_cb=None,
) -> Tuple[Image.Image, Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = CLIPFeatureExtractor(config.clip_model_id).to(device)

    # Always process in victim resolution.
    victim_w, victim_h = victim_img.size
    target_size = (victim_h, victim_w)
    victim = _to_tensor(victim_img, target_size, device)
    if len(semantic_imgs) == 0:
        raise ValueError("At least one semantic image is required.")
    semantic_tensors = [_to_tensor(img, target_size, device) for img in semantic_imgs]
    adv = victim.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([adv], lr=config.lr)
    with torch.no_grad():
        semantic_feats = [
            extractor.encode(semantic_tensor, config.clip_input_size)
            for semantic_tensor in semantic_tensors
        ]
        semantic_feat = torch.stack(semantic_feats, dim=0).mean(dim=0)

    last_losses: Dict[str, float] = {}
    for step in range(1, config.steps + 1):
        optimizer.zero_grad()
        adv_clamped = adv.clamp(0.0, 1.0)
        adv_feat = extractor.encode(adv_clamped, config.clip_input_size)

        cosine = F.cosine_similarity(adv_feat, semantic_feat, dim=-1).mean()
        semantic_loss = 1.0 - cosine
        preserve_loss = F.mse_loss(adv_clamped, victim)
        tv_loss = total_variation(adv_clamped)

        loss = (
            config.semantic_weight * semantic_loss
            + config.preserve_weight * preserve_loss
            + config.tv_weight * tv_loss
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            perturb = (adv - victim).clamp(-config.eps, config.eps)
            adv.copy_((victim + perturb).clamp(0.0, 1.0))

        last_losses = {
            "total": float(loss.item()),
            "semantic": float(semantic_loss.item()),
            "preserve": float(preserve_loss.item()),
            "tv": float(tv_loss.item()),
            "cosine_similarity": float(cosine.item()),
        }
        if progress_cb and (step % config.log_every == 0 or step == config.steps):
            progress_cb(step, config.steps, last_losses)

    return _to_pil(adv.detach()), last_losses
