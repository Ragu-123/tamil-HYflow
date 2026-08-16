import torch
import torch.nn.functional as F

def spectral_l1_loss(pred_spec: torch.Tensor, target_spec: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred_spec, target_spec)

def snr_db(pred_audio: torch.Tensor, target_audio: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    noise = target_audio - pred_audio
    signal_power = (target_audio ** 2).mean(dim=-1)
    noise_power = (noise ** 2).mean(dim=-1) + eps
    return 10.0 * torch.log10(signal_power / noise_power)
