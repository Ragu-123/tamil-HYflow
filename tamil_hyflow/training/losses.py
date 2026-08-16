import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tamil_hyflow.models.prosody import kl_gaussian

class Phase0CodecLoss(nn.Module):
    pass

def _stft_mag(x, n_fft, hop, win):
    if x.ndim == 3:
        x = x.squeeze(1)
    elif x.ndim == 1:
        x = x.unsqueeze(0)
    x = x.float()  # Compute STFT in float32 to prevent ComplexHalf warning and ensure maximum precision
    window = torch.hann_window(win, device=x.device, dtype=x.dtype)
    return torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True).abs()

def multires_stft_loss(pred, target):
    loss = pred.new_zeros(())
    for n_fft, hop, win in ((512,128,512),(1024,256,1024),(2048,512,2048)):
        a = _stft_mag(pred, n_fft, hop, win)
        b = _stft_mag(target, n_fft, hop, win)
        loss = loss + F.l1_loss(a, b)
    return loss

def subband_targets(y, sample_rate=24000):
    if y.ndim == 3:
        y = y.squeeze(1)
    
    # Pure PyTorch FFT subband filtering (100% CUDA, AMP & DDP compatible, zero CUDA IIR bugs)
    n = y.shape[-1]
    y_float = y.float()
    y_fft = torch.fft.rfft(y_float, n=n, dim=-1)
    freqs = torch.fft.rfftfreq(n, d=1.0 / sample_rate, device=y.device)

    # Low band: 0 to 1000 Hz
    mask_low = (freqs <= 1000.0).to(y_fft.dtype)
    low = torch.fft.irfft(y_fft * mask_low, n=n, dim=-1).to(y.dtype)

    # Mid band: 1000 Hz to 8000 Hz
    mask_mid = ((freqs > 1000.0) & (freqs <= 8000.0)).to(y_fft.dtype)
    mid = torch.fft.irfft(y_fft * mask_mid, n=n, dim=-1).to(y.dtype)

    # High band: 8000 Hz to Nyquist
    mask_high = (freqs > 8000.0).to(y_fft.dtype)
    high = torch.fft.irfft(y_fft * mask_high, n=n, dim=-1).to(y.dtype)

    return low, mid, high

def codec_loss(pred, branches, target, subband_weight=1.0, stft_weight=1.0, wave_weight=1.0):
    if target.ndim == 3:
        target = target.squeeze(1)
    if pred.ndim == 3:
        pred = pred.squeeze(1)
    
    # Align length between pred and target
    min_len = min(pred.shape[-1], target.shape[-1])
    pred = pred[..., :min_len]
    target = target[..., :min_len]

    low_t, mid_t, high_t = subband_targets(target)
    b_low = branches["low"][..., :min_len] if branches["low"].ndim == 2 else branches["low"].squeeze(1)[..., :min_len]
    b_mid = branches["mid"][..., :min_len] if branches["mid"].ndim == 2 else branches["mid"].squeeze(1)[..., :min_len]
    b_high = branches["high"][..., :min_len] if branches["high"].ndim == 2 else branches["high"].squeeze(1)[..., :min_len]

    loss_wave = F.l1_loss(pred, target)
    loss_stft = multires_stft_loss(pred, target)
    loss_sub = (
        F.l1_loss(b_low, low_t) +
        F.l1_loss(b_mid, mid_t) +
        F.l1_loss(b_high, high_t)
    )
    total = wave_weight * loss_wave + stft_weight * loss_stft + subband_weight * loss_sub
    return total, {
        "wave": loss_wave.detach(),
        "l1": loss_wave.detach(),
        "mrstft": loss_stft.detach(),
        "stft": loss_stft.detach(),
        "subband": loss_sub.detach(),
    }

def flow_loss(velocity, target_velocity, mask=None):
    if mask is None:
        return F.mse_loss(velocity, target_velocity)
    m = mask.unsqueeze(-1).to(velocity.dtype)
    return ((velocity - target_velocity).pow(2) * m).sum() / m.sum().clamp_min(1.0)

def prosody_reconstruction_loss(frame_pred, target):
    f0, energy, voicing = target
    min_len = min(frame_pred.shape[1], f0.shape[1], energy.shape[1], voicing.shape[1])
    
    f0_pred = frame_pred[:, :min_len, 0]
    energy_pred = frame_pred[:, :min_len, 1]
    voicing_logits = frame_pred[:, :min_len, 2]
    
    f0 = f0[:, :min_len]
    energy = energy[:, :min_len]
    voicing = voicing[:, :min_len]

    voiced = voicing > 0.5
    f0_loss = F.mse_loss(f0_pred[voiced], f0[voiced]) if voiced.any() else f0_pred.new_zeros(())
    energy_loss = F.mse_loss(energy_pred, energy)
    voicing_loss = F.binary_cross_entropy_with_logits(voicing_logits, voicing)
    return f0_loss + energy_loss + voicing_loss, {"f0": f0_loss.detach(), "energy": energy_loss.detach(), "voicing": voicing_loss.detach()}

def total_length_loss(stats, target_frames):
    mu = stats[:, 0]
    return F.mse_loss(mu, target_frames.to(mu.dtype))

def prosody_kl(posterior, prior):
    out = next(iter(posterior.values()))[0].new_zeros(())
    for key in ("u", "p", "w", "s"):
        qz, qmu, qlv = posterior[key]
        _, pmu, plv = prior[key]
        out = out + kl_gaussian(qmu, qlv, pmu, plv)
    return out
