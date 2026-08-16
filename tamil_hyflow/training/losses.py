import torch
import torch.nn.functional as F
import torchaudio
from tamil_hyflow.models.prosody import kl_gaussian

class Phase0CodecLoss(nn.Module):
    pass

def _stft_mag(x, n_fft, hop, win):
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
    low = torchaudio.functional.lowpass_biquad(y, sample_rate, 1000.0)
    mid = torchaudio.functional.highpass_biquad(y, sample_rate, 1000.0)
    mid = torchaudio.functional.lowpass_biquad(mid, sample_rate, 8000.0)
    high = torchaudio.functional.highpass_biquad(y, sample_rate, 8000.0)
    return low, mid, high

def codec_loss(pred, branches, target, subband_weight=1.0, stft_weight=1.0, wave_weight=1.0):
    low_t, mid_t, high_t = subband_targets(target)
    loss_wave = F.l1_loss(pred, target)
    loss_stft = multires_stft_loss(pred, target)
    loss_sub = (
        F.l1_loss(branches["low"], low_t) +
        F.l1_loss(branches["mid"], mid_t) +
        F.l1_loss(branches["high"], high_t)
    )
    total = wave_weight * loss_wave + stft_weight * loss_stft + subband_weight * loss_sub
    return total, {"wave": loss_wave.detach(), "mrstft": loss_stft.detach(), "subband": loss_sub.detach()}

def flow_loss(velocity, target_velocity, mask=None):
    if mask is None:
        return F.mse_loss(velocity, target_velocity)
    m = mask.unsqueeze(-1).to(velocity.dtype)
    return ((velocity - target_velocity).pow(2) * m).sum() / m.sum().clamp_min(1.0)

def prosody_reconstruction_loss(frame_pred, target):
    f0_pred = frame_pred[..., 0]
    energy_pred = frame_pred[..., 1]
    voicing_logits = frame_pred[..., 2]
    f0, energy, voicing = target
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
