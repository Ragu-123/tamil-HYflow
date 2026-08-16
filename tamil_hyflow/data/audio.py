from pathlib import Path
import torch
import torchaudio

def load_audio(path: str | Path, sample_rate: int = 24000, mono: bool = True, normalize: bool = True) -> tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(str(path))
    waveform = waveform.float()
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        sr = sample_rate
    if normalize:
        peak = waveform.abs().amax().clamp_min(1e-8)
        waveform = waveform / peak.clamp(max=1.0)
    return waveform, sr

def trim_silence(waveform: torch.Tensor, threshold: float = 1e-4) -> torch.Tensor:
    energy = waveform.abs().squeeze(0)
    idx = torch.where(energy > threshold)[0]
    if idx.numel() == 0:
        return waveform
    return waveform[:, idx[0]:idx[-1] + 1]

def pad_or_crop(waveform: torch.Tensor, length: int) -> torch.Tensor:
    if waveform.shape[-1] == length:
        return waveform
    if waveform.shape[-1] > length:
        return waveform[..., :length]
    return torch.nn.functional.pad(waveform, (0, length - waveform.shape[-1]))
