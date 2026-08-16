import torch
import torchaudio

@torch.no_grad()
def extract_prosody(waveform, sample_rate=24000, frame_length=1024, hop_length=320):
    x = waveform.mean(dim=1) if waveform.ndim == 3 else waveform
    f0 = torchaudio.functional.detect_pitch_frequency(x, sample_rate=sample_rate, frame_time=1.0 / 25.0)
    energy = x.unfold(-1, frame_length, hop_length).pow(2).mean(dim=-1).clamp_min(1e-8).sqrt()
    energy = torch.nn.functional.pad(energy, (0, max(0, f0.shape[-1] - energy.shape[-1])))[:, :f0.shape[-1]]
    voicing = (f0 > 50.0).float()
    return f0, energy, voicing
