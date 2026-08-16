import torch
import torchaudio

@torch.no_grad()
def extract_prosody(waveform, sample_rate=24000, frame_length=1024, frame_rate=25):
    orig_device = waveform.device
    x = waveform.mean(dim=1) if waveform.ndim == 3 else waveform
    hop_length = int(sample_rate / frame_rate)  # 960 samples for 24kHz @ 25Hz
    
    try:
        f0 = torchaudio.functional.detect_pitch_frequency(x.float(), sample_rate=sample_rate, frame_time=1.0 / float(frame_rate))
    except Exception:
        f0 = torchaudio.functional.detect_pitch_frequency(x.cpu().float(), sample_rate=sample_rate, frame_time=1.0 / float(frame_rate)).to(orig_device)
    
    if x.shape[-1] >= frame_length:
        energy = x.unfold(-1, frame_length, hop_length).pow(2).mean(dim=-1).clamp_min(1e-8).sqrt()
    else:
        energy = x.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
        
    if energy.shape[-1] < f0.shape[-1]:
        energy = torch.nn.functional.pad(energy, (0, f0.shape[-1] - energy.shape[-1]))
    else:
        energy = energy[:, :f0.shape[-1]]
        
    voicing = (f0 > 50.0).float()
    return f0.to(orig_device), energy.to(orig_device), voicing.to(orig_device)
