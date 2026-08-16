import argparse
from pathlib import Path
import torch
import torchaudio
from tamil_hyflow.models.hyflow import TamilHyFlow
from tamil_hyflow.data.text import tokenize_tamil
from tamil_hyflow.data.audio import load_audio

def feature_tensors(text, device):
    tokens = tokenize_tamil(text)
    vals = [[t.cons_id, t.vowel_id, t.length_id, t.class_id, t.word_bound_id, t.punct_id] for t in tokens]
    x = torch.tensor(vals, dtype=torch.long, device=device).T.unsqueeze(0)
    return tuple(x[i] for i in range(6)), torch.ones(1, len(tokens), dtype=torch.bool, device=device)

def main():
    p = argparse.ArgumentParser(description="Tamil-HyFlow Speech Synthesis Inference")
    p.add_argument("--checkpoint", required=True, help="Path to trained TamilHyFlow checkpoint (.pt)")
    p.add_argument("--text", required=True, help="Tamil text to synthesize")
    p.add_argument("--output", required=True, help="Output .wav path")
    p.add_argument("--speaker-audio", required=True, help="Reference speaker .wav audio")
    p.add_argument("--steps", type=int, default=16, help="Euler ODE integration steps (default: 16)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TamilHyFlow().to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    speaker_wav, sr = load_audio(args.speaker_audio, sample_rate=24000)
    speaker_tensor = speaker_wav.unsqueeze(0).to(device)  # (1, 1, samples)

    features, mask = feature_tensors(args.text, device)
    with torch.no_grad():
        h = model.encode_text(features, mask)
        prior = model.prior_pass(h, mask)
        speaker = model.speaker(speaker_tensor)
        n_frames, _ = model.inference_length(h, prior["u"], speaker)
        ta = max(25, int(n_frames[0].item()))
        z = torch.randn(1, ta, 64, device=device)
        ts = torch.linspace(0, 1, args.steps + 1, device=device)
        for t0, t1 in zip(ts[:-1], ts[1:]):
            t = torch.full((1,), float(t0), device=device)
            v, _ = model.cfm_velocity(z, h, prior, speaker, t, mask)
            z = z + (t1 - t0) * v
        audio, _ = model.decode_latent(z)
        audio = audio.squeeze().cpu()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(args.output, audio, 24000)
        print(f"Generated audio saved to {args.output} ({audio.shape[-1] / 24000:.2f}s)")

if __name__ == "__main__":
    main()
