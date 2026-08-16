import argparse
import torch
import torchaudio
from tamil_hyflow.models.hyflow import TamilHyFlow
from tamil_hyflow.data.text import tokenize_tamil

def feature_tensors(text, device):
    tokens = tokenize_tamil(text)
    vals = [[t.cons_id, t.vowel_id, t.length_id, t.class_id, t.word_bound_id, t.punct_id] for t in tokens]
    x = torch.tensor(vals, dtype=torch.long, device=device).T.unsqueeze(0)
    return tuple(x[i] for i in range(6)), torch.ones(1, len(tokens), dtype=torch.bool, device=device)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--speaker-audio", required=True)
    p.add_argument("--steps", type=int, default=8)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TamilHyFlow().to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    speaker_wav, sr = torchaudio.load(args.speaker_audio)
    if sr != 24000:
        speaker_wav = torchaudio.functional.resample(speaker_wav, sr, 24000)
    features, mask = feature_tensors(args.text, device)
    with torch.no_grad():
        h = model.encode_text(features, mask)
        prior = model.prior_pass(h, mask)
        speaker = model.speaker(speaker_wav.unsqueeze(0).to(device))
        n_frames, _ = model.inference_length(h, prior["u"], speaker)
        ta = int(n_frames[0].item())
        z = torch.randn(1, ta, 64, device=device)
        ts = torch.linspace(0, 1, args.steps + 1, device=device)
        for t0, t1 in zip(ts[:-1], ts[1:]):
            t = torch.full((1,), float(t0), device=device)
            v, _ = model.cfm_velocity(z, h, prior, speaker, t, mask)
            z = z + (t1 - t0) * v
        audio, _ = model.decode_latent(z)
        audio = audio.cpu().unsqueeze(0)
        torchaudio.save(args.output, audio, 24000)

if __name__ == "__main__":
    main()
