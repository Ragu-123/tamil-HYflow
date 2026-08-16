import torch
from tamil_hyflow.models.codec import ContinuousAudioEncoder
from tamil_hyflow.models.decoder import MultiBranchSubbandDecoder
from tamil_hyflow.models.hyflow import TamilHyFlow

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio = torch.randn(1, 1, 240000, device=device)
    enc = ContinuousAudioEncoder().to(device).eval()
    dec = MultiBranchSubbandDecoder().to(device).eval()
    with torch.no_grad():
        z = enc(audio)
        y, branches = dec(z)
    assert z.shape == (1, 250, 64), z.shape
    assert y.shape == (1, 240000), y.shape
    print("codec shapes", z.shape, y.shape)
    model = TamilHyFlow().to(device).eval()
    features = tuple(torch.zeros(1, 12, dtype=torch.long, device=device) for _ in range(6))
    mask = torch.ones(1, 12, dtype=torch.bool, device=device)
    with torch.no_grad():
        h = model.encode_text(features, mask)
        prior = model.prior_pass(h, mask)
        speaker = model.speaker(audio)
        z0 = torch.randn(1, 250, 64, device=device)
        t = torch.rand(1, device=device)
        v, weights = model.cfm_velocity(z0, h, prior, speaker, t, mask)
    assert h.shape == (1, 12, 512), h.shape
    assert v.shape == (1, 250, 64), v.shape
    assert weights.shape[-2:] == (250, 12), weights.shape
    print("flow shapes", h.shape, v.shape, weights.shape)

if __name__ == "__main__":
    main()
