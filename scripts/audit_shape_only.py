import torch
from tamil_hyflow.models.codec import ContinuousAudioEncoder
from tamil_hyflow.models.decoder import MultiBranchSubbandDecoder
from tamil_hyflow.models.hyflow import TamilHyFlow

def count(m):
    return sum(p.numel() for p in m.parameters())

def main():
    model = TamilHyFlow().eval()
    print(f"total_params={count(model):,}")
    enc = ContinuousAudioEncoder().eval()
    dec = MultiBranchSubbandDecoder().eval()
    with torch.inference_mode():
        x = torch.randn(1, 1, 24000)
        z = enc(x)
        y, _ = dec(z)
    print(f"codec_input={tuple(x.shape)}")
    print(f"latent={tuple(z.shape)}")
    print(f"decoded={tuple(y.shape)}")
    assert z.shape == (1, 25, 64)
    assert y.shape == (1, 24000)
    print("shape_check=PASS")

if __name__ == "__main__":
    main()
