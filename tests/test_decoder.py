import torch
from tamil_hyflow.models.decoder import MultiBranchSubbandDecoder

def test_decoder_length():
    m = MultiBranchSubbandDecoder().eval()
    z = torch.randn(1, 4, 64)
    with torch.inference_mode():
        y, b = m(z)
    assert y.shape == (1, 3840)
    assert b["low"].shape == (1, 3840)
