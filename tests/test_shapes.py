import torch
from tamil_hyflow.models.hyflow import TamilHyFlow

def test_flow_shapes():
    m = TamilHyFlow()
    feats = tuple(torch.zeros(1, 12, dtype=torch.long) for _ in range(6))
    mask = torch.ones(1, 12, dtype=torch.bool)
    h = m.encode_text(feats, mask)
    prior = m.prior_pass(h, mask)
    audio = torch.randn(1, 1, 240000)
    speaker = m.speaker(audio)
    z = torch.randn(1, 250, 64)
    v, w = m.cfm_velocity(z, h, prior, speaker, torch.rand(1), mask)
    assert h.shape == (1, 12, 512)
    assert v.shape == z.shape
    assert w.shape == (1, 8, 250, 12)
