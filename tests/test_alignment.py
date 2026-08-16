import torch
from tamil_hyflow.models.attention.monotonic import SoftMonotonicCrossAttention

def test_alignment_shape():
    m = SoftMonotonicCrossAttention()
    q = torch.randn(2, 32, 512)
    t = torch.randn(2, 10, 512)
    out, a = m(q, t)
    assert out.shape == q.shape
    assert a.shape == (2, 8, 32, 10)
