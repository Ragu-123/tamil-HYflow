import torch
from tamil_hyflow.models.frontend import TamilStructuralFrontend

def test_frontend_shape():
    m = TamilStructuralFrontend()
    x = [torch.zeros(2, 8, dtype=torch.long) for _ in range(6)]
    y = m(*x)
    assert y.shape == (2, 8, 512)
