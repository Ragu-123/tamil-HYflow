import torch
from tamil_hyflow.models.codec import ContinuousAudioEncoder

def test_codec_rate():
    m = ContinuousAudioEncoder().eval()
    x = torch.randn(1, 1, 24000)
    with torch.inference_mode():
        z = m(x)
    assert z.shape == (1, 25, 64)
