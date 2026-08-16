import torch
import torch.nn as nn

class TotalLengthDistribution(nn.Module):
    def __init__(self, d_model=512, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + latent_dim + 128, 256),
            nn.SiLU(),
            nn.Linear(256, 2),
        )

    def forward(self, h_text, p_u, speaker=None):
        mask = torch.ones(h_text.shape[:2], device=h_text.device, dtype=torch.bool)
        pooled = h_text.mean(dim=1)
        pu = p_u[0]
        if speaker is None:
            speaker = torch.zeros(h_text.shape[0], 128, device=h_text.device, dtype=h_text.dtype)
        return self.net(torch.cat([pooled, pu, speaker], dim=-1))
