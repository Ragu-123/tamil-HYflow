import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianLatentHead(nn.Module):
    def __init__(self, in_dim=512, latent_dim=64):
        super().__init__()
        self.proj = nn.Linear(in_dim, latent_dim * 2)
        self.latent_dim = latent_dim

    def forward(self, x):
        mu, logvar = self.proj(x).chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar.clamp(-12.0, 8.0))
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

class ProsodyPosterior(nn.Module):
    def __init__(self, d_text=512, d_audio=64, latent_dim=64):
        super().__init__()
        self.audio_proj = nn.Linear(d_audio, 256)
        self.text_proj = nn.Linear(d_text, 256)
        self.mix = nn.Linear(512, 256)
        self.u = GaussianLatentHead(256, latent_dim)
        self.p = GaussianLatentHead(256, latent_dim)
        self.w = GaussianLatentHead(256, latent_dim)
        self.s = GaussianLatentHead(256, latent_dim)

    def forward(self, z_audio, h_text, text_mask=None):
        a = self.audio_proj(z_audio).mean(dim=1)
        if text_mask is None:
            t = h_text.mean(dim=1)
        else:
            weights = text_mask.float().unsqueeze(-1)
            t = (h_text * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        fused = torch.tanh(self.mix(torch.cat([a, self.text_proj(t)], dim=-1)))
        return {
            "u": self.u(fused),
            "p": self.p(fused),
            "w": self.w(fused),
            "s": self.s(fused),
        }

class ProsodyPrior(nn.Module):
    def __init__(self, d_text=512, latent_dim=64):
        super().__init__()
        self.u = GaussianLatentHead(d_text, latent_dim)
        self.p = GaussianLatentHead(d_text, latent_dim)
        self.w = GaussianLatentHead(d_text, latent_dim)
        self.s = GaussianLatentHead(d_text, latent_dim)

    def forward(self, h_text, text_mask=None):
        if text_mask is None:
            pooled = h_text.mean(dim=1)
        else:
            m = text_mask.float().unsqueeze(-1)
            pooled = (h_text * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return {
            "u": self.u(pooled),
            "p": self.p(pooled),
            "w": self.w(pooled),
            "s": self.s(pooled),
        }

class ProsodyFusion(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.u = nn.Linear(latent_dim, latent_dim)
        self.p = nn.Linear(latent_dim, latent_dim)
        self.w = nn.Linear(latent_dim, latent_dim)
        self.s = nn.Linear(latent_dim, latent_dim)

    def forward(self, posterior_or_prior, frames):
        z_u = self.u(posterior_or_prior["u"][0]).unsqueeze(1).expand(-1, frames, -1)
        z_p = self.p(posterior_or_prior["p"][0]).unsqueeze(1).expand(-1, frames, -1)
        z_w = self.w(posterior_or_prior["w"][0]).unsqueeze(1).expand(-1, frames, -1)
        z_s = self.s(posterior_or_prior["s"][0]).unsqueeze(1).expand(-1, frames, -1)
        return z_u + z_p + z_w + z_s

class ProsodyReconstruction(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.frame = nn.Sequential(nn.Linear(latent_dim, 64), nn.SiLU(), nn.Linear(64, 3))
        self.global_head = nn.Sequential(nn.Linear(latent_dim * 4, 128), nn.SiLU(), nn.Linear(128, 3))

    def forward(self, local, global_latents):
        frame = self.frame(local)
        g = torch.cat(global_latents, dim=-1)
        return frame, self.global_head(g)

def kl_gaussian(mu_q, logvar_q, mu_p, logvar_p):
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    return 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0).mean()
