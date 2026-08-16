import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .text_encoder import RMSNorm, SwiGLU
from .attention.monotonic import SoftMonotonicCrossAttention

class FlowSelfAttention(nn.Module):
    def __init__(self, d_model=512, heads=8):
        super().__init__()
        self.heads = heads
        self.dim = d_model // heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.dim).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) / math.sqrt(self.dim)
        a = a.softmax(dim=-1)
        return self.out((a @ v).transpose(1, 2).reshape(b, t, d))

class AdaLayerNorm(nn.Module):
    def __init__(self, d_model=512, cond_dim=704):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mod = nn.Linear(cond_dim, d_model * 3)

    def forward(self, x, cond):
        shift, scale, gate = self.mod(cond).chunk(3, dim=-1)
        y = self.norm(x)
        y = y * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return y, gate.unsqueeze(1)

class FlowBlock(nn.Module):
    def __init__(self, d_model=512, heads=8, d_ff=1408, cond_dim=512):
        super().__init__()
        self.n = AdaLayerNorm(d_model, cond_dim)
        self.self_attn = FlowSelfAttention(d_model, heads)
        self.cross = SoftMonotonicCrossAttention(d_model, d_model, heads, d_model // heads)
        self.ff = SwiGLU(d_model, d_ff)
        self.gates = nn.Linear(cond_dim, 3 * d_model)

    def forward(self, x, text, cond, text_mask=None):
        base, _ = self.n(x, cond)
        g_self, g_cross, g_ff = self.gates(cond).chunk(3, dim=-1)
        g_self = g_self.unsqueeze(1).sigmoid()
        g_cross = g_cross.unsqueeze(1).sigmoid()
        g_ff = g_ff.unsqueeze(1).sigmoid()
        x = x + g_self * self.self_attn(base)
        cross, weights = self.cross(x, text, text_mask)
        x = x + g_cross * cross
        x = x + g_ff * self.ff(base + cross)
        return x, weights

class SharedFlowTransformer(nn.Module):
    def __init__(self, layers=8, d_model=512, heads=8, d_ff=1408, latent_dim=64, speaker_dim=128):
        super().__init__()
        self.in_proj = nn.Linear(latent_dim, d_model)
        self.t1 = nn.Linear(d_model, d_model)
        self.t2 = nn.Linear(d_model, d_model)
        self.cond_proj = nn.Linear(64 + speaker_dim + 512, 512)
        self.layers = nn.ModuleList([FlowBlock(d_model, heads, d_ff, 512) for _ in range(layers)])
        self.out = nn.Linear(d_model, latent_dim)

    def timestep_embedding(self, t, dim=512):
        half = dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1))
        x = t[:, None] * freq[None]
        return torch.cat([x.sin(), x.cos()], dim=-1)

    def forward(self, z_t, text_context, prosody, speaker, t, text_mask=None):
        x = self.in_proj(z_t)
        pu = prosody["u"][0]
        te = self.t2(F.silu(self.t1(self.timestep_embedding(t))))
        cond = self.cond_proj(torch.cat([pu, speaker, te], dim=-1))
        last_weights = None
        for layer in self.layers:
            x, last_weights = layer(x, text_context, cond, text_mask)
        return self.out(x), last_weights
