import math
import torch
import torch.nn as nn

class SoftMonotonicCrossAttention(nn.Module):
    def __init__(self, q_dim=512, kv_dim=512, heads=8, head_dim=64):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.q_proj = nn.Linear(q_dim, inner)
        self.k_proj = nn.Linear(kv_dim, inner)
        self.v_proj = nn.Linear(kv_dim, inner)
        self.out_proj = nn.Linear(inner, q_dim)
        self.log_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, q, text, text_mask=None):
        b, ta, _ = q.shape
        tt = text.shape[1]
        qh = self.q_proj(q).view(b, ta, self.heads, self.head_dim).transpose(1, 2)
        kh = self.k_proj(text).view(b, tt, self.heads, self.head_dim).transpose(1, 2)
        vh = self.v_proj(text).view(b, tt, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.head_dim)
        i = torch.arange(ta, device=q.device, dtype=q.dtype)[:, None] / max(ta, 1)
        j = torch.arange(tt, device=q.device, dtype=q.dtype)[None, :] / max(tt, 1)
        alpha = torch.nn.functional.softplus(self.log_alpha) + 1e-4
        prior = -alpha * (j - i).pow(2)
        scores = scores + prior[None, None]
        if text_mask is not None:
            scores = scores.masked_fill(~text_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        out = torch.matmul(weights, vh).transpose(1, 2).reshape(b, ta, -1)
        return self.out_proj(out), weights
