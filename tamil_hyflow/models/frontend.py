import torch
import torch.nn as nn
import torch.nn.functional as F

class TamilStructuralFrontend(nn.Module):
    def __init__(self, cons_classes=20, vowel_classes=13, length_classes=3, class_classes=6, word_classes=2, punct_classes=4, d_model=512):
        super().__init__()
        self.emb_cons = nn.Embedding(cons_classes, 64)
        self.emb_vowel = nn.Embedding(vowel_classes, 64)
        self.emb_length = nn.Embedding(length_classes, 32)
        self.emb_class = nn.Embedding(class_classes, 32)
        self.emb_word = nn.Embedding(word_classes, 16)
        self.emb_punct = nn.Embedding(punct_classes, 16)
        self.proj = nn.Linear(224, d_model)
        self.conv1 = nn.Conv1d(d_model, d_model, 5, padding=2)
        self.conv2 = nn.Conv1d(d_model, d_model, 5, padding=2)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cons_id, vowel_id, length_id, class_id, word_bound_id, punct_id):
        x = torch.cat([
            self.emb_cons(cons_id),
            self.emb_vowel(vowel_id),
            self.emb_length(length_id),
            self.emb_class(class_id),
            self.emb_word(word_bound_id),
            self.emb_punct(punct_id),
        ], dim=-1)
        h = self.proj(x)
        z = self.conv1(h.transpose(1, 2))
        z = F.silu(z)
        z = self.conv2(z).transpose(1, 2)
        return self.norm(h + z)
