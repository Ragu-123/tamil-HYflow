from dataclasses import dataclass
import torch

@dataclass
class Batch:
    audio: torch.Tensor
    audio_mask: torch.Tensor
    cons_id: torch.Tensor
    vowel_id: torch.Tensor
    length_id: torch.Tensor
    class_id: torch.Tensor
    word_bound_id: torch.Tensor
    punct_id: torch.Tensor
    text_mask: torch.Tensor
    speaker_ids: list[str]
    texts: list[str]


def collate_batch(items: list[dict]) -> Batch:
    b = len(items)
    max_audio = max(x["audio"].shape[-1] for x in items)
    max_text = max(len(x["tokens"]) for x in items)
    audio = torch.zeros(b, 1, max_audio)
    audio_mask = torch.zeros(b, max_audio, dtype=torch.bool)
    feats = [torch.zeros(b, max_text, dtype=torch.long) for _ in range(6)]
    text_mask = torch.zeros(b, max_text, dtype=torch.bool)
    for i, item in enumerate(items):
        n = item["audio"].shape[-1]
        audio[i, :, :n] = item["audio"]
        audio_mask[i, :n] = True
        text_mask[i, :len(item["tokens"])] = True
        for j, tok in enumerate(item["tokens"]):
            vals = [tok.cons_id, tok.vowel_id, tok.length_id, tok.class_id, tok.word_bound_id, tok.punct_id]
            for k, value in enumerate(vals):
                feats[k][i, j] = value
    return Batch(audio, audio_mask, *feats, text_mask, [x["speaker_id"] for x in items], [x["text"] for x in items])
