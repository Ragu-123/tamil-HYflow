from pathlib import Path
import torch
from torch.utils.data import Dataset
from .manifest import ManifestRecord, read_manifest
from .audio import load_audio, trim_silence
from .text import tokenize_tamil

class SpeechDataset(Dataset):
    def __init__(self, manifest: str | Path, sample_rate: int = 24000, max_seconds: float = 20.0, trim: bool = False):
        self.records = read_manifest(manifest)
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        self.trim = trim

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        waveform, sr = load_audio(record.audio, self.sample_rate)
        if self.trim:
            waveform = trim_silence(waveform)
        waveform = waveform[..., :self.max_samples]
        tokens = tokenize_tamil(record.text)
        return {
            "audio": waveform,
            "text": record.text,
            "tokens": tokens,
            "speaker_id": record.speaker_id,
            "sample_rate": sr,
            "duration": waveform.shape[-1] / sr,
            "metadata": record.metadata,
        }
