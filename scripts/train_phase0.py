import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tamil_hyflow.data.dataset import SpeechDataset
from tamil_hyflow.data.collate import collate_batch
from tamil_hyflow.models.codec import ContinuousAudioEncoder
from tamil_hyflow.models.decoder import MultiBranchSubbandDecoder
from tamil_hyflow.training.phase0 import Phase0Runner
from tamil_hyflow.utils.config import Config

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    args = p.parse_args()
    cfg = Config.from_json(args.config)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    ds = SpeechDataset(args.manifest, cfg.sample_rate, cfg.max_seconds)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_batch)
    encoder = ContinuousAudioEncoder()
    decoder = MultiBranchSubbandDecoder()
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    runner = Phase0Runner(encoder, decoder, optimizer, device, cfg.grad_clip, cfg.amp)
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(cfg.epochs):
        for batch in dl:
            loss, parts = runner.train_step(batch.audio)
        print(epoch, float(loss), {k: float(v) for k, v in parts.items()})
        torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "epoch": epoch}, Path(cfg.save_dir) / "phase0.pt")

if __name__ == "__main__":
    main()
