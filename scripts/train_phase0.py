import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tamil_hyflow.data.dataset import SpeechDataset
from tamil_hyflow.data.collate import collate_batch
from tamil_hyflow.models.codec import ContinuousAudioEncoder
from tamil_hyflow.models.decoder import MultiBranchSubbandDecoder
from tamil_hyflow.training.phase0 import Phase0Runner
from tamil_hyflow.utils.config import Config

def main():
    p = argparse.ArgumentParser(description="Train Tamil-HyFlow Phase 0 Continuous Acoustic Codec")
    p.add_argument("--config", required=True, help="Path to config JSON")
    p.add_argument("--manifest", required=True, help="Path to training manifest JSONL")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    args = p.parse_args()

    cfg = Config.from_json(args.config)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[Phase 0] Using device: {device}")

    ds = SpeechDataset(args.manifest, cfg.sample_rate, cfg.max_seconds)
    print(f"[Phase 0] Loaded dataset with {len(ds)} utterances")

    # In Kaggle/interactive environments, clamp num_workers to CPU count
    num_workers = min(cfg.num_workers, 2)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )

    encoder = ContinuousAudioEncoder().to(device)
    decoder = MultiBranchSubbandDecoder().to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        print(f"[Phase 0] Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1

    runner = Phase0Runner(encoder, decoder, optimizer, device, cfg.grad_clip, cfg.amp)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Phase 0] Starting training for {cfg.epochs} epochs. Output dir: {save_dir}")
    best_loss = float("inf")

    for epoch in range(start_epoch, cfg.epochs):
        encoder.train()
        decoder.train()
        total_loss = 0.0
        pbar = tqdm(dl, desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=True)

        for step, batch in enumerate(pbar):
            loss, parts = runner.train_step(batch.audio)
            loss_val = float(loss.item())
            total_loss += loss_val

            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "stft": f"{float(parts.get('stft', 0)):.3f}",
                "l1": f"{float(parts.get('l1', 0)):.3f}",
            })

        avg_loss = total_loss / max(1, len(dl))
        print(f"[Epoch {epoch + 1:03d}/{cfg.epochs:03d}] Avg Loss: {avg_loss:.5f}")

        # Save latest checkpoint
        ckpt_data = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": avg_loss,
            "config": cfg.__dict__,
        }
        torch.save(ckpt_data, save_dir / "phase0_latest.pt")
        torch.save(ckpt_data, save_dir / "phase0.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt_data, save_dir / "phase0_best.pt")

        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt_data, save_dir / f"phase0_epoch_{epoch + 1:03d}.pt")

    print(f"[Phase 0] Training complete! Best loss: {best_loss:.5f}. Checkpoints saved in {save_dir}")

if __name__ == "__main__":
    main()
