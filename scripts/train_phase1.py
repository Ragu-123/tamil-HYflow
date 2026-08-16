import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tamil_hyflow.data.dataset import SpeechDataset
from tamil_hyflow.data.collate import collate_batch
from tamil_hyflow.models.hyflow import TamilHyFlow
from tamil_hyflow.training.phase1 import Phase1Runner
from tamil_hyflow.utils.config import Config

def main():
    p = argparse.ArgumentParser(description="Train Tamil-HyFlow Phase 1 Conditional Flow Matching TTS")
    p.add_argument("--config", required=True, help="Path to config JSON")
    p.add_argument("--manifest", required=True, help="Path to training manifest JSONL")
    p.add_argument("--codec-checkpoint", required=True, help="Path to trained Phase 0 codec checkpoint (.pt)")
    p.add_argument("--resume", default=None, help="Path to Phase 1 checkpoint to resume from")
    p.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    args = p.parse_args()

    cfg = Config.from_json(args.config)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[Phase 1] Using device: {device}")

    ds = SpeechDataset(args.manifest, cfg.sample_rate, cfg.max_seconds)
    print(f"[Phase 1] Loaded dataset with {len(ds)} utterances")

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

    model = TamilHyFlow().to(device)

    # Load frozen continuous audio encoder from Phase 0
    print(f"[Phase 1] Loading pre-trained Phase 0 codec from {args.codec_checkpoint}")
    codec_ckpt = torch.load(args.codec_checkpoint, map_location=device)
    encoder_state = codec_ckpt["encoder"] if "encoder" in codec_ckpt else codec_ckpt
    model.audio_encoder.load_state_dict(encoder_state)
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
    model.audio_encoder.eval()

    # Also load decoder weights into model if available
    if "decoder" in codec_ckpt:
        model.decoder.load_state_dict(codec_ckpt["decoder"])

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        print(f"[Phase 1] Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model_state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(model_state, strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1

    runner = Phase1Runner(model, optimizer, device, cfg.grad_clip, cfg.amp)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Phase 1] Starting training for {cfg.epochs} epochs. Output dir: {save_dir}")
    best_loss = float("inf")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        model.audio_encoder.eval()  # Keep codec frozen
        total_loss = 0.0
        pbar = tqdm(dl, desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=True)

        for step, batch in enumerate(pbar):
            loss, parts = runner.train_step(batch)
            loss_val = float(loss.item())
            total_loss += loss_val

            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "flow": f"{float(parts.get('flow', 0)):.3f}",
                "kl": f"{float(parts.get('kl', 0)):.3f}",
                "pros": f"{float(parts.get('prosody', 0)):.3f}",
                "len": f"{float(parts.get('length', 0)):.3f}",
            })

        avg_loss = total_loss / max(1, len(dl))
        print(f"[Epoch {epoch + 1:03d}/{cfg.epochs:03d}] Avg Loss: {avg_loss:.5f}")

        # Save latest checkpoint
        ckpt_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": avg_loss,
            "config": cfg.__dict__,
        }
        torch.save(ckpt_data, save_dir / "phase1_latest.pt")
        torch.save(ckpt_data, save_dir / "phase1.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt_data, save_dir / "phase1_best.pt")

        if (epoch + 1) % args.save_every == 0:
            torch.save(ckpt_data, save_dir / f"phase1_epoch_{epoch + 1:03d}.pt")

    print(f"[Phase 1] Training complete! Best loss: {best_loss:.5f}. Checkpoints saved in {save_dir}")

if __name__ == "__main__":
    main()
