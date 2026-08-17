import argparse
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from tamil_hyflow.data.dataset import SpeechDataset
from tamil_hyflow.data.collate import collate_batch
from tamil_hyflow.models.hyflow import TamilHyFlow
from tamil_hyflow.training.phase1 import Phase1Runner
from tamil_hyflow.utils.config import Config

def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        return True, rank, local_rank, world_size, device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 0, 1, device

def unwrap_model(model):
    return model.module if hasattr(model, "module") else model

def main():
    p = argparse.ArgumentParser(description="Train Tamil-HyFlow Phase 1 Conditional Flow Matching TTS (Single & Multi-GPU)")
    p.add_argument("--config", required=True, help="Path to config JSON")
    p.add_argument("--manifest", required=True, help="Path to training manifest JSONL")
    p.add_argument("--codec-checkpoint", required=True, help="Path to trained Phase 0 codec checkpoint (.pt)")
    p.add_argument("--resume", default=None, help="Path to Phase 1 checkpoint to resume from")
    p.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    p.add_argument("--dp", action="store_true", help="Use torch.nn.DataParallel for multi-GPU without torchrun")
    args = p.parse_args()

    is_dist, rank, local_rank, world_size, device = init_distributed()
    is_main = (rank == 0)

    cfg = Config.from_json(args.config)
    if is_main:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        mode = f"DDP ({world_size} processes)" if is_dist else (f"DataParallel ({num_gpus} GPUs)" if (args.dp and num_gpus > 1) else f"Single Device ({device})")
        print(f"[Phase 1] Training Mode: {mode} | Primary Device: {device}")

    ds = SpeechDataset(args.manifest, cfg.sample_rate, cfg.max_seconds)
    if is_main:
        print(f"[Phase 1] Loaded dataset with {len(ds)} utterances")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if is_dist else None
    num_workers = cfg.num_workers

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=True
    )

    model = TamilHyFlow().to(device)

    # Load frozen continuous audio encoder from Phase 0
    if is_main:
        print(f"[Phase 1] Loading pre-trained Phase 0 codec from {args.codec_checkpoint}")
    codec_ckpt = torch.load(args.codec_checkpoint, map_location=device)
    encoder_state = codec_ckpt["encoder"] if "encoder" in codec_ckpt else codec_ckpt
    model.audio_encoder.load_state_dict(encoder_state)
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
    model.audio_encoder.eval()

    if "decoder" in codec_ckpt:
        model.decoder.load_state_dict(codec_ckpt["decoder"])

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        if is_main:
            print(f"[Phase 1] Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model_state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(model_state, strict=False)
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    if is_dist:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=True)
    elif args.dp and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    runner = Phase1Runner(model, optimizer, device, cfg.grad_clip, cfg.amp)
    save_dir = Path(cfg.save_dir)
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Phase 1] Starting training for {cfg.epochs} epochs. Output dir: {save_dir}")

    best_loss = float("inf")

    for epoch in range(start_epoch, cfg.epochs):
        if is_dist and sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        raw_m = unwrap_model(model)
        raw_m.audio_encoder.eval()

        total_loss = 0.0
        pbar = tqdm(dl, desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=True, disable=not is_main)

        for step, batch in enumerate(pbar):
            loss, parts = runner.train_step(batch)
            loss_val = float(loss.item())
            total_loss += loss_val

            if is_main:
                pbar.set_postfix({
                    "loss": f"{loss_val:.4f}",
                    "flow": f"{float(parts.get('flow', 0)):.3f}",
                    "kl": f"{float(parts.get('kl', 0)):.3f}",
                    "pros": f"{float(parts.get('prosody', 0)):.3f}",
                    "len": f"{float(parts.get('length', 0)):.3f}",
                })

        avg_loss = total_loss / max(1, len(dl))
        if is_main:
            print(f"[Epoch {epoch + 1:03d}/{cfg.epochs:03d}] Avg Loss: {avg_loss:.5f}")

            raw_model = unwrap_model(model)
            ckpt_data = {
                "model": raw_model.state_dict(),
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

    if is_main:
        print(f"[Phase 1] Training complete! Best loss: {best_loss:.5f}. Checkpoints saved in {save_dir}")

    if is_dist:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
