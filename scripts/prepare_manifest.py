import argparse
import json
import os
from pathlib import Path
import random
import torchaudio
from tqdm import tqdm
from tamil_hyflow.data.manifest import ManifestRecord, write_manifest

def parse_speaker(wav_path: Path, audio_root: Path, default_speaker: str = "unknown") -> str:
    rel = wav_path.relative_to(audio_root)
    if len(rel.parts) > 1:
        return rel.parts[0]
    stem = wav_path.stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return default_speaker

def find_transcript(wav_path: Path, audio_root: Path, text_root: Path) -> Path | None:
    rel = wav_path.relative_to(audio_root)
    # 1. Direct match via relative path
    txt = text_root / rel.with_suffix(".txt")
    if txt.exists():
        return txt
    # 2. Direct match in text_root by stem
    txt = text_root / f"{wav_path.stem}.txt"
    if txt.exists():
        return txt
    # 3. Subfolder match by stem
    txt = text_root / rel.parent / f"{wav_path.stem}.txt"
    if txt.exists():
        return txt
    # 4. Case-insensitive check (.TXT)
    txt = text_root / rel.with_suffix(".TXT")
    if txt.exists():
        return txt
    txt = text_root / f"{wav_path.stem}.TXT"
    if txt.exists():
        return txt
    return None

def get_audio_info(wav_path: Path) -> tuple[float, int] | None:
    # 1. Fast Python standard library wave parser (0 dependencies, pure C/Python)
    try:
        import wave
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            frames = wf.getnframes()
            if sr > 0:
                return frames / float(sr), sr
    except Exception:
        pass

    # 2. Soundfile parser
    try:
        import soundfile as sf
        info = sf.info(str(wav_path))
        if info.samplerate > 0:
            return info.duration, info.samplerate
    except Exception:
        pass

    # 3. Torchaudio parser
    try:
        info = torchaudio.info(str(wav_path))
        if info.sample_rate > 0:
            return info.num_frames / float(info.sample_rate), info.sample_rate
    except Exception:
        pass

    return None

def main():
    p = argparse.ArgumentParser(description="Prepare JSONL manifest for Tamil-HyFlow training")
    p.add_argument("--audio-root", required=True, help="Directory containing .wav audio files")
    p.add_argument("--text-root", required=True, help="Directory containing .txt transcript files")
    p.add_argument("--output", required=True, help="Path to output manifest .jsonl")
    p.add_argument("--val-output", default=None, help="Path to validation manifest .jsonl (optional)")
    p.add_argument("--val-ratio", type=float, default=0.0, help="Fraction of data for validation (e.g. 0.05)")
    p.add_argument("--speaker", default="unknown", help="Default speaker identifier")
    p.add_argument("--min-duration", type=float, default=0.5, help="Minimum audio duration in seconds")
    p.add_argument("--max-duration", type=float, default=20.0, help="Maximum audio duration in seconds")
    p.add_argument("--max-items", type=int, default=None, help="Limit number of items to process")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = p.parse_args()

    audio_root = Path(args.audio_root)
    text_root = Path(args.text_root)

    print(f"Scanning audio files in: {audio_root}")
    print(f"Matching transcripts in: {text_root}")

    wav_files = sorted(list(audio_root.rglob("*.wav")))
    if not wav_files:
        wav_files = sorted(list(audio_root.rglob("*.WAV")))
    print(f"Found {len(wav_files)} total audio files.")

    if args.max_items is not None and args.max_items > 0:
        wav_files = wav_files[:args.max_items]
        print(f"Limiting to first {len(wav_files)} files for manifest preparation.")

    records = []
    skipped_notxt = 0
    skipped_empty = 0
    skipped_dur = 0
    skipped_audio_err = 0

    for wav in tqdm(wav_files, desc="Processing files"):
        txt_path = find_transcript(wav, audio_root, text_root)
        if txt_path is None or not txt_path.exists():
            skipped_notxt += 1
            continue

        try:
            text = txt_path.read_text(encoding="utf-8").strip()
        except Exception:
            try:
                text = txt_path.read_text(encoding="latin-1").strip()
            except Exception:
                skipped_empty += 1
                continue

        if not text:
            skipped_empty += 1
            continue

        audio_info = get_audio_info(wav)
        if audio_info is None:
            skipped_audio_err += 1
            continue

        duration, sample_rate = audio_info

        if duration < args.min_duration or duration > args.max_duration:
            skipped_dur += 1
            continue

        speaker = parse_speaker(wav, audio_root, args.speaker)

        records.append(ManifestRecord(
            audio=str(wav.resolve()),
            text=text,
            speaker_id=speaker,
            language="ta",
            sample_rate=sample_rate,
            duration=round(duration, 3),
            metadata={"filename": wav.name}
        ))

    print(f"\nManifest Summary:")
    print(f"  Valid records: {len(records)}")
    print(f"  Skipped (no transcript): {skipped_notxt}")
    print(f"  Skipped (empty text): {skipped_empty}")
    print(f"  Skipped (duration out of bounds): {skipped_dur}")
    print(f"  Skipped (audio read error): {skipped_audio_err}")

    if not records:
        print("Warning: No records found!")
        # Still create empty manifest so downstream doesn't crash on open
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        return

    speakers = set(r.speaker_id for r in records)
    total_hours = sum(r.duration for r in records if r.duration) / 3600.0
    print(f"  Unique speakers: {len(speakers)}")
    print(f"  Total audio: {total_hours:.2f} hours")

    if args.val_ratio > 0 and args.val_output:
        random.seed(args.seed)
        shuffled = list(records)
        random.shuffle(shuffled)
        val_count = max(1, int(len(shuffled) * args.val_ratio))
        val_records = shuffled[:val_count]
        train_records = shuffled[val_count:]

        write_manifest(train_records, args.output)
        write_manifest(val_records, args.val_output)
        print(f"Wrote {len(train_records)} train records to {args.output}")
        print(f"Wrote {len(val_records)} val records to {args.val_output}")
    else:
        write_manifest(records, args.output)
        print(f"Wrote {len(records)} records to {args.output}")

if __name__ == "__main__":
    main()
