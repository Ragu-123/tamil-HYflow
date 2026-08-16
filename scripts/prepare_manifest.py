import argparse
from pathlib import Path
import torchaudio
from tamil_hyflow.data.manifest import ManifestRecord, write_manifest

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio-root", required=True)
    p.add_argument("--text-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--speaker", default="unknown")
    args = p.parse_args()
    audio_root = Path(args.audio_root)
    text_root = Path(args.text_root)
    records = []
    for wav in sorted(audio_root.rglob("*.wav")):
        rel = wav.relative_to(audio_root)
        txt = text_root / rel.with_suffix(".txt")
        if not txt.exists():
            continue
        text = txt.read_text(encoding="utf-8").strip()
        if not text:
            continue
        info = torchaudio.info(str(wav))
        duration = info.num_frames / info.sample_rate
        speaker = args.speaker
        if rel.parts:
            speaker = rel.parts[0]
        records.append(ManifestRecord(str(wav.resolve()), text, speaker, "ta", info.sample_rate, duration, {}))
    write_manifest(records, args.output)
    print(f"wrote {len(records)} records")

if __name__ == "__main__":
    main()
