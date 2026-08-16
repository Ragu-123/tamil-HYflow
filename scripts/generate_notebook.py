import json
from pathlib import Path

def build_notebook():
    cells = []

    def md(source):
        lines = [line + "\n" for line in source.strip().splitlines()]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        }

    def code(source):
        lines = [line + "\n" for line in source.strip().splitlines()]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        }

    # Cell 1: Header
    cells.append(md("""# 🎙️ Tamil-HyFlow: Next-Generation Tamil Text-to-Speech
### *Modular Flow Matching with Continuous Latents on IISc-MILE Tamil Corpus*

Tamil-HyFlow is a Tamil-first TTS architecture engineered for paired `(audio, text, speaker)` speech data.
It eliminates the need for phoneme duration labels, word/syllable timestamps, forced alignment, MAS, or external ASR models.

---

### 🏛️ Architecture Pipeline
```text
Tamil Text
  └──> Tamil Structural Frontend (Unicode NFC, Consonants, Vowels, Modifiers, Length, Phonetic Class)
        └──> Bidirectional Text Transformer
              ├──> Hierarchical Prosody Prior (Utterance, Phrase, Word, Syllable)
              ├──> Total Length Distribution (Frame Duration Predictor)
              ├──> Soft Monotonic Alignment Field (Differentiable dynamic attention)
              └──> Shared Conditional Flow Transformer (Velocity Field v_t)
                    └──> 25 Hz x 64-D Continuous Acoustic Latent
                          └──> Multi-Branch Subband Decoder
                                └──> 24 kHz Speech Waveform
```

---

### 🎯 Training Stages:
- **Phase 0 (Continuous Latent Representation)**: Trains the audio encoder and multi-branch subband decoder to represent 24 kHz waveforms in a continuous 25 Hz × 64-D latent space.
- **Phase 1 (Flow Matching & Alignment)**: Freezes the Phase 0 encoder and trains the structural Tamil frontend, prosody prior/posterior, soft monotonic alignment field, and flow transformer."""))

    # Cell 2: GPU Check
    cells.append(md("""## 1. ⚙️ Hardware Environment & GPU Verification"""))
    cells.append(code("""import os
import sys
import torch
import torchaudio

print("=" * 60)
print(f"Python Version    : {sys.version.split()[0]}")
print(f"PyTorch Version   : {torch.__version__}")
print(f"Torchaudio Version: {torchaudio.__version__}")
print(f"CUDA Available    : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device Name       : {torch.cuda.get_device_name(0)}")
    print(f"Device Count      : {torch.cuda.device_count()}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total VRAM        : {total_mem:.2f} GB")
else:
    device = torch.device("cpu")
    print("WARNING: GPU not detected. Training will run on CPU (slower).")
print("=" * 60)"""))

    # Cell 3: Clone Repo & Install
    cells.append(md("""## 2. 📦 Setup Tamil-HyFlow Repository & Dependencies"""))
    cells.append(code("""import os
import sys
from pathlib import Path

# Setup repository path
REPO_URL = "https://github.com/Ragu-123/tamil-HYflow.git"

if Path("tamil_hyflow").exists():
    print("Found local tamil_hyflow codebase in current directory.")
else:
    print(f"Cloning Tamil-HyFlow from {REPO_URL}...")
    !git clone {REPO_URL}
    if Path("tamil-HYflow").exists():
        os.chdir("tamil-HYflow")
    elif Path("tamil-HyFlow").exists():
        os.chdir("tamil-HyFlow")

print(f"Current Working Directory: {os.getcwd()}")

# Install dependencies and editable package
!pip install -q soundfile tqdm
!pip install -e .

# Add to python path
if "." not in sys.path:
    sys.path.insert(0, ".")"""))

    # Cell 4: Smoke Test
    cells.append(md("""## 3. 🧪 Architecture Smoke Test
Verify the Tamil structural frontend, continuous audio encoder, subband decoder, prosody network, and flow matching velocity field."""))
    cells.append(code("""from tamil_hyflow.models.codec import ContinuousAudioEncoder
from tamil_hyflow.models.decoder import MultiBranchSubbandDecoder
from tamil_hyflow.models.hyflow import TamilHyFlow
from tamil_hyflow.data.text import tokenize_tamil, normalize_text

print("[1/3] Testing Tamil Structural Frontend...")
sample_text = "வணக்கம்! தமிழ் மொழியில் குரல் உருவாக்கம்."
tokens = tokenize_tamil(sample_text)
print(f"Input: '{sample_text}' -> Extracted {len(tokens)} linguistic/syllabic tokens:")
for t in tokens[:6]:
    print(f"  - Surface: '{t.text}', Consonant: {t.cons_id}, Vowel: {t.vowel_id}, Length: {t.length_id}, Class: {t.class_id}")

print("\n[2/3] Testing Continuous Codec (24kHz <-> 25Hz x 64D)...")
dummy_audio = torch.randn(1, 1, 24000 * 2, device=device)  # 2 seconds @ 24kHz
encoder = ContinuousAudioEncoder().to(device).eval()
decoder = MultiBranchSubbandDecoder().to(device).eval()
with torch.no_grad():
    z = encoder(dummy_audio)
    y_rec, branches = decoder(z)
print(f"  Audio in: {dummy_audio.shape} -> Latent: {z.shape} (25 Hz) -> Audio out: {y_rec.shape}")

print("\n[3/3] Testing Full TamilHyFlow Pipeline...")
hyflow = TamilHyFlow().to(device).eval()
features = tuple(torch.zeros(1, len(tokens), dtype=torch.long, device=device) for _ in range(6))
mask = torch.ones(1, len(tokens), dtype=torch.bool, device=device)
with torch.no_grad():
    h = hyflow.encode_text(features, mask)
    prior = hyflow.prior_pass(h, mask)
    speaker = hyflow.speaker(dummy_audio)
    t = torch.rand(1, device=device)
    v, weights = hyflow.cfm_velocity(z, h, prior, speaker, t, mask)
print(f"  Text Repr: {h.shape}, Speaker Vector: {speaker.shape}, CFM Velocity: {v.shape}")
print("\n✅ All Tamil-HyFlow architecture modules verified successfully!")"""))

    # Cell 5: Dataset Discovery
    cells.append(md("""## 4. 🔍 Dataset Discovery & Inspection (IISc-MILE Tamil ASR Corpus)
The dataset contains thousands of high-fidelity Tamil speech recordings and corresponding text transcripts."""))
    cells.append(code("""import glob
from pathlib import Path

# Search common Kaggle input directory structures
CANDIDATE_PATHS = [
    Path("/kaggle/input/datasets/raghavanmuthuraman/iisc-mile-tamil-asr-corpus/mile_tamil_asr_corpus"),
    Path("/kaggle/input/iisc-mile-tamil-asr-corpus/mile_tamil_asr_corpus"),
    Path("/kaggle/input/iisc-mile-tamil-asr-corpus"),
    Path("/kaggle/input/mile_tamil_asr_corpus"),
    Path("/kaggle/input/mile-tamil-asr-corpus"),
]

DATASET_ROOT = None
for p in CANDIDATE_PATHS:
    if p.exists():
        DATASET_ROOT = p
        break

if DATASET_ROOT is None:
    # Deep search across /kaggle/input
    search_results = list(Path("/kaggle/input").rglob("audio_files"))
    if search_results:
        DATASET_ROOT = search_results[0].parent.parent
    else:
        print("Searching all .wav files in /kaggle/input...")
        all_wavs = list(Path("/kaggle/input").rglob("*.wav"))
        if all_wavs:
            DATASET_ROOT = all_wavs[0].parent.parent

print(f"📍 Detected Dataset Root: {DATASET_ROOT}")

# Locate train and test splits
TRAIN_AUDIO = DATASET_ROOT / "train" / "audio_files" if (DATASET_ROOT / "train" / "audio_files").exists() else DATASET_ROOT / "audio_files"
TRAIN_TRANS = DATASET_ROOT / "train" / "trans_files" if (DATASET_ROOT / "train" / "trans_files").exists() else DATASET_ROOT / "trans_files"
TEST_ROOT = DATASET_ROOT / "test"

print(f"  - Train Audio Directory     : {TRAIN_AUDIO} (Exists: {TRAIN_AUDIO.exists()})")
print(f"  - Train Transcript Directory: {TRAIN_TRANS} (Exists: {TRAIN_TRANS.exists()})")
print(f"  - Test Directory            : {TEST_ROOT} (Exists: {TEST_ROOT.exists()})")

wav_count = len(list(TRAIN_AUDIO.glob("*.wav")))
txt_count = len(list(TRAIN_TRANS.glob("*.txt")))
print(f"\n📊 Discovered {wav_count} training audio (.wav) files and {txt_count} transcript (.txt) files.")"""))

    # Cell 6: Manifest Preparation
    cells.append(md("""## 5. 📝 Generate Dataset Manifests (Train & Validation Splits)"""))
    cells.append(code("""import json
from tamil_hyflow.data.manifest import read_manifest

MANIFEST_DIR = Path("/kaggle/working/manifests")
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MANIFEST = MANIFEST_DIR / "train_manifest.jsonl"
VAL_MANIFEST = MANIFEST_DIR / "val_manifest.jsonl"

# We can prepare manifest with train/val split (e.g. 5% validation)
# Set max_items to None for full dataset, or an integer (e.g. 5000) for rapid pilot experiments
MAX_ITEMS = None  # None for full training set

limit_arg = f"--max-items {MAX_ITEMS}" if MAX_ITEMS else ""

!python scripts/prepare_manifest.py \\
    --audio-root "{TRAIN_AUDIO}" \\
    --text-root "{TRAIN_TRANS}" \\
    --output "{TRAIN_MANIFEST}" \\
    --val-output "{VAL_MANIFEST}" \\
    --val-ratio 0.05 \\
    --min-duration 0.5 \\
    --max-duration 15.0 \\
    {limit_arg}

train_records = read_manifest(TRAIN_MANIFEST)
val_records = read_manifest(VAL_MANIFEST)

print(f"\\n✅ Manifest creation complete!")
print(f"  Train Utterances: {len(train_records):,}")
print(f"  Val Utterances  : {len(val_records):,}")

print(f"\\n🔍 Sample Train Record:")
sample = train_records[0]
print(f"  Audio Path: {sample.audio}")
print(f"  Speaker ID: {sample.speaker_id}")
print(f"  Duration  : {sample.duration}s")
print(f"  Text      : {sample.text}")"""))

    # Cell 7: Audio & Text Inspection
    cells.append(md("""## 6. 🎧 Data Exploration: Audio Playback & Waveform Display"""))
    cells.append(code("""import IPython.display as ipd
import matplotlib.pyplot as plt
import torchaudio

# Pick a sample audio file from manifest
sample_record = train_records[0]
waveform, sr = torchaudio.load(sample_record.audio)

print(f"Utterance: {sample_record.text}")
print(f"Speaker  : {sample_record.speaker_id}")
print(f"Duration : {sample_record.duration:.2f} seconds | Sample Rate: {sr} Hz")

# Display Audio Player in Notebook
display(ipd.Audio(waveform.numpy(), rate=sr))

# Plot waveform and spectrogram
fig, axes = plt.subplots(2, 1, figsize=(12, 5))
axes[0].plot(waveform[0].numpy(), color="#0284c7")
axes[0].set_title(f"Waveform: {sample_record.speaker_id} - '{sample_record.text[:40]}...'")
axes[0].set_xlim(0, waveform.shape[1])
axes[0].set_ylabel("Amplitude")

spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80)(waveform)
axes[1].imshow(spec[0].log2().clamp(-10, 10).numpy(), aspect="auto", origin="lower", cmap="magma")
axes[1].set_title("Log-Mel Spectrogram")
axes[1].set_ylabel("Mel Frequency")
axes[1].set_xlabel("Frames")
plt.tight_layout()
plt.show()"""))

    # Cell 8: Phase 0 Codec Config & Training
    cells.append(md(r"""## 7. 🚀 Phase 0: Train Continuous Acoustic Latent Codec
Phase 0 learns continuous representations:
$$\text{Waveform (24 kHz)} \longrightarrow \mathbf{z} \in \mathbb{R}^{25 \text{ Hz} \times 64\text{-D}} \longrightarrow \text{Waveform (24 kHz)}$$
Trained with Multi-Resolution STFT Spectral Loss and Subband Filterbank Consistency."""))
    cells.append(code("""import json

PHASE0_CONFIG_PATH = Path("/kaggle/working/configs/kaggle_phase0.json")
PHASE0_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

phase0_cfg = {
    "sample_rate": 24000,
    "latent_rate": 25,
    "latent_dim": 64,
    "batch_size": 4,          # Adjust based on GPU VRAM (4 or 8 recommended for T4/P100)
    "num_workers": 2,
    "epochs": 10,             # Number of Phase 0 epochs
    "lr": 2e-4,
    "weight_decay": 0.01,
    "max_seconds": 10.0,
    "grad_clip": 1.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "amp": True,              # Mixed Precision Training
    "save_dir": "/kaggle/working/checkpoints/phase0"
}

with open(PHASE0_CONFIG_PATH, "w") as f:
    json.dump(phase0_cfg, f, indent=2)

print(f"Phase 0 Configuration saved to {PHASE0_CONFIG_PATH}:")
print(json.dumps(phase0_cfg, indent=2))

# Run Phase 0 Training
!python scripts/train_phase0.py \\
    --config "{PHASE0_CONFIG_PATH}" \\
    --manifest "{TRAIN_MANIFEST}" \\
    --save-every 1"""))

    # Cell 9: Phase 0 Evaluation & Reconstruction Listening
    cells.append(md(r"""## 8. 🔬 Phase 0 Codec Evaluation: Original vs. Reconstructed Audio
Let's inspect the continuous acoustic latent compression quality ($24\text{ kHz} \leftrightarrow 25\text{ Hz}\times 64\text{D}$)."""))
    cells.append(code("""from tamil_hyflow.models.codec import ContinuousAudioEncoder
from tamil_hyflow.models.decoder import MultiBranchSubbandDecoder
from tamil_hyflow.data.audio import load_audio

phase0_ckpt_path = Path(phase0_cfg["save_dir"]) / "phase0_latest.pt"

if phase0_ckpt_path.exists():
    encoder = ContinuousAudioEncoder().to(device).eval()
    decoder = MultiBranchSubbandDecoder().to(device).eval()
    
    ckpt = torch.load(phase0_ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    print(f"Loaded Phase 0 Checkpoint from {phase0_ckpt_path} (Epoch {ckpt.get('epoch', 0)})")

    test_audio_path = train_records[0].audio
    audio_orig, sr = load_audio(test_audio_path, target_sr=24000)
    audio_tensor = audio_orig.unsqueeze(0).to(device)

    with torch.no_grad():
        z = encoder(audio_tensor)
        rec_audio, _ = decoder(z)

    rec_np = rec_audio.squeeze().cpu().numpy()
    orig_np = audio_orig.squeeze().cpu().numpy()

    print(f"Original Audio Shape: {orig_np.shape} | Latent Shape: {z.shape} (Compression: {orig_np.shape[0] / (z.shape[1] * 64):.1f}x)")
    
    print("\n🎧 Original Audio (24 kHz):")
    display(ipd.Audio(orig_np, rate=24000))
    
    print("🎧 Phase 0 Reconstructed Audio (Decoded from 25 Hz Latent):")
    display(ipd.Audio(rec_np, rate=24000))
else:
    print(f"Checkpoint {phase0_ckpt_path} not found yet. Train Phase 0 first.")"""))

    # Cell 10: Phase 1 Flow Matching Config & Training
    cells.append(md(r"""## 9. 🌊 Phase 1: Conditional Flow Matching TTS Training
In Phase 1, we freeze the continuous acoustic encoder and train:
- **Tamil Structural Phonetic/Syllabic Frontend**
- **Bidirectional Text Transformer**
- **Hierarchical Prosody Prior Network** ($\mathbf{u}, \mathbf{p}, \mathbf{w}, \mathbf{s}$)
- **Soft Monotonic Cross-Attention Alignment Field**
- **Shared Conditional Flow Transformer** (Vector Field $v_t$)"""))
    cells.append(code("""PHASE1_CONFIG_PATH = Path("/kaggle/working/configs/kaggle_phase1.json")

phase1_cfg = {
    "sample_rate": 24000,
    "latent_rate": 25,
    "latent_dim": 64,
    "batch_size": 2,          # Batch size for Phase 1 flow model
    "num_workers": 2,
    "epochs": 15,             # Number of Phase 1 epochs
    "lr": 1e-4,
    "weight_decay": 0.01,
    "max_seconds": 10.0,
    "grad_clip": 1.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "amp": True,
    "save_dir": "/kaggle/working/checkpoints/phase1"
}

with open(PHASE1_CONFIG_PATH, "w") as f:
    json.dump(phase1_cfg, f, indent=2)

print(f"Phase 1 Configuration saved to {PHASE1_CONFIG_PATH}:")
print(json.dumps(phase1_cfg, indent=2))

# Train Phase 1 Model
!python scripts/train_phase1.py \\
    --config "{PHASE1_CONFIG_PATH}" \\
    --manifest "{TRAIN_MANIFEST}" \\
    --codec-checkpoint "{phase0_ckpt_path}" \\
    --save-every 1"""))

    # Cell 11: Inference & Speech Synthesis
    cells.append(md("""## 10. 🗣️ Speech Synthesis Demo (Tamil Text-to-Speech Inference)
Enter any Tamil sentence and synthesize speech using the trained Tamil-HyFlow flow ODE solver!"""))
    cells.append(code("""from tamil_hyflow.models.hyflow import TamilHyFlow
from tamil_hyflow.data.text import tokenize_tamil
from tamil_hyflow.data.audio import load_audio

def synthesize_tamil_speech(
    text: str,
    reference_audio_path: str,
    checkpoint_path: str,
    output_path: str = "/kaggle/working/output_tamil_synth.wav",
    steps: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    dev = torch.device(device)
    model = TamilHyFlow().to(dev).eval()
    
    ckpt = torch.load(checkpoint_path, map_location=dev)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    print(f"Loaded TamilHyFlow checkpoint: {checkpoint_path}")

    # Process reference speaker audio
    ref_wav, sr = load_audio(reference_audio_path, target_sr=24000)
    ref_tensor = ref_wav.unsqueeze(0).to(dev)

    # Process Tamil input text
    tokens = tokenize_tamil(text)
    vals = [[t.cons_id, t.vowel_id, t.length_id, t.class_id, t.word_bound_id, t.punct_id] for t in tokens]
    feat_tensor = torch.tensor(vals, dtype=torch.long, device=dev).T.unsqueeze(0)
    features = tuple(feat_tensor[i] for i in range(6))
    mask = torch.ones(1, len(tokens), dtype=torch.bool, device=dev)

    print(f"Synthesizing '{text}' ({len(tokens)} linguistic units) with {steps} Flow Euler steps...")

    with torch.no_grad():
        h = model.encode_text(features, mask)
        prior = model.prior_pass(h, mask)
        speaker = model.speaker(ref_tensor)
        
        # Predict duration / number of latent frames
        n_frames, _ = model.inference_length(h, prior["u"], speaker)
        num_acoustic_frames = int(n_frames[0].item())
        print(f"Predicted audio length: {num_acoustic_frames} latent frames (~{num_acoustic_frames / 25.0:.2f} seconds)")

        # Initial standard Gaussian latent noise z_0 ~ N(0, I)
        z = torch.randn(1, num_acoustic_frames, 64, device=dev)
        
        # Euler Integration along Vector Field v_t with tqdm progress
        from tqdm import tqdm
        time_steps = torch.linspace(0, 1, steps + 1, device=dev)
        for t0, t1 in tqdm(zip(time_steps[:-1], time_steps[1:]), total=steps, desc="Flow ODE Sampling", leave=False):
            t_curr = torch.full((1,), float(t0), device=dev)
            v, _ = model.cfm_velocity(z, h, prior, speaker, t_curr, mask)
            z = z + (t1 - t0) * v

        # Decode latent trajectory to 24 kHz waveform
        audio_out, _ = model.decode_latent(z)
        audio_cpu = audio_out.squeeze().cpu()

    # Save output audio
    torchaudio.save(output_path, audio_cpu.unsqueeze(0), 24000)
    print(f"Saved synthesized audio to {output_path}")
    return audio_cpu.numpy(), 24000

# Example sentences in Tamil to test:
test_sentences = [
    "வணக்கம்! இது தமிழ்-ஹைஃப்ளோ குரல் அமைப்பு.",
    "செயற்கை நுண்ணறிவு தமிழ் மொழியை மிக அழகாக பேசுகிறது.",
    "இன்று வானிலை மிகவும் இனிமையாகவும் அமைதியாகவும் உள்ளது."
]

phase1_ckpt_path = Path(phase1_cfg["save_dir"]) / "phase1_latest.pt"

if phase1_ckpt_path.exists():
    for idx, sentence in enumerate(test_sentences):
        print("=" * 60)
        print(f"Tamil Prompt [{idx+1}]: {sentence}")
        audio_np, sr = synthesize_tamil_speech(
            text=sentence,
            reference_audio_path=train_records[0].audio,
            checkpoint_path=str(phase1_ckpt_path),
            output_path=f"/kaggle/working/synth_sample_{idx+1}.wav",
            steps=16
        )
        display(ipd.Audio(audio_np, rate=sr))
else:
    print(f"Phase 1 checkpoint {phase1_ckpt_path} not found. Run Phase 1 training first.")"""))

    # Cell 12: Export & Checkpoint Archive
    cells.append(md("""## 11. 💾 Package & Download Model Checkpoints
Save the trained model weights, configurations, and generated manifests into a zip archive."""))
    cells.append(code("""import shutil
from pathlib import Path

ARTIFACTS_DIR = Path("/kaggle/working/tamil_hyflow_release")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Copy checkpoints and manifests
shutil.copytree("/kaggle/working/checkpoints", ARTIFACTS_DIR / "checkpoints", dirs_exist_ok=True)
shutil.copytree("/kaggle/working/configs", ARTIFACTS_DIR / "configs", dirs_exist_ok=True)
shutil.copytree("/kaggle/working/manifests", ARTIFACTS_DIR / "manifests", dirs_exist_ok=True)

zip_path = "/kaggle/working/tamil_hyflow_checkpoints.zip"
shutil.make_archive("/kaggle/working/tamil_hyflow_checkpoints", 'zip', ARTIFACTS_DIR)

print(f"📦 Successfully created checkpoint archive: {zip_path}")
print(f"   Size: {Path(zip_path).stat().st_size / (1024*1024):.2f} MB")
print("You can download this archive from the Kaggle Output section!")"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook

if __name__ == "__main__":
    nb = build_notebook()
    out_dir = Path("notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "tamil_hyflow_kaggle_training.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
        
    with open("tamil_hyflow_kaggle_training.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
        
    print(f"Wrote clean notebook with {len(nb['cells'])} cells.")
