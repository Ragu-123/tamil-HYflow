# Tamil-HyFlow

Tamil-HyFlow is a modular, from-scratch Tamil-first text-to-speech research implementation built around a low-rate continuous acoustic latent and conditional flow matching.

The implementation is designed for paired `(audio, text, speaker)` data. It does not require phoneme durations, word timestamps, syllable timestamps, forced alignment, MAS, or ASR training labels.

## Architecture

```text
Tamil text
  -> Tamil structural frontend
  -> bidirectional text transformer
  -> hierarchical prosody prior
  -> total length distribution
  -> soft monotonic alignment field
  -> shared conditional flow transformer
  -> 25 Hz x 64-D acoustic latent
  -> multi-branch subband decoder
  -> 24 kHz waveform
```

Training uses a training-only audio encoder to construct 25 Hz x 64-D targets.

## Repository layout

```text
tamil_hyflow/
  data/
    manifest.py
    text.py
    audio.py
    dataset.py
    collate.py
  models/
    frontend.py
    text_encoder.py
    prosody.py
    speaker.py
    length.py
    codec.py
    flow.py
    decoder.py
    hyflow.py
    attention/
      monotonic.py
  training/
    losses.py
    engine.py
    phase0.py
    phase1.py
  utils/
    checkpoint.py
    config.py
    seed.py
    metrics.py
configs/
  base.json
  mile_phase0.json
  mile_phase1.json
scripts/
  prepare_manifest.py
  train_phase0.py
  train_phase1.py
  infer.py
  audit_model.py
  smoke_test.py
tests/
  test_frontend.py
  test_codec.py
  test_alignment.py
  test_decoder.py
  test_shapes.py
docs/
  architecture.md
```

## Dataset contract

Each training item is normalized to:

- `audio`: waveform file
- `text`: transcript
- `speaker_id`: stable speaker identifier
- optional metadata

The IISc-MILE loader accepts JSONL manifests and resamples audio to 24 kHz.

## Phase 0

Phase 0 learns the continuous audio representation:

`24 kHz waveform -> 25 Hz x 64-D latent -> 24 kHz waveform`

Run:

```bash
python scripts/prepare_manifest.py --audio-root /path/to/audio --text-root /path/to/text --output data/mile.jsonl
python scripts/train_phase0.py --config configs/mile_phase0.json --manifest data/mile.jsonl
```

## Phase 1

Phase 1 requires a Phase 0 encoder checkpoint. It trains text, prosody, speaker conditioning, soft monotonic alignment, and the shared flow transformer.

```bash
python scripts/train_phase1.py --config configs/mile_phase1.json --manifest data/mile.jsonl --codec-checkpoint checkpoints/phase0.pt
```

## Kaggle Notebook Quickstart

A notebook is included: [`tamil_hyflow_kaggle_training.ipynb`](file:///c:/Users/SEC/Downloads/Tamil%20asr/Tamil-HyFlow/tamil_hyflow_kaggle_training.ipynb) (and [`notebooks/tamil_hyflow_kaggle_training.ipynb`](file:///c:/Users/SEC/Downloads/Tamil%20asr/Tamil-HyFlow/notebooks/tamil_hyflow_kaggle_training.ipynb)).

### Steps on Kaggle:
1. **Create a New Notebook** on Kaggle with **GPU accelerator (T4 x 2 or P100)**.
2. **Attach Dataset**: Add `raghavanmuthuraman/iisc-mile-tamil-asr-corpus` (`mile_tamil_asr_corpus`).
3. **Upload or clone** the repo:
   ```bash
   !git clone https://github.com/Ragu-123/tamil-HYflow.git
   cd tamil-HYflow
   pip install -e .
   ```
4. **Run Phase 0 (Codec)**:
   Trains the 25 Hz continuous audio latent encoder and multi-branch subband decoder on the IISc-MILE speech recordings.
5. **Run Phase 1 (Flow Matching TTS)**:
   Trains Tamil phonetic frontend, prosody prior, soft monotonic alignment field, and conditional flow matching transformer.
6. **Inference & Audio Synthesis**:
   Synthesize custom Tamil text into 24 kHz speech using Euler ODE flow sampling and listen directly in the notebook with `IPython.display.Audio`.

## Smoke test

```bash
python scripts/smoke_test.py
```

## Audit

```bash
python scripts/audit_model.py
```
