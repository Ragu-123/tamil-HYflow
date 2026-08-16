# Tamil-HyFlow v13 Implementation Specification

## Data requirement

Required paired supervision is `(audio, text, speaker_id)` when speaker identity exists. No token-duration labels, phoneme alignments, word timestamps, syllable timestamps, MAS targets, or ASR outputs are required.

## Fixed targets

- Audio sampling rate: 24000 Hz
- Acoustic latent rate: 25 Hz
- Acoustic latent width: 64
- Text model width: 512
- Text layers: 6
- Text heads: 8
- Flow layers: 8
- Flow heads: 8
- Flow FFN width: 1408
- Stochastic hierarchical prosody: utterance, phrase, word, structural-token
- Total output length predicted as a distribution, not token durations

## Tamil structural representation

For each normalized Tamil structural unit, the frontend stores six categorical features: consonant/base, vowel modifier, vowel length, consonant class, word boundary, punctuation.

Embeddings are combined and projected to 512 dimensions, followed by a local two-layer Conv1D co-articulation block and RMSNorm.

## Text encoder

Bidirectional Transformer with RMSNorm, 8-head self-attention, and SwiGLU FFN.

## Training-only continuous audio encoder

The encoder downsamples 24000 Hz audio by 960x using strides 12, 4, 4, 5 and projects to 64 channels. Therefore 960 samples correspond to one latent frame and one second of audio corresponds to 25 latent frames.

## Hierarchical prosody

Training posterior:

`q(P | X, T)`

Inference prior:

`p(P | T)`

Each hierarchy produces a 64-dimensional sample through a 128-dimensional mean/log-variance head.

Auxiliary waveform-derived targets include F0, energy, and voicing. These are computed from the waveform and are not manual labels.

## Total length distribution

The model predicts `mu_L` and `logvar_L` from text context and utterance prosody. Total latent frame count is sampled at inference:

`T_a = round(mu_L + exp(0.5 logvar_L) * epsilon)`

with `epsilon ~ N(0, 1)` and a minimum frame count.

## Soft monotonic alignment field

For audio frame `i` and text token `j`:

`M[i,j] = -alpha * (j/N - i/T_a)^2`

The field is added to cross-attention logits. It is a learned soft monotonic inductive bias and is not a mathematical guarantee of monotonicity.

## Shared flow transformer

Let `z1` be the frozen Phase-0 audio latent and `z0 ~ N(0, I)`. Sample `t ~ U(0,1)`:

`zt = (1-t) z0 + t z1`

`ut = z1 - z0`

The flow network predicts `v_theta(zt, t, text, prosody, speaker)` and minimizes:

`L_CFM = E ||v_theta - ut||^2`

## Decoder

The decoder has three parallel learned branches. Subband targets are low 0-1 kHz, mid 1-8 kHz, and high 8-12 kHz because the 24 kHz waveform Nyquist limit is 12 kHz.

Each branch upsamples by 5 x 4 x 4 x 12 = 960.

## Important implementation rule

The dataset does not provide token-level acoustic alignment. The architecture therefore never constructs a token duration table. Text-to-audio correspondence is learned inside the flow model from the soft monotonic attention prior.
