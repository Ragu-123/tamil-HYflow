import torch
from tamil_hyflow.models.codec import ContinuousAudioEncoder
from tamil_hyflow.models.decoder import MultiBranchSubbandDecoder
from tamil_hyflow.models.hyflow import TamilHyFlow

def count(module):
    return sum(p.numel() for p in module.parameters())

def main():
    model = TamilHyFlow()
    print("TamilHyFlow total parameters:", count(model))
    print("Text:", count(model.text))
    print("Audio encoder:", count(model.audio_encoder))
    print("Decoder:", count(model.decoder))
    print("Posterior:", count(model.posterior))
    print("Prior:", count(model.prior))
    print("Prosody fusion:", count(model.prosody_fusion))
    print("Prosody reconstruction:", count(model.prosody_rec))
    print("Speaker:", count(model.speaker))
    print("Length:", count(model.length))
    print("Flow:", count(model.flow))

if __name__ == "__main__":
    main()
