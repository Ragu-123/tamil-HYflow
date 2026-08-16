import torch
import torch.nn as nn
from .text_encoder import TamilTextEncoder
from .codec import ContinuousAudioEncoder
from .decoder import MultiBranchSubbandDecoder
from .prosody import ProsodyPosterior, ProsodyPrior, ProsodyFusion, ProsodyReconstruction, kl_gaussian
from .speaker import SpeakerReferenceEncoder
from .length import TotalLengthDistribution
from .flow import SharedFlowTransformer

class TamilHyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.text = TamilTextEncoder()
        self.audio_encoder = ContinuousAudioEncoder()
        self.decoder = MultiBranchSubbandDecoder()
        self.posterior = ProsodyPosterior()
        self.prior = ProsodyPrior()
        self.prosody_fusion = ProsodyFusion()
        self.prosody_rec = ProsodyReconstruction()
        self.speaker = SpeakerReferenceEncoder()
        self.length = TotalLengthDistribution()
        self.flow = SharedFlowTransformer()

    def encode_text(self, features, text_mask=None):
        return self.text(features, text_mask)

    def encode_audio(self, audio):
        return self.audio_encoder(audio)

    def decode_latent(self, z):
        return self.decoder(z)

    def posterior_pass(self, z, h_text, text_mask=None):
        return self.posterior(z, h_text, text_mask)

    def prior_pass(self, h_text, text_mask=None):
        return self.prior(h_text, text_mask)

    def cfm_velocity(self, z_t, h_text, prosody, speaker, t, text_mask=None):
        return self.flow(z_t, h_text, prosody, speaker, t, text_mask)

    @staticmethod
    def cfm_pair(z1, t=None):
        b = z1.shape[0]
        if t is None:
            t = torch.rand(b, device=z1.device, dtype=z1.dtype)
        z0 = torch.randn_like(z1)
        zt = (1.0 - t[:, None, None]) * z0 + t[:, None, None] * z1
        target = z1 - z0
        return z0, zt, target, t

    def inference_length(self, h_text, p_u, speaker, min_frames=25, max_frames=2500):
        stats = self.length(h_text, p_u, speaker)
        mu = stats[:, 0]
        sigma = stats[:, 1].clamp(-8, 8).exp().sqrt()
        n = (mu + sigma * torch.randn_like(mu)).round().long()
        return n.clamp(min_frames, max_frames), stats

    def posterior_kl(self, posterior, prior):
        total = 0.0
        for key in ("u", "p", "w", "s"):
            q = posterior[key]
            p = prior[key]
            total = total + kl_gaussian(q[1], q[2], p[1], p[2])
        return total
