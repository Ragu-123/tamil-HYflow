import torch
from tamil_hyflow.training.losses import codec_loss

class Phase0Runner:
    def __init__(self, encoder, decoder, optimizer, device, grad_clip=1.0, amp=True):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
        self.amp = amp and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

    def train_step(self, audio):
        audio = audio.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=self.amp):
            z = self.encoder(audio)
            pred, branches = self.decoder(z)
            loss, parts = codec_loss(pred, branches, audio)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.detach(), parts
