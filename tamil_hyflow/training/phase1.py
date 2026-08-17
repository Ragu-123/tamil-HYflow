import torch
from tamil_hyflow.training.losses import flow_loss, prosody_kl, prosody_reconstruction_loss, total_length_loss
from tamil_hyflow.utils.prosody import extract_prosody

class Phase1Runner:
    def __init__(self, model, optimizer, device, grad_clip=1.0, amp=True, kl_weight=0.01, prosody_weight=1.0, length_weight=0.1):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
        self.amp = amp and device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.kl_weight = kl_weight
        self.prosody_weight = prosody_weight
        self.length_weight = length_weight

    def train_step(self, batch):
        m = self.model.module if hasattr(self.model, "module") else self.model
        audio = batch.audio.to(self.device)
        features = tuple(x.to(self.device) for x in (batch.cons_id, batch.vowel_id, batch.length_id, batch.class_id, batch.word_bound_id, batch.punct_id))
        text_mask = batch.text_mask.to(self.device)

        autocast_ctx = torch.amp.autocast("cuda", enabled=self.amp) if hasattr(torch, "amp") else torch.cuda.amp.autocast(enabled=self.amp)
        with autocast_ctx:
            with torch.no_grad():
                z1 = m.encode_audio(audio)
            h = m.encode_text(features, text_mask)
            posterior = m.posterior_pass(z1, h, text_mask)
            prior = m.prior_pass(h, text_mask)
            speaker = m.speaker(audio)
            p_fused = m.prosody_fusion(posterior, z1.shape[1])
            rec_local, global_rec = m.prosody_rec(p_fused, [posterior[k][0] for k in ("u", "p", "w", "s")])
            f0, energy, voicing = extract_prosody(audio, 24000)
            z0, zt, target, t = m.cfm_pair(z1)
            velocity, weights = m.cfm_velocity(zt, h, posterior, speaker, t, text_mask)
            rec_loss, _ = prosody_reconstruction_loss(rec_local, (f0[:, :rec_local.shape[1]], energy[:, :rec_local.shape[1]], voicing[:, :rec_local.shape[1]]))
            kl = prosody_kl(posterior, prior)
            target_frames = torch.tensor([z1.shape[1]] * audio.shape[0], device=self.device, dtype=torch.float32)
            stats = m.length(h, posterior["u"], speaker)
            length = total_length_loss(stats, target_frames)
            flow = flow_loss(velocity, target)
            loss = flow + self.kl_weight * kl + self.prosody_weight * rec_loss + self.length_weight * length

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.detach(), {"flow": flow.detach(), "kl": kl.detach(), "prosody": rec_loss.detach(), "length": length.detach()}
