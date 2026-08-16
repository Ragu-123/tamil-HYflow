from pathlib import Path
import torch
from tqdm import tqdm
from tamil_hyflow.utils.checkpoint import save_checkpoint

class Trainer:
    def __init__(self, model, optimizer, device, grad_clip=1.0, amp=True):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
        self.amp = amp and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.global_step = 0

    def step(self, loss):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=self.amp):
            scaled = loss
        self.scaler.scale(scaled).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.global_step += 1

    def save(self, path, epoch):
        save_checkpoint(path, self.model, self.optimizer, epoch=epoch, step=self.global_step)
