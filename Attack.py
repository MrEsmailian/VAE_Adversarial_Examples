import torch
import torch.nn as nn

class PGDTargetedAttack:
    def __init__(self, model, epsilon=8/255, alpha=2/255, steps=40, restarts=1, random_start=True, clip_min=-1.0, clip_max=1.0, device="cpu"):
        self.model = model.eval()
        self.device = device
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.restarts = restarts
        self.random_start = random_start

        self.clip_min = clip_min
        self.clip_max = clip_max

        self.criterion = nn.CrossEntropyLoss(reduction="none")

        for p in self.model.parameters():
            p.requires_grad_(False)

    def __call__(self, images, target_labels):
        images = images.to(self.device)
        target_labels = target_labels.to(self.device)
        batch_size = images.size(0)
        best_adv = images.clone()
        best_loss = torch.full((batch_size,), float("inf"), device=self.device)

        for _ in range(self.restarts):
            adv = images.clone().detach()
            if self.random_start:
                delta = torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
                adv = images + delta
                adv = torch.clamp(adv, self.clip_min, self.clip_max)
                delta = torch.clamp(adv - images, -self.epsilon, self.epsilon)
                adv = (images + delta).detach()

            for _ in range(self.steps):
                adv.requires_grad_(True)
                logits = self.model(adv)
                loss = self.criterion(logits, target_labels).mean()
                grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
                adv = adv.detach() - self.alpha * grad.sign()
                delta = torch.clamp(adv - images, -self.epsilon, self.epsilon)
                adv = torch.clamp(images + delta, self.clip_min, self.clip_max).detach()

            with torch.no_grad():
                logits = self.model(adv)
                losses = self.criterion(logits, target_labels)
                better = losses < best_loss
                best_loss[better] = losses[better]
                best_adv[better] = adv[better]

        return best_adv

import torch
import torch.nn as nn


class PGDLatentAttack:
    def __init__(self, encoder, epsilon=8/255, alpha=2/255, steps=40, restarts=1, random_start=True, clip_min=-1.0, clip_max=1.0, device="cpu"):
        self.encoder = encoder.eval()
        self.device = device
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.restarts = restarts
        self.random_start = random_start
        self.clip_min = clip_min
        self.clip_max = clip_max
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def __call__(self, source_images, target_image):
        source_images = source_images.to(self.device)
        if target_image.dim() == 3:
            target_image = target_image.unsqueeze(0)
        target_image = target_image.to(self.device)
        with torch.no_grad():
            target_mu, _ = self.encoder(target_image)
        target_mu = target_mu.expand(source_images.size(0), -1)
        batch_size = source_images.size(0)
        best_adv = source_images.clone()
        best_loss = torch.full((batch_size,), float("inf"), device=self.device)

        for _ in range(self.restarts):
            adv = source_images.clone().detach()
            if self.random_start:
                delta = torch.empty_like(source_images).uniform_(-self.epsilon, self.epsilon)
                adv = source_images + delta
                adv = torch.clamp(adv, self.clip_min, self.clip_max,)
                delta = torch.clamp(adv - source_images, -self.epsilon, self.epsilon,)
                adv = (source_images + delta).detach()

            for _ in range(self.steps):
                adv.requires_grad_(True)
                adv_mu, _ = self.encoder(adv)
                loss = ((adv_mu - target_mu) ** 2).mean()
                grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
                adv = adv.detach() - self.alpha * grad.sign()
                delta = torch.clamp(adv - source_images, -self.epsilon, self.epsilon)
                adv = torch.clamp(source_images + delta, self.clip_min, self.clip_max).detach()

            with torch.no_grad():
                adv_mu, _ = self.encoder(adv)
                losses = ((adv_mu - target_mu) ** 2).flatten(1).mean(dim=1)
                better = losses < best_loss
                best_loss[better] = losses[better]
                best_adv[better] = adv[better]

        return best_adv