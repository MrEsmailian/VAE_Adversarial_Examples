import torch
import torch.nn as nn


class PGDTargetedAttack:
    def __init__(self, model, epsilon=8/255, alpha=2/255, steps=40, random_start=True, clip_min=-1.0, clip_max=1.0, device='cpu'):
        self.model = model
        self.device = device

        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

        self.clip_min = clip_min
        self.clip_max = clip_max

        self.criterion = nn.CrossEntropyLoss()

    def __call__(self, images, target_labels):
        self.model.eval()

        images = images.to(self.device)
        target_labels = target_labels.to(self.device)

        adv = images.detach().clone()

        if self.random_start:
            adv += torch.empty_like(adv).uniform_(-self.epsilon, self.epsilon)
            adv = torch.clamp(adv, self.clip_min, self.clip_max)

        for _ in range(self.steps):
            adv.requires_grad_(True)

            logits = self.model(adv)
            loss = self.criterion(logits, target_labels)

            grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
            
            adv = adv.detach() - self.alpha * grad.sign()

            delta = torch.clamp(adv - images, min=-self.epsilon, max=self.epsilon)
            adv = torch.clamp(images + delta, self.clip_min, self.clip_max).detach()

        return adv