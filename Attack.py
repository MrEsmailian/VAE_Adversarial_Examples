import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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


class LatentAttack:
    def __init__(self, encoder, lr=0.1, iterations=1000, lambda_weight=1.0, device='cpu'):
        self.encoder = encoder.eval()
        self.lr = lr
        self.iterations = iterations
        self.lambda_weight = lambda_weight
        self.device = device

    def generate(self, x_orig, x_target):
        x_orig = x_orig.to(self.device)
        x_target = x_target.to(self.device)
        for param in self.encoder.parameters():
            param.requires_grad = False
        with torch.no_grad():
            z_target = self.encoder(x_target)
            if isinstance(z_target, tuple):
                z_target = z_target[0] 
                
        delta = torch.zeros_like(x_orig, requires_grad=True)
        optimizer = optim.Adam([delta], lr=self.lr)

        loss_latents, loss_dists = [], []
        for i in range(self.iterations):
            optimizer.zero_grad()
            x_adv = x_orig + delta
            z_adv = self.encoder(x_adv)
            if isinstance(z_adv, tuple):
                z_adv = z_adv[0]
            # loss_latent = torch.sqrt(torch.sum((z_adv - z_target) ** 2, dim=1) + 1e-8)
            # loss_dist = torch.sqrt(torch.sum(delta ** 2, dim=(1, 2, 3)) + 1e-8)
            loss_latent = torch.sum((z_adv - z_target) ** 2, dim=1)
            loss_dist = torch.sum(delta ** 2, dim=(1, 2, 3))
            batch_loss = torch.mean((self.lambda_weight * loss_dist) + loss_latent)
            loss_latents.append(loss_latent.mean().item())
            loss_dists.append(loss_dist.mean().item())
            batch_loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta.copy_(torch.clamp(x_orig + delta, 0, 1.0) - x_orig)
                
        return (x_orig + delta).detach(), loss_latents, loss_dists
    

class LVAEAttack:
    def __init__(self, encoder, decoder, lr=0.1, iterations=1000, lambda_weight=1.0):
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self.iterations = iterations
        self.lambda_weight = lambda_weight

    def generate(self, x_orig, x_target):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        with torch.no_grad():
            z_t = self.encoder(x_target)
            if isinstance(z_t, tuple):
                z_t = z_t[0] 
            x_hat_target = self.decoder(z_t)
        delta = torch.zeros_like(x_orig, requires_grad=True)
        
        optimizer = optim.Adam([delta], lr=self.lr)

        for i in range(self.iterations):
            optimizer.zero_grad()
            
            x_adv = x_orig + delta
            z_adv = self.encoder(x_adv)
            if isinstance(z_adv, tuple):
                z_adv = z_adv[0]
                
            x_hat_adv = self.decoder(z_adv)
            loss_recon = F.mse_loss(x_hat_adv, x_hat_target, reduction='sum')
            loss_dist = torch.norm(delta, p=2)
            loss = (self.lambda_weight * loss_dist) + loss_recon
            loss.backward()
            optimizer.step()

        return torch.clamp(x_orig + delta, 0, 1).detach()