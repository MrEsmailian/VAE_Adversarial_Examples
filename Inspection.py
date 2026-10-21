import torch
import lpips
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

class Evaluate:
    @staticmethod
    def _prepare_target(target_image, device):
        target_image = target_image.to(device)

        if target_image.dim() == 3:
            target_image = target_image.unsqueeze(0)
        if target_image.ndim != 4 or target_image.size(0) != 1:
            raise ValueError('target_image must have shape (C, H, W) or (1, C, H, W)')

        return target_image

    @staticmethod
    def _get_representation(vae, images, space):
        mu, _ = vae.encoder(images)

        if space == 'latent':
            return mu
        elif space == 'reconstruction':
            return vae.decoder(mu)
        else:
            raise ValueError("space must be 'latent' or 'reconstruction'")

    @staticmethod
    def _get_distances(vae, attack, dataloader, target_image, space='latent', device='cpu'):
        device = torch.device(device)
        vae.to(device)
        vae.eval()
        target_image = Evaluate._prepare_target(target_image, device)

        with torch.inference_mode():
            target_rep = Evaluate._get_representation(vae, target_image, space)

        source_distances = []
        target_distances = []

        for source_images, _ in dataloader:
            source_images = source_images.to(device)
            adv_images, _, _ = attack.generate(source_images, target_image)

            with torch.inference_mode():
                source_rep = Evaluate._get_representation(vae, source_images, space)
                adv_rep = Evaluate._get_representation(vae, adv_images, space)

                dist_to_source = ((adv_rep - source_rep).flatten(1).norm(p=2, dim=1))
                dist_to_target = ((adv_rep - target_rep).flatten(1).norm(p=2, dim=1))

                source_distances.append(dist_to_source.cpu())
                target_distances.append(dist_to_target.cpu())

        if not source_distances:
            raise ValueError('dataloader is empty')

        return torch.cat(source_distances), torch.cat(target_distances)

    @staticmethod
    def targeted_pgd(model, attack, dataloader, target_label, correct_only=False):
        device = next(model.parameters()).device
        model.eval()

        successes = 0
        eligible = 0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            targets = torch.full_like(labels, target_label)

            with torch.inference_mode():
                clean_preds = model(images).argmax(dim=1)

            valid = clean_preds != targets
            if correct_only:
                valid &= (clean_preds == labels) & (labels != targets)

            if not valid.any().item():
                continue

            adv_images = attack(images, targets)

            with torch.inference_mode():
                adv_preds = model(adv_images).argmax(dim=1)
                successes += ((adv_preds == targets) & valid).sum().item()
                eligible += valid.sum().item()

        if eligible == 0:
            return 0.0, 0.0

        mean = successes / eligible
        variance = mean * (1.0 - mean)
        return mean, variance

    @staticmethod
    def latent_attack_success(vae, attack, dataloader, target_image, sensitivity=1, device='cpu'):
        dist_to_source, dist_to_target = Evaluate._get_distances(vae, attack, dataloader, target_image, 'latent', device)

        successes = (sensitivity * dist_to_target < dist_to_source).float()

        return successes.mean().item(), successes.var(unbiased=False).item()

    @staticmethod
    def reconstruction_attack_success(vae, attack, dataloader, target_image, sensitivity=1, device='cpu'):
        dist_to_source, dist_to_target = Evaluate._get_distances(vae, attack, dataloader, target_image, 'reconstruction', device)

        successes = (sensitivity * dist_to_target < dist_to_source).float()

        return successes.mean().item(), successes.var(unbiased=False).item()

    @staticmethod
    def latent_distance(vae, attack, dataloader, target_image, relative=True, device='cpu'):
        dist_to_source, dist_to_target = Evaluate._get_distances(vae, attack, dataloader, target_image, 'latent', device)

        if relative:
            eps = 1e-8
            distances = torch.log(dist_to_target.clamp(min=eps)) - torch.log(dist_to_source.clamp(min=eps))
        else:
            distances = dist_to_target

        return distances.mean().item(), distances.var(unbiased=False).item()

    @staticmethod
    def reconstruction_distance(vae, attack, dataloader, target_image, relative=True, device='cpu'):
        dist_to_source, dist_to_target = Evaluate._get_distances(vae, attack, dataloader, target_image, 'reconstruction', device)

        if relative:
            eps = 1e-8
            distances = torch.log(dist_to_target.clamp(min=eps)) - torch.log(dist_to_source.clamp(min=eps))
        else:
            distances = dist_to_target

        return distances.mean().item(), distances.var(unbiased=False).item()

    @staticmethod
    def perturbation_magnitude(attack, dataloader, target_image, lp_norm='inf', device='cpu'):
        target_image = Evaluate._prepare_target(target_image, device)
        l_distances = []

        for source_images, _ in dataloader:
            source_images = source_images.to(device)
            adv_images, _, _ = attack.generate(source_images, target_image)

            with torch.inference_mode():
                perturbations = (adv_images - source_images).flatten(1)
                l = perturbations.norm(p=lp_norm, dim=1)
                l_distances.append(l.cpu())

        if not l_distances:
            raise ValueError("dataloader is empty")

        l_distances = torch.cat(l_distances)

        return l_distances.mean().item(), l_distances.var(unbiased=False).item()

    @staticmethod
    def perceptual_lpips(vae, attack, dataloader, target_image, comparison='source', device='cpu'):
        if comparison not in ('source', 'target'):
            raise ValueError("comparison must be 'source' or 'target'")

        vae.to(device).eval()
        metric = lpips.LPIPS(net='alex').to(device).eval()
        target_image = Evaluate._prepare_target(target_image, device)
        distances = []

        if comparison == 'target':
            with torch.inference_mode():
                target_recon = Evaluate._get_representation(vae, target_image, 'reconstruction')

        for source_images, _ in dataloader:
            source_images = source_images.to(device)
            adv_images, _, _ = attack.generate(source_images, target_image)

            with torch.inference_mode():
                if comparison == 'source':
                    image1, image2 = source_images, adv_images
                else:
                    image1 = Evaluate._get_representation(vae, adv_images, 'reconstruction')
                    image2 = target_recon.expand_as(image1)

                image1 = image1 * 2 - 1
                image2 = image2 * 2 - 1

                if image1.shape[1] == 1:
                    image1 = image1.repeat(1, 3, 1, 1)
                    image2 = image2.repeat(1, 3, 1, 1)

                image1 = F.interpolate(image1, size=(64, 64), mode='bilinear', align_corners=False)
                image2 = F.interpolate(image2, size=(64, 64), mode='bilinear', align_corners=False)

                scores = metric(image1, image2).flatten()
                distances.append(scores.cpu())

        if not distances:
            raise ValueError("dataloader is empty")

        distances = torch.cat(distances)
        return distances.mean().item(), distances.var(unbiased=False).item()

    @staticmethod
    def perceptual_ssim(vae, attack, dataloader, target_image, comparison='source', device='cpu'):
        if comparison not in ('source', 'target'):
            raise ValueError("comparison must be 'source' or 'target'")

        vae.to(device).eval()
        target_image = Evaluate._prepare_target(target_image, device)
        scores_list = []

        if comparison == 'target':
            with torch.inference_mode():
                target_recon = Evaluate._get_representation(vae, target_image, 'reconstruction')

        for source_images, _ in dataloader:
            source_images = source_images.to(device)
            adv_images, _, _ = attack.generate(source_images, target_image)

            with torch.inference_mode():
                if comparison == 'source':
                    image1, image2 = source_images, adv_images
                else:
                    image1 = Evaluate._get_representation(vae, adv_images, 'reconstruction')
                    image2 = target_recon.expand_as(image1)

                scores = ssim(
                    image1, image2,
                    data_range=1.0,
                    reduction='none'
                )
                scores_list.append(scores.flatten().cpu())

        if not scores_list:
            raise ValueError("dataloader is empty")

        scores = torch.cat(scores_list)
        return scores.mean().item(), scores.var(unbiased=False).item()