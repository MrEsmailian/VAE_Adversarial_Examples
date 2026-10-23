import torch
import lpips
import math
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchmetrics.functional.image import structural_similarity_index_measure as ssim


class Evaluator:
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
        target_image = Evaluator._prepare_target(target_image, device)

        with torch.inference_mode():
            target_rep = Evaluator._get_representation(vae, target_image, space)

        source_distances = []
        target_distances = []

        for source_images, _ in dataloader:
            source_images = source_images.to(device)
            adv_images, _, _ = attack.generate(source_images, target_image)

            with torch.inference_mode():
                source_rep = Evaluator._get_representation(vae, source_images, space)
                adv_rep = Evaluator._get_representation(vae, adv_images, space)

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
        dist_to_source, dist_to_target = Evaluator._get_distances(vae, attack, dataloader, target_image, 'latent', device)

        successes = (sensitivity * dist_to_target < dist_to_source).float()

        return successes.mean().item(), successes.var(unbiased=False).item()

    @staticmethod
    def reconstruction_attack_success(vae, attack, dataloader, target_image, sensitivity=1, device='cpu'):
        dist_to_source, dist_to_target = Evaluator._get_distances(vae, attack, dataloader, target_image, 'reconstruction', device)

        successes = (sensitivity * dist_to_target < dist_to_source).float()

        return successes.mean().item(), successes.var(unbiased=False).item()

    @staticmethod
    def latent_distance(vae, attack, dataloader, target_image, relative=True, device='cpu'):
        dist_to_source, dist_to_target = Evaluator._get_distances(vae, attack, dataloader, target_image, 'latent', device)

        if relative:
            eps = 1e-8
            distances = torch.log(dist_to_target.clamp(min=eps)) - torch.log(dist_to_source.clamp(min=eps))
        else:
            distances = dist_to_target

        return distances.mean().item(), distances.var(unbiased=False).item()

    @staticmethod
    def reconstruction_distance(vae, attack, dataloader, target_image, relative=True, device='cpu'):
        dist_to_source, dist_to_target = Evaluator._get_distances(vae, attack, dataloader, target_image, 'reconstruction', device)

        if relative:
            eps = 1e-8
            distances = torch.log(dist_to_target.clamp(min=eps)) - torch.log(dist_to_source.clamp(min=eps))
        else:
            distances = dist_to_target

        return distances.mean().item(), distances.var(unbiased=False).item()

    @staticmethod
    def perturbation_magnitude(attack, dataloader, target_image, lp_norm='inf', device='cpu'):
        target_image = Evaluator._prepare_target(target_image, device)
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
        target_image = Evaluator._prepare_target(target_image, device)
        distances = []

        if comparison == 'target':
            with torch.inference_mode():
                target_recon = Evaluator._get_representation(vae, target_image, 'reconstruction')

        for source_images, _ in dataloader:
            source_images = source_images.to(device)
            adv_images, _, _ = attack.generate(source_images, target_image)

            with torch.inference_mode():
                if comparison == 'source':
                    image1, image2 = source_images, adv_images
                else:
                    image1 = Evaluator._get_representation(vae, adv_images, 'reconstruction')
                    image2 = target_recon.expand_as(image1)

                image1 = image1 * 2 - 1
                image2 = image2 * 2 - 1

                if image1.shape[1] == 1:
                    image1 = image1.repeat(1, 3, 1, 1)
                    image2 = image2.repeat(1, 3, 1, 1)

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
        target_image = Evaluator._prepare_target(target_image, device)
        scores_list = []

        if comparison == 'target':
            with torch.inference_mode():
                target_recon = Evaluator._get_representation(vae, target_image, 'reconstruction')

        for source_images, _ in dataloader:
            source_images = source_images.to(device)
            adv_images, _, _ = attack.generate(source_images, target_image)

            with torch.inference_mode():
                if comparison == 'source':
                    image1, image2 = source_images, adv_images
                else:
                    image1 = Evaluator._get_representation(vae, adv_images, 'reconstruction')
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


class Visualizer:
    @staticmethod
    def show_batch(loader, class_names=None, n_cols=8, show_label=False, max_attrs=3):
        images, labels = next(iter(loader))
        batch_size = len(images)
        n_rows = math.ceil(batch_size / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), squeeze=False,)

        axes = axes.ravel()
        for ax in axes:
            ax.axis("off")

        for i, (img, label) in enumerate(zip(images, labels)):
            if img.shape[0] == 1:
                axes[i].imshow(img.squeeze(), cmap="gray")
            else:
                axes[i].imshow(img.permute(1, 2, 0))

            if label.ndim == 0:
                label = label.item()
                title = class_names[label] if class_names is not None else str(label)
            else:
                active = torch.where(label == 1)[0].tolist()
                if class_names is not None:
                    attrs = [class_names[j] for j in active[:max_attrs]]
                else:
                    attrs = [str(j) for j in active[:max_attrs]]
                if len(active) > max_attrs:
                    attrs.append("...")
                title = "\n".join(attrs)
            if show_label:
                axes[i].set_title(title, fontsize=10)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_reconstructions(model, dataloader, num_images=-1, n_cols=16):
        model.eval()

        images, _ = next(iter(dataloader))
        if num_images == -1:
            num_images = len(images)

        if num_images > len(images):
            raise ValueError(
                f"num_images ({num_images}) exceeds batch size ({len(images)})."
            )
        images = images[:num_images].to(model.device)
        with torch.no_grad():
            reconstructions, _, _ = model(images)
        images = images.cpu()
        reconstructions = reconstructions.cpu()
        if n_cols == -1:
            n_cols = num_images
        n_rows = math.ceil(num_images / n_cols)

        fig, axes = plt.subplots(2 * n_rows, n_cols, figsize=(2 * n_cols, 4 * n_rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for i in range(num_images):
            row = (i // n_cols) * 2
            col = i % n_cols
            img = images[i]
            img = img.permute(1,2,0)
            axes[row, col].imshow(img)
            rec = reconstructions[i]
            rec = rec.permute(1,2,0)
            axes[row+1,col].imshow(rec)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_curves(curve_data, title, x_axis, y_axis, save=True):
        _, length = list(curve_data.items())[0]
        length = range(1, len(length) + 1)

        plt.figure(figsize=(8, 5))
        for key in curve_data:
            plt.plot(length, curve_data[key], label=key, linewidth=2)

        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.xticks(length)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(f'.\\Output\\{title}.svg', format="svg", bbox_inches="tight")
        plt.show()

    @staticmethod
    def visualize_attack_reconstruction(images, adv_images_dict, vae, title, num_images=16, save=True):
        device = next(vae.parameters()).device
        vae.eval()

        images = images[:num_images].to(device)
        num_images = len(images)

        attack_names = list(adv_images_dict.keys())
        num_attacks = len(attack_names)
        num_cols = 2 * (num_attacks + 1)

        with torch.no_grad():
            recon_images, _, _ = vae(images)

            adv_images = {
                name: imgs[:num_images].to(device)
                for name, imgs in adv_images_dict.items()
            }

            recon_adv = {
                name: vae(imgs)[0]
                for name, imgs in adv_images.items()
            }

        # Move everything to CPU
        images = images.cpu()
        recon_images = recon_images.cpu()
        adv_images = {name: imgs.cpu() for name, imgs in adv_images.items()}
        recon_adv = {name: imgs.cpu() for name, imgs in recon_adv.items()}

        fig, axes = plt.subplots(
            num_images, num_cols,
            figsize=(num_cols, num_images),
            squeeze=False
        )

        # Column titles
        column_titles = (["Original"] + attack_names + ["Original"] + attack_names)

        for j, name in enumerate(column_titles):
            axes[0, j].set_title(name, fontsize=8)

        for i in range(num_images):
            imgs = (
                [images[i]] +
                [adv_images[name][i] for name in attack_names] +
                [recon_images[i]] +
                [recon_adv[name][i] for name in attack_names]
            )

            for j, img in enumerate(imgs):
                if img.shape[0] == 1:
                    axes[i, j].imshow(img.squeeze(0), cmap="gray", vmin=0, vmax=1)
                else:
                    axes[i, j].imshow(img.permute(1, 2, 0).clamp(0, 1))

                axes[i, j].axis("off")

        group_width = num_attacks + 1

        fig.text(0.25, 0.965, "Input Images", ha="center", fontsize=11, fontweight="bold")
        fig.text(0.75, 0.965, "Reconstructions", ha="center", fontsize=11, fontweight="bold")
        fig.suptitle(title, fontsize=13, y=1.01)

        plt.tight_layout(rect=[0, 0, 1, 0.94], pad=0.2, w_pad=0.1, h_pad=0.1)
        if save:
            fig.savefig(f'.\\Output\\{title}.svg', format="svg", bbox_inches="tight")
        plt.show()

    @staticmethod
    def show_image(image, title, save=True):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
            if image.ndim == 3:
                image = image.permute(1, 2, 0)
            image = image.numpy()

        plt.figure(figsize=(3, 3))
        plt.imshow(image.squeeze())
        plt.title(title)
        plt.axis('off')
        if save:
            plt.savefig(f'.\\Output\\{title}.svg', format="svg", bbox_inches="tight")
        plt.show()

    @staticmethod
    def show_images(images: dict, title, save=True):
        n = len(images)
        if n == 0:
            return

        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        axes = np.atleast_1d(axes)

        for ax, (key, image) in zip(axes, images.items()):
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu()
                if image.ndim == 3:
                    image = image.permute(1, 2, 0)
                image = image.numpy()

            image = image.squeeze()
            ax.imshow(image)
            ax.set_title(str(key))
            ax.axis('off')

        fig.suptitle(title)
        plt.tight_layout()
        if save:
            fig.savefig(f'.\\Output\\{title}.svg', format="svg", bbox_inches="tight")
        plt.show()