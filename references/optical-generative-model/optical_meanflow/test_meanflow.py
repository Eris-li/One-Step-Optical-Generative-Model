"""
Test MeanFlow (Teacher Model) for MNIST generation
Based on: https://github.com/zhuyu-cs/MeanFlow

Usage:
    Training:  python debug_scripts/test_meanflow.py --mode train --num_epochs 10
    Generate:  python debug_scripts/test_meanflow.py --mode generate --checkpoint <path> --specific_digit 7
"""
import os
import sys
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.func import jvp
from diffusers.utils import make_image_grid
from tqdm.auto import tqdm
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Optical-MeanFlow', 'MeanFlow'))

from sit import SiT


class SILoss:
    def __init__(
            self,
            path_type="linear",
            weighting="uniform",
            time_sampler="logit_normal",
            time_mu=-0.4,
            time_sigma=1.0,
            ratio_r_not_equal_t=0.75,
            adaptive_p=1.0,
            label_dropout_prob=0.1,
            cfg_omega=1.0,
            cfg_kappa=0.0,
            cfg_min_t=0.0,
            cfg_max_t=0.8,
    ):
        self.weighting = weighting
        self.path_type = path_type
        self.time_sampler = time_sampler
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.ratio_r_not_equal_t = ratio_r_not_equal_t
        self.label_dropout_prob = label_dropout_prob
        self.adaptive_p = adaptive_p
        self.cfg_omega = cfg_omega
        self.cfg_kappa = cfg_kappa
        self.cfg_min_t = cfg_min_t
        self.cfg_max_t = cfg_max_t

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t = 1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def sample_time_steps(self, batch_size, device):
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * self.time_sigma + self.time_mu
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")

        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]

        fraction_equal = 1.0 - self.ratio_r_not_equal_t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        r = torch.where(equal_mask, t, r)

        return r, t

    def __call__(self, model, images, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        else:
            model_kwargs = model_kwargs.copy()

        batch_size = images.shape[0]
        device = images.device

        unconditional_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if model_kwargs.get('y') is not None and self.label_dropout_prob > 0:
            y = model_kwargs['y'].clone()
            batch_size = y.shape[0]
            num_classes = model.num_classes
            dropout_mask = torch.rand(batch_size, device=y.device) < self.label_dropout_prob

            y[dropout_mask] = num_classes
            model_kwargs['y'] = y
            unconditional_mask = dropout_mask

        r, t = self.sample_time_steps(batch_size, device)

        noises = torch.randn_like(images)

        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(-1, 1, 1, 1))
        z_t = alpha_t * images + sigma_t * noises

        v_t = d_alpha_t * images + d_sigma_t * noises
        time_diff = (t - r).view(-1, 1, 1, 1)

        u_target = torch.zeros_like(v_t)

        u = model(z_t, r, t, **model_kwargs)

        cfg_time_mask = (t >= self.cfg_min_t) & (t <= self.cfg_max_t) & (~unconditional_mask)

        if model_kwargs.get('y') is not None and cfg_time_mask.any():
            cfg_indices = torch.where(cfg_time_mask)[0]
            no_cfg_indices = torch.where(~cfg_time_mask)[0]

            u_target = torch.zeros_like(v_t)

            if len(cfg_indices) > 0:
                cfg_z_t = z_t[cfg_indices]
                cfg_v_t = v_t[cfg_indices]
                cfg_r = r[cfg_indices]
                cfg_t = t[cfg_indices]
                cfg_time_diff = time_diff[cfg_indices]

                cfg_kwargs = {}
                for k, v in model_kwargs.items():
                    if torch.is_tensor(v) and v.shape[0] == batch_size:
                        cfg_kwargs[k] = v[cfg_indices]
                    else:
                        cfg_kwargs[k] = v

                cfg_y = cfg_kwargs.get('y')
                num_classes = model.num_classes

                cfg_z_t_batch = torch.cat([cfg_z_t, cfg_z_t], dim=0)
                cfg_t_batch = torch.cat([cfg_t, cfg_t], dim=0)
                cfg_t_end_batch = torch.cat([cfg_t, cfg_t], dim=0)
                cfg_y_batch = torch.cat([cfg_y, torch.full_like(cfg_y, num_classes)], dim=0)

                cfg_combined_kwargs = cfg_kwargs.copy()
                cfg_combined_kwargs['y'] = cfg_y_batch

                with torch.no_grad():
                    cfg_combined_u_at_t = model(cfg_z_t_batch, cfg_t_batch, cfg_t_end_batch, **cfg_combined_kwargs)
                    cfg_u_cond_at_t, cfg_u_uncond_at_t = torch.chunk(cfg_combined_u_at_t, 2, dim=0)
                    cfg_v_tilde = (self.cfg_omega * cfg_v_t +
                                   self.cfg_kappa * cfg_u_cond_at_t +
                                   (1 - self.cfg_omega - self.cfg_kappa) * cfg_u_uncond_at_t)

                def fn_current_cfg(z, cur_r, cur_t):
                    return model(z, cur_r, cur_t, **cfg_kwargs)

                primals = (cfg_z_t, cfg_r, cfg_t)
                tangents = (cfg_v_tilde, torch.zeros_like(cfg_r), torch.ones_like(cfg_t))
                _, cfg_dudt = jvp(fn_current_cfg, primals, tangents)

                cfg_u_target = cfg_v_tilde - cfg_time_diff * cfg_dudt
                u_target[cfg_indices] = cfg_u_target

            if len(no_cfg_indices) > 0:
                no_cfg_z_t = z_t[no_cfg_indices]
                no_cfg_v_t = v_t[no_cfg_indices]
                no_cfg_r = r[no_cfg_indices]
                no_cfg_t = t[no_cfg_indices]
                no_cfg_time_diff = time_diff[no_cfg_indices]

                no_cfg_kwargs = {}
                for k, v in model_kwargs.items():
                    if torch.is_tensor(v) and v.shape[0] == batch_size:
                        no_cfg_kwargs[k] = v[no_cfg_indices]
                    else:
                        no_cfg_kwargs[k] = v

                def fn_current_no_cfg(z, cur_r, cur_t):
                    return model(z, cur_r, cur_t, **no_cfg_kwargs)

                primals = (no_cfg_z_t, no_cfg_r, no_cfg_t)
                tangents = (no_cfg_v_t, torch.zeros_like(no_cfg_r), torch.ones_like(no_cfg_t))
                _, no_cfg_dudt = jvp(fn_current_no_cfg, primals, tangents)

                no_cfg_u_target = no_cfg_v_t - no_cfg_time_diff * no_cfg_dudt
                u_target[no_cfg_indices] = no_cfg_u_target
        else:
            primals = (z_t, r, t)
            tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))

            def fn_current(z, cur_r, cur_t):
                return model(z, cur_r, cur_t, **model_kwargs)

            _, dudt = jvp(fn_current, primals, tangents)

            u_target = v_t - time_diff * dudt

        error = u - u_target.detach()
        loss_mid = torch.sum((error ** 2).reshape(error.shape[0], -1), dim=-1)
        if self.weighting == "adaptive":
            weights = 1.0 / (loss_mid.detach() + 1e-3).pow(self.adaptive_p)
            loss = weights * loss_mid
        else:
            loss = loss_mid
        loss_mean_ref = torch.mean((error ** 2))
        return loss, loss_mean_ref


@torch.no_grad()
def meanflow_sampler(model, latents, y=None, cfg_scale=1.0, num_steps=1):
    batch_size = latents.shape[0]
    device = latents.device

    do_cfg = y is not None and cfg_scale > 1.0
    if do_cfg:
        num_classes = model.num_classes
        null_y = torch.full_like(y, num_classes)

    if num_steps == 1:
        r = torch.zeros(batch_size, device=device)
        t = torch.ones(batch_size, device=device)

        if do_cfg:
            z_combined = torch.cat([latents, latents], dim=0)
            r_combined = torch.cat([r, r], dim=0)
            t_combined = torch.cat([t, t], dim=0)
            y_combined = torch.cat([y, null_y], dim=0)

            u_combined = model(z_combined, r_combined, t_combined, y=y_combined)
            u_cond, u_uncond = u_combined.chunk(2, dim=0)

            u = u_uncond + cfg_scale * (u_cond - u_uncond)
        else:
            u = model(latents, r, t, y=y)

        x0 = latents - u

    else:
        z = latents

        time_steps = torch.linspace(1, 0, num_steps + 1, device=device)

        for i in range(num_steps):
            t_cur = time_steps[i]
            t_next = time_steps[i + 1]

            t = torch.full((batch_size,), t_cur, device=device)
            r = torch.full((batch_size,), t_next, device=device)

            if do_cfg:
                z_combined = torch.cat([z, z], dim=0)
                r_combined = torch.cat([r, r], dim=0)
                t_combined = torch.cat([t, t], dim=0)
                y_combined = torch.cat([y, null_y], dim=0)

                u_combined = model(z_combined, r_combined, t_combined, y=y_combined)
                u_cond, u_uncond = u_combined.chunk(2, dim=0)

                u = u_uncond + cfg_scale * (u_cond - u_uncond)
            else:
                u = model(z, r, t, y=y)

            z = z - (t_cur - t_next) * u

        x0 = z

    return x0


class MeanFlowGenerator:
    def __init__(self, output_dir="./debug_scripts/meanflow_checkpoints"):
        self.output_dir = output_dir
        self.model = None
        self.ema_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_fn = None

        self.config = {
            'input_size': 32,
            'patch_size': 2,
            'in_channels': 1,
            'hidden_size': 192,
            'decoder_hidden_size': 192,
            'depth': 4,
            'num_heads': 4,
            'mlp_ratio': 4.0,
            'class_dropout_prob': 0.1,
            'num_classes': 10,
        }
        self.block_kwargs = {
            'qk_norm': False,
            'fused_attn': False,
        }

    def _create_model(self):
        model = SiT(**self.config, **self.block_kwargs)
        return model

    @staticmethod
    @torch.no_grad()
    def _update_ema(ema_model, model, decay=0.9999):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.mul_(decay).add_(param.data, alpha=1 - decay)

        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)

    def train(self, num_epochs=10, batch_size=128, learning_rate=1e-4,
              target_digit=None, save_interval=5):
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Training MeanFlow on {self.device}")
        print(f"Target digit: {target_digit if target_digit is not None else 'ALL'}")

        self.model = self._create_model().to(self.device)
        self.ema_model = deepcopy(self.model).to(self.device)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        self.loss_fn = SILoss(
            time_sampler="logit_normal",
            time_mu=-0.4,
            time_sigma=1.0,
            ratio_r_not_equal_t=0.75,
            weighting="adaptive",
            label_dropout_prob=0.1,
            cfg_omega=1.0,
            cfg_kappa=0.0,
        )

        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

        if target_digit is not None:
            indices = [i for i, (_, label) in enumerate(dataset) if label == target_digit]
            subset = torch.utils.data.Subset(dataset, indices)
            train_dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            print(f"Filtered to digit {target_digit}: {len(subset)} samples")
        else:
            train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            eps=1e-8,
        )

        import math
        total_steps = len(train_dataloader) * num_epochs
        warmup_steps = int(0.05 * total_steps)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )

        self.model.train()

        global_step = 0
        for epoch in range(num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
            epoch_loss = 0

            for step, batch in enumerate(train_dataloader):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                model_kwargs = {'y': labels}

                loss, loss_ref = self.loss_fn(self.model, images, model_kwargs)

                loss.mean().backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                self._update_ema(self.ema_model, self.model)

                if global_step < warmup_steps:
                    warmup_scheduler.step()
                else:
                    lr_scheduler.step()

                global_step += 1
                epoch_loss += loss.mean().item()
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    "loss_opt": f"{loss.mean().item():.4f}",
                    "loss_ref": f"{loss_ref.item():.4f}",
                    "lr": f"{current_lr:.2e}"
                })
                progress_bar.update(1)

            progress_bar.close()
            avg_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch+1} Complete, Avg loss_opt: {avg_loss:.4f}, Last loss_ref: {loss_ref.item():.4f}")

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"meanflow_epoch{epoch+1}.pth")
                self.generate(f"{self.output_dir}/meanflow_epoch{epoch+1}.png")

        self.save_checkpoint("meanflow_final.pth")
        print(f"Training complete!")
        return self

    def save_checkpoint(self, filename):
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema_model.state_dict() if self.ema_model is not None else None,
            'config': self.config,
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.config = checkpoint.get('config', self.config)
        self.model = self._create_model().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        ema_state_dict = checkpoint.get('ema_state_dict')
        if ema_state_dict is not None:
            self.ema_model = self._create_model().to(self.device)
            self.ema_model.load_state_dict(ema_state_dict)
            self.ema_model.eval()
        else:
            self.ema_model = None
        print(f"Loaded MeanFlow from {checkpoint_path}")
        return self

    @torch.no_grad()
    def generate(self, output_path="meanflow_output.png", num_images=9,
                 num_inference_steps=1, seed=42, specific_digit=None, cfg_scale=1.5):
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or train() first.")

        model_for_sampling = self.ema_model if self.ema_model is not None else self.model
        model_for_sampling.eval()

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        if specific_digit is not None:
            class_sample = torch.tensor([specific_digit] * num_images,
                                       dtype=torch.int64, device=self.device)
        else:
            class_sample = torch.randint(0, 10, size=(num_images,),
                                        dtype=torch.int64, device=self.device)

        noise = torch.randn(num_images, self.config['in_channels'],
                           self.config['input_size'], self.config['input_size'],
                           device=self.device, generator=generator)

        x0 = meanflow_sampler(model_for_sampling, noise, y=class_sample,
                             cfg_scale=cfg_scale, num_steps=num_inference_steps)

        x0 = (x0 * 0.5 + 0.5).clamp(0, 1)
        x0 = (x0 * 255).byte().cpu().permute(0, 2, 3, 1).numpy()
        pil_images = [Image.fromarray(img.squeeze(), mode="L") for img in x0]

        cols = min(num_images, 3)
        rows = (num_images + cols - 1) // cols
        image_grid = make_image_grid(pil_images, rows=rows, cols=cols)
        image_grid.save(output_path)
        print(f"Saved {num_images} images to {output_path}")
        return image_grid


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test MeanFlow Model")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "generate"])
    parser.add_argument("--output_dir", type=str, default="./debug_scripts/meanflow_checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_inference_steps", type=int, default=1,
                        help="Number of steps for MeanFlow generation (1 = single step)")
    parser.add_argument("--specific_digit", type=int, default=None, help="Generate specific digit (0-9)")
    parser.add_argument("--target_digit", type=int, default=None, help="Train on specific digit only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    args = parser.parse_args()

    model = MeanFlowGenerator(output_dir=args.output_dir)

    if args.mode == "train":
        model.train(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            target_digit=args.target_digit,
            save_interval=args.save_interval
        )
        model.generate(f"{args.output_dir}/meanflow_final.png")

    elif args.mode == "generate":
        if args.checkpoint is None:
            checkpoint_path = os.path.join(args.output_dir, "meanflow_final.pth")
        else:
            checkpoint_path = args.checkpoint

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            print("Please train first: python debug_scripts/test_meanflow.py --mode train")
            return

        model.load(checkpoint_path)
        digit_str = f"_digit{args.specific_digit}" if args.specific_digit is not None else ""
        model.generate(
            f"{args.output_dir}/meanflow_generated{digit_str}.png",
            num_images=9,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            specific_digit=args.specific_digit,
            cfg_scale=args.cfg_scale
        )


if __name__ == "__main__":
    main()
