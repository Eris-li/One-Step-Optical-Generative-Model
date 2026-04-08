"""
Test DDPM (Teacher Diffusion Model) using original repository code
Usage:
    Training:  python debug_scripts/test_ddpm.py --mode train --num_epochs 10
    Generate: python debug_scripts/test_ddpm.py --mode generate --checkpoint <path> --specific_digit 7
"""
import os
import sys
import torch
import torch.nn.functional as F
import math
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from tqdm.auto import tqdm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline_costum import DDPMPipeline_Costum_ClsEmb


class DDPMGenerator:
    """DDPM Generator that uses standard diffusers library"""

    def __init__(self, output_dir="./debug_scripts/ddpm_checkpoints"):
        self.output_dir = output_dir
        self.model = None
        self.scheduler = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = {
            'sample_size': 32,
            'in_channels': 1,
            'out_channels': 1,
            'down_block_types': ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            'up_block_types': ("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            'block_out_channels': (64, 128, 256, 512),
            'layers_per_block': 2,
            'num_train_timesteps': 1000,
            'num_class_embeds': 10,
        }

    def train(self, num_epochs=10, batch_size=128, learning_rate=1e-4,
              num_inference_steps=50, target_digit=None, save_interval=5):
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Training DDPM on {self.device}")
        print(f"Target digit: {target_digit if target_digit is not None else 'ALL'}")

        self.model = UNet2DModel(**self.config).to(self.device)

        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.config['num_train_timesteps'],
            beta_schedule="linear",
            prediction_type="epsilon"
        )

        preprocess = transforms.Compose([
            transforms.Resize((self.config['sample_size'], self.config['sample_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        full_dataset = datasets.MNIST(root='./data', train=True, transform=preprocess, download=True)

        if target_digit is not None:
            target_indices = (full_dataset.targets == target_digit).nonzero().squeeze()
            if target_indices.dim() == 0:
                target_indices = target_indices.unsqueeze(0)
            train_dataset = torch.utils.data.Subset(full_dataset, target_indices)
            print(f"Filtered to digit {target_digit}: {len(train_dataset)} samples")
        else:
            train_dataset = full_dataset
            print(f"Training on all digits: {len(train_dataset)} samples")

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=0, pin_memory=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(train_dataloader) * 2,
            num_training_steps=len(train_dataloader) * num_epochs
        )

        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(num_epochs):
            self.model.train()
            progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")

            for step, batch in enumerate(train_dataloader):
                clean_images = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                noise = torch.randn_like(clean_images)
                batch_size_cur = clean_images.shape[0]

                timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps, (batch_size_cur,),
                    device=self.device, dtype=torch.int64
                )

                noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)
                noise_pred = self.model(noisy_images, timesteps, class_labels=labels, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()

                progress_bar.update(1)
                if step % 100 == 0:
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            progress_bar.close()
            print(f"Epoch {epoch+1} Complete, Loss: {loss.item():.4f}")

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"ddpm_epoch{epoch+1}.pth")
                self.generate(os.path.join(self.output_dir, f"ddpm_epoch{epoch+1}.png"), num_inference_steps=num_inference_steps)

        self.save_checkpoint("ddpm_final.pth")
        self.generate(os.path.join(self.output_dir, "ddpm_final.png"), num_inference_steps=num_inference_steps)
        print(f"Training complete!")
        return self

    def save_checkpoint(self, filename):
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, filename)
        torch.save({
            'unet_state_dict': self.model.state_dict(),
            'config': self.config
        }, save_path)
        print(f"Saved: {save_path}")

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.config = checkpoint.get('config', self.config)

        self.model = UNet2DModel(**self.config).to(self.device)
        self.model.load_state_dict(checkpoint['unet_state_dict'])
        self.model.eval()

        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.config['num_train_timesteps'],
            beta_schedule="linear",
            prediction_type="epsilon"
        )

        print(f"Loaded DDPM from {checkpoint_path}")
        return self

    def generate(self, output_path="ddpm_output.png", num_images=9,
                 num_inference_steps=50, seed=42, specific_digit=None):
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or train() first.")

        self.model.eval()

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        if specific_digit is not None:
            class_sample = torch.tensor([specific_digit] * num_images,
                                       dtype=torch.int64, device=self.device)
        else:
            class_sample = torch.randint(0, 10, size=(num_images,),
                                        dtype=torch.int64, device=self.device)

        pipeline = DDPMPipeline_Costum_ClsEmb(unet=self.model, scheduler=self.scheduler)
        pipeline.set_progress_bar_config(disable=True)

        images = pipeline(
            batch_size=num_images,
            class_sample=class_sample,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type="pil"
        ).images

        rows = 3
        cols = (num_images + rows - 1) // rows
        w, h = images[0].size
        grid = Image.new('L', (cols * w, rows * h))
        for i, img in enumerate(images):
            grid.paste(img, (i % cols * w, i // cols * h))

        grid.save(output_path)
        print(f"Saved {num_images} images to {output_path}")
        return grid

    def generate_with_fixed_noise(self, output_path="ddpm_output.png",
                                  noise=None, labels=None,
                                  num_inference_steps=50):
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or train() first.")

        self.model.eval()

        num_images = noise.shape[0] if noise is not None else 9

        scheduler = DDIMScheduler(
            num_train_timesteps=self.scheduler.config.num_train_timesteps,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        scheduler.set_timesteps(num_inference_steps)

        if noise is None:
            noise = torch.randn(num_images, self.config['in_channels'],
                               self.config['sample_size'], self.config['sample_size'],
                               device=self.device)

        if labels is None:
            labels = torch.randint(0, 10, size=(num_images,), device=self.device)

        image = noise

        for i, t in enumerate(scheduler.timesteps):
            with torch.no_grad():
                model_pred = self.model(image, t.unsqueeze(0).to(image.device),
                                        class_labels=labels,
                                        return_dict=False)[0]

            image = scheduler.step(model_pred, t, image, eta=0.0).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = (image * 255).byte().cpu().permute(0, 2, 3, 1).numpy()
        pil_images = [Image.fromarray(img.squeeze(), mode="L") for img in image]

        rows = 3
        cols = (num_images + rows - 1) // rows
        w, h = pil_images[0].size
        grid = Image.new('L', (cols * w, rows * h))
        for i, img in enumerate(pil_images):
            grid.paste(img, (i % cols * w, i // cols * h))

        grid.save(output_path)
        print(f"Saved {num_images} DDPM images to {output_path}")
        return grid


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test DDPM (Teacher Model)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "generate"])
    parser.add_argument("--output_dir", type=str, default="./debug_scripts/ddpm_checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--target_digit", type=int, default=None, help="Train on specific digit (0-9)")
    parser.add_argument("--specific_digit", type=int, default=None, help="Generate specific digit")
    parser.add_argument("--num_images", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generator = DDPMGenerator(output_dir=args.output_dir)

    if args.mode == "train":
        generator.train(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            target_digit=args.target_digit
        )

    elif args.mode == "generate":
        if args.checkpoint is None:
            checkpoint_path = os.path.join(args.output_dir, "ddpm_final.pth")
        else:
            checkpoint_path = args.checkpoint

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return

        generator.load(checkpoint_path)
        digit_str = f"_digit{args.specific_digit}" if args.specific_digit is not None else ""
        generator.generate(
            f"{args.output_dir}/ddpm_generated{digit_str}.png",
            num_images=args.num_images,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            specific_digit=args.specific_digit
        )


if __name__ == "__main__":
    main()
