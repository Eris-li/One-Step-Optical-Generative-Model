"""
Test Optical Generative Network using original repository code
Uses: Generator_ClsEmd_light, D2nnModel from test_example_mnist.py

Usage:
    Training (with DDPM distillation):  python debug_scripts/test_optical.py --mode train --num_epochs 10
    Generate: python debug_scripts/test_optical.py --mode generate --checkpoint <path>
    Compare:  python debug_scripts/test_optical.py --mode compare --ddpm_checkpoint <path>
"""
import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_example_mnist import Generator_ClsEmd_light, D2nnModel, TestConfig
from initialization import extract_material_parameter
from pipeline_costum import DDPMPipeline_Costum_ClsEmb
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler


class OpticalGenerativeModel:
    """Optical Generative Model using original test_example_mnist.py classes"""

    def __init__(self, output_dir="./debug_scripts/optical_checkpoints"):
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = self._get_default_config()
        self.generator_e = None
        self.generator_d = None
        self.teacher_pipeline = None

    def _get_default_config(self):
        config = {
            'img_size': 32,
            'in_channel': 1,
            'num_classes': 10,
            'dim_expand_ratio': 128,
            'c': 299792458,
            'ridx_air': 1.0,
            'wlength_vc': 520e-9,
            'num_masks': 1,
            'object_mask_dist': 12.01e-2,
            'mask_mask_dist': 2.0e-2,
            'mask_sensor_dist': 9.64e-2,
            'mask_base_thick': 1.0e-3,
            'total_x_num': 800,
            'total_y_num': 800,
            'mask_x_num': 400,
            'mask_y_num': 400,
            'dx': 8e-6,
            'dy': 8e-6,
            'obj_x_num': 320,
            'obj_y_num': 320,
            'mask_init_method': 'zero',
        }
        return config

    def _create_models(self):
        freq = self.config['c'] / self.config['wlength_vc']
        ridx_mask, attenu_factor = extract_material_parameter(freq, mask_amp_modulation=False)

        generator_e = Generator_ClsEmd_light(
            img_size=self.config['img_size'],
            in_channel=self.config['in_channel'],
            num_classes=self.config['num_classes'],
            dim_expand_ratio=self.config['dim_expand_ratio']
        ).to(self.device)

        generator_d = D2nnModel(
            img_size=self.config['img_size'],
            in_channel=self.config['in_channel'],
            c=self.config['c'],
            freq=freq,
            num_masks=self.config['num_masks'],
            wlength_vc=self.config['wlength_vc'],
            ridx_air=self.config['ridx_air'],
            ridx_mask=ridx_mask,
            attenu_factor=attenu_factor,
            total_x_num=self.config['total_x_num'],
            total_y_num=self.config['total_y_num'],
            mask_x_num=self.config['mask_x_num'],
            mask_y_num=self.config['mask_y_num'],
            mask_init_method=self.config['mask_init_method'],
            mask_base_thick=self.config['mask_base_thick'],
            dx=self.config['dx'],
            dy=self.config['dy'],
            object_mask_dist=self.config['object_mask_dist'],
            mask_mask_dist=self.config['mask_mask_dist'],
            mask_sensor_dist=self.config['mask_sensor_dist'],
            obj_x_num=self.config['obj_x_num'],
            obj_y_num=self.config['obj_y_num']
        ).to(self.device)

        return generator_e, generator_d

    def load_teacher(self, checkpoint_path):
        """Load DDPM teacher model for distillation"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        saved_config = checkpoint.get('config', {})
        saved_state_dict = checkpoint.get('unet_state_dict', {})

        has_class_embedding = 'class_embedding.weight' in saved_state_dict

        default_config = {
            'sample_size': 32,
            'in_channels': 1,
            'out_channels': 1,
            'down_block_types': ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            'up_block_types': ("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            'block_out_channels': (64, 128, 256, 512),
            'layers_per_block': 2,
            'num_train_timesteps': 1000,
        }

        if has_class_embedding:
            default_config['num_class_embeds'] = 10

        unet_config = {**default_config, **saved_config}

        teacher_unet = UNet2DModel(**unet_config).to(self.device)
        teacher_unet.load_state_dict(saved_state_dict)
        teacher_unet.eval()

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=unet_config.get('num_train_timesteps', 1000),
            beta_schedule="linear",
            prediction_type="epsilon"
        )

        self.use_class_conditioning = has_class_embedding
        self.teacher_pipeline = DDPMPipeline_Costum_ClsEmb(
            unet=teacher_unet,
            scheduler=noise_scheduler
        ).to(self.device)

        print(f"Loaded DDPM teacher from {checkpoint_path}")
        print(f"  Class conditioning: {'enabled' if has_class_embedding else 'disabled'}")
        return self

    def train(self, num_epochs=10, batch_size=64, learning_rate=2e-4,
              teacher_checkpoint=None, num_inference_steps=50,
              noise_perturb=1e-4, save_interval=5):
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Training Optical Model on {self.device}")
        print(f"Device params: {self.config['total_x_num']}x{self.config['total_y_num']}")

        if teacher_checkpoint is not None:
            self.load_teacher(teacher_checkpoint)
        else:
            print("WARNING: No teacher checkpoint provided, training without distillation!")
            self.teacher_pipeline = None

        self.generator_e, self.generator_d = self._create_models()

        print(f"Generator_E params: {sum(p.numel() for p in self.generator_e.parameters()):,}")
        print(f"Generator_D params: {sum(p.numel() for p in self.generator_d.parameters()):,}")

        preprocess = transforms.Compose([
            transforms.Resize((self.config['img_size'], self.config['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        train_dataset = datasets.MNIST(root='./data', train=True, transform=preprocess, download=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=0, pin_memory=True)

        optimizer = torch.optim.AdamW(
            list(self.generator_e.parameters()) + list(self.generator_d.parameters()),
            lr=learning_rate, weight_decay=1e-4
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(train_dataloader) * 2,
            num_training_steps=len(train_dataloader) * num_epochs
        )

        for epoch in range(num_epochs):
            self.generator_e.train()
            self.generator_d.train()
            progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")

            for step, batch in enumerate(train_dataloader):
                clean_images = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                if self.teacher_pipeline is not None:
                    with torch.no_grad():
                        generator = torch.Generator(device=self.device)
                        self.teacher_pipeline.set_progress_bar_config(disable=True)
                        teacher_output, noises_init, _ = self.teacher_pipeline(
                            batch_size=clean_images.shape[0],
                            class_sample=labels,
                            num_inference_steps=num_inference_steps,
                            output_type='tensor',
                            generator=generator,
                            return_dict=False
                        )
                        teacher_output = (teacher_output / 2. + 0.5).clamp(0, 1)
                        noises_init = noises_init + noise_perturb * torch.randn_like(noises_init)
                else:
                    noises_init = torch.randn_like(clean_images)
                    teacher_output = (clean_images + 1) / 2

                gen_img, _ = self.generator_e(noises_init, labels)
                d2nn_img = self.generator_d(gen_img)

                loss = F.mse_loss(d2nn_img, teacher_output)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator_e.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.generator_d.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()

                progress_bar.update(1)
                if step % 100 == 0:
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            progress_bar.close()
            print(f"Epoch {epoch+1} Complete, Loss: {loss.item():.4f}")

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"optical_epoch{epoch+1}.pth")
                self.generate(f"{self.output_dir}/optical_epoch{epoch+1}.png")

        self.save_checkpoint("optical_final.pth")
        print(f"Training complete!")
        return self

    def save_checkpoint(self, filename):
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, filename)
        torch.save({
            'ge_model_state_dict': self.generator_e.state_dict(),
            'gd_model_state_dict': self.generator_d.state_dict(),
            'config': self.config
        }, save_path)
        print(f"Saved: {save_path}")

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.config = checkpoint.get('config', self._get_default_config())

        self.generator_e, self.generator_d = self._create_models()
        self.generator_e.load_state_dict(checkpoint['ge_model_state_dict'])
        self.generator_d.load_state_dict(checkpoint['gd_model_state_dict'])
        self.generator_e.eval()
        self.generator_d.eval()

        print(f"Loaded Optical model from {checkpoint_path}")
        return self

    def generate(self, output_path="optical_output.png", num_images=9,
                 seed=42, specific_digit=None):
        if self.generator_e is None or self.generator_d is None:
            raise ValueError("Model not loaded. Call load() or train() first.")

        self.generator_e.eval()
        self.generator_d.eval()

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        with torch.no_grad():
            noise = torch.randn(num_images, self.config['in_channel'],
                              self.config['img_size'], self.config['img_size'],
                              device=self.device, generator=generator)

            if specific_digit is not None:
                labels = torch.tensor([specific_digit] * num_images, device=self.device)
            else:
                labels = torch.randint(0, 10, size=(num_images,), device=self.device)

            gen_img, _ = self.generator_e(noise, labels)
            d2nn_img = self.generator_d(gen_img)

            d2nn_img_min = torch.min(torch.min(d2nn_img, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
            d2nn_img_max = torch.max(torch.max(d2nn_img, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
            d2nn_img = (d2nn_img - d2nn_img_min) / (d2nn_img_max - d2nn_img_min + 1e-8)

            d2nn_img = (d2nn_img * 255).byte().cpu().permute(0, 2, 3, 1).numpy()
            pil_images = [Image.fromarray(img.squeeze(), mode="L") for img in d2nn_img]

            cols = min(num_images, 3)
            rows = (num_images + cols - 1) // cols
            image_grid = make_image_grid(pil_images, rows=rows, cols=cols)
            image_grid.save(output_path)

        digit_str = specific_digit if specific_digit is not None else "all"
        print(f"Saved {num_images} optical images (digit={digit_str}) to {output_path}")
        return image_grid

    def generate_with_fixed_noise(self, output_path="optical_output.png",
                                  noise=None, labels=None):
        if self.generator_e is None or self.generator_d is None:
            raise ValueError("Model not loaded. Call load() or train() first.")

        self.generator_e.eval()
        self.generator_d.eval()

        num_images = noise.shape[0] if noise is not None else 9

        with torch.no_grad():
            if noise is None:
                noise = torch.randn(num_images, self.config['in_channel'],
                                  self.config['img_size'], self.config['img_size'],
                                  device=self.device)

            if labels is None:
                labels = torch.randint(0, 10, size=(num_images,), device=self.device)

            gen_img, _ = self.generator_e(noise, labels)
            d2nn_img = self.generator_d(gen_img)

            d2nn_img_min = torch.min(torch.min(d2nn_img, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
            d2nn_img_max = torch.max(torch.max(d2nn_img, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
            d2nn_img = (d2nn_img - d2nn_img_min) / (d2nn_img_max - d2nn_img_min + 1e-8)

            d2nn_img = (d2nn_img * 255).byte().cpu().permute(0, 2, 3, 1).numpy()
            pil_images = [Image.fromarray(img.squeeze(), mode="L") for img in d2nn_img]

            cols = min(num_images, 3)
            rows = (num_images + cols - 1) // cols
            image_grid = make_image_grid(pil_images, rows=rows, cols=cols)
            image_grid.save(output_path)

        print(f"Saved {num_images} optical images to {output_path}")
        return image_grid


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Optical Generative Network")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "generate", "compare"])
    parser.add_argument("--output_dir", type=str, default="./debug_scripts/optical_checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ddpm_checkpoint", type=str, default="./debug_scripts/ddpm_checkpoints/ddpm_final.pth")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="DDPM inference steps for teacher")
    parser.add_argument("--noise_perturb", type=float, default=1e-4,
                        help="Noise perturbation for distillation")
    parser.add_argument("--specific_digit", type=int, default=None, help="Generate specific digit (0-9)")
    parser.add_argument("--num_images", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = OpticalGenerativeModel(output_dir=args.output_dir)

    if args.mode == "train":
        model.train(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            teacher_checkpoint=args.ddpm_checkpoint,
            num_inference_steps=args.num_inference_steps,
            noise_perturb=args.noise_perturb
        )
        model.generate(f"{args.output_dir}/optical_final.png")

    elif args.mode == "generate":
        if args.checkpoint is None:
            checkpoint_path = os.path.join(args.output_dir, "optical_final.pth")
        else:
            checkpoint_path = args.checkpoint

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            print("Please train first: python debug_scripts/test_optical.py --mode train")
            return

        model.load(checkpoint_path)
        digit_str = f"_digit{args.specific_digit}" if args.specific_digit is not None else ""
        model.generate(
            f"{args.output_dir}/optical_generated{digit_str}.png",
            specific_digit=args.specific_digit
        )

    elif args.mode == "compare":
        print("=" * 60)
        print("Comparing DDPM vs Optical Generation")
        print("=" * 60)

        num_images = args.num_images
        seed = args.seed

        generator = torch.Generator(device=model.device)
        generator.manual_seed(seed)
        noise = torch.randn(num_images, model.config['in_channel'],
                          model.config['img_size'], model.config['img_size'],
                          device=model.device, generator=generator)

        if args.specific_digit is not None:
            labels = torch.tensor([args.specific_digit] * num_images, device=model.device)
        else:
            labels = torch.randint(0, 10, size=(num_images,), device=model.device)

        from debug_scripts.test_ddpm import DDPMGenerator

        ddpm_gen = DDPMGenerator(output_dir=os.path.dirname(args.ddpm_checkpoint))
        if os.path.exists(args.ddpm_checkpoint):
            ddpm_gen.load(args.ddpm_checkpoint)
            ddpm_gen.generate_with_fixed_noise(
                f"{args.output_dir}/ddpm_compare.png",
                noise=noise, labels=labels
            )
        else:
            print(f"DDPM checkpoint not found: {args.ddpm_checkpoint}")

        optical_path = args.checkpoint if args.checkpoint else os.path.join(args.output_dir, "optical_final.pth")
        if os.path.exists(optical_path):
            model.load(optical_path)
            model.generate_with_fixed_noise(
                f"{args.output_dir}/optical_compare.png",
                noise=noise, labels=labels
            )
        else:
            print(f"Optical checkpoint not found: {optical_path}")


if __name__ == "__main__":
    main()
