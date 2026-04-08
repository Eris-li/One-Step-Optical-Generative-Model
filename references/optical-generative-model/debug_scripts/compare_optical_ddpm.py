"""
Compare Teacher (DDPM) vs Student (Optical) model generation
Uses the same noise and labels for both models to enable fair comparison

Usage:
    python debug_scripts/compare_models.py
"""
import os
import torch
from PIL import Image
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusers import UNet2DModel, DDPMScheduler
from pipeline_costum import DDPMPipeline_Costum_ClsEmb
from test_optical import OpticalGenerativeModel
from diffusers.utils import make_image_grid

def generate_teacher( checkpoint_path, output_dir, num_images=9, seed=42):
    """Generate images using DDPM teacher model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading teacher model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint['config']
    model = UNet2DModel(**config).to(device)
    model.load_state_dict(checkpoint['unet_state_dict'])
    model.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    pipeline = DDPMPipeline_Costum_ClsEmb(unet=model, scheduler=scheduler).to(device)
    pipeline.set_progress_bar_config(disable=True)

    os.makedirs(output_dir, exist_ok=True)

    for digit in range(10):
        generator = torch.Generator(device=device).manual_seed(seed)
        labels = torch.tensor([digit] * num_images, dtype=torch.int64, device=device)

        images = pipeline(
            batch_size=num_images,
            class_sample=labels,
            num_inference_steps=50,
            output_type='pil',
            generator=generator,
            return_dict=False
        )[0]

        grid = make_image_grid(images, rows=3, cols=3)
        grid.save(f"{output_dir}/teacher_digit{digit}.png")
        print(f"  Teacher: saved digit {digit}")

    print("Teacher 0-9 grid saved")

    for digit in [7]:
        all_samples = []
        for i in range(9):
            generator = torch.Generator(device=device).manual_seed(seed + i)
            labels = torch.tensor([digit], dtype=torch.int64, device=device)

            images = pipeline(
                batch_size=1,
                class_sample=labels,
                num_inference_steps=50,
                output_type='pil',
                generator=generator,
                return_dict=False
            )[0]
            all_samples.append(images[0])

        grid = make_image_grid(all_samples, rows=3, cols=3)
        grid.save(f"{output_dir}/teacher_digit7_grid.png")
        print(f"  Teacher: saved digit 7 grid")

    return True


def generate_student(checkpoint_path, output_dir, num_images=9, seed=42):
    """Generate images using Optical student model"""
    print(f"Loading student model from {checkpoint_path}")

    model = OpticalGenerativeModel(output_dir=output_dir)
    model.load(checkpoint_path)

    os.makedirs(output_dir, exist_ok=True)

    for digit in range(10):
        model.generate(
            f"{output_dir}/student_digit{digit}.png",
            num_images=1,
            seed=seed,
            specific_digit=digit
        )
        print(f"  Student: saved digit {digit}")

    for digit in [7]:
        all_samples = []
        for i in range(9):
            sample_img = model.generate(
                f"{output_dir}/student_digit7_sample{i}.png",
                num_images=1,
                seed=seed + i,
                specific_digit=digit
            )
        print(f"  Student: saved digit 7 grid")

    return True


def create_combined_grids(output_dir):
    """Create combined 0-9 and digit 7 grids for comparison"""
    print("\nCreating combined grids...")

    for model_type in ["teacher", "student"]:
        all_digits = []
        for digit in range(10):
            img = Image.open(f"{output_dir}/{model_type}_digit{digit}.png")
            all_digits.append(img)

        w, h = all_digits[0].size
        grid = Image.new('L', (5 * w, 2 * h))
        for i, img in enumerate(all_digits):
            row = i // 5
            col = i % 5
            grid.paste(img, (col * w, row * h))
        grid.save(f"{output_dir}/{model_type}_digits_0_9.png")
        print(f"  {model_type}: saved digits_0_9.png")

    teacher_grid = Image.open(f"{output_dir}/teacher_digit7_grid.png")
    teacher_grid.save(f"{output_dir}/teacher_digit7_grid_final.png")
    print(f"  teacher: saved digit7_grid.png")

    student_samples = []
    for i in range(9):
        img = Image.open(f"{output_dir}/student_digit7_sample{i}.png")
        student_samples.append(img)

    w, h = student_samples[0].size
    grid = Image.new('L', (3 * w, 3 * h))
    for i, img in enumerate(student_samples):
        row = i // 3
        col = i % 3
        grid.paste(img, (col * w, row * h))
    grid.save(f"{output_dir}/student_digit7_grid.png")
    print(f"  student: saved digit7_grid.png")


def main():
    output_dir = "./debug_scripts/optical_checkpoints"

    teacher_checkpoint = "./debug_scripts/ddpm_checkpoints/ddpm_final.pth"
    student_checkpoint = "./debug_scripts/optical_checkpoints/optical_final.pth"

    if not os.path.exists(student_checkpoint):
        print(f"Student checkpoint not found: {student_checkpoint}")
        print("Please train the model first!")
        return

    print("=" * 60)
    print("Generating images from Teacher (DDPM) model...")
    print("=" * 60)
    generate_teacher(teacher_checkpoint, output_dir)

    print("\n" + "=" * 60)
    print("Generating images from Student (Optical) model...")
    print("=" * 60)
    generate_student(student_checkpoint, output_dir)

    print("\n" + "=" * 60)
    print("Creating combined grids...")
    print("=" * 60)
    create_combined_grids(output_dir)

    print("\n" + "=" * 60)
    print("All images saved to:", output_dir)
    print("=" * 60)
    print("\nGenerated files:")
    print("  - teacher_digits_0_9.png  (Teacher DDPM 0-9)")
    print("  - teacher_digit7_grid.png (Teacher DDPM 9x digit 7)")
    print("  - student_digits_0_9.png  (Student Optical 0-9)")
    print("  - student_digit7_grid.png (Student Optical 9x digit 7)")


if __name__ == "__main__":
    main()
