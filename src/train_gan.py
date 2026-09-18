"""
train_gan.py
-------------
Adversarial training loop for the GAN defined in gan_models.py.

Dataset
-------
We train on the "digits" dataset bundled with scikit-learn: 1,797
8x8 grayscale images of handwritten digits (0-9). This dataset ships
with scikit-learn itself, so training is fully reproducible offline
with no external download (MNIST/Fashion-MNIST work as drop-in
replacements -- see the README for how to swap them in).

Training dynamics
------------------
Each step:
  1. Sample a batch of real images and an equal-size batch of latent
     noise vectors.
  2. Update the Discriminator to better separate real vs. fake
     (generator-produced) images using binary cross-entropy.
  3. Update the Generator to produce images the Discriminator is more
     likely to classify as real (non-saturating generator loss).

Sample grids of generated images are saved every SAMPLE_INTERVAL
epochs to output/generated_samples/ so training progress can be
inspected visually.
"""

import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

from gan_models import Generator, Discriminator, weights_init

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LATENT_DIM = 32
IMG_SIZE = 8            # 8x8 images from the digits dataset
IMG_DIM = IMG_SIZE * IMG_SIZE
HIDDEN_DIM = 128
BATCH_SIZE = 64
NUM_EPOCHS = 300
LR = 2e-4
BETA1 = 0.5              # Adam beta1, standard for GAN training
SAMPLE_INTERVAL = 25     # epochs between saved sample grids
N_SAMPLE_IMAGES = 16
SEED = 42

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "generated_samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_data() -> DataLoader:
    """Load the sklearn digits dataset and normalize images to [-1, 1]."""
    digits = load_digits()
    images = digits.images.astype(np.float32)          # shape (N, 8, 8), range [0, 16]
    images = images / 16.0                              # -> [0, 1]
    images = images * 2.0 - 1.0                          # -> [-1, 1] to match Tanh output
    images = images.reshape(len(images), -1)             # flatten to (N, 64)

    tensor_images = torch.from_numpy(images)
    dataset = TensorDataset(tensor_images)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


def save_sample_grid(generator: Generator, fixed_noise: torch.Tensor, epoch: int, device: torch.device) -> None:
    """Generate a grid of images from fixed noise and save it as a PNG."""
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise.to(device)).cpu().numpy()
    generator.train()

    fake = (fake + 1.0) / 2.0  # back to [0, 1] for plotting
    fake = fake.reshape(-1, IMG_SIZE, IMG_SIZE)

    grid_size = int(np.sqrt(N_SAMPLE_IMAGES))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake[i], cmap="gray")
        ax.axis("off")
    fig.suptitle(f"Epoch {epoch}", fontsize=10)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch:04d}.png")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def train() -> dict:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = load_data()

    generator = Generator(LATENT_DIM, IMG_DIM, HIDDEN_DIM).to(device)
    discriminator = Discriminator(IMG_DIM, HIDDEN_DIM).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))

    fixed_noise = torch.randn(N_SAMPLE_IMAGES, LATENT_DIM)

    history = {"epoch": [], "loss_d": [], "loss_g": [], "d_real_acc": [], "d_fake_acc": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss_d, epoch_loss_g = 0.0, 0.0
        epoch_real_acc, epoch_fake_acc = 0.0, 0.0
        n_batches = 0

        for (real_images,) in dataloader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # --- Train Discriminator ---
            opt_d.zero_grad()

            real_logits = discriminator(real_images)
            loss_d_real = criterion(real_logits, real_labels)

            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            fake_images = generator(noise).detach()
            fake_logits = discriminator(fake_images)
            loss_d_fake = criterion(fake_logits, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            # --- Train Generator ---
            opt_g.zero_grad()

            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            gen_images = generator(noise)
            gen_logits = discriminator(gen_images)
            # Non-saturating loss: train G to make D output "real" for fakes.
            loss_g = criterion(gen_logits, real_labels)
            loss_g.backward()
            opt_g.step()

            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            epoch_real_acc += (torch.sigmoid(real_logits) > 0.5).float().mean().item()
            epoch_fake_acc += (torch.sigmoid(fake_logits) < 0.5).float().mean().item()
            n_batches += 1

        avg_loss_d = epoch_loss_d / n_batches
        avg_loss_g = epoch_loss_g / n_batches
        avg_real_acc = epoch_real_acc / n_batches
        avg_fake_acc = epoch_fake_acc / n_batches

        history["epoch"].append(epoch)
        history["loss_d"].append(avg_loss_d)
        history["loss_g"].append(avg_loss_g)
        history["d_real_acc"].append(avg_real_acc)
        history["d_fake_acc"].append(avg_fake_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{NUM_EPOCHS} | "
                f"D loss: {avg_loss_d:.4f} | G loss: {avg_loss_g:.4f} | "
                f"D(real) acc: {avg_real_acc:.2f} | D(fake) acc: {avg_fake_acc:.2f}"
            )

        if epoch % SAMPLE_INTERVAL == 0 or epoch == 1 or epoch == NUM_EPOCHS:
            save_sample_grid(generator, fixed_noise, epoch, device)

    # Persist final model weights and training history for the report.
    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    torch.save(generator.state_dict(), os.path.join(ckpt_dir, "generator.pt"))
    torch.save(discriminator.state_dict(), os.path.join(ckpt_dir, "discriminator.pt"))
    with open(os.path.join(ckpt_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history


def plot_training_curves(history: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["epoch"], history["loss_d"], label="Discriminator loss")
    axes[0].plot(history["epoch"], history["loss_g"], label="Generator loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Adversarial training losses")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["d_real_acc"], label="D accuracy on real")
    axes[1].plot(history["epoch"], history["d_fake_acc"], label="D accuracy on fake")
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Ideal equilibrium (0.5)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Discriminator accuracy")
    axes[1].set_title("Discriminator accuracy over training")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "output", "training_curves.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    history = train()
    plot_training_curves(history)
    print("\nTraining complete. Samples saved to output/generated_samples/")
    print("Training curves saved to output/training_curves.png")
