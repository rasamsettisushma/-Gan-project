# Generative Adversarial Network (GAN) — Synthetic Digit Generation

Week 6 · Task 2 — Generative Adversarial Networks

This repository implements a fully-connected GAN in PyTorch that learns to
generate synthetic handwritten-digit images from random noise, trained
adversarially against a discriminator.

## 1. Overview

A GAN consists of two competing neural networks trained simultaneously:

- **Generator (G):** takes a random noise vector `z` (drawn from a standard
  normal distribution) and maps it to a synthetic image. Its goal is to
  produce images realistic enough to fool the discriminator.
- **Discriminator (D):** takes an image (real or generated) and outputs the
  probability that it is real. Its goal is to correctly tell real images
  apart from the generator's fakes.

The two networks are trained in alternation with opposing objectives — a
minimax game. As training progresses, G should get better at producing
realistic images, and D should get better at spotting fakes, until (ideally)
D can no longer do better than random guessing (50% accuracy) on the
generator's output.

## 2. Dataset

We use the **`digits` dataset bundled with scikit-learn**: 1,797 8x8
grayscale images of handwritten digits (0–9). This dataset ships with
`scikit-learn` itself and requires no external download, which keeps the
project fully reproducible offline.

Images are normalized from `[0, 16]` to `[-1, 1]` to match the generator's
`Tanh` output activation, and flattened to 64-dimensional vectors for the
fully-connected architecture.

> **Swapping in MNIST or Fashion-MNIST:** the task brief suggests MNIST or
> Fashion-MNIST (28x28 images). To use either, replace `load_data()` in
> `src/train_gan.py` with a `torchvision.datasets.MNIST` (or
> `FashionMNIST`) loader, and change `IMG_SIZE = 28` in the hyperparameters.
> No other code changes are required — the model and training loop are
> already sized generically from `IMG_DIM = IMG_SIZE * IMG_SIZE`.

## 3. Architecture

Both networks are simple multi-layer perceptrons (MLPs) rather than
convolutional networks, chosen so the model trains in seconds on CPU while
still demonstrating every core GAN mechanism clearly. See
`docs/gan_report.md` for the case for/against this choice.

**Generator** (`src/gan_models.py`):

```
z (latent_dim=32)
  -> Linear(32, 128)   -> BatchNorm -> LeakyReLU(0.2)
  -> Linear(128, 256)  -> BatchNorm -> LeakyReLU(0.2)
  -> Linear(256, 64)   -> Tanh
  -> synthetic image (8x8, flattened, range [-1, 1])
```

**Discriminator** (`src/gan_models.py`):

```
image (64,)
  -> Linear(64, 256)  -> LeakyReLU(0.2) -> Dropout(0.3)
  -> Linear(256, 128) -> LeakyReLU(0.2) -> Dropout(0.3)
  -> Linear(128, 1)   -> (raw logit; BCEWithLogitsLoss applies sigmoid internally)
```

## 4. Hyperparameters

| Hyperparameter        | Value   | Notes                                              |
|------------------------|---------|-----------------------------------------------------|
| Latent dimension       | 32      | Size of the random noise vector fed to G            |
| Hidden dimension       | 128     | Base width of hidden layers                         |
| Batch size             | 64      |                                                       |
| Epochs                 | 300     |                                                       |
| Learning rate          | 2e-4    | Same for G and D, Adam optimizer                     |
| Adam beta1             | 0.5     | Standard for GAN training (vs. default 0.9)          |
| Loss function          | BCEWithLogitsLoss | Binary cross-entropy on raw logits (numerically stable) |
| Weight init            | N(0, 0.02) | Applied to Linear/BatchNorm layers, standard for GANs |

## 5. Training Process

Each training step:

1. Sample a real batch of images and a batch of random noise.
2. **Update D:** compute D's loss on real images (label = 1) and on
   generator-produced fake images (label = 0), sum them, and step the
   discriminator's optimizer.
3. **Update G:** generate a new batch of fake images and compute G's loss
   using the *non-saturating* objective — i.e., train G to make D output
   "real" (label = 1) for its fakes, rather than minimizing `log(1 - D(fake))`
   directly, which provides stronger gradients early in training.

Every 25 epochs, a fixed noise batch is passed through the generator and
saved as an image grid to `output/generated_samples/`, so the same 16
latent vectors can be tracked visually as they evolve throughout training.

## 6. Repository Structure

```
README.md
requirements.txt
src/
  gan_models.py            # Generator, Discriminator, weight init
  train_gan.py             # Data loading, training loop, sample/plot saving
output/
  generated_samples/       # PNG grids of generated digits at each checkpointed epoch
  training_curves.png      # Generator/Discriminator loss and accuracy curves
  generator.pt             # Final trained generator weights
  discriminator.pt         # Final trained discriminator weights
  training_history.json    # Per-epoch loss/accuracy log
docs/
  gan_report.md             # Written report: training dynamics, challenges, results
```

## 7. How to Run

```bash
pip install -r requirements.txt
cd src
python train_gan.py
```

Training takes well under a minute on CPU for the default 300 epochs.
Generated sample grids and training curves will appear under `output/`.

## 8. Results Summary

After 300 epochs, the discriminator's accuracy on real and fake images both
settle close to the ~55-65% range (versus starting near 100%/0%), and the
generated digit samples show clearly digit-like strokes and loops rather
than noise. Full discussion, including mode-collapse checks, is in
[`docs/gan_report.md`](docs/gan_report.md).
