# GAN Training Report

## 1. Objective

Implement and train a Generative Adversarial Network to generate synthetic
handwritten-digit images from random noise, and analyze the quality and
dynamics of the training process.

## 2. Setup

- **Dataset:** scikit-learn's bundled `digits` dataset — 1,797 8x8 grayscale
  images of handwritten digits (0–9), normalized to `[-1, 1]`.
- **Architecture:** fully-connected Generator (32 → 128 → 256 → 64, Tanh
  output) and Discriminator (64 → 256 → 128 → 1, raw logit output).
- **Loss:** binary cross-entropy on logits (`BCEWithLogitsLoss`), non-saturating
  generator objective.
- **Optimizer:** Adam, lr = 2e-4, β1 = 0.5, for both networks.
- **Epochs:** 300, batch size 64.

Full configuration is in `src/train_gan.py`; architecture code is in
`src/gan_models.py`.

## 3. Training Dynamics

Three phases were visible across the 300-epoch run (see
`output/training_curves.png`):

1. **Epochs 1–10 — Discriminator dominance.** At epoch 1, the discriminator
   correctly classified ~97% of real images and ~96% of fakes (i.e. only 4%
   fake-accuracy for the generator's side) — it started out able to trivially
   separate real digits from unstructured noise-like output. Generator loss
   was low at first only because the discriminator hadn't yet learned to
   penalize it strongly.
2. **Epochs 10–50 — Rapid equilibration.** Within the first 10–20 epochs, the
   discriminator's accuracy on both real and fake images dropped sharply
   toward the 55–65% range, and generator loss rose from 0.66 to ~0.83 as it
   was forced to work harder to fool an improving discriminator. This is the
   expected signature of healthy adversarial training: neither network
   collapses or "wins" outright.
3. **Epochs 50–300 — Stable plateau.** From epoch ~50 onward, both losses and
   both discriminator-accuracy curves flatten and oscillate mildly around a
   stable point (D loss ≈ 1.30, G loss ≈ 0.80, D accuracy on real/fake both
   ≈ 0.58–0.64). This is close to — though not exactly at — the theoretical
   equilibrium of 50% discriminator accuracy, indicating the generator is
   producing images that are frequently, though not always, mistaken for
   real.

Visually, generated samples (`output/generated_samples/`) confirm this
progression: epoch 1 samples are blob-like, roughly digit-shaped noise with
no consistent structure; by epoch 100–300, samples show clear curved and
straight strokes, closed loops (resembling 0/6/8/9-like shapes), and
consistent stroke widths, though fine detail remains blurred (an expected
limitation of a small MLP generator on a fully-connected architecture, as
opposed to a convolutional one).

## 4. Challenges Faced and How They Were Addressed

- **Discriminator overpowering the generator early on.** Early in training,
  the discriminator's near-perfect real/fake separation (epoch 1: 97%/96%)
  risked vanishing gradients for the generator, since `log(1 - D(G(z)))` is
  flat when `D(G(z))` is close to 0. This was addressed by using the
  standard **non-saturating generator loss** — training G to maximize
  `log(D(G(z)))` (equivalently, minimizing BCE against label 1) instead of
  minimizing `log(1 - D(G(z)))` directly — which supplies much stronger
  gradients when the generator is performing poorly.
- **Training instability / oscillation.** GANs are known to oscillate rather
  than converge monotonically. Using **Adam with β1 = 0.5** (rather than the
  default 0.9) reduces momentum-driven overshoot, which is a standard
  mitigation and produced the relatively smooth, mildly oscillating plateau
  seen after epoch 50, rather than divergent swings.
- **Mode collapse risk.** Mode collapse — where the generator produces only
  a small number of distinct outputs regardless of the input noise — was
  checked for by visually inspecting each saved 4x4 sample grid: the 16
  images at every checkpoint show visually distinct digit-like shapes rather
  than 16 near-identical images, suggesting mode collapse did not occur to a
  severe degree in this run. Using `BatchNorm1d` in the generator's hidden
  layers also helped here, since batch statistics couple the outputs within
  a batch and discourage the generator from collapsing all noise vectors to
  one output.
- **No external dataset dependency.** The task suggests MNIST or
  Fashion-MNIST, both of which require a network download. To keep the
  project runnable in constrained or offline environments, the
  scikit-learn-bundled `digits` dataset (8x8 images, same 0–9 label space,
  no download) was used instead. The README documents the exact change
  needed to swap in MNIST/Fashion-MNIST if a GPU and larger images are
  available.

## 5. Results

- Discriminator accuracy on real and fake images converged from
  ~97%/4% (epoch 1) to a stable ~58–64% band (epoch 50 onward) — much
  closer to the ideal 50% equilibrium than to the discriminator "winning"
  outright.
- Generated samples show clear, recognizable digit-like stroke structure
  by epoch 100–300, with visibly diverse shapes across the 16-image sample
  grid (no evidence of severe mode collapse).
- The chosen MLP architecture trains a full 300-epoch run in well under a
  minute on CPU, at the cost of blurrier, less sharp digit detail than a
  convolutional (DCGAN-style) architecture would produce on the larger
  28x28 MNIST images.

## 6. Possible Improvements

- Swap in a convolutional Generator/Discriminator (DCGAN-style) and MNIST's
  28x28 images for sharper, more detailed samples.
- Add a learning-rate schedule or use two discriminator steps per generator
  step if oscillation is worse on a different dataset/architecture.
- Track a quantitative diversity metric across generated batches (e.g.
  pairwise pixel-distance among generated samples) to detect subtler mode
  collapse than visual inspection alone catches.
