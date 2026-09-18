"""
gan_models.py
--------------
Model definitions for a simple fully-connected GAN, built with PyTorch.

The Generator maps a latent noise vector z ~ N(0, 1) of dimension
`latent_dim` to a flattened image of dimension `img_dim`.

The Discriminator maps a flattened image to a single scalar logit
representing how "real" the discriminator believes the image is.

These architectures are intentionally simple (MLP-based rather than
convolutional) so the model trains quickly on small grayscale images
without needing a GPU or an external dataset download. The same
training loop in train_gan.py works unchanged with a convolutional
Generator/Discriminator if you swap in a larger image dataset
(e.g. MNIST 28x28 or Fashion-MNIST) later.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Maps random noise -> a synthetic flattened image in [-1, 1]."""

    def __init__(self, latent_dim: int, img_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_dim = img_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_dim * 2, img_dim),
            nn.Tanh(),  # output scaled to [-1, 1] to match normalized real images
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """Maps a flattened image -> a single real/fake logit."""

    def __init__(self, img_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.img_dim = img_dim

        self.net = nn.Sequential(
            nn.Linear(img_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, 1),
            # No sigmoid here: BCEWithLogitsLoss applies it internally,
            # which is more numerically stable than sigmoid + BCELoss.
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.net(img)


def weights_init(m: nn.Module) -> None:
    """Standard GAN weight initialization (helps avoid early mode collapse)."""
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
