from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor

EPS = 1e-8


def sample_from_posterior(
    *,
    images: Tensor,
    sigma: Tensor,
    #
    D: int | str = "inf",
    #
    seed: Optional[int]=None
):
    if D == "inf": # EDM case
        perturbation_x = torch.randn_like(images) * sigma.view(-1, 1, 1, 1)
    else: # PFGMPP case
        assert isinstance(D, int)
        data_dim = np.prod(images.shape[1:])

        # Convert sigma to r
        r = sigma.reshape(-1) * D**0.5

        # Sample from inverse-beta distribution
        samples_norm = np.random.beta(
            a=data_dim / 2.,
            b=D / 2.,
            size=images.shape[0],
        ).astype(np.double)
        samples_norm = np.clip(samples_norm, 1e-3, 1-1e-3)
        inverse_beta = samples_norm / (1 - samples_norm + EPS)
        inverse_beta = torch.from_numpy(inverse_beta).to(images.device).double()

        # Sampling from p_r(R) by change-of-variable
        R = r * torch.sqrt(inverse_beta + EPS)
        R = R.view(len(samples_norm), -1)

        # Uniformly sample the angle component
        gaussian = torch.randn(len(images), data_dim).to(images.device)
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)

        # Construct the perturbation for x
        perturbation_x = unit_gaussian * R
        perturbation_x = perturbation_x.float()

        perturbation_x = perturbation_x.view_as(images)

    return images + perturbation_x

def sample_from_prior(
    *,
    sample_size: int,
    shape: tuple,
    sigma_max: float,
    D: int | str,
    #
    device: torch.device,
    dtype: torch.dtype,
    #
    seed: Optional[int]=None
):
    images = torch.zeros(sample_size, *shape, device=device, dtype=dtype)
    sigma = torch.full((sample_size,), sigma_max, device=device, dtype=dtype)
    return sample_from_posterior(images=images, sigma=sigma, D=D, seed=seed)
