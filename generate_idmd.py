import glob
import os
import pickle
import re

import click
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import tqdm
from torch import Tensor
from torch.distributions import Beta
from torchvision.utils import make_grid, save_image

import dnnlib
from idmd.utils.pfgmpp_utils import sample_from_posterior
from torch_utils import distributed as dist
from torch_utils import misc

#----------------------------------------------------------------------------
# Proposed IDMD sampler.

def get_sigma_schedule(*, mode: str, n_steps: int, sigma_min: float, sigma_max: float):
    if mode == "uniform":
        sigma_schedule = np.linspace(sigma_min, sigma_max, n_steps + 1)[1:][::-1]
    else:
        raise NotImplementedError
    return sigma_schedule

def idmd_sampler(
    net: nn.Module,
    latents: Tensor,
    class_labels: Tensor | None = None,
    #
    sigma_min: float = 0.002,
    sigma_max: float = 80.,
    D: int | str = "inf",
    #
    num_steps: int = 1,
    multistep_mode: str = "uniform",
):
    sigma_schedule = get_sigma_schedule(
        mode=multistep_mode,
        n_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max
    )
    x_noisy = latents.to(torch.float64)
    for i, sigma in enumerate(sigma_schedule):
        sigma = torch.tensor([sigma] * len(x_noisy), dtype=x_noisy.dtype, device=x_noisy.device)
        x_denoised = net(x_noisy, sigma, class_labels).to(torch.float64)
        if i < len(sigma_schedule) - 1:
            sigma_next = sigma_schedule[i + 1]
            sigma_next = torch.tensor([sigma_next] * len(x_noisy), dtype=x_noisy.dtype, device=x_noisy.device)
            x_noisy = sample_from_posterior(images=x_denoised, sigma=sigma_next, D=D)
    return x_denoised

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]
        self.seeds = seeds
        self.device = device

    def randn(self, size, sigma_max, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]) * sigma_max

    def rand_beta_prime(self, size, sigma_max, D, **kwargs):
        # sample from beta_prime (N/2, D/2)
        # print(f"N:{N}, D:{D}")
        assert size[0] == len(self.seeds)
        N = np.prod(size[1:])
        latent_list = []
        beta_gen = Beta(torch.FloatTensor([N / 2.]), torch.FloatTensor([D / 2.]))
        for seed in self.seeds:
            torch.manual_seed(seed)
            sample_norm = beta_gen.sample().to(kwargs['device']).double()
            # inverse beta distribution
            inverse_beta = sample_norm / (1-sample_norm)

            sample_norm = torch.sqrt(inverse_beta) * sigma_max * np.sqrt(D)
            gaussian = torch.randn(N).to(sample_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2)
            init_sample = unit_gaussian * sample_norm
            latent_list.append(init_sample.reshape((1, *size[1:])))

        latent = torch.cat(latent_list, dim=0)
        return latent

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--save_images',             help='only save a batch images for grid visualization',                     is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--sigma_min', type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True, help='Minimum noise level')
@click.option('--sigma_max', type=click.FloatRange(min=0, min_open=True), default=80, show_default=True, help='Maximum noise level')
@click.option('--aug_dim', type=str, default="inf", show_default=True, help='Divergence dimension D for IDMD')

@click.option('--num_steps', type=click.IntRange(min=1), default=1, show_default=True, help='Number of sampling steps')
@click.option('--multistep_mode', type=str, default="uniform", show_default=True, help='Multistep mode for IDMD')

def main(
    network_pkl,
    outdir,
    seeds,
    subdirs,
    save_images,
    class_idx,
    max_batch_size,
    sigma_min,
    sigma_max,
    aug_dim,
    num_steps,
    multistep_mode,
    device=torch.device('cuda'),
):
    """Generate images using IDMD sampler."""

    assert network_pkl.startswith('http') or os.path.isfile(network_pkl)

    dist.init()

    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Load network
    dist.print0(f'Loading network from "{network_pkl}"...')
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device).eval()
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Create output directory
    os.makedirs(outdir, exist_ok=True)
    vis_dir = os.path.join(outdir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    # Loop over batches
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        rnd = StackedRandomGenerator(device, batch_seeds)

        # Create latents (TODO: pfgmpp latents)
        if aug_dim == "inf": # EDM case
            latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], sigma_max=sigma_max, device=device)
        else: # PFGMPP case
            latents = rnd.rand_beta_prime([batch_size, net.img_channels, net.img_resolution, net.img_resolution], sigma_max=sigma_max, D=aug_dim, device=device)

        # Create labels if applicable
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[
                rnd.randint(net.label_dim, size=[batch_size], device=device)
            ]
        if class_idx is not None and class_labels is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Run IDMD sampler
        with torch.no_grad():
            images = idmd_sampler(
                net=net,
                latents=latents,
                class_labels=class_labels,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                D=aug_dim,
                num_steps=num_steps,
                multistep_mode=multistep_mode,
            )

        # Save batch grid (if requested)
        if save_images:
            images_ = (images + 1) / 2
            grid = make_grid(images_, nrow=int(np.sqrt(batch_size)), padding=2)
            save_image(grid, os.path.join(vis_dir, 'grid.png'))
            dist.print0("Saved visualization grid.")
            break  # Exit after first batch if only visualizing

        # Save individual images
        images_np = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed - seed % 1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    # Done
    torch.distributed.barrier()
    dist.print0('Done.')


if __name__ == "__main__":
    main()
