import json
import os
import re
import warnings

import click
import torch

import dnnlib
from idmd.constants import EXPERIMENTS_DIR
from torch_utils import distributed as dist
from training import distillation_loop

warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

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
# Main options
@click.option('--outdir',        help='Where to save the results',                     metavar='DIR',    type=str, required=True)
@click.option('--teacher_pkl',   help='Teacher Model (EDM-like format)',              metavar='PKL|URL', type=str, required=True)
@click.option('--sigma_min',     help='',                           metavar='FLOAT', type=float, default=0.002, show_default=True)
@click.option('--sigma_max',     help='',                           metavar='FLOAT', type=float, default=80.0, show_default=True)
@click.option('--aug_dim',       help='PFGM++ D parameter',                           metavar='STR|INT', type=str, default="inf", show_default=True)
@click.option('--precond',       help='edm | vp | ve',                                metavar='STR', type=str, default="edm", show_default=True)

# Hyperparameters
@click.option('--duration',      help='Training duration',                            metavar='ITERS',  type=click.FloatRange(min=0, min_open=True), default=20_000, show_default=True)
@click.option('--batch',         help='Total batch size',                             metavar='INT',    type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU',                     metavar='INT',    type=click.IntRange(min=1))
@click.option('--student_lr',    help='Student Learning rate',                        metavar='FLOAT',  type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--generator_lr',  help='Generator Learning rate',                      metavar='FLOAT',  type=click.FloatRange(min=0, min_open=True), default=1e-5, show_default=True)
@click.option('--generator_beta1', help="Generator Adam's beta1",                     metavar='FLOAT',  type=click.FloatRange(min=0, min_open=False), default=0.0, show_default=True)
@click.option('--generator_beta2', help="Generator Adam's beta2 Learning rate",       metavar='FLOAT',  type=click.FloatRange(min=0, min_open=False), default=0.999, show_default=True)
@click.option('--remove_dropout', help='Remove dropout from teacher',    metavar='BOOL',   type=bool, default=True, show_default=True)
@click.option('--n_student_updates', help='',    metavar='INT',   type=int, default=5, show_default=True)

# Performance-related
@click.option('--ls',            help='Loss scaling',                                metavar='FLOAT',  type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking',                    metavar='BOOL',   type=bool, default=True, show_default=True)

# I/O-related
@click.option('--desc',          help='String to include in result dir name',         metavar='STR',    type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                      is_flag=True)
@click.option('--tick',          help='How often to print progress',                  metavar='ITERS',  type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--snap',          help='How often to save snapshots',                  metavar='TICKS',  type=click.IntRange(min=1), default=5, show_default=True)
@click.option('--seed',          help='Random seed [default: random]',                metavar='INT',    type=int)
@click.option('--resume_pkl',    help='Resume from previous training state',          metavar='PKL|URL',    type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                               is_flag=True)

def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()

    # Main options
    if dist.get_rank() != 0:
        c.run_dir = None
    else:
        teacher_name = opts.teacher_pkl.split('/')[-1].split('.')[0]
        c.run_dir = os.path.join(EXPERIMENTS_DIR, teacher_name, opts.outdir)

    c.teacher_pkl = opts.teacher_pkl
    c.preprocess_edm_net_kwargs = {"remove_dropout": opts.remove_dropout}
    c.sigma_min = opts.sigma_min
    c.sigma_max = opts.sigma_max
    c.D = opts.aug_dim
    c.n_student_updates = opts.n_student_updates

    if opts.resume_pkl is not None:
        c.resume_pkl = opts.resume_pkl

    # Hyperparameters
    c.student_optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.student_lr, betas=[0.9,0.999], eps=1e-8)
    c.generator_optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.generator_lr, betas=[opts.generator_beta1,opts.generator_beta2], eps=1e-8)

    c.loss_kwargs = dnnlib.EasyDict()
    if opts.precond == 'edm':
        c.loss_kwargs.class_name = 'training.idmd_loss.EDMLoss'
        c.loss_kwargs.D = opts.aug_dim
    else:
        raise NotImplementedError()

    # Training options.
    c.total_iter = max(opts.duration, 1)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(iters_per_tick=opts.tick, snapshot_ticks=opts.snap)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)


    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    distillation_loop.distillation_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
