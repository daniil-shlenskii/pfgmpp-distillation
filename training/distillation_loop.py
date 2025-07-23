import copy
import json
import os
import pickle
import time

import numpy as np
import psutil
import torch

import dnnlib
from generate_idmd import idmd_sampler
from idmd.utils.pfgmpp_utils import sample_from_prior
from torch_utils import distributed as dist
from torch_utils import misc, training_stats

#----------------------------------------------------------------------------

def distillation_loop(
    run_dir             = '.',      # Output directory.
    teacher_pkl         = None,     # Pretrained teacher network pickle.
    loss_kwargs         = {},       # Options for loss function.
    generator_optimizer_kwargs    = {},       # Options for generator optimizer.
    student_optimizer_kwargs    = {},       # Options for student optimizer.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_iter          = 15_000,   # Training duration, measured in thousands of training images.
    lr_rampup_iter      = 1_000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    iters_per_tick      = 500,       # Interval of progress prints.
    snapshot_ticks      = 2,       # How often to save network snapshots, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_iter         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    D                   = 128,
    n_student_updates = 1,
    generator_num_steps = 1,
    generator_multistep_mode = "uniform",
):
    # Translate iteration into kimg
    def iter_to_kimg(iter: int):
        return iter * batch_size // 1000
    total_kimg = iter_to_kimg(total_iter)
    lr_rampup_kimg = iter_to_kimg(lr_rampup_iter)
    kimg_per_tick = iter_to_kimg(iters_per_tick)
    resume_kimg = iter_to_kimg(resume_iter)

    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Construct networks
    dist.print0('Loading teacher model...')
    assert teacher_pkl is not None, "teacher_pkl must be specified"
    if dist.get_rank() != 0:
        torch.distributed.barrier() # rank 0 goes first
    with dnnlib.util.open_url(teacher_pkl, verbose=(dist.get_rank() == 0)) as f:
        data = pickle.load(f)
    if dist.get_rank() == 0:
        torch.distributed.barrier() # other ranks follow
    teacher_net = data['ema'].eval().requires_grad_(False).to(device)
    del data # conserve memory

    dist.print0('Creating student model...')
    student_net = copy.deepcopy(teacher_net).train().requires_grad_(True).to(device)

    dist.print0('Creating generator model...')
    generator_net = copy.deepcopy(teacher_net).train().requires_grad_(True).to(device)
    generator_net.num_steps = generator_num_steps
    generator_net.multistep_mode = generator_multistep_mode


    if dist.get_rank() == 0:
        B = batch_size // dist.get_world_size()
        with torch.no_grad():
            images = torch.zeros([B, generator_net.img_channels, generator_net.img_resolution, generator_net.img_resolution], device=device)
            sigma = torch.ones([B], device=device)
            labels = torch.zeros([B, generator_net.label_dim], device=device)
            misc.print_module_summary(generator_net, [images, sigma, labels], max_nesting=2)

    # Setup optimizers.
    dist.print0('Setting up optimizer...')
    loss_kwargs.D = D
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.idmd_loss.EDMLoss

    student_optimizer = dnnlib.util.construct_class_by_name(params=student_net.parameters(), **student_optimizer_kwargs) # subclass of torch.optim.Optimizer
    student_ddp = torch.nn.parallel.DistributedDataParallel(student_net, device_ids=[device], broadcast_buffers=False)

    generator_optimizer = dnnlib.util.construct_class_by_name(params=generator_net.parameters(), **generator_optimizer_kwargs) # subclass of torch.optim.Optimizer
    generator_ddp = torch.nn.parallel.DistributedDataParallel(generator_net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(generator_net).eval().requires_grad_(False)

    cur_nimg = resume_kimg * 1000
    cur_tick = 0

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['generator'], dst_module=generator_net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['student'], dst_module=student_net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        generator_optimizer.load_state_dict(data['generator_optimizer_state'])
        student_optimizer.load_state_dict(data['student_optimizer_state'])
        cur_nimg = data['cur_nimg']
        cur_tick = cur_nimg // (1000 * kimg_per_tick)
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    while True:
        # Update student
        for _ in range(n_student_updates):
            # Accumulate gradients.
            student_optimizer.zero_grad(set_to_none=True)
            for round_idx in range(num_accumulation_rounds):
                with misc.ddp_sync(student_ddp, (round_idx == num_accumulation_rounds - 1)):
                    latents = sample_from_prior(
                        sample_size=batch_gpu,
                        shape=(teacher_net.img_channels, teacher_net.img_resolution, teacher_net.img_resolution),
                        sigma_max=teacher_net.sigma_max,
                        D=D,
                        device=device,
                        dtype=torch.float32,
                    )
                    labels = None
                    if teacher_net.label_dim > 0:
                        label_indices = torch.randint(low=0, high=teacher_net.label_dim, size=(batch_gpu,), device=device)
                        labels = torch.eye(teacher_net.label_dim, device=device)[label_indices]
                    with torch.no_grad():
                        images = idmd_sampler(
                            net=generator_ddp.module,
                            latents=latents,
                            class_labels=labels,
                            sigma_min=teacher_net.sigma_min,
                            sigma_max=teacher_net.sigma_max,
                            D=D,
                            num_steps=generator_num_steps,
                            multistep_mode=generator_multistep_mode,
                        )
                    student_loss = loss_fn.primal_loss(
                        net=student_ddp,
                        images=images,
                        labels=labels
                    )
                    student_loss.sum().mul(loss_scaling / (batch_size // dist.get_world_size())).backward()

            for g in student_optimizer.param_groups:
                g['lr'] = student_optimizer_kwargs['lr'] * min(cur_nimg * n_student_updates / max(lr_rampup_kimg * 1000, 1e-8), 1)
            for param in student_net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            student_optimizer.step()

        # Update generator
        generator_optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(generator_ddp, (round_idx == num_accumulation_rounds - 1)):
                latents = sample_from_prior(
                    sample_size=batch_gpu,
                    shape=(teacher_net.img_channels, teacher_net.img_resolution, teacher_net.img_resolution),
                    sigma_max=teacher_net.sigma_max,
                    D=D,
                    device=device,
                    dtype=torch.float32,
                )
                labels = None
                if teacher_net.label_dim > 0:
                    label_indices = torch.randint(low=0, high=teacher_net.label_dim, size=(batch_gpu,), device=device)
                    labels = torch.eye(teacher_net.label_dim, device=device)[label_indices]
                images = idmd_sampler(
                    net=generator_ddp,
                    latents=latents,
                    class_labels=labels,
                    sigma_min=teacher_net.sigma_min,
                    sigma_max=teacher_net.sigma_max,
                    D=D,
                    num_steps=generator_num_steps,
                    multistep_mode=generator_multistep_mode,
                )
                distillation_loss = loss_fn.distillation_loss(
                    teacher_net=teacher_net,
                    student_net=student_ddp.module,
                    images=images,
                    labels=labels
                )
                training_stats.report('Loss/loss', distillation_loss)
                dist.print0("loss:", distillation_loss.mean().item())
                distillation_loss.sum().mul(loss_scaling / (batch_size // dist.get_world_size())).backward()

        # Update weights.
        for g in generator_optimizer.param_groups:
            g['lr'] = generator_optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in generator_net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        generator_optimizer.step()

        # TODO: Update EMA

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(
                ema=ema,
                generator=generator_net,
                student=student_net,
                generator_optimizer_state=generator_optimizer.state_dict(),
                student_optimizer_state=student_optimizer.state_dict(),
                cur_nimg=cur_nimg,
            )
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
            if stats_jsonl is not None:
                stats_jsonl.close()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
