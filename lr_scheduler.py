from timm.scheduler.cosine_lr import CosineLRScheduler


def create_scheduler(args, optimizer, n_iter_per_epoch):
    num_steps = int(args.epochs * n_iter_per_epoch)
    print("From create_scheduler: {}".format(num_steps))
    lr_scheduler = None
    if args.sched == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            # cycle_mul=getattr(args, 'lr_cycle_mul', 1.),
            t_mul=getattr(args, 'lr_cycle_mul', 1.),
            lr_min=args.min_lr,
            # decay_rate=args.decay_rate,
            cycle_decay=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_steps,
            cycle_limit=getattr(args, 'lr_cycle_limit', 1),
            t_in_epochs=False,
            noise_range_t=None,
            noise_pct=getattr(args, 'lr_noise_pct', 0.67),
            noise_std=getattr(args, 'lr_noise_std', 1.),
            noise_seed=getattr(args, 'seed', 42),
        )
    else:
        raise NotImplementedError(f'{args.sched} scheduler is not implemented.')

    return lr_scheduler


# lr_scheduler = CosineLRScheduler(
#             optimizer,
#             t_initial=num_epochs * iter_per_epoch,
#             cycle_mul=getattr(args, 'lr_cycle_mul', 1.),
#             lr_min=args.min_lr,
#             cycle_decay=args.decay_rate,
#             warmup_lr_init=args.warmup_lr,
#             warmup_t=warmup_t,
#             cycle_limit=getattr(args, 'lr_cycle_limit', 1),
#             t_in_epochs=False,
#             noise_range_t=noise_range,
#             noise_pct=getattr(args, 'lr_noise_pct', 0.67),
#             noise_std=getattr(args, 'lr_noise_std', 1.),
#             noise_seed=getattr(args, 'seed', 42),