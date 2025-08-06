import os
import torch
import torch.multiprocessing as mp
from my_utils.config_utils import  get_hparams

# ======================= 主函数 ==========================
def main():
    """
    可选单gpu或多gpu训练
    """
    # 加载 config.yaml
    hps = get_hparams()

    # 多卡训练
    if hps.train.distributed:
        n_gpus = torch.cuda.device_count()
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "40000"

        mp.spawn(
            run_distributed,   # run(rank, n_gpus, hps)
            nprocs=n_gpus,
            args=(n_gpus, hps),
        )
    else:
        # 单卡训练
        run_single_gpu(hps)

# ============================= 多卡训练 ==============================
def run_distributed(rank, n_gpus, hps):
    global global_step

    # ========== 初始化 ==========
    torch.cuda.set_device(rank)
    backend_str = "gloo" if platform.system().lower() == "windows" else "nccl"
    dist.init_process_group(
        backend=backend_str,
        init_method="env://",
        world_size=n_gpus,
        rank=rank
    )
    torch.manual_seed(hps.train.seed)

    # ========== 日志和 TensorBoard 只在主进程启用 ==========
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    else:
        logger = None
        writer = None
        writer_eval = None

    # ========== 数据加载 ==========
    train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=False,  # 分布式由 sampler 控制
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )

    if rank == 0:
        eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=8,
            shuffle=False,
            batch_size=hps.train.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    else:
        eval_loader = None

    # ========== 模型构建 ==========
    net_g = utils.load_class(hps.train.train_class)(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    # ========== Teacher 模型（可选） ==========
    try:
        teacher = getattr(hps.train, "teacher")
        if rank == 0:
            logger.info(f"Has teacher model: {teacher}")
        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
        utils.load_teacher(teacher, net_g)
    except:
        net_g = DDP(net_g, device_ids=[rank])
        if rank == 0:
            logger.info("No teacher model.")

    net_d = DDP(net_d, device_ids=[rank])

    # ========== checkpoint 加载 ==========
    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0

    # ========== 学习率调度器 + 混精度 ==========
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    # ========== 正式训练 ==========
    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        train_and_evaluate(
            rank=rank,
            epoch=epoch,
            hps=hps,
            nets=[net_g, net_d],
            optims=[optim_g, optim_d],
            schedulers=[scheduler_g, scheduler_d],
            scaler=scaler,
            loaders=[train_loader, eval_loader],
            logger=logger,
            writers=[writer, writer_eval],
        )
        scheduler_g.step()
        scheduler_d.step()

# ============================ 单卡训练 ============================
def run_single_gpu(hps):
    global global_step

    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)

    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(0)  # 单卡固定 GPU 0

    # ========== 数据加载 ==========
    train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=True,   # ✅ 单卡直接 shuffle
        pin_memory=True,
        collate_fn=collate_fn,
        batch_size=hps.train.batch_size
    )

    eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(
        eval_dataset,
        num_workers=8,
        shuffle=False,
        batch_size=hps.train.batch_size,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # ========== 模型构建 ==========
    net_g = utils.load_class(hps.train.train_class)(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).cuda()
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda()

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    # ========== Teacher 模型（可选） ==========
    try:
        teacher = getattr(hps.train, "teacher")
        logger.info(f"Has teacher model: {teacher}")
        utils.load_teacher(teacher, net_g)
    except:
        logger.info("No teacher model.")

    # ========== 加载 checkpoint ==========
    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0

    # ========== 学习率调度器 + 混精度 ==========
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    # ========== 正式训练 ==========
    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_and_evaluate(
            rank=0,
            epoch=epoch,
            hps=hps,
            nets=[net_g, net_d],
            optims=[optim_g, optim_d],
            schedulers=[scheduler_g, scheduler_d],
            scaler=scaler,
            loaders=[train_loader, eval_loader],
            logger=logger,
            writers=[writer, writer_eval],
        )

        scheduler_g.step()
        scheduler_d.step()


if __name__ == "__main__":
    main()
