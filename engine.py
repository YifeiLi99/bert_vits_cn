








# ========================= 单轮训练模块 =================================
def train_one_epoch(
    rank, epoch, hps, net_g, net_d, optim_g, optim_d,
    scheduler_g, scheduler_d, scaler, train_loader, logger, writer
):
    """
    单轮训练流程。用于多卡或单卡训练。

    参数：
        rank: 当前进程编号
        epoch: 当前 epoch 编号
        hps: 超参数配置
        net_g/net_d: Generator 和 Discriminator
        optim_g/optim_d: 对应优化器
        scheduler_g/scheduler_d: 学习率调度器
        scaler: 混合精度训练 scaler
        train_loader: 训练集 DataLoader
        logger: 日志记录器
        writer: TensorBoard 写入器
    """
    global global_step
    net_g.train()
    net_d.train()
    train_loader.batch_sampler.set_epoch(epoch)

    loader = tqdm.tqdm(train_loader, desc='Loading train data') if rank == 0 else train_loader

    for batch_idx, (x, x_lengths, bert, spec, spec_lengths, y, y_lengths) in enumerate(loader):
        # === 将数据搬到当前 rank 所属 GPU 上 ===
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
        bert = bert.cuda(rank, non_blocking=True)

        # === Generator 正向传播 ===
        with autocast(enabled=hps.train.fp16_run):
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
            (z, z_p, z_r, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, bert, spec, spec_lengths)

            mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels,
                                    hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), hps.data.filter_length, hps.data.n_mel_channels,
                                              hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                              hps.data.mel_fmin, hps.data.mel_fmax)
            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)

            # === Discriminator 推理 + 损失 ===
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc

        # === 判别器反向传播 ===
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        # === Generator 训练 ===
        with autocast(enabled=hps.train.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_kl_r = 0 if z_r is None else kl_loss(z_r, logs_p, m_q, logs_q, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_kl_r

        # === 生成器反向传播 ===
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        # === 日志记录 ===
        if rank == 0 and global_step % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]["lr"]
            losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl, loss_kl_r]
            logger.info(f"Train Epoch: {epoch} [{100. * batch_idx / len(train_loader):.0f}%]")
            logger.info([global_step, lr])
            logger.info(f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f}")
            logger.info(f"loss_mel={loss_mel:.3f}, loss_dur={loss_dur:.3f}, loss_kl={loss_kl:.3f}, loss_kl_r={loss_kl_r:.3f}")

            scalar_dict = {
                "loss/g/total": loss_gen_all,
                "loss/d/total": loss_disc_all,
                "learning_rate": lr,
                "grad_norm_d": grad_norm_d,
                "grad_norm_g": grad_norm_g,
                "loss/g/fm": loss_fm,
                "loss/g/mel": loss_mel,
                "loss/g/dur": loss_dur,
                "loss/g/kl": loss_kl,
                "loss/g/kl_r": loss_kl_r,
                **{f"loss/g/{i}": v for i, v in enumerate(losses_gen)},
                **{f"loss/d_r/{i}": v for i, v in enumerate(losses_disc_r)},
                **{f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)},
            }

            image_dict = {
                "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                "all/attn": utils.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy()),
            }

            utils.summarize(writer=writer, global_step=global_step,
                            images=image_dict, scalars=scalar_dict)

        global_step += 1

    scheduler_g.step()
    scheduler_d.step()

    if rank == 0:
        logger.info(f"====> Epoch: {epoch}")



# ========================== 验证模块 =====================================
def evaluate(hps, generator, eval_loader, writer_eval, device):
    generator.eval()

    with torch.no_grad():
        try:
            x, x_lengths, bert, spec, spec_lengths, y, y_lengths = next(iter(eval_loader))
        except StopIteration:
            print("[WARNING] Empty eval_loader, skipping evaluate.")
            return

        x = x.to(device)[:1]
        x_lengths = x_lengths.to(device)[:1]
        bert = bert.to(device)[:1]
        spec = spec.to(device)[:1]
        spec_lengths = spec_lengths.to(device)[:1]
        y = y.to(device)[:1]
        y_lengths = y_lengths.to(device)[:1]

        y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, bert, max_len=1000)
        y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

        gt_mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
        pred_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )

    image_dict = {
        f"gen/mel_{global_step}": utils.plot_spectrogram_to_numpy(
            pred_mel[0].cpu().numpy()
        )
    }
    audio_dict = {
        f"gen/audio_{global_step}": y_hat[0, :, : y_hat_lengths[0]]
    }

    if global_step == 0:
        image_dict["gt/mel"] = utils.plot_spectrogram_to_numpy(gt_mel[0].cpu().numpy())
        audio_dict["gt/audio"] = y[0, :, : y_lengths[0]]

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )

    generator.train()



# ========================= 调度模块（训练+验证） ===========================
def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers,
    scaler, loaders, logger, writers
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    writer, writer_eval = writers if writers is not None else (None, None)

    # === 执行训练轮 ===
    train_one_epoch(
        rank, epoch, hps,
        net_g, net_d,
        optim_g, optim_d,
        scheduler_g, scheduler_d,
        scaler, train_loader,
        logger, writer,
    )

    # === 可选执行验证 + 模型保存 ===
    if rank == 0 and global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval)
        utils.save_checkpoint(
            net_g, optim_g, hps.train.learning_rate, epoch,
            os.path.join(hps.model_dir, f"G_{global_step}.pth")
        )
        utils.save_checkpoint(
            net_d, optim_d, hps.train.learning_rate, epoch,
            os.path.join(hps.model_dir, f"D_{global_step}.pth")
        )



