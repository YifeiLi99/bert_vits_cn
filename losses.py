import torch

def feature_loss(fmap_r, fmap_g):
    """
    Feature Matching Loss（特征匹配损失）
    让生成器输出与真实样本在判别器内部的特征图尽可能接近。
    Args:
        fmap_r: 判别器对真实音频提取的特征图列表（多层次）
        fmap_g: 判别器对生成音频提取的特征图列表（多层次）
    Returns:
        loss: 累加的L1特征匹配损失
    """
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):  # 遍历多尺度特征图
        for rl, gl in zip(dr, dg):       # 遍历每一层特征图
            rl = rl.float().detach()     # 真实特征图不参与梯度传播
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))  # L1 loss
    return loss * 2  # 可调权重


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    判别器损失（LSGAN）
    真实样本判别为1，生成样本判别为0
    Args:
        disc_real_outputs: 判别器对真实样本的输出列表
        disc_generated_outputs: 判别器对生成样本的输出列表
    Returns:
        loss: 判别器总loss
        r_losses: 每层对真实样本的loss
        g_losses: 每层对生成样本的loss
    """
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)  # 真实样本距离1的平方差
        g_loss = torch.mean(dg ** 2)        # 生成样本距离0的平方差
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """
    生成器损失（LSGAN）
    生成器希望判别器输出为1
    Args:
        disc_outputs: 判别器对生成样本的输出列表
    Returns:
        loss: 生成器总loss
        gen_losses: 每层loss列表
    """
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)  # 目标是让生成样本接近1
        gen_losses.append(l)
        loss += l
    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    KL Divergence Loss (VAE用)
    让z_p分布与先验分布(m_p, logs_p)接近。
    Args:
        z_p, logs_q: 后验分布采样结果与对数方差 [B, H, T]
        m_p, logs_p: 先验分布的均值与对数方差 [B, H, T]
        z_mask: 有效帧mask [B, 1, T]
    Returns:
        l: 平均KL散度
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    # KL散度计算（逐元素）
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)  # 按有效mask加权求和
    l = kl / torch.sum(z_mask)   # 归一化
    return l
