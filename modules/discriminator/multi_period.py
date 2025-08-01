import torch.nn as nn
from modules.discriminator.periodic import DiscriminatorP
from modules.discriminator.scale import DiscriminatorS

class MultiPeriodDiscriminator(nn.Module):
    """
    多周期判别器（Multi-Period Discriminator, MPD）
    结合多个周期尺度与时间尺度的判别器，用于更全面地评估语音生成质量。
    组成：
    - 1 个尺度判别器 DiscriminatorS（从时域尺度捕捉细节）
    - 多个周期判别器 DiscriminatorP（从周期性结构捕捉节奏感）
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()

        # 周期判别器的周期配置，分别从不同节奏尺度感知（奇素数周期）
        periods = [2, 3, 5, 7, 11]

        # 创建判别器：1 个尺度 + 5 个周期
        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]  # 全局尺度特征
        discs += [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods]  # 不同周期特征

        # 模块列表封装
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        """
        判别器前向传播，对真实语音 y 和合成语音 y_hat 同时评估。

        Args:
            y:      Tensor [B, 1, T]，真实语音波形
            y_hat:  Tensor [B, 1, T]，生成语音波形

        Returns:
            y_d_rs: List[Tensor]，所有判别器对真实语音的判别结果（loss使用）
            y_d_gs: List[Tensor]，所有判别器对生成语音的判别结果（loss使用）
            fmap_rs: List[List[Tensor]]，所有判别器的中间特征（真实），用于 FM loss
            fmap_gs: List[List[Tensor]]，所有判别器的中间特征（生成），用于 FM loss
        """
        y_d_rs, y_d_gs = [], []     # 判别分数
        fmap_rs, fmap_gs = [], []   # 特征图（feature map）

        # 依次通过 6 个判别器（1个S + 5个P）
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)       # 判别真实语音
            y_d_g, fmap_g = d(y_hat)   # 判别合成语音

            # 记录结果
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
