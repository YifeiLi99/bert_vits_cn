import torch
import torch.nn as nn
from modules.discriminator import DiscriminatorS, DiscriminatorP

class MultiPeriodDiscriminator(nn.Module):
    """
    多周期判别器（Multi-Period Discriminator, MPD）
    聚合了：
    - 1 个尺度判别器（DiscriminatorS）
    - 多个周期判别器（DiscriminatorP，周期 ∈ [2, 3, 5, 7, 11]）
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()

        periods = [2, 3, 5, 7, 11]

        # 1 个尺度判别器
        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]

        # 5 个周期判别器
        discs += [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]

        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        """
        Args:
            y: [B, 1, T]，真实波形
            y_hat: [B, 1, T]，生成波形

        Returns:
            y_d_rs: List[判别器输出]，对真实语音的打分结果
            y_d_gs: List[判别器输出]，对合成语音的打分结果
            fmap_rs: List[中间特征]，用于 feature matching loss（真实）
            fmap_gs: List[中间特征]，用于 feature matching loss（生成）
        """
        y_d_rs, y_d_gs = [], []
        fmap_rs, fmap_gs = [], []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)       # 判别真实语音
            y_d_g, fmap_g = d(y_hat)   # 判别生成语音
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
