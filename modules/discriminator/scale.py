import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from modules.resblock import LRELU_SLOPE


class DiscriminatorS(nn.Module):
    """
    尺度判别器（Scale Discriminator）
    使用多层 1D 卷积（包含分组卷积）处理原始波形，擅长捕捉局部细节变化。
    多尺度结构适用于建模不同采样率或频率分辨率。
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        # 选择正则化方法：默认 weight_norm，可选 spectral_norm（更稳定）
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        # 多层 1D 卷积网络（使用较大卷积核，含分组卷积 group conv）
        self.convs = nn.ModuleList([
            # 第一层：输入通道 1 → 输出通道 16，卷积核 15，步长 1，保持时序长度不变
            norm_f(nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7)),

            # 第二层：16 → 64，较大卷积核，步长 4，group=4，进行下采样
            norm_f(nn.Conv1d(16, 64, kernel_size=41, stride=4, groups=4, padding=20)),

            # 第三层：64 → 256，继续使用 group conv
            norm_f(nn.Conv1d(64, 256, kernel_size=41, stride=4, groups=16, padding=20)),

            # 第四层：256 → 1024，继续下采样
            norm_f(nn.Conv1d(256, 1024, kernel_size=41, stride=4, groups=64, padding=20)),

            # 第五层：保持通道数 1024，group=256
            norm_f(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, groups=256, padding=20)),

            # 第六层：卷积核小，细化特征
            norm_f(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
        ])

        # 最后一层卷积：输出通道设为 1（判别得分）
        self.conv_post = norm_f(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        """
        Args:
            x: [B, 1, T] 波形输入
        Returns:
            x: [B, -1] 判别得分向量（flatten）
            fmap: list，特征图（用于 feature matching loss）
        """
        fmap = []  # 存储中间特征图

        # 多层卷积 → LeakyReLU 激活 → 收集中间特征
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        # 最后一层卷积输出
        x = self.conv_post(x)
        fmap.append(x)

        # 展平为一维向量作为判别结果
        x = torch.flatten(x, 1, -1)  # shape: [B, T']

        return x, fmap
