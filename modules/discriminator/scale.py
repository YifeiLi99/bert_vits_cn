import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm, spectral_norm
from modules import modules  # 提供 LRELU_SLOPE

class DiscriminatorS(nn.Module):
    """
    尺度判别器（Scale Discriminator）
    使用多层 1D 卷积（包含分组卷积）处理原始波形，擅长捕捉局部细节变化。
    多尺度结构适用于建模不同采样率或频率分辨率。
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        # 多层 1D 卷积网络（包含分组卷积 group conv）
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, kernel_size=15, stride=1, padding=7)),                     # 保留 T 长度
            norm_f(Conv1d(16, 64, kernel_size=41, stride=4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, kernel_size=41, stride=4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, kernel_size=41, stride=4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, kernel_size=41, stride=4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
        ])

        # 输出判别值的卷积层
        self.conv_post = norm_f(Conv1d(1024, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        """
        Args:
            x: [B, 1, T] 波形输入
        Returns:
            score: [B, -1] 判别得分向量
            fmap: list，特征图（用于 feature matching loss）
        """
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)  # 展平输出作为判别结果

        return x, fmap
