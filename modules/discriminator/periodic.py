import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d
from torch.nn.utils import weight_norm, spectral_norm
from .commons import get_padding  # padding计算函数
from modules import modules             # 提供 LRELU_SLOPE

class DiscriminatorP(nn.Module):
    """
    周期判别器（Periodic Discriminator）
    每个实例对应一个周期 period，例如：2, 3, 5, 7 等
    将 wave 切分为 [B, 1, T//period, period] 的 2D 输入
    """

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm

        # 选择正则化方法
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        # 5 层卷积块（2D卷积，仅在 time 轴有 stride）
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])

        # 最后一层输出（卷积 → 判别得分）
        self.conv_post = norm_f(Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0)))

    def forward(self, x):
        """
        Args:
            x: [B, 1, T] 波形数据
        Returns:
            score: [B, T'] 判别得分
            fmap: list，特征图（用于 feature matching loss）
        """
        fmap = []

        # reshape 成 [B, 1, T//P, P]
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")  # 反射填充
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)  # reshape 成 2D

        # 多层卷积 + 激活
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)

        # 最后一层卷积
        x = self.conv_post(x)
        fmap.append(x)

        # flatten → 最终判别 score 向量
        x = torch.flatten(x, 1, -1)  # [B, -1]

        return x, fmap
