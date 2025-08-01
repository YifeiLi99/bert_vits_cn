import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.resblock import LRELU_SLOPE
from commons.commons import get_padding
from torch.nn.utils import weight_norm, spectral_norm

class DiscriminatorP(nn.Module):
    """
    周期判别器（Periodic Discriminator）
    用于捕捉语音中周期性结构（如基频）的特征。
    每个实例专注于一个给定的周期 P，将输入波形按周期展开为二维形式进行卷积判别。
    """

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        """
        Args:
            period (int): 设定的周期长度（例如 2, 3, 5, 7 等）
            kernel_size (int): 卷积核大小（仅 time 维度参与）
            stride (int): 步幅（只作用于 time 维度）
            use_spectral_norm (bool): 是否使用谱归一化代替权重归一化
        """
        super().__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm

        # 根据配置选择正则化方式
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        # 构造5层2D卷积网络（仅 time 方向滑动）
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])

        # 输出层（生成最终判别 score）
        self.conv_post = norm_f(
            nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )

    def forward(self, x):
        """
        Args:
            x: [B, 1, T] 音频波形张量
        Returns:
            x: [B, -1] 判别器输出（flatten）
            fmap: List[Tensor] 各层中间特征图，用于 feature matching loss
        """
        fmap = []

        # 处理周期不整除的情况，进行反射填充以保证 T 能被 period 整除
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), mode="reflect")
            t += n_pad

        # 变换为 2D 格式：[B, 1, T // period, period]
        x = x.view(b, c, t // self.period, self.period)

        # 多层卷积 + LeakyReLU 激活 + 保存 fmap
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        # 最终判别输出层
        x = self.conv_post(x)
        fmap.append(x)

        # 展平输出，得到判别结果（用于 GAN loss）
        x = torch.flatten(x, 1, -1)  # [B, -1]

        return x, fmap
