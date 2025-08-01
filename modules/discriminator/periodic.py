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
        """
        每一层是一个 2D 卷积。
        kernel_size=(k,1) 说明只在时间维度做滑动，不跨越周期维度。
        stride=(s,1) 同样表示只在时间轴做步进。
        特征图维度从 1 → 32 → 128 → 512 → 1024 → 1024，逐渐加深表示能力。
        目的是挖掘跨周期的周期性特征（pitch, harmonics）
        """
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
            # 反射填充不会引入明显的新频率分量，比零填充更自然
            x = F.pad(x, (0, n_pad), mode="reflect")
            t += n_pad

        # reshape 为周期维度的二维格式 ：[B, 1, T//period, period]
        # 用 2D 卷积提取周期相关特征，把“周期结构”显式建模进网络结构里
        x = x.view(b, c, t // self.period, self.period)

        # 多层卷积 + LeakyReLU 激活 + 保存 fmap
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            # 输出都记录进 fmap 列表中，供后续用于 Feature Matching Loss
            fmap.append(x)

        # 最终判别输出层
        x = self.conv_post(x)
        # 得到一个单通道的“判别热力图”，反映每个位置是否像真实音频
        fmap.append(x)

        # 展平输出，得到判别结果（用于 GAN loss）
        x = torch.flatten(x, 1, -1)  # [B, -1]

        return x, fmap
