import torch
import torch.nn as nn
import math
from modules.conv import DDSConv
from modules.flow.spline_flow import piecewise_rational_quadratic_transform

class ConvFlow(nn.Module):
    def __init__(
        self,
        in_channels,        # 输入通道数（通常是 latent z 的维度）
        filter_channels,    # 卷积层的中间维度
        kernel_size,        # 卷积核大小（用于 DDSConv）
        n_layers,           # 卷积层堆叠层数
        num_bins=10,        # 分段数量（用于 piecewise rational quadratic spline）
        tail_bound=5.0,     # 尾部边界（定义输入支持的范围）
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2  # 输入分成两半做 coupling

        # 前置 1x1 卷积，用于将 x0 的信息映射为高维特征
        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)

        # 多层可控扩张卷积（可参考 Glow / VITS DDSConv 结构）
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)

        # 最后线性映射，用于输出 spline 的参数：宽度、高度、斜率
        # 对于每个通道：num_bins 段 × 3 参数（除了第一个高度没斜率，故少1）
        self.proj = nn.Conv1d(
            filter_channels, self.half_channels * (num_bins * 3 - 1), 1
        )

        # 初始化为恒等映射，便于 early training 稳定
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        # Split x along channel dim into two halves
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)

        # 1x1 conv + DDSConv 对 x0 编码，得到 spline 参数
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask  # 最终输出 spline 参数

        # reshape 为 spline 参数格式：[B, C, T, num_params]
        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [B, C, T, param_per_bin]

        # 提取 spline 所需的三个参数组：
        unnormalized_widths = h[..., :self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins: 2 * self.num_bins] / math.sqrt(
            self.filter_channels
        )
        unnormalized_derivatives = h[..., 2 * self.num_bins:]  # 尾部斜率

        # 使用 piecewise rational quadratic spline 进行流变换
        # 参考 Durkan et al. (2020) Neural Spline Flows
        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",             # 尾部线性映射
            tail_bound=self.tail_bound  # 输入值限制在 [-tail_bound, tail_bound]
        )

        # 拼接变换后的 x1 和 x0，乘上 mask
        x = torch.cat([x0, x1], 1) * x_mask

        # 对 log Jacobian 行列式求和作为 flow loss
        logdet = torch.sum(logabsdet * x_mask, [1, 2])

        if not reverse:
            return x, logdet  # 正向：返回变换结果和 logdet
        else:
            return x  # 逆向：仅返回 x
