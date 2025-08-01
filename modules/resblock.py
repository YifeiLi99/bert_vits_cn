import torch
import torch.nn as nn
from torch.nn import Conv1d
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.nn import functional as F
from commons.commons import get_padding, init_weights
LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        """
        构建一个三组残差卷积块，每组包含两个卷积层。
        - 第一层为膨胀卷积（dilated conv），感受野逐渐增大；
        - 第二层为普通卷积，用于特征整合和残差回路。

        Args:
            channels: 输入/输出通道数（不变）
            kernel_size: 卷积核大小
            dilation: 三层膨胀率，默认为 (1, 3, 5)
        """
        super(ResBlock1, self).__init__()

        # convs1：3 层不同膨胀率的卷积（感受野增大），每层都带权重归一化
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,  # 输入通道
                        channels,  # 输出通道
                        kernel_size,  # 卷积核大小
                        stride=1,
                        dilation=dilation[0],  # 第一层 dilation=1
                        padding=get_padding(kernel_size, dilation[0])
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],  # 第二层 dilation=3
                        padding=get_padding(kernel_size, dilation[1])
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],  # 第三层 dilation=5
                        padding=get_padding(kernel_size, dilation[2])
                    )
                ),
            ]
        )
        # 初始化卷积权重
        self.convs1.apply(init_weights)

        # convs2：普通卷积（dilation=1），与 convs1 一一对应
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1)
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1)
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1)
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        """
        Args:
            x: 输入特征 [B, C, T]
            x_mask: 掩码张量 [B, 1, T]（用于处理 padding）
        Returns:
            x: 输出特征，残差连接后的结果
        """
        # 三组（conv1, conv2）组成的残差块
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)  # 激活
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)  # 第一个卷积：膨胀卷积
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)  # 第二个卷积：标准卷积

            # 残差连接
            x = xt + x

        # 最后再乘一次掩码（确保 padding 区为 0）
        if x_mask is not None:
            x = x * x_mask

        return x

    def remove_weight_norm(self):
        """
        推理阶段移除 weight normalization，提高效率
        """
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        """
        第二种残差卷积块（相比 ResBlock1 简化了一层结构）：
        包含两层带膨胀率的卷积结构 + 残差连接，用于局部感受建模。

        Args:
            channels: 输入/输出通道数（保持一致）
            kernel_size: 卷积核大小（通常为 3）
            dilation: 两层卷积的膨胀率（默认分别为 1 和 3）
        """
        super(ResBlock2, self).__init__()

        # 两层卷积，每层都有 weight_norm 和 dilation（感受野逐渐扩大）
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels, channels,
                        kernel_size,
                        stride=1,
                        dilation=dilation[0],  # 第一层 dilation
                        padding=get_padding(kernel_size, dilation[0])
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels, channels,
                        kernel_size,
                        stride=1,
                        dilation=dilation[1],  # 第二层 dilation 更大
                        padding=get_padding(kernel_size, dilation[1])
                    )
                ),
            ]
        )

        # 初始化权重
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        """
        前向传播：
        每层都进行 LeakyReLU -> 卷积 -> 残差连接，并可选乘上 mask。

        Args:
            x: 输入特征 [B, C, T]
            x_mask: 掩码 [B, 1, T]（屏蔽 padding 区域）
        Returns:
            x: 残差更新后的特征
        """
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x  # 残差连接（跳跃连接）

        if x_mask is not None:
            x = x * x_mask  # 最后统一应用掩码

        return x

    def remove_weight_norm(self):
        """
        推理前移除卷积中的 weight norm（提升推理效率）
        """
        for l in self.convs:
            remove_weight_norm(l)
