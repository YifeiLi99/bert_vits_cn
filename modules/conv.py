import torch.nn as nn

class ConvReluNorm(nn.Module):
    """
    多层 1D 卷积 + LayerNorm + ReLU + Dropout + 残差连接模块。
    适用于序列建模任务，如 TTS 中处理编码器或持续性特征。
    """

    def __init__(
        self,
        in_channels,        # 输入通道数
        hidden_channels,    # 隐藏层通道数（卷积输出通道）
        out_channels,       # 输出通道数
        kernel_size,        # 卷积核大小
        n_layers,           # 卷积层数（应 ≥2）
        p_dropout           # dropout 概率
    ):
        super().__init__()
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))

        # 第一层卷积
        self.conv_layers.append(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        )
        self.norm_layers.append(LayerNorm(hidden_channels))

        # 后续 n_layers - 1 层卷积
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
            )
            self.norm_layers.append(LayerNorm(hidden_channels))

        # 最后一层 projection，将 hidden_channels → out_channels
        self.proj = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        self.proj.weight.data.zero_()  # 初始化为0
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        """
        Args:
            x: 输入张量，[B, C, T]
            x_mask: 掩码张量，[B, 1, T]，用于屏蔽 padding 部分
        Returns:
            x: 输出特征，[B, C_out, T]
        """
        x_org = x  # 残差路径
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)    # 应用掩码
            x = self.norm_layers[i](x)             # LayerNorm
            x = self.relu_drop(x)                  # ReLU + Dropout

        x = x_org + self.proj(x)                   # 残差连接 + 输出投影
        return x * x_mask                          # 再次屏蔽 padding 区


class DDSConv(nn.Module):
    """
    DDSConv: Dilated & Depthwise-Separable Convolution Module
    用于高效提取时序上下文信息，融合空洞卷积与残差结构。
    """

    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
        """
        Args:
            channels: 输入/输出通道数
            kernel_size: 卷积核大小（通常为 3 或 5）
            n_layers: 堆叠的层数
            p_dropout: dropout 概率
        """
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()  # 深度可分离卷积（带 dilation）
        self.convs_1x1 = nn.ModuleList()  # Pointwise 卷积（通道融合）
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()

        for i in range(n_layers):
            dilation = kernel_size**i  # 空洞倍数指数递增
            padding = (kernel_size * dilation - dilation) // 2  # 保持输出长度不变
            self.convs_sep.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    groups=channels,     # depthwise
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, kernel_size=1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x, x_mask, g=None):
        """
        Args:
            x: 输入张量 [B, C, T]
            x_mask: 掩码张量 [B, 1, T]
            g: 可选的条件向量 [B, C, 1]（如 speaker embedding）
        Returns:
            x: 输出张量 [B, C, T]
        """
        if g is not None:
            x = x + g  # 加入条件信息（如 speaker embedding）

        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)

            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)

            y = self.drop(y)
            x = x + y  # 残差连接
        return x * x_mask

