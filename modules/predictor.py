import torch
import torch.nn as nn
import norm

class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        """
        持续时间预测模块（log-duration）
        输入为 TextEncoder 的输出，预测每个时间步的持续时长（以帧数计）

        Args:
            in_channels: 输入通道数，通常是 TextEncoder 的输出维度
            filter_channels: 卷积层中间维度
            kernel_size: 卷积核大小（通常为 3 或 5）
            p_dropout: dropout 概率（防止过拟合）
            gin_channels: 可选条件通道（如情绪 embedding 的维度）
        """
        super().__init__()

        self.drop = nn.Dropout(p_dropout)

        # 第一层 1D 卷积 + LayerNorm
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = norm.LayerNorm(filter_channels)

        # 第二层 1D 卷积 + LayerNorm
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = norm.LayerNorm(filter_channels)

        # 投影层，输出 1 个通道（表示 log-duration）
        self.proj = nn.Conv1d(filter_channels, 1, kernel_size=1)

        # 如果启用条件输入（如情感），加入 1x1 conv 以适配维度并加和到主输入
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, kernel_size=1)

    def forward(self, x, x_mask, g=None):
        """
        前向传播

        Args:
            x: 输入序列，[B, C, T]，来自 TextEncoder
            x_mask: 输入掩码，[B, 1, T]，标记有效 token
            g: 条件向量（可选），如情感 embedding，[B, gin_channels, T]

        Returns:
            log_durations: [B, 1, T]，预测每个 token 的对数持续时长
        """
        x = x.detach()  # 阻断梯度，防止影响 TextEncoder（因为这是辅助模块）
        if g is not None:
            g = g.detach()
            x = x + self.cond(g)  # 融合条件向量（如情感信息）

        # 第一层卷积 + 激活 + LayerNorm + dropout
        x = self.conv_1(x * x_mask)  # 注意加了 mask，只对有效 token 做卷积
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)

        # 第二层卷积 + 激活 + LayerNorm + dropout
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        # 输出 log-duration（经过线性变换）
        x = self.proj(x * x_mask)

        return x * x_mask  # 输出时保持 mask，确保 padding 不被污染
