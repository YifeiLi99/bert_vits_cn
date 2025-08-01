import torch
import torch.nn as nn
from norm import LayerNorm

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

        # 随机将部分神经元置为 0，防止过拟合
        self.drop = nn.Dropout(p_dropout)

        # 输入：[B, in_channels, T]（注意是 1D 卷积，时间轴卷积）   输出：[B, filter_channels, T]
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        # LayerNorm 对特征维度归一化（更稳定）
        self.norm_1 = LayerNorm(filter_channels)

        # 输入是第一层的输出，再卷积一次
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        # 归一化第二层的输出，帮助模型稳定训练
        self.norm_2 = LayerNorm(filter_channels)

        # 用一个 1×1 卷积把通道数压缩成 1，表示每个位置的 对数持续时间   输出维度是 [B, 1, T]
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
        # 不希望对齐误差影响到 TextEncoder，这里切断反向传播路径
        x = x.detach()
        # 如果有条件向量，加入主输入
        if g is not None:
            # 避免梯度误传
            g = g.detach()
            # 把条件向量 g（如情感）通过线性映射变换维度后加到 x 上
            x = x + self.cond(g)

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
        # 得到最终输出 [B, 1, T]，每个 token 的 log-duration
        x = self.proj(x * x_mask)

        return x * x_mask  # 输出时保持 mask，确保 padding 不被污染
