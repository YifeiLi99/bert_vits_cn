import torch
import torch.nn as nn
from models import modules  # 假设你有 LayerNorm 定义在 commons/modules.py 中


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        """
        预测每个输入 token 的持续时间（帧数）

        in_channels (int): 输入通道数（通常是 TextEncoder 输出 dim）
        filter_channels (int): 卷积层中间维度
        kernel_size (int): 卷积核大小
        p_dropout (float): dropout 概率
        gin_channels (int): 条件信息通道（如情感 embedding），可选
        """
        super().__init__()

        self.drop = nn.Dropout(p_dropout)

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = modules.LayerNorm(filter_channels)

        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = modules.LayerNorm(filter_channels)

        self.proj = nn.Conv1d(filter_channels, 1, kernel_size=1)  # 输出一个标量，即 log-duration

        if gin_channels != 0:
            # 如果需要条件信息（如情感 embedding），映射到输入维度后加和
            self.cond = nn.Conv1d(gin_channels, in_channels, kernel_size=1)

    def forward(self, x, x_mask, g=None):
        """
        x (Tensor): shape [B, C, T]，TextEncoder 的输出
        x_mask (Tensor): shape [B, 1, T]，输入掩码
        g (Tensor, optional): 条件向量（如情绪 embedding）

        Returns: Tensor: log-duration，shape [B, 1, T]
        """
        x = x.detach()  # 避免梯度流入
        if g is not None:
            g = g.detach()
            x = x + self.cond(g)

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)

        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        x = self.proj(x * x_mask)
        return x * x_mask  # 输出时也乘上 mask，保持 padding 对齐
