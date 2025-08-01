import torch
import torch.nn as nn

class ElementwiseAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # 初始化两个可学习参数：
        # m 是偏移量 μ，[C, 1]
        # logs 是对数尺度 logσ，[C, 1]
        self.m = nn.Parameter(torch.zeros(channels, 1))      # 初始化为 0，表示初始无偏移
        self.logs = nn.Parameter(torch.zeros(channels, 1))   # 初始化为 0，表示初始缩放为 1（因为 exp(0)=1）

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            # 正向过程：仿射变换 y = μ + σ * x，其中 σ = exp(logs)
            y = self.m + torch.exp(self.logs) * x  # 对每个通道进行独立仿射变换
            y = y * x_mask  # 仅保留有效时间步（掩码位置为1）

            # 计算 log-determinant，用于 Flow Loss
            # 因为变换是可分离的，对每个元素来说，Jacobian 行列式就是 σ（对数就是 logs）
            logdet = torch.sum(self.logs * x_mask, [1, 2])  # 对时间和通道求和
            return y, logdet
        else:
            # 反向过程：x = (y - μ) / σ = (x - m) * exp(-logs)
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x