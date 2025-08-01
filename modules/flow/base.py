import torch
import torch.nn as nn

class Log(nn.Module):
    # 对数变换模块（Flow 中的一种可逆变换）
    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            # 正向：执行 log(x) 操作，同时乘上 mask 避免 padding 干扰
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask  # clamp 避免 log(0)
            # 计算 log-determinant，用于 flow 模块的 NLL 损失
            logdet = torch.sum(-y, [1, 2])  # 对 channel 和 time 两个维度求和
            return y, logdet
        else:
            # 反向：恢复 x = exp(y)，并乘 mask 屏蔽 padding 区域
            x = torch.exp(x) * x_mask
            return x


class Flip(nn.Module):
    # 通道翻转模块（打乱维度结构，提高耦合层建模能力）
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])  # 在 channel 维度上翻转（dim=1）

        if not reverse:
            # 正向：翻转后返回 logdet=0（因为不改变体积）
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            # 反向：再翻一次即可恢复原样
            return x
