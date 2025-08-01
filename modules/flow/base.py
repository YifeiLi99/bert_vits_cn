import torch
import torch.nn as nn

class Log(nn.Module):
    # 对数变换模块（Flow 中的一种可逆变换）
    # 正向： 对输入取对数（log），输出变换后值和对数行列式（log-determinant，简称 logdet）
    # 反向： 取指数（exp），用于反变换恢复原值
    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            # 正向：执行 log(x) 操作，同时乘上 mask 避免 padding 干扰
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask  # clamp 防止取 log 时出现负数或 0，避免 -inf 或 NaN
            # 计算 log-determinant，用于 flow 模块的 NLL 损失
            logdet = torch.sum(-y, [1, 2])  # 对 channel 和 time 两个维度求和
            # y：变换后输出   logdet：变换的 log-determinant，用于 Flow 的 NLL loss 计算
            return y, logdet
        else:
            # 反向：恢复 x = exp(y)，并乘 mask 屏蔽 padding 区域
            x = torch.exp(x) * x_mask
            return x


class Flip(nn.Module):
    # 通道翻转模块（打乱维度结构，提高耦合层建模能力）
    # 耦合层每次只处理一半，翻转之后处理另一半
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])  # 在 channel 维度上翻转（dim=1）

        if not reverse:
            # 正向：翻转后返回 logdet=0（因为不改变体积）
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            # 反向：再翻一次即可恢复原样
            return x
