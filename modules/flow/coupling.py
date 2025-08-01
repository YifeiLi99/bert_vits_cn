import torch
import torch.nn as nn
from modules.flow.base import Flip
from modules.norm import WN

class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,           # 输入通道数（通常是 latent z 的通道数）
        hidden_channels,    # 内部耦合网络的隐藏通道数
        kernel_size,        # 卷积核大小
        dilation_rate,      # 卷积的膨胀率（决定感受野）
        n_layers,           # 每个 Coupling Layer 中的网络层数
        n_flows=4,          # Flow 层的数量，每个 flow = 1个 ResidualCouplingLayer + 1个 Flip
        gin_channels=0      # 条件输入通道数（如情感 embedding 的通道数）
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            # 每个 flow 包括两个模块：Residual Coupling + Flip（共两层）
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True  # 表示只预测加性偏移（不缩放）
                )
            )
            self.flows.append(Flip())  # 添加 Flip，用于打乱顺序，实现两半交替学习

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        进行 Flow 的正向或反向变换。
        Args:
            x: [B, C, T]，表示 latent z（或其变换）
            x_mask: [B, 1, T]，表示有效长度的掩码
            g: 可选条件向量 [B, gin_channels, T]，如情感 embedding
            reverse: 是否执行逆变换（通常在 inference 时用）
        Returns:
            x: Flow 变换后的输出
        """
        if not reverse:
            # 正向变换（训练阶段）
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=False)
        else:
            # 逆向变换（推理阶段，生成 latent z_p）
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=True)
        return x

    def remove_weight_norm(self):
        """
        推理前移除 Coupling Layer 中所有的 weight norm（防止不稳定）
        注意：只处理偶数 index（flow[i * 2]），因为奇数是 Flip，无需操作
        """
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()



class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()