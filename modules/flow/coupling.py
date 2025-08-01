import torch
import torch.nn as nn
from modules.flow.base import Flip
from modules.norm import WN

class ResidualCouplingBlock(nn.Module):
    """
    Flow 模块的完整组合体，由多个 ResidualCouplingLayer + Flip 顺序堆叠构成
    模块的封装器和调度器，负责把多个 flow unit 组合并执行流程控制
    """
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
        # 保存构造参数
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        # 用 ModuleList 存放 flow 层（包括 ResidualCouplingLayer 和 Flip），方便迭代管理
        self.flows = nn.ModuleList()

        # 每个 flow 包括两个模块：Residual Coupling + Flip（共两层）
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True  # 表示只预测加性偏移（不缩放），计算简单稳定
                )
            )
            # 添加 Flip，用于打乱顺序，实现两半交替学习
            self.flows.append(Flip())

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
                # 只关心变换后的 x，忽略 logdet
                x, _ = flow(x, x_mask, g=g, reverse=False)
        else:
            # 逆向变换（推理阶段，生成 latent z_p）
            for flow in reversed(self.flows):
                # 将 flow 层倒序，并执行逆变换（即从标准正态分布还原到 latent z）
                x = flow(x, x_mask, g=g, reverse=True)
        return x

    def remove_weight_norm(self):
        """
        推理前移除 Coupling Layer 中所有的 weight norm（防止不稳定）
        注意：只处理偶数 index（flow[i * 2]），因为奇数是 Flip，不含权重，无需处理
        """
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()


class ResidualCouplingLayer(nn.Module):
    """
    可逆变换单元
    对输入 x 的一半特征进行可逆仿射变换，而另一半用于预测参数
    """
    def __init__(
        self,
        channels,           # 输入通道数，必须为偶数，便于一分为二
        hidden_channels,    # 中间隐藏层通道数（非线性建模）
        kernel_size,        # 卷积核大小
        dilation_rate,      # 卷积膨胀率
        n_layers,           # 层数（WN 中用）
        p_dropout=0,        # dropout 概率
        gin_channels=0,     # 条件输入通道（如情感 embedding）
        mean_only=False     # 是否只建模均值（不建模缩放 log σ）
    ):
        # 输入通道必须是偶数，便于一分为二（x0, x1），用于 coupling
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()

        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2  # 输入一分为二
        self.mean_only = mean_only          # 是否只输出均值（默认同时建模 μ 和 logσ）

        # 预处理层：将 x 的一半映射到 hidden_channels
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)

        # 主干网络：WaveNet 模式残差网络，用于非线性建模
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )

        # 输出层：输出 m（或者 m 和 logs），通道数视 mean_only 决定
        self.post = nn.Conv1d(
            hidden_channels,
            self.half_channels * (2 - mean_only),  # 如果只预测 mean，则输出一组；否则输出 mean 和 log_std
            1
        )
        # 初始化为恒等变换（加速 early-stage 收敛）
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        正向或反向的 Flow 变换。
        Args:
            x: [B, C, T]，输入特征
            x_mask: 掩码，用于屏蔽 padding，形状 [B, 1, T]
            g: 可选条件向量（如情感 embedding），[B, gin_channels, T]
            reverse: 是否进行逆向传播（用于采样）
        Returns:
            如果正向：返回变换后的 x 和 logdet（对数行列式）
            如果反向：仅返回逆向还原后的 x
        """
        # 将输入沿通道维一分为二（Coupling机制核心）  一半作为条件特征（x0），另一半为被变换对象（x1）
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)

        # 用 x0 预测 x1 的转换参数
        h = self.pre(x0) * x_mask  # 预处理 + mask   仅处理有效部分
        h = self.enc(h, x_mask, g=g)  # 进入主干 Wavenet 编码器，提取特征并注入条件向量（g），输出特征 h
        stats = self.post(h) * x_mask  # 用 post 卷积层将特征 h 映射为 m/logs（或只 m），并乘掩码

        if not self.mean_only:
            # 若输出两组通道，则按通道维拆成 μ 和 logσ
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            # 若只预测 μ，则 logσ = 0，相当于 σ = 1（无缩放）
            m = stats
            logs = torch.zeros_like(m)

        # 正向变换（训练/推理）
        if not reverse:
            # 正向：x1 → z1（可逆变换）
            x1 = m + x1 * torch.exp(logs) * x_mask
            # 把变换后的 x1 与未变换的 x0 合并
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])  # 对 log σ 累加，得到对数行列式
            # 返回 logdet（用于 flow 的概率密度计算）
            return x, logdet
        else:
            # 逆向：z1 → x1，逆向恢复
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            # 合并后返回
            x = torch.cat([x0, x1], 1)
            return x

    def remove_weight_norm(self):
        """
        为了加速推理时计算，移除 WN 网络内部的权重归一化
        """
        self.enc.remove_weight_norm()
