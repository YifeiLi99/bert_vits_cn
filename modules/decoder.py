import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from models import modules  # 提供 ResBlock1/2、LRELU_SLOPE、init_weights

class Generator(nn.Module):
    def __init__(
        self,
        initial_channel,              # 输入 latent z 的通道数
        resblock,                     # 残差块类型："1" or "2"
        resblock_kernel_sizes,       # 每个残差块的 kernel_size 列表
        resblock_dilation_sizes,     # 每个残差块的 dilation 配置（列表套列表）
        upsample_rates,              # 上采样比例（2, 2, 2, 2）→ ×16
        upsample_initial_channel,    # 上采样前的通道数
        upsample_kernel_sizes,       # 每层上采样的卷积核大小
        gin_channels=0               # 条件编码的通道数（如情感向量）
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # 前置 1D 卷积（对输入 latent z 编码）
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, kernel_size=7, padding=3)

        # 选择残差块类型
        resblock_cls = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        # 上采样卷积层（使用转置卷积逐步放大时序维度）
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2**(i + 1)),
                        kernel_size=k,
                        stride=u,
                        padding=(k - u) // 2
                    )
                )
            )

        # 残差模块（每个上采样层后接多个残差块）
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(resblock_cls(ch, k, d))

        # 后置 1D 卷积（映射为单通道波形）
        self.conv_post = Conv1d(ch, 1, kernel_size=7, padding=3, bias=False)

        # 初始化上采样权重
        self.ups.apply(modules.init_weights)

        # 条件 embedding 接入（如情感编码）
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, kernel_size=1)

    def forward(self, x, g=None):
        """
        Args:
            x: [B, C, T]，latent 编码
            g: [B, gin_channels, T]，条件向量（可选）
        Returns:
            waveform: [B, 1, T]，生成的音频波形
        """
        x = self.conv_pre(x)  # 初始卷积

        if g is not None:
            x = x + self.cond(g)

        # 多层上采样 + 残差块堆叠
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)

            xs = None
            for j in range(self.num_kernels):
                res_out = self.resblocks[i * self.num_kernels + j](x)
                xs = res_out if xs is None else xs + res_out
            x = xs / self.num_kernels  # 多分支平均

        x = F.leaky_relu(x, modules.LRELU_SLOPE)
        x = self.conv_post(x)
        x = torch.tanh(x)  # 输出限制在 [-1, 1]

        return x

    def remove_weight_norm(self):
        """推理前移除权重归一化"""
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
