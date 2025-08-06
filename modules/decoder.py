import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from modules.resblock import ResBlock1, ResBlock2, LRELU_SLOPE
from commons import init_weights

class Generator(nn.Module):
    """
    解码器（Decoder） 用于将 latent 编码转换为音频波形
    """
    def __init__(
        self,
        initial_channel,              # 输入 latent z 的通道数（如 posterior encoder 输出）
        resblock,                     # 残差块类型，"1" 表示 ResBlock1，"2" 表示 ResBlock2
        resblock_kernel_sizes,       # 每个残差块的卷积核大小（列表）
        resblock_dilation_sizes,     # 每个残差块的 dilation（感受野）配置（嵌套列表）
        upsample_rates,              # 每层上采样倍率（如 [2, 2, 2, 2] 总共上采样16倍）
        upsample_initial_channel,    # 上采样阶段的初始通道数
        upsample_kernel_sizes,       # 每层上采样卷积核大小（对应上采样倍率）
        gin_channels=0               # 条件向量通道数（如情感 embedding）
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)     # 每层上采样后有几个并行残差块（通常为 3），即多分支平均结构数量
        self.num_upsamples = len(upsample_rates)          # 总共有多少次上采样，例如 [2,2,2,2] 是 4 层，总共放大 16 倍

        # 1. 前置卷积：将输入 latent z 编码成上采样输入
        # kernel=7 保证足够大感受野，padding=3 保持长度不变
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, kernel_size=7, padding=3)

        # 2. 选择使用的残差块结构类型
        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2

        # 3. 构建每一层上采样模块（使用转置卷积）
        self.ups = nn.ModuleList()
        # 循环创建每一层上采样模块，对应的倍率是 u，卷积核大小是 k
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                # weight_norm 提高训练稳定性
                weight_norm(
                    # ConvTranspose1d 实现反卷积上采样，通道数逐层减半
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),              # 输入通道数逐层减半
                        upsample_initial_channel // (2**(i + 1)),        # 输出通道数再减半
                        kernel_size=k,
                        stride=u,
                        padding=(k - u) // 2                             # 计算 padding 保证对齐
                    )
                )
            )

        # 4. 残差块模块，每个上采样层后接多个残差块
        self.resblocks = nn.ModuleList()
        # 为每层上采样后创建多个并行残差块
        # 不同 dilation 用于建模不同时间尺度的特征
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i + 1))    # 当前残差块的通道数
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(resblock_cls(ch, k, d))

        # 5. 后置卷积 + 输出限制：将上采样和残差块输出映射为单通道（音频波形）
        self.conv_post = nn.Conv1d(ch, 1, kernel_size=7, padding=3, bias=False)

        # 6. 初始化上采样卷积的权重
        self.ups.apply(init_weights)

        # 7. 条件输入（如情感 embedding）接入通道
        if gin_channels != 0:
            # 如果模型有条件编码（如情绪、说话人等），则添加一个线性映射通道来与主干相加
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, kernel_size=1)

    def forward(self, x, g=None):
        """
        正向生成音频波形
        Args:
            x: [B, C, T]，latent 编码（decoder 输入）
            g: [B, gin_channels, T]，条件向量（如情感编码，可选）
        Returns:
            waveform: [B, 1, T]，合成的音频波形
        """
        # 对输入 latent 先进行卷积编码
        x = self.conv_pre(x)

        # 如果提供条件向量 g（如情感 embedding），则进行加法融合（broadcast add）
        if g is not None:
            x = x + self.cond(g)

        # 上采样 + 残差块堆叠处理
        for i in range(self.num_upsamples):
            # 每一层上采样前进行激活，然后用 ConvTranspose1d 进行 2× 放大
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            # 每一层上采样后，接多个并行残差块并求平均
            # 增强建模能力，特别是对不同频段建模（高频/低频）
            xs = None
            for j in range(self.num_kernels):
                res_out = self.resblocks[i * self.num_kernels + j](x)
                xs = res_out if xs is None else xs + res_out
            x = xs / self.num_kernels

        # 最后一层卷积 + tanh 限幅
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        x = torch.tanh(x)  # 输出范围控制在 [-1, 1]，对应音频波形

        return x

    def remove_weight_norm(self):
        """
        推理阶段必须移除 weight_norm，否则对输出会造成干扰
        """
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
