import torch
from torch import nn
import math
import attentions
import commons as commons
from norm import WN

class TextEncoder(nn.Module):
    """
    把离散的音素序列 + 可选的 BERT embedding，转化为连续的特征向量（μ, logσ），供后续 flow 和 posterior encoder 使用
    """
    def __init__(
            self,
            n_vocab,  # 词表大小（音素数量）
            out_channels,  # 输出通道数，等于 VAE 的 z 维度（例如 192）
            hidden_channels,  # Transformer 中间隐层维度（通常等于 embedding dim）
            filter_channels,  # 前馈层的维度（通常比 hidden 大，例如 4x）
            n_heads,  # 多头注意力的头数（通常为 2~8）
            n_layers,  # Transformer 层数（如 6）
            kernel_size,  # FFN 中卷积核大小（如 5）
            p_dropout  # dropout 概率（如 0.1）
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # 嵌入层：将离散 token（音素 ID）转为向量
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        # 将嵌入矩阵 self.emb.weight 按照均值为 0，标准差为 1 / sqrt(hidden_channels) 的正态分布进行初始化
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        # BERT 情感向量线性映射层（256 → hidden_channels）
        self.emb_bert = nn.Linear(256, hidden_channels)

        # Transformer 编码器：包含多层注意力机制 + 前馈网络
        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )

        # 投影层：从上下文特征 → 生成 μ 与 logσ
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, kernel_size=1)

    def forward(self, x, x_lengths, bert):
        """
        x: [B, T]，音素索引序列（int64）
        x_lengths: 有效长度，排除 padding 部分
        bert: [B, T, 256]，BERT 情感 embedding（可选）

        return:
        x: 编码后的上下文表示
        m: μ（变分编码器均值）
        logs: logσ（变分编码器对数方差）
        x_mask: 掩码张量 [B, 1, T]
        """

        # 查嵌入表，并缩放，保持 variance 稳定
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [B, T, H]

        # 叠加 BERT 向量（如果有），提供语义上下文
        if bert is not None:
            b = self.emb_bert(bert)  # [B, T, H]
            # 与 phoneme embedding 相加，增强音素
            x = x + b

        # 转置维度：用于后续 conv/attention
        x = torch.transpose(x, 1, -1)  # [B, H, T]

        # 构建掩码：用于 attention、conv 掩蔽 padding 部分
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        # Transformer 编码
        x = self.encoder(x * x_mask, x_mask)  # [B, H, T]

        # 输出 μ 和 logσ
        stats = self.proj(x) * x_mask         # [B, 2H, T]
        # 切成两半：前一半是 μ，后一半是 logσ
        m, logs = torch.split(stats, self.out_channels, dim=1)  # 各 [B, H, T]

        return x, m, logs, x_mask


class PosteriorEncoder(nn.Module):
    """
    变分后验编码器
    将真实 mel 频谱 → 编码成 (μ, logσ) → 再通过 reparameterization trick 得到 latent 变量 z
    """
    def __init__(
        self,
        in_channels,        # 输入通道数（mel 频谱维度）
        out_channels,       # 输出通道数（latent 变量维度）
        hidden_channels,    # WN 网络的中间通道数
        kernel_size,        # WN 卷积核大小
        dilation_rate,      # WN 膨胀率
        n_layers,           # WN 卷积层数
        gin_channels=0      # 条件向量（如情感 embedding）通道数，若无则为 0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # 初始 1x1 Conv：维度对齐   把原始输入从 mel 维度映射到 hidden 维度，方便接后续残差卷积网络
        self.pre = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        # Wavenet 样式的残差卷积网络（来自 Glow/VITS 中 WN）
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels
        )

        # 输出 μ 和 logσ（双倍输出通道）
        # 输出通道 = 2 × out_channels，对应变分分布的两个参数：μ：均值  logσ：对数标准差
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, kernel_size=1)

    def forward(self, x, x_lengths, g=None):
        """
        Args:
            x: [B, mel_dim, T]，GT mel 频谱
            x_lengths: [B]，每个样本的有效长度（用于 mask）
            g: 条件向量 [B, gin_channels, T]，如情感 embedding
        Returns:
            z: 采样得到的 latent 表征
            m: μ
            logs: logσ
            x_mask: [B, 1, T] 有效长度 mask
        """
        # 生成一个形状 [B, 1, T] 的 mask，标出哪些是有效部分
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.pre(x) * x_mask  # 通道投影
        x = self.enc(x, x_mask, g=g)  # 经过 WaveNet 残差网络进行建模

        # stats：输出 μ 和 logσ 的拼接结果
        stats = self.proj(x) * x_mask  # [B, 2*out_channels, T]
        # split：拆成两部分
        m, logs = torch.split(stats, self.out_channels, dim=1)

        # 采样 z ∼ N(μ, σ²)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask

        # 返回这四项，用于后续 loss 计算或 flow 解码
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        """
        VITS 训练时使用了 weight_norm 加速收敛，但推理时需要移除
        推理前做网络瘦身
        """
        self.enc.remove_weight_norm()
