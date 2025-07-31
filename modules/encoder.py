class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab,          # 词表大小（音素表）
        out_channels,     # 输出通道数，通常等于 posterior encoder 输出的维度
        hidden_channels,  # transformer 的隐层维度
        filter_channels,  # transformer 前馈层的维度
        n_heads,          # 多头注意力数量
        n_layers,         # transformer 层数
        kernel_size,      # 前馈卷积核大小
        p_dropout         # dropout 概率
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # 音素嵌入层（用于离散 token）
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        # BERT 情感向量线性映射层（256 → hidden_channels）
        self.emb_bert = nn.Linear(256, hidden_channels)

        # Transformer 编码器：包含多层注意力机制 + 前馈网络
        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )

        # 输出投影：预测 μ 和 logσ
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, kernel_size=1)

    def forward(self, x, x_lengths, bert):
        """
        Args:
            x: [B, T]，音素索引序列（int64）
            x_lengths: [B]，每个样本的有效长度
            bert: [B, T, 256]，BERT 情感 embedding（可选）
        Returns:
            x: 编码后的上下文表示
            m: μ（变分编码器均值）
            logs: logσ（变分编码器对数方差）
            x_mask: 掩码张量 [B, 1, T]
        """

        # 嵌入 → 乘以 √d 作为缩放
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [B, T, H]

        # 若启用 BERT，叠加其特征向量
        if bert is not None:
            b = self.emb_bert(bert)  # [B, T, H]
            x = x + b

        # 转置维度：用于后续 conv/attention
        x = torch.transpose(x, 1, -1)  # [B, H, T]

        # 构建掩码：用于 attention、conv 掩蔽 padding 部分
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        # 编码器处理
        x = self.encoder(x * x_mask, x_mask)  # [B, H, T]

        # 输出 μ 和 logσ
        stats = self.proj(x) * x_mask         # [B, 2H, T]
        m, logs = torch.split(stats, self.out_channels, dim=1)  # 各 [B, H, T]

        return x, m, logs, x_mask


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,        # 输入通道数（mel 频谱维度）
        out_channels,       # 输出通道数（latent 变量维度）
        hidden_channels,    # 中间通道数
        kernel_size,        # WN 卷积核大小
        dilation_rate,      # WN 膨胀率
        n_layers,           # WN 层数
        gin_channels=0      # 条件向量（如情感 embedding）通道数
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # 1x1 conv，初步投影到隐藏空间
        self.pre = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        # Wavenet 样式的残差卷积网络（来自 Glow/VITS 中 WN）
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels
        )

        # 输出 μ 和 logσ（双倍输出通道）
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, kernel_size=1)

    def forward(self, x, x_lengths, g=None):
        """
        Args:
            x: [B, mel_dim, T]，GT mel 频谱
            x_lengths: [B]，每个样本的有效长度
            g: 条件向量 [B, gin_channels, T]，如情感 embedding
        Returns:
            z: 采样得到的 latent 表征
            m: μ
            logs: logσ
            x_mask: [B, 1, T] 有效长度 mask
        """
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.pre(x) * x_mask  # 初步卷积
        x = self.enc(x, x_mask, g=g)  # Wavenet 编码

        stats = self.proj(x) * x_mask  # [B, 2*out_channels, T]
        m, logs = torch.split(stats, self.out_channels, dim=1)

        # 采样 z ∼ N(μ, σ²)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask

        return z, m, logs, x_mask

    def remove_weight_norm(self):
        """推理前移除权重归一化"""
        self.enc.remove_weight_norm()
