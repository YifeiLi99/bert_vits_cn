import torch
import torch.nn as nn
import math
from modules.encoder import TextEncoder, PosteriorEncoder
from modules.decoder import Generator
from modules.predictor import DurationPredictor
from modules.flow.coupling import ResidualCouplingBlock
from commons.commons import rand_slice_segments, sequence_mask, generate_path

class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training （VITS主模块）
    包含：
    - 文本编码器 TextEncoder
    - 后验编码器 PosteriorEncoder
    - Flow 模块 ResidualCouplingBlock
    - 时长预测器 DurationPredictor
    - 声码器 Generator
    """

    def __init__(
        self,
        n_vocab,                     # 词表大小（字符/音素总数）
        spec_channels,               # Mel谱维度，一般为80
        segment_size,                # 训练时从波形中截取的长度（用于提升训练稳定性）
        inter_channels,              # 编码器与生成器之间的通道数
        hidden_channels,             # 模型内部隐藏通道数
        filter_channels,            # 前馈网络通道（用于注意力模块）
        n_heads,                    # 多头注意力的头数
        n_layers,                   # transformer层数
        kernel_size,                # transformer卷积核大小
        p_dropout,                  # dropout概率
        resblock,                   # 残差块类型 "1" or "2"
        resblock_kernel_sizes,     # 残差块卷积核大小列表
        resblock_dilation_sizes,   # 残差块dilation配置
        upsample_rates,            # 上采样比例列表（乘起来是总倍数）
        upsample_initial_channel,  # 上采样前的通道数
        upsample_kernel_sizes,     # 上采样卷积核大小列表
        n_speakers=0,              # 说话人数量（多说话人建模）
        gin_channels=0,            # 条件编码的通道数（如情感/说话人）
        use_sdp=False,             # 是否使用可变形 self-dur predictor（保留字段）
        **kwargs
    ):
        super().__init__()

        # 保存初始化参数
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        # 文本编码器（text → latent表示）
        self.enc_p = TextEncoder(
            n_vocab, inter_channels, hidden_channels,
            filter_channels, n_heads, n_layers,
            kernel_size, p_dropout
        )

        # 声码器（z → waveform）
        self.dec = Generator(
            inter_channels, resblock, resblock_kernel_sizes,
            resblock_dilation_sizes, upsample_rates,
            upsample_initial_channel, upsample_kernel_sizes,
            gin_channels=gin_channels
        )

        # 后验编码器（mel → z）
        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels,
            5, 1, 16, gin_channels=gin_channels
        )

        # flow网络（用于latent对齐）
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )

        # 时长预测器
        self.dp = DurationPredictor(
            hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
        )

        # 说话人embedding
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(self, x, x_lengths, bert, y, y_lengths, sid=None):
        """
        前向传播（训练阶段）

        Args:
            x: 文本token
            x_lengths: 文本长度
            bert: BERT embedding 输入（可选）
            y: mel谱（Ground Truth）
            y_lengths: mel谱长度
            sid: 说话人ID（可选）

        Returns:
            o: 生成波形
            l_length: duration loss
            attn: 对齐矩阵
            ids_slice: 截取的片段索引
            x_mask, y_mask: mask
            latent项们
        """

        # 文本编码器（prior）
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, bert)

        # 条件信息（如说话人向量）
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)
        else:
            g = None

        # 后验编码器（posterior）
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)

        # z 通过 flow 变换成 z_p
        z_p = self.flow(z, y_mask, g=g)

        # -------- 计算对齐矩阵（monotonic alignment） --------
        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)  # 方差倒数
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        # -------- 时长损失 --------
        w = attn.sum(2)
        logw_ = torch.log(w + 1e-6) * x_mask
        logw = self.dp(x, x_mask, g=g)
        l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)

        # -------- 对齐后展平（取出每个音素对应的 z）--------
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        # -------- 切片训练片段，送入 decoder --------
        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)

        # -------- 推理路径 z_r --------
        z_r = m_p + torch.randn_like(m_p) * torch.exp(logs_p)
        z_r = self.flow(z_r, y_mask, g=g, reverse=True)

        return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, z_r, m_p, logs_p, m_q, logs_q)

    def infer(self, x, x_lengths, bert, sid=None, noise_scale=1, length_scale=1, max_len=None):
        """
        推理过程

        Args:
            x: 文本输入
            x_lengths: 文本长度
            bert: BERT特征（可选）
            sid: 说话人ID（可选）
            noise_scale: 控制采样时的随机性（越大越多样化）
            length_scale: 控制语速（越大语速越慢）
            max_len: 限制最大生成长度

        Returns:
            o: 合成波形
            attn: 对齐矩阵
            y_mask: mask
            latent组
        """
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, bert)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)
        else:
            g = None

        # 预测 duration → 得到 attention 路径
        logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)

        # 根据 attention 对齐，获取 m_p, logs_p（逐音素）
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        # 根据采样获取 z → 反向 flow → decoder
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)

        return o, attn, y_mask, (z, z_p, m_p, logs_p)
