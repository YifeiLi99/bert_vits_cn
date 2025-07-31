import torch
import torch.nn as nn
from modules.norm import LayerNorm
import math
from torch.nn import functional as F
import commons.commons as commons

# Encoder模块：基于Transformer结构的多层堆叠编码器
# 输入：[B, H, T] 特征
# 输出：同尺寸编码结果
class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels,      # 输入/输出通道数 H，通常为 embedding dim（如 192）
        filter_channels,      # 前馈网络中间层维度，通常为 H 的 4 倍（如 768）
        n_heads,              # 多头注意力的头数（如 2、4、8）
        n_layers,             # Transformer 层数（如 6）
        kernel_size=1,        # FFN 中卷积核大小（可为 1 或 3、5）
        p_dropout=0.0,        # dropout 概率（如 0.1）
        window_size=4,        # 注意力窗口大小（如启用局部 attention，可控制范围）
        **kwargs              # 为兼容未来接口扩展
    ):
        super().__init__()

        # 参数保存
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        # 通用 Dropout 层（用于每层后）
        self.drop = nn.Dropout(p_dropout)

        # 多层模块容器初始化
        self.attn_layers = nn.ModuleList()       # 存放多层注意力模块
        self.norm_layers_1 = nn.ModuleList()     # 每层注意力后的 LayerNorm
        self.ffn_layers = nn.ModuleList()        # 前馈网络模块
        self.norm_layers_2 = nn.ModuleList()     # 每层 FFN 后的 LayerNorm

        # 构造每一层的 Attention + FFN + Norm
        for i in range(self.n_layers):
            # 多头注意力层（支持 window 限制）
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,      # Q/K/V 的维度
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size
                )
            )

            # 残差连接前的归一化层（用于 attention）
            self.norm_layers_1.append(LayerNorm(hidden_channels))

            # 前馈网络（可使用卷积或MLP实现）
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout
                )
            )

            # 残差连接前的归一化层（用于 FFN）
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        """
        x: [B, H, T] 特征向量（如：经过embedding后的文本序列）
        x_mask: [B, 1, T] 掩码张量（padding位置为0）

        return
        x: 编码后的结果 [B, H, T]
        """

        # 构造 attention 掩码：形状 [B, T, T]，用于 self-attention 中防止查看 padding
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)

        # 输入前先屏蔽掉 padding 部分（清零）
        x = x * x_mask

        # 遍历每一层 Transformer
        for i in range(self.n_layers):
            # ==== 1. 多头注意力层 ====
            y = self.attn_layers[i](x, x, attn_mask)  # Self-attention: Q=K=V=x
            y = self.drop(y)                          # dropout
            x = self.norm_layers_1[i](x + y)          # 残差 + 归一化（Pre-Norm）

            # ==== 2. 前馈网络层 ====
            y = self.ffn_layers[i](x, x_mask)         # FFN，支持mask内卷积或 MLP
            y = self.drop(y)                          # dropout
            x = self.norm_layers_2[i](x + y)          # 残差 + 归一化

        # 再次屏蔽 padding 部分，防止后续模块读取无效位置
        x = x * x_mask

        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels,              # 输入/输出特征维度（一般为 hidden_dim）
        out_channels,          # 最终输出维度（可与输入不同）
        n_heads,               # 注意力头数
        p_dropout=0.0,         # dropout 概率
        window_size=None,      # 相对位置编码的窗口大小（开启后仅限 self-attention）
        heads_share=True,      # 所有 head 是否共享位置编码
        block_length=None,     # 局部块 attention 范围（用于 block mask）
        proximal_bias=False,   # 是否加入位置接近性偏置（鼓励注意近邻）
        proximal_init=False,   # 是否将 K 初始化为 Q（模仿 Transformer TTS 初始化方式）
    ):
        super().__init__()
        assert channels % n_heads == 0  # 每个 head 的维度必须可以整除

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None  # 用于保存注意力权重，便于外部访问

        self.k_channels = channels // n_heads  # 每个 head 的通道数

        # Q, K, V 的投影卷积（1x1 Conv 相当于 Linear）
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)

        # 输出卷积：拼接后还原成 out_channels
        self.conv_o = nn.Conv1d(channels, out_channels, 1)

        self.drop = nn.Dropout(p_dropout)

        # ===== 相对位置编码（用于局部 attention）=====
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )

        # Xavier初始化
        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

        # 可选：将 conv_k 初始化为 conv_q（Proximal Attention 技巧）
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        """
        x: query 输入 [B, C, T]
        c: context，K/V 来源（通常等于 x）
        attn_mask: [B, T, T]，padding 掩码 + 可选局部 mask
        """
        q = self.conv_q(x)  # [B, C, T]
        k = self.conv_k(c)
        v = self.conv_v(c)

        # 注意力主计算逻辑（含 mask、softmax、相对位置等）
        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # 将 [B, C, T] → [B, n_heads, T, k_channels]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        # Scaled Dot-Product Attention
        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))  # [B, H, T_t, T_s]

        # === 相对位置编码（仅限 self-attention）===
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention 仅支持 self-attention"
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query / math.sqrt(self.k_channels), key_relative_embeddings)
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local

        # === 加入 Proximal Bias ===
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias 仅支持 self-attention"
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)

        # === 加入掩码 ===
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)  # 屏蔽 padding
            if self.block_length is not None:
                assert t_s == t_t
                block_mask = (
                    torch.ones_like(scores)
                    .triu(-self.block_length)
                    .tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)

        # softmax + dropout
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)

        # 输出 = attention 加权 value
        output = torch.matmul(p_attn, value)

        # === 相对位置编码：加权 value ===
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)

        # 还原回 [B, C, T] 结构
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    # === 以下是相对位置的辅助函数 ===

    def _matmul_with_relative_values(self, x, y):
        """
        x: [B, H, L, M]
        y: [1 or H, M, D]
        return: [B, H, L, D]
        """
        return torch.matmul(x, y.unsqueeze(0))

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [B, H, L, D]
        y: [1 or H, M, D]
        return: [B, H, L, M]
        """
        return torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))

    def _get_relative_embeddings(self, relative_embeddings, length):
        # 为了对齐序列长度，动态截取/填充位置编码
        max_relative_position = 2 * self.window_size + 1
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:
            padded_relative_embeddings = relative_embeddings
        return padded_relative_embeddings[:, slice_start_position:slice_end_position]

    def _relative_position_to_absolute_position(self, x):
        """
        将相对位置 logits 转换为绝对位置矩阵
        输入: [B, H, L, 2L-1] → 输出: [B, H, L, L]
        """
        batch, heads, length, _ = x.size()
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        将绝对注意力权重转换为相对位置形式
        输入: [B, H, L, L] → 输出: [B, H, L, 2L-1]
        """
        batch, heads, length, _ = x.size()
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """
        Proximal Bias：让 attention 更偏向邻近位置
        返回：[1, 1, L, L] 的偏置矩阵
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    def __init__(
        self,
        in_channels,         # 输入通道数（通常等于 hidden_dim）
        out_channels,        # 输出通道数（通常等于 in_channels）
        filter_channels,     # 中间层维度（类似 transformer FFN 的扩展层）
        kernel_size,         # 卷积核大小（通常为1或3）
        p_dropout=0.0,       # dropout 概率
        activation=None,     # 激活函数类型（支持 gelu 或 relu）
        causal=False,        # 是否使用因果卷积（适用于自回归结构）
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        # 决定 padding 模式
        if causal:
            self.padding = self._causal_padding  # 只向左 pad（自回归）
        else:
            self.padding = self._same_padding    # 左右对称 pad（非自回归）

        # 前馈网络第一层：输入 → filter_channels
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        # 前馈网络第二层：filter_channels → 输出
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)

        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        """
        输入:
            x: [B, C, T]
            x_mask: [B, 1, T]，padding 掩码，非 1 的位置为 padding
        输出:
            x: [B, C, T]，经过两层卷积和激活后的结果
        """

        # 第一层卷积：mask 掩盖无效位置 → padding → conv
        x = self.conv_1(self.padding(x * x_mask))

        # 激活函数（支持 gelu 或 relu）
        if self.activation == "gelu":
            # 近似 GELU = x * sigmoid(1.702 * x)
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)

        x = self.drop(x)  # dropout

        # 第二层卷积
        x = self.conv_2(self.padding(x * x_mask))

        # 输出仍乘上 mask，屏蔽 padding 部分
        return x * x_mask

    def _causal_padding(self, x):
        """
        因果 padding：仅向左补齐，确保当前位置不看到未来帧
        输入: [B, C, T]
        返回: pad 后的 [B, C, T + pad]
        """
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

    def _same_padding(self, x):
        """
        对称 padding：左右两边各补一半，保持输出长度不变
        """
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x


