import torch
import torch.nn as nn
from modules.norm import LayerNorm

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
        输入：
            x: [B, H, T] 特征向量（如：经过embedding后的文本序列）
            x_mask: [B, 1, T] 掩码张量（padding位置为0）
        返回：
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
