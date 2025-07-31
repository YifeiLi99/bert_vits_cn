import torch
import math
from torch.nn import functional as F

def init_weights(m, mean=0.0, std=0.01):
    """
    初始化权重函数：用于 Conv 层的权重初始化
    Args:
        m: 模型中的一个 module
        mean: 均值（默认0.0）
        std: 标准差（默认0.01）
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)  # 正态分布初始化


def get_padding(kernel_size, dilation=1):
    """
    自动计算 padding 数值，保持卷积后输出长度不变
    Args:
        kernel_size: 卷积核大小
        dilation: 扩张系数（默认为1）
    Returns:
        需要添加的 padding 数值（左右总和）
    """
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    """
    将二维 padding 形状 [[a,b],[c,d],...] 转换为 PyTorch F.pad 支持的一维列表格式
    Args:
        pad_shape: 形如 [[pad_left, pad_right], ...]
    Returns:
        展平后的 padding 参数列表，顺序为最后一维优先，例如 [c,d,a,b]
    """
    l = pad_shape[::-1]  # 逆序，使最后一维先展开
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def intersperse(lst, item):
    """
    在列表中插入间隔项，例如音素之间插入停顿符号
    Args:
        lst: 原始列表，如 [a, b, c]
        item: 插入项，如 <blank>
    Returns:
        插入后的新列表，如 [<blank>, a, <blank>, b, <blank>, c, <blank>]
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def kl_divergence(m_p, logs_p, m_q, logs_q):
    """
    计算 KL 散度 KL(P || Q)，用于变分编码器损失
    Args:
        m_p, logs_p: 分布 P 的均值和 log 方差
        m_q, logs_q: 分布 Q 的均值和 log 方差
    Returns:
        KL 散度（逐点张量）
    """
    kl = (logs_q - logs_p) - 0.5
    kl += 0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    return kl


def rand_gumbel(shape):
    """
    从 Gumbel 分布中采样（避免 log(0) 的数值不稳定）
    Args:
        shape: 输出张量形状
    Returns:
        Gumbel 噪声张量
    """
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001  # 避免取到 log(0)
    return -torch.log(-torch.log(uniform_samples))  # Gumbel 分布采样公式


def rand_gumbel_like(x):
    """
    生成与张量 x 同形状的 Gumbel 噪声
    """
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g


def slice_segments(x, ids_str, segment_size=4):
    """
    按照指定起点提取固定长度 segment（用于音频片段切分）
    Args:
        x: [B, C, T] 的输入序列
        ids_str: 起始位置列表 [B]
        segment_size: 片段长度
    Returns:
        ret: [B, C, segment_size]，提取的片段
    """
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    """
    从序列中**随机切片**固定长度 segment，用于训练中做局部训练片段
    Args:
        x: [B, C, T]，输入序列
        x_lengths: 每个样本的有效长度（可选）
        segment_size: 切片长度
    Returns:
        ret: [B, C, segment_size]，随机切出的片段
        ids_str: [B]，每个样本的起始位置
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1  # 起点范围上限
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    生成 1D 的位置编码（Position Encoding），用于 transformer 输入加 positional bias
    使用 sin + cos 编码，类似于 Transformer 原始论文的做法
    Args:
        length: 序列长度 T
        channels: 通道数 H（必须为偶数）
        min_timescale: 最小时间尺度
        max_timescale: 最大时间尺度
    Returns:
        signal: [1, channels, length] 的位置编码张量
    """
    position = torch.arange(length, dtype=torch.float)                      # [T]
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1)
    inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment)
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)      # [num_timescales, T]
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)  # [H, T]
    signal = F.pad(signal, [0, 0, 0, channels % 2])                         # 若通道数为奇数，补1维
    signal = signal.view(1, channels, length)                              # [1, H, T]
    return signal

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """
    为输入添加位置编码（Add型）
    输入 x: [B, C, T]，添加位置编码后仍为 [B, C, T]
    signal: torch.Tensor
    """
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    """
    为输入添加位置编码（Concat型）
    输入 x: [B, C, T]，输出 [B, 2C, T]（若 axis=1）
    signal: torch.Tensor
    """
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length):
    """
    生成下三角 mask（用于 causal self-attention）
    返回: [1, 1, T, T]
    """
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    """
    高效计算 gated activation：tanh(a+b) * sigmoid(a+b)
    用于 WaveNet / HiFi-GAN 中的门控结构
    input_a, input_b: [B, 2C, T]
    n_channels: [C]
    返回: [B, C, T]
    """
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def convert_pad_shape(pad_shape):
    """
    将二维 padding 形状转换为 PyTorch F.pad 所需格式
    例：[[0,0], [0,0], [2,1]] → [1,2,0,0,0,0]
    """
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def shift_1d(x):
    """
    时间维上向右平移一格，填充 0（如：decoder teacher forcing）
    输入: [B, C, T]
    输出: [B, C, T]，x[:, :, 1:] 向右移一位
    """
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(length, max_length=None):
    """
    为变长序列生成 padding mask（如：用于 attention）
    Args:
        length: [B] 每个样本的有效长度
        max_length: 可选，最大长度
    Returns:
        mask: [B, T]，有效位置为 True
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    """
    根据对齐 duration 推出 attention path（离散1-1对齐）
    用于 duration-based attention，例如 Glow-TTS, VITS
    Args:
        duration: [B, 1, T_x]，每个 token 对应的帧数
        mask: [B, 1, T_y, T_x]，有效区域掩码
    Returns:
        path: [B, 1, T_y, T_x]，hard-alignment 矩阵
    """
    device = duration.device
    b, _, t_y, t_x = mask.shape

    # 每个 token 的累计结束位置
    cum_duration = torch.cumsum(duration, -1)                      # [B, 1, T_x]
    cum_duration_flat = cum_duration.view(b * t_x)

    # 为每个 token 生成 attention 区域
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)    # [B*T_x, T_y]
    path = path.view(b, t_x, t_y)                                  # [B, T_x, T_y]

    # 相邻差分，得到每个 token 的起止帧
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]

    # 转置为 [B, 1, T_y, T_x]，并乘掩码
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def clip_grad_value_(parameters, clip_value, norm_type=2):
    """
    梯度裁剪函数（按 value 而非 norm 裁剪）
    Args:
        parameters: 参数列表
        clip_value: 截断区间 [-clip_value, +clip_value]
        norm_type: 用于 norm 统计的范数类型（不影响截断本身）
    Returns:
        total_norm: 所有梯度的 norm（未裁剪前）
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)              # 获取当前参数的梯度 norm
        total_norm += param_norm.item() ** norm_type          # 累加平方和
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)  # 裁剪值
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm
