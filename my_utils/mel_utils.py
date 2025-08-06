import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

# 16-bit PCM 最大幅值，用于归一化
MAX_WAV_VALUE = 32768.0

# ==============================
# 动态范围压缩 (对数域)
# ==============================
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    对输入特征进行动态范围压缩 (对数域)
    防止小于clip_val的值导致log(0)错误

    参数：
        x: 输入Tensor
        C: 压缩因子（默认1）
        clip_val: 最小阈值（防止log(0)）
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    """
    对动态范围压缩后的特征进行反变换（指数域）

    参数：
        x: 输入Tensor
        C: 压缩时使用的压缩因子（默认1）
    """
    return torch.exp(x) / C

# ==============================
# 频谱归一化 / 反归一化
# ==============================
def spectral_normalize_torch(magnitudes):
    """
    将频谱幅度值进行对数域归一化
    """
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    """
    将归一化频谱幅度值反归一化（指数域还原）
    """
    output = dynamic_range_decompression_torch(magnitudes)
    return output

# ==============================
# Mel滤波器与Hann窗缓存字典
# 便于避免重复计算
# ==============================
mel_basis = {}
hann_window = {}

# ==============================
# STFT谱图提取
# ==============================
def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    """
    输入音频波形 y，计算STFT频谱（幅度谱）

    参数：
        y: 输入波形Tensor (B, T)
        n_fft: FFT点数
        sampling_rate: 采样率
        hop_size: 帧移（每次滑动的长度）
        win_size: 窗口长度
        center: 是否在中心对齐padding
    返回：
        spec: 幅度谱 (B, n_fft/2+1, T')
    """
    # 检查音频幅度是否超限
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device

    # Hann窗缓存，加速重复调用
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    # 反射padding，确保STFT边缘完整
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    ).squeeze(1)

    # 计算STFT
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,  # 兼容老版PyTorch
    )

    # 复数谱转幅度谱
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec

# ==============================
# 频谱 -> Mel谱图 (不含STFT)
# ==============================
def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    """
    将输入频谱 spec 转换为 Mel谱图

    参数：
        spec: 输入频谱Tensor (B, n_fft/2+1, T')
        n_fft: FFT点数
        num_mels: Mel滤波器数量
        sampling_rate: 采样率
        fmin: 最小频率
        fmax: 最大频率
    返回：
        mel_spec: Mel谱图 (B, num_mels, T')
    """
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device

    # Mel滤波器缓存，加速重复调用
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )

    # 频谱 x Mel滤波器 = Mel谱图
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)

    # 归一化
    spec = spectral_normalize_torch(spec)
    return spec

# ==============================
# 波形 -> Mel谱图 (完整流程)
# ==============================
def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    """
    一步到位将波形y转换为Mel谱图

    参数：
        y: 输入波形Tensor (B, T)
        n_fft: FFT点数
        num_mels: Mel滤波器数量
        sampling_rate: 采样率
        hop_size: 帧移
        win_size: 窗口长度
        fmin: 最小频率
        fmax: 最大频率
        center: 是否中心对齐
    返回：
        mel_spec: Mel谱图 (B, num_mels, T')
    """
    # 幅度检查
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_size) + "_" + dtype_device

    # Mel滤波器缓存
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=y.dtype, device=y.device
        )

    # Hann窗缓存
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    # 反射padding
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    ).squeeze(1)

    # STFT计算
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    # 复数谱转幅度谱
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    # 频谱 -> Mel谱图
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)

    # 归一化
    spec = spectral_normalize_torch(spec)

    return spec
