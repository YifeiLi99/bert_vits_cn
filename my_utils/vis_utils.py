import matplotlib
import matplotlib.pylab as plt
import numpy as np
import logging

# 全局配置matplotlib，只初始化一次
matplotlib.use("Agg")
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

def plot_spectrogram_to_numpy(spectrogram):
    """
    将Mel谱图绘制为numpy格式的RGB图片
    Args:
        spectrogram (ndarray): [Channels, Frames] 的频谱图
    Returns:
        ndarray: 绘制后的RGB图像 (H, W, 3)
    """
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")  # 绘制频谱图
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    # 将绘图内容渲染到canvas并转为numpy array
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # 从buffer读取图像数据
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # 重塑为 (H, W, 3)
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    """
    将Attention对齐矩阵绘制为numpy格式的RGB图片
    Args:
        alignment (ndarray): [Encoder timestep, Decoder timestep] 的对齐矩阵
        info (str): 附加的标签信息
    Returns:
        ndarray: 绘制后的RGB图像 (H, W, 3)
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect="auto", origin="lower", interpolation="none")  # 绘制对齐图
    fig.colorbar(im, ax=ax)

    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info  # 可选的额外信息

    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data
