from pathlib import Path
import numpy as np
import torch
from scipy.io.wavfile import read  # 注意：这个API只接受str路径

def load_wav_to_torch(full_path):
    """
    读取wav文件并转为Tensor
    Args:
        full_path (str or Path): wav文件路径
    Returns:
        Tensor: 音频数据
        int: 采样率
    """
    full_path = Path(full_path)  # 确保是Path对象
    sampling_rate, data = read(str(full_path))  # scipy不支持Path对象，转str
    # 转换为float32 Tensor并返回
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    """
    从文本文件中读取文件路径与文本对
    Args:
        filename (str or Path): 标注文件路径
        split (str): 分隔符
    Returns:
        list: 形如[[path1, text1], [path2, text2], ...] 的列表
    """
    filename = Path(filename)  # 转为Path对象
    filepaths_and_text = []

    # 以utf-8编码打开文件
    with filename.open(encoding="utf-8") as f:
        for line in f:
            path_text = line.strip().split(split)  # 按分隔符拆分路径和文本
            filepaths_and_text.append(path_text)

    return filepaths_and_text
