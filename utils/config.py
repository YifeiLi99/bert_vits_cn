# utils/config.py

from omegaconf import OmegaConf
from pathlib import Path
import torch

# ========= 全局缓存 =========
_hps_cache = None

def get_hparams(config_path="config.yaml"):
    """
    加载或获取缓存的配置文件（OmegaConf 对象）
    可在任意模块调用，避免重复读取。
    """
    global _hps_cache
    if _hps_cache is None:
        config_path = Path(config_path).resolve()
        _hps_cache = OmegaConf.load(config_path)
    return _hps_cache


def reload_hparams(config_path="config.yaml"):
    """
    强制重新加载配置（如用于测试或切换配置）
    """
    global _hps_cache
    config_path = Path(config_path).resolve()
    _hps_cache = OmegaConf.load(config_path)
    return _hps_cache


def get_device():
    """
    获取当前设备，只支持GPU训练。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        raise RuntimeError("CUDA not available! GPU is required for training.")
