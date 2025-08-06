from pathlib import Path
import subprocess
import logging
from omegaconf import OmegaConf
import torch

logger = logging.getLogger(__name__)  # 规范写法，获取模块级logger
# ========= 全局缓存 =========
_hps_cache = None

class HParams:
    """支持属性访问的参数类，等价于字典"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)  # 递归支持嵌套字典
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

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

def check_git_hash(model_dir):
    """
    检查当前代码库的git commit hash是否与logs/xxx/githash一致
    """
    source_dir = Path(__file__).resolve().parent  # 获取当前文件所在目录
    git_dir = source_dir / ".git"

    if not git_dir.exists():
        logger.warning(f"{source_dir} is not a git repository, hash comparison skipped.")
        return

    # 获取当前git hash
    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = Path(model_dir) / "githash"
    if path.exists():
        saved_hash = path.read_text().strip()
        if saved_hash != cur_hash:
            logger.warning(f"git hash mismatch: {saved_hash[:8]}(saved) != {cur_hash[:8]}(current)")
    else:
        path.write_text(cur_hash)

