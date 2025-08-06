from pathlib import Path
import importlib

def load_class(full_class_name):
    """
    根据完整类名字符串动态加载类对象
    Args:
        full_class_name (str): 形如 'module.submodule.ClassName'
    Returns:
        cls: 加载到的类对象
    """
    # 先在当前模块的全局变量中查找
    if full_class_name in globals():
        return globals()[full_class_name]

    # 若字符串中含有'.'，则尝试动态导入
    if "." in full_class_name:
        module_name, cls_name = full_class_name.rsplit('.', 1)  # 以最后一个.拆分模块与类名
        mod = importlib.import_module(module_name)              # 动态导入模块
        cls = getattr(mod, cls_name)                            # 从模块中获取类对象
        return cls

    # 找不到则返回 None
    return None


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    """
    查找目录下最新的checkpoint文件（按数字排序）
    Args:
        dir_path (str or Path): checkpoint目录
        regex (str): 文件匹配模式（默认G_*.pth）
    Returns:
        str: 最新的checkpoint文件路径（str格式）
    """
    dir_path = Path(dir_path)  # 确保是Path对象
    matched_files = list(dir_path.glob(regex))  # 使用Pathlib自带的glob方法

    if not matched_files:
        print(f"No checkpoint files found in {dir_path}")
        return None

    # 根据文件名中的数字部分排序（如G_100.pth -> 100）
    matched_files.sort(key=lambda f: int("".join(filter(str.isdigit, f.stem))))
    latest_ckpt = matched_files[-1]
    print(f"Latest checkpoint: {latest_ckpt}")
    return str(latest_ckpt)  # 返回字符串路径（以兼容torch.load等API）
