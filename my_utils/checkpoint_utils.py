from pathlib import Path
import torch
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)  # 获取当前模块的logger实例

def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint_path = Path(checkpoint_path)  # Pathlib路径对象
    assert checkpoint_path.is_file(), f"{checkpoint_path} not found."

    # torch.load 只接受str路径，所以这里转str
    checkpoint_dict = torch.load(str(checkpoint_path), map_location="cpu")

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]

    # 如果传入了优化器，恢复优化器状态
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])

    saved_state_dict = checkpoint_dict["model"]
    # 判断模型是否是DataParallel封装（多GPU）
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    new_state_dict = {}
    # 只更新checkpoint中有的参数，其余保留原模型的
    for k, v in state_dict.items():
        if k in saved_state_dict:
            new_state_dict[k] = saved_state_dict[k]
        else:
            logger.info(f"{k} is not in the checkpoint, using existing value.")
            new_state_dict[k] = v

    # 将新state_dict加载回模型
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)

    logger.info(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration})")
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    logger.info(f"Saving model and optimizer state at iteration {iteration} to {checkpoint_path}")

    # 获取模型参数
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    # torch.save 也只能接受str路径
    torch.save({
        "model": state_dict,
        "iteration": iteration,
        "optimizer": optimizer.state_dict(),
        "learning_rate": learning_rate,
    }, str(checkpoint_path))


def load_model(checkpoint_path, model):
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.is_file(), f"{checkpoint_path} not found."

    checkpoint_dict = torch.load(str(checkpoint_path), map_location="cpu")
    saved_state_dict = checkpoint_dict["model"]

    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        if k in saved_state_dict:
            new_state_dict[k] = saved_state_dict[k]
        else:
            logger.info(f"{k} is not in the checkpoint, using existing value.")
            new_state_dict[k] = v

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)

    return model


def save_model(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({'model': state_dict}, str(checkpoint_path))


def load_teacher(checkpoint_path, model):
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.is_file(), f"{checkpoint_path} not found."

    checkpoint_dict = torch.load(str(checkpoint_path), map_location="cpu")
    saved_state_dict = checkpoint_dict['model']

    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    new_state_dict = {}
    # 只加载 enc_q 和 flow 的参数，其他参数保留不变
    for k, v in state_dict.items():
        if k.startswith('enc_q') or k.startswith('flow'):
            new_state_dict[k] = saved_state_dict.get(k, v)  # 若ckpt里没有，也保留原值
        else:
            new_state_dict[k] = v

    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)

    return model
