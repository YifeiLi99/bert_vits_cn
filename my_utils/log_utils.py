from pathlib import Path
import logging

def get_logger(model_dir, filename="train.log"):
    """
    初始化一个写文件+控制台的logger
    Args:
        model_dir (str or Path): 日志文件保存的目录
        filename (str): 日志文件名 (默认 train.log)
    Returns:
        logger: logging.Logger 对象
    """
    model_dir = Path(model_dir)
    logger = logging.getLogger(model_dir.name)  # 用目录名作为logger name
    logger.setLevel(logging.DEBUG)

    # 只添加一次Handler，避免重复绑定
    if not logger.handlers:
        model_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在

        formatter = logging.Formatter(
            "%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s"
        )

        # 文件日志Handler
        file_handler = logging.FileHandler(str(model_dir / filename))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 控制台Handler（可选）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
    """
    将训练指标写入 TensorBoard
    Args:
        writer: TensorBoard SummaryWriter
        global_step: 当前迭代步数
        scalars: 标量（如loss）字典
        histograms: 直方图数据
        images: 图片数据
        audios: 音频数据
        audio_sampling_rate: 音频采样率
    """
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)
