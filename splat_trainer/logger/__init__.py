from .logger import Logger, NullLogger
from .tensorboard import TensorboardLogger

def WandbLogger(*args, **kwargs):
    from .wandb import WandbLogger
    return WandbLogger(*args, **kwargs)


__all__ = ["WandbLogger", "TensorboardLogger", "Logger", "NullLogger"]