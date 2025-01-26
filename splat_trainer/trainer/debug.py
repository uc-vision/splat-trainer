import pandas as pd
from taichi_splatting import Rendering
import torch
from splat_trainer.debug.optim import print_stats, print_table
from splat_trainer.logger.logger import HistoryLogger
from splat_trainer.scene.point_statistics import PointStatistics
from splat_trainer.trainer.trainer import Trainer
from splat_trainer.util.containers import mean_rows


from contextlib import contextmanager
from splat_trainer.logger import Logger

@contextmanager
def set_logger(trainer: Trainer, logger: Logger):
    """Temporarily sets a logger on a trainer and restores the original afterwards.
    
    Args:
        trainer: The trainer to modify
        logger: The logger to temporarily use
    """
    original_logger = trainer.logger
    trainer.logger = logger
    try:
        yield
    finally:
        trainer.logger = original_logger


def batch_summary(trainer:Trainer):
    stats = PointStatistics.new_zeros(trainer.scene.num_points, trainer.device)
    def get_heuristics(image_idx:int, rendering:Rendering):
      stats.add_rendering(rendering)

    logger = HistoryLogger()
    with set_logger(trainer, logger):
      trainer.evaluate_backward_with(torch.arange(len(trainer.camera_table), device=trainer.device), get_heuristics)

    metrics = logger.flatten()

    print_stats(trainer.all_parameters)
    print_stats(stats.to_tensordict())
    print_table(pd.DataFrame([mean_rows(metrics)]))