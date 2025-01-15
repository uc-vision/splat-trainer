import pandas as pd
from taichi_splatting import Rendering
import torch
from splat_trainer.debug.optim import print_stats, print_table
from splat_trainer.scene.point_statistics import PointStatistics
from splat_trainer.trainer.trainer import Trainer
from splat_trainer.util.containers import mean_rows


def batch_summary(trainer:Trainer):
    stats = PointStatistics.new_zeros(trainer.scene.num_points, trainer.device)
    def get_heuristics(image_idx:int, rendering:Rendering):
      stats.add_rendering(rendering)

    metrics = trainer.evaluate_backward_with(torch.arange(len(trainer.camera_table), device=trainer.device), get_heuristics)

    print_stats(trainer.all_parameters)
    print_stats(stats.to_tensordict())
    print_table(pd.DataFrame([mean_rows(metrics)]))