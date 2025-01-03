from dataclasses import replace
import itertools
import json
from pathlib import Path
from time import time
from typing import List
import torch
from tqdm import tqdm

from splat_trainer.controller.controller import DisabledConfig, DisabledController
from splat_trainer.debug.optim import dump_optimizer, optimizer_state, print_params, print_stats
from splat_trainer.scene.scene import GaussianScene, PointStatistics
from splat_trainer.scripts.checkpoint import arguments, with_trainer
from splat_trainer.trainer import Trainer

from tensordict import TensorClass

import pandas as pd


def prune_with(scene:GaussianScene, heuristics:PointStatistics, n_prune:int, min_views:int=5):
  
  # prune_cost = heuristics.prune_cost  / heuristics.points_in_view
  prune_cost = heuristics.prune_cost
  prune_cost[heuristics.points_in_view < min_views] = torch.inf


  prune_idx = torch.argsort(prune_cost)[:n_prune]

  keep_mask = torch.ones(scene.num_points, dtype=torch.bool, device=scene.device)
  keep_mask[prune_idx] = False

  scene.split_and_prune(keep_mask, split_idx=torch.arange(0, dtype=torch.int64, device=scene.device))
  return heuristics[keep_mask]
  

def evaluate_with_steps(trainer:Trainer, train:bool):
  # Train for a few steps

  if train:
    trainer = trainer.clone()

    pbar = tqdm(trainer.dataset.train(shuffle=True), desc="Training")
    with torch.enable_grad():
      for camera_view in pbar:
      
        filename, camera_params, image_idx, image = trainer.load_data(camera_view)
        metrics =  trainer.training_step(filename, camera_params, image_idx, image)

        metrics = {k:f"{v:.4f}" for k, v in metrics.items() if k in ['l1', 'ssim', 'reg', 't']}
        pbar.set_postfix(**metrics)

  metrics = trainer.evaluate()
  return metrics


def main():
  parser = arguments()
  parser.add_argument("--repeats", type=int, default=1, help="Number of times to repeat each benchmark")
  parser.add_argument("--max_prune", type=float, default=0.25, help="Total proportion of points to prune")
  parser.add_argument("--prune_steps", type=int, default=5, help="Number of steps to evaluate")
  parser.add_argument("--train", action="store_true", help="Train after pruning")

  parser.add_argument("--clustered", action="store_true", help="Use clustered views")

  parser.add_argument("--min_views", type=int, default=5, help="Minimum number of views to consider pruning a point")
  parser.add_argument("--batch_size", type=int, default=60, help="Number of views to evaluate at a time")

  parser.add_argument("--output", type=str, default=None, help="Output file")
  args = parser.parse_args()


  def f(trainer:Trainer):

    
    n = trainer.scene.num_points
    trainer.controller = DisabledController()
    trainer.config = replace(trainer.config, controller=DisabledConfig())
    
    heuristics:PointStatistics = PointStatistics.new_zeros(n, device=trainer.device)

    if args.clustered:
      batch_idx = trainer.select_cluster()
    else:
      batch_idx = torch.randint(0, len(trainer.camera_table), (args.batch_size,))

    heuristics, metrics = trainer.evaluate_batch(batch_idx)


    num_seen = (heuristics.points_in_view > 0).sum().item()
    num_above_min = (heuristics.points_in_view >= args.min_views).sum().item()
    print(f"Evaluated {len(batch_idx)} views, seen {num_seen} points, {num_above_min} points above min views")

    trainer.zero_grad()

    prune_size = int(args.max_prune * n / args.prune_steps)
    levels = [100 * args.max_prune * i / args.prune_steps for i in range(args.prune_steps + 1)]
    evals = []
    for step in range(args.prune_steps + 1):

      metrics = evaluate_with_steps(trainer, args.train) #step > 0 and args.train)
      print(f"Pruned {levels[step]:.1f}% points, {metrics}")
      evals.append({**metrics, "n": trainer.scene.num_points, "level": levels[step]})

      heuristics = prune_with(trainer.scene, heuristics, prune_size, min_views=args.min_views)

    df = pd.DataFrame(evals)
    print(df.to_string(float_format=lambda x: f"{x:.3f}"))


    if args.output:
      paths = trainer.paths()
      out_file = paths.workspace / args.output
      with open(out_file, "w") as f:
        json.dump(evals, f)

      print(f"Saved to {out_file}")

  with_trainer(f, args)
