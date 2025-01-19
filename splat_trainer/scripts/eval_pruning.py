from dataclasses import replace
import json
from taichi_splatting import Gaussians3D
import torch
from tqdm import tqdm

from splat_trainer.controller import DisabledConfig
from splat_trainer.controller.threshold_controller import take_n
from splat_trainer.scene.point_statistics import PointStatistics
from splat_trainer.scene.scene import GaussianScene
from splat_trainer.scripts.checkpoint import arguments, with_trainer
from splat_trainer.trainer import Trainer

import pandas as pd
from splat_trainer.util.containers import mean_rows


def prune_with(scene:GaussianScene, heuristics:PointStatistics, n_prune:int, min_views:int=5
               ) -> torch.Tensor:
  
  prune_cost = heuristics.prune_cost  / (heuristics.points_in_view + min_views).sqrt()
  # prune_cost = heuristics.prune_cost
  # prune_cost[heuristics.points_in_view < min_views] = torch.inf

  prune_mask = take_n(prune_cost, n_prune, descending=False)

  scene.split_and_prune(~prune_mask, split_idx=torch.arange(0, dtype=torch.int64, device=scene.device))
  return prune_mask
  

def evaluate_with_training(trainer:Trainer, train:bool) -> dict:
  # Train for a few steps

  if train is True:
    trainer = trainer.clone()

    pbar = tqdm(trainer.dataset.train(shuffle=True), desc="Training")
    with torch.enable_grad():
      for camera_view in pbar:
        camera_view = trainer.load_data(camera_view)   
           
        trainer.training_step([camera_view])
        trainer.update_progress()


  train = trainer.dataset.train(shuffle=False)
  metrics = [eval.metrics for eval in tqdm(trainer.evaluations(train), desc="Evaluating", total=len(train))]


def show_pruning(trainer:Trainer, cloud:Gaussians3D, prune_mask:torch.Tensor):
    from splat_viewer.viewer import show_workspace
    from splat_viewer.gaussians import Workspace, Gaussians

    paths = trainer.paths()
    workspace = Workspace.load(paths.workspace)
    gaussian_labelled = Gaussians.from_gaussians3d(cloud).with_labels(prune_mask.unsqueeze(1).int())
  
    show_workspace(workspace, gaussians=gaussian_labelled)

def main():
  parser = arguments()
  parser.add_argument("--repeats", type=int, default=1, help="Number of times to repeat each benchmark")
  parser.add_argument("--max_prune", type=float, default=0.25, help="Total proportion of points to prune")
  parser.add_argument("--prune_steps", type=int, default=5, help="Number of steps to evaluate")
  parser.add_argument("--train", action="store_true", help="Train after pruning")
  parser.add_argument("--opacity", action="store_true", help="Use opacity for pruning")

  parser.add_argument("--show", action="store_true", help="Show results in a viewer")

  parser.add_argument("--clustered", action="store_true", help="Use clustered views")

  parser.add_argument("--min_views", type=int, default=5, help="Minimum number of views to consider pruning a point")
  parser.add_argument("--view_proportion", type=float, default=1.0, help="Proportion of views to evaluate at a time")

  parser.add_argument("--output", type=str, default=None, help="Output file")
  args = parser.parse_args()


  def f(trainer:Trainer):

    torch.random.manual_seed(0)
    n = trainer.scene.num_points

    cloud = trainer.load_cloud()
    trainer = trainer.replace(controller=DisabledConfig())

    n_views = len(trainer.camera_table)
    batch_idx = torch.randint(0, n_views, (int(n_views * args.view_proportion),))

    # Compute pruning heuristics in the backward pass
    heuristics:PointStatistics = PointStatistics.new_zeros(n, device=trainer.device)
    trainer.evaluate_backward_with(trainer.load_batch(batch_idx), 
                  lambda _, rendering: heuristics.add_rendering(rendering))
    
    if args.opacity:
      gaussians = trainer.scene.gaussians
      heuristics.prune_cost[:] = (1 - gaussians.alpha.squeeze(1))

    num_seen = (heuristics.points_in_view > 0).sum().item()
    num_above_min = (heuristics.points_in_view >= args.min_views).sum().item()
    print(f"Evaluated {len(batch_idx)} views, seen {num_seen} points, {num_above_min} points above min views")

    trainer.zero_grad()

    prune_size = int(args.max_prune * n / args.prune_steps)
    levels = [100 * args.max_prune * i / args.prune_steps for i in range(args.prune_steps + 1)]
    evals = []

    prune_mask = None

    for step in range(args.prune_steps + 1):
        metrics = evaluate_with_training(trainer, args.train)
        metrics["prune_max"] = heuristics.prune_cost.min().item()

        metrics_str = [f"{k}={v:.4g}" for k, v in metrics.items()]

        print(f"Pruned {levels[step]:.1f}% points, {', '.join(metrics_str)}")
        evals.append({**metrics, "n": trainer.scene.num_points, "level": levels[step]})

        if args.show and prune_mask is not None:
          show_pruning(trainer, cloud, prune_mask)
          cloud = cloud[~prune_mask]

        if step < args.prune_steps:
          prune_mask = prune_with(trainer.scene, heuristics, prune_size, min_views=args.min_views)
          heuristics = heuristics[~prune_mask]



    df = pd.DataFrame(evals)
    print(df.to_string(float_format=lambda x: f"{x:.3f}"))


    if args.output:
      paths = trainer.paths()
      out_file = paths.workspace / args.output
      with open(out_file, "w") as f:
        json.dump(evals, f)

      print(f"Saved to {out_file}")

  with_trainer(f, args)
