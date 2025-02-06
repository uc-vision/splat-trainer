import argparse
from pathlib import Path
import hydra
import numpy as np
from omegaconf import OmegaConf
from taichi_splatting import TaichiQueue
from termcolor import colored
import termcolor
from splat_trainer.logger.logger import Logger
from splat_trainer import config

from taichi_splatting import TaichiQueue

import torch
import torch._logging

import os

from splat_trainer.viewer.viewer import Viewer
import logging




def cfg_from_args():
  parser = argparse.ArgumentParser()

  # General arguments
  parser.add_argument("overrides", nargs="*", help="hydra overrides var=value")
  parser.add_argument("--debug", action="store_true", help="Enable taichi debugging")
  parser.add_argument("--show_config", action="store_true", help="Show config")

  # Dataset group
  dataset_group = parser.add_argument_group("Dataset")
  dataset_group.add_argument("--scan", type=str, default=None, help="Scan json scene file to load")
  dataset_group.add_argument("--colmap", type=str, default=None, help="Colmap scene to load")
  dataset_group.add_argument("--image_scale", type=float, default=None, help="Image scale")
  dataset_group.add_argument("--resize_longest", type=int, default=None, help="Resize longest side")
  dataset_group.add_argument("--far", type=float, default=None, help="Set far plane")
  dataset_group.add_argument("--near", type=float, default=None, help="Set near plane")

  # Training group
  training_group = parser.add_argument_group("Training")
  training_group.add_argument("--target_points", type=int, default=None, help="Target point count")
  training_group.add_argument("--total_steps", type=int, default=None, help="Number of total training steps")
  training_group.add_argument("--training_scale", type=float, default=1.0, help="Scale the number of steps by a constant factor")
  training_group.add_argument("--eval_steps", type=int, default=None, help="Number of steps between evaluations")
  
  training_group.add_argument("--initial_points", type=int, default=None, help="Start with N points in the point cloud (add random points to make up the difference)")
  training_group.add_argument("--limit_points", type=int, default=None, help="Limit the number of points from the dataset to N")
  
  
  training_group.add_argument("--random_points", type=int, default=None, help="Use N random points only")

  training_group.add_argument("--tcnn", action="store_true", help="Use tcnn scene")
  training_group.add_argument("--bilateral", action="store_true", help="Use bilateral color correction")

  training_group.add_argument("--no_controller", action="store_true", help="Disable controller prune/split")
  training_group.add_argument("--mcmc", action="store_true", help="Use MCMC controller")

  training_group.add_argument("--vis", action="store_true", help="Enable web viewer")


  # Rendering group
  rendering_group = parser.add_argument_group("Rendering")
  rendering_group.add_argument("--antialias", action="store_true", help="Use antialiasing")
  rendering_group.add_argument("--no_autotune", action="store_true", help="Disable autotuning for easier development")


  # Output group
  output_group = parser.add_argument_group("Output")
  output_group.add_argument("--project", type=str, required=True, help="Project name")
  output_group.add_argument("--run", type=str, default=None, help="Name for this run")
  output_group.add_argument("--base_path", type=str, default=None, help="Base output path")
  output_group.add_argument("--checkpoint", action="store_true", help="Save checkpoints")

  output_group.add_argument("--non_interactive", action="store_true", help="Disable progress bars")



  output_group.add_argument("--wandb", action="store_true", help="Use wandb logging")
  output_group.add_argument("--log_details", action="store_true", help="Log detailed histograms for points/gradients etc.")


  args = parser.parse_args()

  overrides = args.overrides

  # General arguments
  if args.debug:
    overrides.append(f"debug={args.debug}")

  # Dataset group
  if args.scan is not None:
    
    overrides.append("dataset=scan")
    overrides.append(f"dataset.scan_file={os.path.abspath(args.scan)}")

  if args.colmap is not None:
    overrides.append("dataset=colmap")
    overrides.append(f"dataset.base_path={os.path.abspath(args.colmap)}")

  if args.image_scale is not None:
    overrides.append(f"dataset.image_scale={args.image_scale}")
    overrides.append("dataset.resize_longest=null")
  
  if args.resize_longest is not None:
    overrides.append(f"dataset.resize_longest={args.resize_longest}")
    overrides.append("dataset.image_scale=null")

  if args.far is not None:
    overrides.append(f"far={args.far}")

  if args.near is not None:
    overrides.append(f"near={args.near}")

  # Training group
  if args.target_points is not None:
    overrides.append(f"trainer.target_points={args.target_points}")


  if args.total_steps is not None:
    overrides.append(f"trainer.total_steps={args.total_steps}")

  if args.eval_steps is not None:
    overrides.append(f"trainer.eval_steps={args.eval_steps}")

  if args.training_scale is not None:
    overrides.append(f"training_scale={args.training_scale}")

  # Pointcloud initialisation from dataset

  if args.initial_points is not None:
    overrides.append(f"trainer.cloud_init.initial_points={args.initial_points}")

  if args.limit_points is not None:
    overrides.append(f"trainer.cloud_init.limit_points={args.limit_points}")

  if args.random_points is not None:
    assert not args.limit_points, "Cannot use both --limit_points and --random_points"
    assert not args.initial_points, "Cannot use both --initial_points and --random_points"

    overrides.append("trainer.cloud_init.limit_points=0")
    overrides.append(f"trainer.cloud_init.initial_points={args.random_points}")


  # Scene
  if args.tcnn:
    overrides.append("scene=tcnn")


  if args.bilateral:
    overrides.append("color_corrector=bilateral")

  if args.vis:
    overrides.append("viewer=splatview")


  if args.no_controller:
    overrides.append("controller=disabled")

  if args.mcmc:
    overrides.append("controller=mcmc")

  if args.antialias:
    overrides.append("trainer.antialias=true")

  if args.no_autotune:
    overrides.append("trainer.scene.autotune=false")

  # Output group
  if args.wandb is True:
    overrides.append("logger=wandb")

  if args.log_details:
    overrides.append("trainer.log_details=true")

  if args.checkpoint:
    overrides.append("trainer.save_checkpoints=true")

  if args.non_interactive:
    overrides.append("trainer.interactive=false")
  
  base_path = Path(args.base_path) if args.base_path is not None else Path.cwd()
  args.base_path, run_path, args.run = config.setup_project(args.project, args.run, base_path=base_path)
  os.chdir(str(run_path))

  overrides += config.make_overrides(run_name=args.run, project=args.project, base_path=args.base_path)
  config.add_resolvers()

  
  hydra.initialize(config_path="../config", version_base="1.2")
  cfg = hydra.compose(config_name="config", overrides=overrides)

  if args.show_config:
    print(config.pretty(cfg))

  return cfg

def train_with_config(cfg) -> dict | str:
  import taichi as ti
  from splat_trainer.trainer import Trainer

  torch.set_grad_enabled(False)
  torch.set_float32_matmul_precision('highest')
  
  # suppress triton and torch dynamo verbosity
  # torch._logging.set_logs(dynamo=logging.CRITICAL, inductor=logging.CRITICAL)

  torch.set_printoptions(precision=4, sci_mode=False)
  np.set_printoptions(precision=4, suppress=True)

  output_path = Path.cwd()
  print(f"Output path {colored(output_path, 'light_green')}")

  with open(output_path / "config.yaml", "w") as f:
      OmegaConf.save(cfg, f)

  logger:Logger = hydra.utils.instantiate(cfg.logger)
  logger.log_config(OmegaConf.to_container(cfg, resolve=True))

  if not cfg.interactive:
    os.environ["TQDM_DISABLE"] = "True"


  trainer = None
  result = None

  try:
    TaichiQueue.init(arch=ti.cuda, debug=cfg.debug, device_memory_GB=0.1, threaded=True)
    
    train_config = hydra.utils.instantiate(cfg.trainer, _convert_="object")
    dataset = hydra.utils.instantiate(cfg.dataset, _convert_="object")
  
    trainer = Trainer.initialize(train_config, dataset, logger)
    viewer:Viewer = hydra.utils.instantiate(cfg.viewer).create_viewer(trainer, enable_training=True)

    result = trainer.train()

    # allow viewer to run if enabled
    viewer.spin()
  except KeyboardInterrupt:
    pass

  finally:
    if trainer is not None:
      trainer.close()
    
    logger.close()
  return result
    
def main():
  cfg = cfg_from_args()
  train_with_config(cfg)


if __name__ == "__main__":
  main()