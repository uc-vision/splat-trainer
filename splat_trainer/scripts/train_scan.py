import argparse
from pathlib import Path
import hydra
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored
from splat_trainer.logger.logger import Logger
from splat_trainer import config

from taichi_splatting import TaichiQueue

import torch
import os

from splat_trainer.viewer.viewer import Viewer


config.add_resolvers()



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

  # Training group
  training_group = parser.add_argument_group("Training")
  training_group.add_argument("--target", type=int, default=None, help="Target point count")
  training_group.add_argument("--no_alpha", action="store_true", help="Fix point alpha=1.0 in training")
  training_group.add_argument("--steps", type=int, default=None, help="Number of training steps")
  
  training_group.add_argument("--add_points", type=int, default=None, help="Add random background points")
  training_group.add_argument("--limit_points", type=int, default=None, help="Limit the number of points from the dataset to N")
  training_group.add_argument("--random_points", type=int, default=None, help="Initialise with N random points only")

  training_group.add_argument("--tcnn", action="store_true", help="Use tcnn scene")
  training_group.add_argument("--sh", action="store_true", help="Use spherical harmonics scene")
  training_group.add_argument("--bilateral", action="store_true", help="Use bilateral color correction")

  training_group.add_argument("--vis", action="store_true", help="Enable web viewer")


  # Rendering group
  rendering_group = parser.add_argument_group("Rendering")
  rendering_group.add_argument("--antialias", action="store_true", help="Use antialiasing")


  # Output group
  output_group = parser.add_argument_group("Output")
  output_group.add_argument("--project", type=str, required=True, help="Project name")
  output_group.add_argument("--run", type=str, default=None, help="Name for this run")
  output_group.add_argument("--base_path", type=str, default=None, help="Base output path")
  output_group.add_argument("--checkpoint", action="store_true", help="Save checkpoints")
  output_group.add_argument("--wandb", action="store_true", help="Use wandb logging")

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

  # Training group
  if args.target is not None:
    overrides.append("controller=target")
    overrides.append(f"trainer.controller.target_count={args.target}")
  
  if args.no_alpha:
    overrides.append("trainer.initial_alpha=1.0")
    overrides.append("trainer.scene.learning_rates.alpha_logit=0.0")

  if args.steps is not None:
    overrides.append(f"trainer.steps={args.steps}")

  assert args.add_points is None or args.random_points is None, "Cannot specify both background and random points"
  assert args.limit_points is None or args.random_points is None, "Cannot specify both limit and random points"

  if args.add_points is not None:
    overrides.append(f"trainer.initial_points={args.add_points}")
    overrides.append("trainer.add_initial_points=true")

  if args.limit_points is not None:
    overrides.append(f"trainer.limit_points={args.limit_points}")


  if args.random_points is not None:
    overrides.append(f"trainer.initial_points={args.random_points}")
    overrides.append("trainer.load_dataset_cloud=false")


  if args.tcnn:
    overrides.append("scene=tcnn")

  if args.sh:
    overrides.append("scene=sh")

  if args.bilateral:
    overrides.append("color_corrector=bilateral")

  if args.vis:
    overrides.append("viewer=splatview")

  if args.antialias:
    overrides.append("trainer.antialias=true")

  # Output group
  if args.wandb is not None:
    overrides.append("logger=wandb")

  if args.checkpoint:
    overrides.append("trainer.save_checkpoints=true")
  
  run_path, args.run = config.setup_project(args.project, args.run, base_path=args.base_path)
  os.chdir(str(run_path))

  overrides += [f"run_name={args.run}", f"project={args.project}", f"base_path={args.base_path}"]

  hydra.initialize(config_path="../config", version_base="1.2")
  cfg = hydra.compose(config_name="config", overrides=overrides)

  if args.show_config:
    print(config.pretty(cfg))

  return cfg

def train_with_config(cfg) -> dict | str:
  import taichi as ti
  from splat_trainer.trainer import Trainer

  torch.set_grad_enabled(False)
  torch.set_float32_matmul_precision('high')

  torch.set_printoptions(precision=4, sci_mode=False)
  np.set_printoptions(precision=4, suppress=True)

  output_path = Path.cwd()
  print(f"Output path {colored(output_path, 'light_green')}")
  
  with open(output_path / "config.yaml", "w") as f:
      OmegaConf.save(cfg, f)

  logger:Logger = hydra.utils.instantiate(cfg.logger)
  logger.log_config(OmegaConf.to_container(cfg, resolve=True))

  trainer = None
  result = None

  try:
    TaichiQueue.stop()
    TaichiQueue.init(arch=ti.cuda, debug=cfg.debug, device_memory_GB=0.1, threaded=True)
    
    train_config = hydra.utils.instantiate(cfg.trainer, _convert_="object")
    dataset = hydra.utils.instantiate(cfg.dataset)
  
    trainer = Trainer.initialize(train_config, dataset, logger)
    trainer.warmup()

    viewer: Viewer = hydra.utils.instantiate(cfg.viewer).create_viewer(trainer, enable_training=True)
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