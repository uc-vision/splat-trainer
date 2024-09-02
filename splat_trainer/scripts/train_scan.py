import argparse
import os
from pathlib import Path
import signal
import traceback
import hydra

from omegaconf import OmegaConf
from termcolor import colored
from splat_trainer.util import config

import numpy as np
import torch


config.add_resolvers()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("overrides", nargs="*", help="hydra overrides var=value")
  parser.add_argument("--debug", action="store_true", help="Enable taichi debugging")

  parser.add_argument("--target", type=int, default=None, help="Target point count")
  parser.add_argument("--image_scale", type=float, default=None, help="Image scale")
  parser.add_argument("--resize_longest", type=float, default=None, help="Resize longest side")

  parser.add_argument("--no_alpha", action="store_true", help="Fix point alpha=1.0 in training")

  parser.add_argument("--steps", type=int, default=None, help="Number of training steps")
  parser.add_argument("--background_points", type=int, default=None, help="Number of background points")

  parser.add_argument("--scan", type=str, default=None, help="Scan json scene file to load")
  parser.add_argument("--colmap", type=str, default=None, help="Colmap scene to load")

  parser.add_argument("--project", type=str, default=None, help="Use project name")
  parser.add_argument("--output_path", type=str, default=None, help="Override output path")

  parser.add_argument("--random_points", type=int, default=None,  help="Initialise with N random points only")
  parser.add_argument("--checkpoint", type=int, default=None, help="Save checkpoints every N evaluations")


  parser.add_argument("--wandb", action="store_true", help="Use wandb logging")
  args = parser.parse_args()

  
  hydra.initialize(config_path="../config", version_base="1.2")
  overrides = args.overrides

  if args.debug:
    overrides.append(f"debug={args.debug}")

  if args.scan is not None:
    overrides.append("dataset=scan")
    overrides.append(f"dataset.scan_file={args.scan}")

  if args.colmap is not None:
    overrides.append("dataset=colmap")
    overrides.append(f"dataset.base_path={args.colmap}")

  if args.target is not None:
    overrides.append("controller=target")
    overrides.append(f"trainer.controller.target_count={args.target}")

  if args.image_scale is not None:
    overrides.append(f"dataset.image_scale={args.image_scale}")
    overrides.append(f"dataset.resize_longest=null")
  
  if args.resize_longest is not None:
    overrides.append(f"dataset.resize_longest={args.resize_longest}")
    overrides.append(f"dataset.image_scale=null")
  
  
  if args.no_alpha:
    overrides.append(f"trainer.initial_alpha=1.0")
    overrides.append(f"trainer.scene.learning_rates.alpha_logit=0.0")

  if args.steps is not None:
    overrides.append(f"trainer.steps={args.steps}")

  if args.background_points is not None:
    overrides.append(f"trainer.background_points={args.background_points}")

  if args.wandb is not None:
    overrides.append("logger=wandb")

  
  if args.project is not None:
    overrides.append(f"project={args.project}")

  cfg = hydra.compose(config_name="config", overrides=overrides)

  if args.output_path is not None:
    output_path = Path(args.output_path) / cfg.project 
    output_path.mkdir(parents=True, exist_ok=True)
    os.chdir(str(output_path))

  train_with_config(cfg)

def train_with_config(cfg):
  import taichi as ti
  from splat_trainer.trainer import Trainer

  torch.set_grad_enabled(False)
  torch.set_printoptions(precision=4, sci_mode=False)
  np.set_printoptions(precision=4, suppress=True)
  torch.set_float32_matmul_precision('high')

  # print(config.pretty(cfg))
  print(f"Output path {colored(Path.cwd(), 'light_green')}")


  logger = hydra.utils.instantiate(cfg.logger, _partial_=True)(log_config=OmegaConf.to_container(cfg, resolve=True))
  # logger = hydra.utils.instantiate(cfg.logger) 

  trainer = None

  try:
    ti.init(arch=ti.cuda, debug=cfg.debug, device_memory_GB=0.1)
    
    train_config = hydra.utils.instantiate(cfg.trainer)

    dataset = hydra.utils.instantiate(cfg.dataset)

    trainer = Trainer.initialize(train_config, dataset, logger)

    result = trainer.train()
    if cfg.wait_exit:
      input("Press Enter to continue...")

  except KeyboardInterrupt:
    pass
  except Exception:
    # print exception and stack trace and exit
     traceback.print_exc()

  if trainer is not None:
    trainer.close()
  
  logger.close()
  return result
    
if __name__ == "__main__":
  main()