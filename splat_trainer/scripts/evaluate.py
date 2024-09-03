from argparse import ArgumentParser
import os
from pathlib import Path
from typing import Optional

import hydra

from omegaconf import OmegaConf
from splat_trainer.logger.logger import NullLogger
from splat_trainer.trainer import Trainer
import torch

import taichi as ti


def find_checkpoint(path:Path, checkpoint:Optional[int]=None):
  checkpoint_path = path / "checkpoints"
  checkpoints = list(checkpoint_path.glob("*.pt"))

  numbers = [int(x.stem.split("_")[1]) for x in checkpoints]
  checkpoint_dict = dict(zip(numbers, checkpoints))

  n = max(numbers) if checkpoint is None else checkpoint
  if n not in checkpoint_dict:
    raise FileNotFoundError(f"Checkpoint {n} not found, options are {sorted(numbers)}")

  return checkpoint_dict[n]


def init_from_checkpoint(config, output_path:Path, step:Optional[int]=None):


  checkpoint = find_checkpoint(output_path, step)
  print(f"Loading checkpoint {checkpoint}")

  state_dict = torch.load(checkpoint, weights_only=True)

  dataset = hydra.utils.instantiate(config.dataset)  
  train_config = hydra.utils.instantiate(config.trainer)

  logger = hydra.utils.instantiate(config.logger)

  trainer = Trainer.from_state_dict(train_config, dataset, logger, state_dict)
  return trainer

def main():
  parser = ArgumentParser()
  parser.add_argument("output_path", type=Path,  help="Path to output folder from splat-trainer")
  parser.add_argument("--step", type=int, default=None, help="Checkpoint from step to evaluate")
  parser.add_argument("--debug", action="store_true", help="Enable debug in taichi")
  parser.add_argument("--scale_images", type=float, default=None, help="Scale images relative to training size")

  parser.add_argument("--override", type=str, nargs="*", help="Override config values")

  args = parser.parse_args()
  overrides = args.override or []
  
  ti.init(arch=ti.cuda, debug=args.debug)

  config = OmegaConf.load(args.output_path / "config.yaml")
  print(OmegaConf.to_yaml(config))

  os.chdir(str(args.output_path))

  for override in overrides:
    key, value = override.split("=")
    OmegaConf.update(config, key, value)

  if args.scale_images is not None:
    dataset = config.dataset
    if dataset.image_scale is not None:
      dataset.image_scale *= args.scale_images
    
    if config.dataset.resize_longest is not None:
      dataset.resize_longest = int(dataset.resize_longest * args.scale_images)

  

  trainer = init_from_checkpoint(config, args.output_path, args.step)
  result = trainer.evaluate()

  print(result)

  trainer.close()
  trainer.logger.close()

  print("Done")


    


if __name__ == "__main__":
  main()