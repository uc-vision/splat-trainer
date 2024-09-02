from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import hydra

from omegaconf import OmegaConf
from splat_trainer.logger.logger import NullLogger
from splat_trainer.trainer import Trainer
import torch


def find_checkpoint(path:Path, checkpoint:Optional[int]=None):
  checkpoint_path = path / "checkpoints"
  checkpoints = list(checkpoint_path.glob("*.pt"))

  numbers = [int(x.stem.split("_")[1]) for x in checkpoints]
  checkpoint_dict = dict(zip(numbers, checkpoints))

  n = max(numbers) if checkpoint is None else checkpoint
  if n not in checkpoint_dict:
    raise FileNotFoundError(f"Checkpoint {n} not found, options are {sorted(numbers)}")

  return checkpoint_dict[n]




def main():
  parser = ArgumentParser()
  parser.add_argument("path", type=Path,  help="Path to output folder from splat-trainer")
  parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint to evaluate")
  
  args = parser.parse_args()

  config = OmegaConf.load(args.path / "config.yaml")
  print(OmegaConf.to_yaml(config))

  checkpoint = find_checkpoint(args.path, args.checkpoint)
  print(f"Loading checkpoint {checkpoint}")

  state_dict = torch.load(checkpoint, weights_only=True)

  dataset = hydra.utils.instantiate(config.dataset)  
  train_config = hydra.utils.instantiate(config.trainer)
  trainer = Trainer.from_state_dict(train_config, dataset, NullLogger(), state_dict)

    


if __name__ == "__main__":
  main()