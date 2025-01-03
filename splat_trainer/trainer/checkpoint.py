
from pathlib import Path
from typing import Optional

import hydra
import torch



def find_checkpoint(path:Path, checkpoint:Optional[int]=None):
  checkpoint_path = path / "checkpoint"
  checkpoints = list(checkpoint_path.glob("*.pt"))

  if len(checkpoints) == 0:
    raise FileNotFoundError(f"No checkpoints found in {checkpoint_path}")

  numbers = [int(x.stem.split("_")[1]) for x in checkpoints]
  checkpoint_dict = dict(zip(numbers, checkpoints))

  n = max(numbers) if checkpoint is None else checkpoint
  if n not in checkpoint_dict:
    raise FileNotFoundError(f"Checkpoint {n} not found, options are {sorted(numbers)}")

  return checkpoint_dict[n]


def load_checkpoint(splat_path:Path, step:Optional[int]=None):
  if splat_path.is_dir():
    workspace_path = splat_path
    checkpoint = find_checkpoint(workspace_path, step)
  else:
    assert splat_path.is_file(), f"Checkpoint {splat_path} not found"

    workspace_path = splat_path.parent.parent
    checkpoint = splat_path

  
  print(f"Loading checkpoint {checkpoint}")
  state_dict = torch.load(checkpoint, weights_only=True)

  return state_dict, workspace_path


def init_from_checkpoint(config, state_dict):
  from splat_trainer.trainer import Trainer

  dataset = hydra.utils.instantiate(config.dataset, _convert_="object")  
  train_config = hydra.utils.instantiate(config.trainer, _convert_="object")

  logger = hydra.utils.instantiate(config.logger)
  trainer = Trainer.from_state_dict(train_config, dataset, logger, state_dict)
  return trainer
