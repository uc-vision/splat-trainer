from argparse import ArgumentParser
import os
from pathlib import Path
from time import time
from typing import Optional

import hydra

import numpy as np
from omegaconf import DictConfig, OmegaConf
from splat_trainer.trainer import Trainer
import torch

import taichi as ti

from splat_trainer.config import add_resolvers, setup_project
from splat_trainer.viewer import SplatviewConfig


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


def init_from_checkpoint(config, splat_path:Path, step:Optional[int]=None):

  checkpoint = find_checkpoint(splat_path, step)
  print(f"Loading checkpoint {checkpoint}")

  state_dict = torch.load(checkpoint, weights_only=True)

  dataset = hydra.utils.instantiate(config.dataset)  
  train_config = hydra.utils.instantiate(config.trainer)

  logger = hydra.utils.instantiate(config.logger)

  trainer = Trainer.from_state_dict(train_config, dataset, logger, state_dict)
  return trainer



def get_path(dotted_path:str, config:DictConfig):
  keys = dotted_path.split(".")
  for key in keys:
    if key not in config:
      raise AttributeError(f"Key {key} not found in config")
    
    config = config[key]
  return config

def with_trainer(f, args):

  overrides = args.override or []
  
  torch.set_grad_enabled(False)
  torch.set_float32_matmul_precision('high')

  torch.set_printoptions(precision=4, sci_mode=False)
  np.set_printoptions(precision=4, suppress=True)

  ti.init(arch=ti.cuda, debug=args.debug)

  add_resolvers()

  args.splat_path = args.splat_path.absolute()
  config = OmegaConf.load(args.splat_path / "config.yaml")


  run_path, args.run = setup_project(config.project, args.run or config.run_name, config.base_path)
  os.chdir(str(run_path))

  for override in overrides:
    key, value = override.split("=")

    existing_type = type(get_path(key, config))
    OmegaConf.update(config, key, existing_type(value))

  scale_images = getattr(args, "scale_images", None)
  resize_longest = getattr(args, "resize_longest", None)

  if scale_images is not None:
    dataset = config.dataset
    if dataset.image_scale is not None:
      dataset.image_scale *= scale_images

  if resize_longest is not None:
    dataset.resize_longest = int(dataset.resize_longest * resize_longest)

  print(OmegaConf.to_yaml(config))
  trainer = init_from_checkpoint(config, args.splat_path, args.step)
  print(trainer)

  try:
    f(trainer)
  except KeyboardInterrupt:
    pass

  trainer.close()

  print("Done")


def arguments():
  parser = ArgumentParser()
  parser.add_argument("splat_path", type=Path,  help="Path to output folder from splat-trainer")
  parser.add_argument("--step", type=int, default=None, help="Checkpoint from step to evaluate")
  parser.add_argument("--debug", action="store_true", help="Enable debug in taichi")

  parser.add_argument("--scale_images", type=float, default=None, help="Scale images relative to training size")
  parser.add_argument("--run", type=str, default=None, help="Name for this run")

  parser.add_argument("--override", type=str, nargs="*", help="Override config values")
  return parser

def evaluate():
  parser = arguments()
  args = parser.parse_args()

  def f(trainer):
    result = trainer.evaluate()
    print(result)

  with_trainer(f, args)



def visualize():
  parser = arguments()
  parser.add_argument("--eval", action="store_true", help="Evaluate the model before visualizing")

  parser.add_argument("--port", type=int, default=8000, help="Port to run the web viewer on")
  parser.add_argument("--host", type=str, default="localhost", help="Host to run the web viewer on")

  args = parser.parse_args()
    
  def f(trainer):

    config = SplatviewConfig(port=args.port, host=args.host)
    viewer = config.create_viewer(trainer)

    if args.eval:
      result = trainer.evaluate()
      print(result)

    viewer.spin()

  with_trainer(f, args)

    
def resume():
  parser = arguments()
  parser.add_argument("--vis", action="store_true", help="Run the web viewer")
  parser.add_argument("--port", type=int, default=8000, help="Port to run the web viewer on")
  parser.add_argument("--host", type=str, default="localhost", help="Host to run the web viewer on")
  args = parser.parse_args()

  def f(trainer):

    if args.vis:
      config = SplatviewConfig(port=args.port, host=args.host)
      viewer = config.create_viewer(trainer)

    trainer.train()

    if args.vis:
      viewer.spin()

  with_trainer(f, args)



if __name__ == "__main__":
  evaluate()