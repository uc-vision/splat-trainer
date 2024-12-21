from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
from typing import Callable, Optional

import hydra

import numpy as np
from omegaconf import DictConfig, OmegaConf
from taichi_splatting import TaichiQueue
from splat_trainer.trainer import Trainer
from taichi_splatting import TaichiQueue
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

def with_trainer(f:Callable[[Trainer], None], args:Namespace):

  overrides = args.override or []
  
  torch.set_grad_enabled(False)
  torch.set_float32_matmul_precision('high')

  torch.set_printoptions(precision=4, sci_mode=False)
  np.set_printoptions(precision=4, suppress=True)

  TaichiQueue.init(arch=ti.cuda, debug=args.debug, threaded=False)

  add_resolvers()

  state_dict, workspace_path = load_checkpoint(args.splat_path, args.step)
  config = OmegaConf.load(workspace_path / "config.yaml")

  if not args.enable_logging:
    config.logger = OmegaConf.create({"_target_": "splat_trainer.logger.NullLogger"})

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
  trainer = init_from_checkpoint(config, state_dict)
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
  parser.add_argument("--enable_logging", action="store_true", help="Enable logging")
  
  parser.add_argument("--override", type=str, nargs="*", help="Override config values")
  return parser

def evaluate():
  parser = arguments()
  args = parser.parse_args()

  def f(trainer):
    result = trainer.evaluate()
    print(result)

  with_trainer(f, args)


def add_viewer_args(parser:ArgumentParser):
  parser.add_argument("--eval", action="store_true", help="Evaluate the model before visualizing")
  parser.add_argument("--port", type=int, default=8000, help="Port to run the web viewer on")
  parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web viewer on")

def start_with_viewer(trainer:Trainer, args:Namespace, enable_training: bool = False):
  trainer.warmup()

  config = SplatviewConfig(port=args.port, host=args.host)
  viewer = config.create_viewer(trainer, enable_training)

  if args.eval:
    result = trainer.evaluate()
    print(result)

  return viewer

def visualize():
  parser = arguments()
  add_viewer_args(parser)

  args = parser.parse_args()
    
  def f(trainer:Trainer):

    viewer = start_with_viewer(trainer, args)
    viewer.spin()

  with_trainer(f, args)
    
def resume():
  parser = arguments()
  add_viewer_args(parser)
  args = parser.parse_args()

  def f(trainer:Trainer):

    if args.vis:
      viewer = start_with_viewer(trainer, args, enable_training=True)

    trainer.train()

    if viewer is not None:
      viewer.spin()

  with_trainer(f, args)


  

if __name__ == "__main__":
  resume()
