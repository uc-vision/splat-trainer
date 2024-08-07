import argparse
import signal
import traceback
import hydra

from omegaconf import OmegaConf
from splat_trainer.util import config

import numpy as np
import torch
import debugpy


config.add_resolvers()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("overrides", nargs="*", help="hydra overrides var=value")
  parser.add_argument("--debug_port", type=int, default=None, help="Enable python remote debugger on port")
  parser.add_argument("--target", type=int, default=None, help="Target point count")
  parser.add_argument("--image_scale", type=float, default=None, help="Image scale")
  parser.add_argument("--steps", type=int, default=None, help="Number of training steps")

  args = parser.parse_args()

  if args.debug_port is not None:
    debugpy.listen(("localhost", args.debug_port))
    print(f"Waiting for debugger attach on port {args.debug_port}")
    debugpy.wait_for_client()

  hydra.initialize(config_path="../config", version_base="1.2")
  cfg = hydra.compose(config_name="config", overrides=args.overrides)

  if args.target is not None:
    cfg.trainer.controller.target_count = args.target

  if args.image_scale is not None:
    cfg.dataset.image_scale = args.image_scale

  if args.steps is not None:
    cfg.trainer.steps = args.steps

  train_with_config(cfg)

def train_with_config(cfg):
  import taichi as ti
  from splat_trainer.trainer import Trainer

  torch.set_grad_enabled(False)
  torch.set_printoptions(precision=4, sci_mode=False)
  np.set_printoptions(precision=4, suppress=True)
  torch.set_float32_matmul_precision('high')
  
  print(config.pretty(cfg))
  logger = hydra.utils.instantiate(cfg.logger, _partial_=True)(log_config=OmegaConf.to_container(cfg, resolve=True))
  trainer = None


  try:
    ti.init(arch=ti.cuda, debug=cfg.debug, device_memory_GB=0.1)
    
    train_config = hydra.utils.instantiate(cfg.trainer)
    dataset = hydra.utils.instantiate(cfg.dataset)

    trainer = Trainer(dataset, train_config, logger)

    trainer.train()
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
    
if __name__ == "__main__":
  main()