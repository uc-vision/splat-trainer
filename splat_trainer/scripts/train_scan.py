import hydra
import argparse

from omegaconf import OmegaConf
from splat_trainer.util import config
import signal
import sys

import numpy as np
import torch


config.add_resolvers()


@hydra.main(config_name="config", config_path="../config", version_base="1.2")
def main(cfg):
  import taichi as ti
  from splat_trainer.trainer import Trainer

  torch.set_printoptions(precision=4, sci_mode=False)
  np.set_printoptions(precision=4, suppress=True)

  print(config.pretty(cfg))
  
  ti.init(arch=ti.cuda, debug=cfg.debug)
  logger = hydra.utils.instantiate(cfg.logger, _partial_=True)(log_config=OmegaConf.to_container(cfg, resolve=True))
  
  train_config = hydra.utils.instantiate(cfg.trainer)
  dataset = hydra.utils.instantiate(cfg.dataset)

  trainer = Trainer(dataset, train_config, logger)

  def signal_handler(sig, frame):
      trainer.close()
      sys.exit(0)

  signal.signal(signal.SIGINT, signal_handler)


  trainer.train()
  
if __name__ == "__main__":
  main()