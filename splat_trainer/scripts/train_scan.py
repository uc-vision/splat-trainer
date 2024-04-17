import traceback
import hydra

from omegaconf import OmegaConf
from splat_trainer.util import config

import numpy as np
import torch


config.add_resolvers()


@hydra.main(config_name="config", config_path="../config", version_base="1.2")
def main(cfg):
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