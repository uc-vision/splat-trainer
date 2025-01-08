import hydra
from omegaconf import DictConfig, OmegaConf

import taichi as ti

from splat_trainer.scripts.train_scan import train_with_config
from splat_trainer.trainer import Trainer

OmegaConf.register_new_resolver("format", lambda x: f"{x:04d}")

@hydra.main(config_path="../config", version_base="1.2", config_name="bayesian_optimize")
def main(cfg : DictConfig) -> None:
  
  result = train_with_config(cfg)
  return result
  


if __name__ == "__main__":
    main()