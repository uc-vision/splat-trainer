from hydra import initialize, compose
from hydra.utils import instantiate
import argparse

from omegaconf import OmegaConf

from splat_trainer.util import config
from splat_trainer.trainer import Trainer
import taichi as ti


def main():
  parser = argparse.ArgumentParser("Train a Gaussian Point Cloud Scene")
  parser.add_argument("scan_file", type=str)
  

  parser.add_argument(
      'overrides',
      nargs='*',
      help=(
        'Any key=value arguments to override config values '
        '(use dots for.nested=overrides)'
      ),
  )

  
  args = parser.parse_args()
  print(args)

  config.add_resolvers()
  initialize('../config', version_base="1.2")
  cfg = compose('config.yaml', 
                [*args.overrides, "dataset=scan", f"dataset.scan_file={args.scan_file}"] )
  
  print(config.pretty(cfg))

  ti.init(arch=ti.cuda, debug=cfg.debug)

  logger = instantiate(cfg.logger, _partial_=True)(log_config=OmegaConf.to_container(cfg, resolve=True))
  
  train_config = instantiate(cfg.trainer)
  dataset = instantiate(cfg.dataset)


  trainer = Trainer(dataset, train_config, logger)
  trainer.train()
  
if __name__ == "__main__":
  main()