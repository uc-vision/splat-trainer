from hydra import initialize, compose
from hydra.utils import instantiate
import argparse
from splat_trainer.trainer import GaussianPointCloudTrainer



def main():
  parser = argparse.ArgumentParser("Train a Gaussian Point Cloud Scene")
  parser.add_argument("scan_file", type=str)

  parser.add_argument(
      'overrides',
      nargs='*',
      help=(
        'Any key=value arguments to override config values '
        '(use dots for.nested=overrides)'
      )
  )

  args = parser.parse_args()
  print(args)

  initialize('../config', version_base="1.2")
  cfg = compose('config.yaml', 
                [*args.overrides, "dataset=scan", f"dataset.filename={args.scan_file}"] )
  

  dataset=instantiate(cfg.dataset)

  trainer = GaussianPointCloudTrainer(
    loss_function=instantiate(cfg.loss),
    config=instantiate(cfg.trainer))

  trainer.train()

if __name__ == "__main__":
  main()