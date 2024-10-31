import hydra
from omegaconf import DictConfig, OmegaConf
import re
import sys

import taichi as ti

from splat_trainer.scripts.train_scan import train_with_config
from splat_trainer.trainer import Trainer


def find_dataset_config(name:str, test_datasets:DictConfig):
  for k, collection in test_datasets.items():
    if name in collection.scenes:
      config = OmegaConf.merge(collection.common,
        collection.scenes[name])
      
      return config

  raise AttributeError(f"Scene {name} not found in test datasets")


OmegaConf.register_new_resolver("sanitize", lambda x: x.replace("/", "__"))
OmegaConf.register_new_resolver(
    "extract",
    lambda x: "; ".join([
        f"{(item.split('=')[0]).split('.')[-1]}={item.split('=')[1]}" 
        if bool(re.match(r'^-?\d+(\.\d+)?$', item.split('=')[1])) or item.split('=')[1] in ['True', 'False', 'true', 'false'] 
        else item.split('=')[1]
        for item in x
    ]) if x else ""
)
OmegaConf.register_new_resolver("format", lambda x: f"{x:04d}")



@hydra.main(config_path="../config", version_base="1.2", config_name="grid_search")
def main(cfg : DictConfig) -> None:
  dataset_cfg = find_dataset_config(cfg.test_scene, cfg.test_datasets)
  OmegaConf.update(cfg, "dataset", dataset_cfg, force_add=True)

  result = train_with_config(cfg)

  # save result to yaml - TODO: use a different wandb logger to upload result too?


  return result
  


if __name__ == "__main__":
    main()