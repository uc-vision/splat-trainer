import hydra
from omegaconf import DictConfig, OmegaConf

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



@hydra.main(config_path="../config", version_base="1.2", config_name="grid_search")
def main(cfg : DictConfig) -> dict | str:
    
  dataset_cfg = find_dataset_config(cfg.test_scene, cfg.test_datasets)
  OmegaConf.update(cfg, "dataset", dataset_cfg, force_add=True)
  
  result = train_with_config(cfg)
  print(result)

  return result


if __name__ == "__main__":
    main()