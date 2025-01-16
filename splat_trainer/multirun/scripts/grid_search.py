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
    "run_name",
    lambda x: "; ".join([
        f"{'.'.join((item.split('=')[0]).split('.')[-2:])}={float(item.split('=')[1]):.4f}"
        if bool(re.match(r'^-?\d+(\.\d+)?$', item.split('=')[1]))
        else f"{'.'.join((item.split('=')[0]).split('.')[-2:])}={item.split('=')[1]}"
        for item in x
    ]) if x else ""
)

OmegaConf.register_new_resolver("format", lambda x: f"{x:04d}")

OmegaConf.register_new_resolver(
    "project_name", 
    lambda x: "__".join([
        ".".join((item.split('=')[0]).split('.')[-2:])
        for item in x if 'test_scene' not in item
    ]) if x else ""
)

OmegaConf.register_new_resolver(
    "group_name",
    lambda x: "; ".join([
        "{}={}".format(
            ".".join((item.split('=')[0]).split('.')[-2:]),
            "{:.3f}".format(float(value)) if re.match(r'^-?\d+(\.\d+)?$', value) and '.' in value else value
        )
        for item in x
        if 'test_scene' not in item and (value := item.split('=')[1])  # Use walrus operator here
    ]) if x else ""
)

OmegaConf.register_new_resolver(
    "conditional_run_name",
    lambda optimize_algorithm, group_name, job_num, task_name:
      f"{group_name}__{job_num}__{task_name}"
      if optimize_algorithm == "grid_search"
      else f"{job_num}__{task_name}"
    
)



@hydra.main(config_path="../../config", version_base="1.2", config_name="grid_search")
def main(cfg : DictConfig) -> None:
  dataset_cfg = find_dataset_config(cfg.test_scene, cfg.test_datasets)
  OmegaConf.update(cfg, "dataset", dataset_cfg, force_add=True)

  result = train_with_config(cfg)

  return result
  


if __name__ == "__main__":
    main()