import subprocess

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import redis

from splat_trainer.scripts.train_scan import train_with_config




def find_dataset_config(name:str, test_datasets:DictConfig):
  for k, collection in test_datasets.items():
    if name in collection.scenes:
      config = OmegaConf.merge(collection.common,
        collection.scenes[name])
      
      return config

  raise AttributeError(f"Scene {name} not found in test datasets")



@hydra.main(config_path="../../config", version_base="1.2", config_name="config")
def main(cfg : DictConfig) -> None:

  if cfg.algorithm in ['grid-search']:
    dataset_cfg = find_dataset_config(cfg.test_scene, cfg.test_datasets)
    OmegaConf.update(cfg, "dataset", dataset_cfg, force_add=True)
    
    result = train_with_config(cfg)
    
    return result
  
  else:
    sweeper_params = HydraConfig.get().sweeper.params
  
    def flatten_cfg(d, parent_key=''):
      items=[]
      for k, v in d.items():
          new_key = f"{parent_key}.{k}" if parent_key else k
          if isinstance(v, (dict, DictConfig)):
              items.extend(flatten_cfg(v, new_key))
          elif isinstance(v, list):
              items.append((new_key, f"[{', '.join(map(str, v))}]"))
          else:
              items.append((new_key, v))
      return items
    
    cfg = OmegaConf.to_container(cfg, throw_on_missing=False, resolve=True)
    override_strings = [f"{k}={v}" for k, v in flatten_cfg(cfg) if v and k in sweeper_params or k in ['project']]
    
    dir = f"{HydraConfig.get().sweep.dir}/{HydraConfig.get().sweep.subdir}"
    group = f'{int(cfg["logger"]["group"]):04d}'
    subprocess.run(['splat-trainer-multirun', '+multirun=grid_search', f'hydra.sweep.dir={dir}', f'logger.group={group}', *override_strings])
    
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    average_result = redis_client.hgetall("multirun_result:1")

    assert average_result, "Error: 'multirun_result:1' not found or empty in Redis. Check if the key exists and data is stored correctly."
      
    results = float(average_result['train_psnr'])

    return results
    


if __name__ == "__main__":
    main()