import logging
import subprocess

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import redis
import pandas as pd

from splat_trainer.scripts.train_scan import train_with_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


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
    cfg_dict = OmegaConf.to_container(cfg, throw_on_missing=False, resolve=True)
    df = pd.json_normalize(cfg_dict)
    override_strings = [f"{k}={df[k].iloc[0]}" for k in sweeper_params.keys() if k in df.columns ]
    
    sweep_dir = f"{HydraConfig.get().sweep.dir}/{HydraConfig.get().sweep.subdir}"
    subprocess.run([
        'splat-trainer-multirun', 
        '+multirun=grid_search', 
        f'hydra.sweep.dir={sweep_dir}', 
        f'logger.group={int(cfg.logger.group):04d}', 
        f'project={cfg.project}',
        *override_strings
      ], check=True)
    
    redis_db = redis.Redis(host='localhost', port=cfg.conn.redis.port, decode_responses=True)
    average_result = redis_db.hgetall(cfg.conn.graphite.result_key)

    assert average_result, "Error: 'multirun_result:1' not found or empty in Redis."
      
    results = float(average_result['train_psnr'])

    return results
    


if __name__ == "__main__":
    main()