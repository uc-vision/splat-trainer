import subprocess

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import redis

from splat_trainer import config



config.add_resolvers()

OmegaConf.register_new_resolver("format", lambda x: f"{x:04d}")

@hydra.main(config_path="../config", version_base="1.2", config_name="bayesian_optimize")
def main(cfg : DictConfig) -> None:
  
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
  
  override_strings = [f"{k}={v}" for k, v in flatten_cfg(cfg) if k in sweeper_params or k in ['project', 'group_name', 'trainer.optimizer', 'optimize_algorithm']]

  subprocess.run(['grid-search-trainer', *override_strings])
  
  redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
  averaged_results = redis_client.hgetall("multirun_result:1")
  print("averaged_results: ", averaged_results)
  
  results = float(averaged_results['train_psnr'])

  return results

  


if __name__ == "__main__":
    main()