import datetime
import logging
from pathlib import Path
from typing import Any

import redis
import socket
import wandb
from omegaconf import DictConfig
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback

from splat_trainer.multirun.util import get_sweep_params_dict
from splat_trainer.multirun.deploy import flush_db
from splat_trainer.logger.logger import Logger




class LogBestResult(Callback):
    def __init__(self, redis_port: int, redis_db_num: int, project: str, sweep_params: DictConfig) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.project = project
        self.redis_host = socket.gethostname()
        self.redis_port = redis_port
        self.redis_db_num = redis_db_num
        self.sweep_params = sweep_params
        self.redis_url = f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db_num}"
        
        self.run = wandb.init(project=project, 
                        group="best_result",
                        name="best_result", 
                        dir=Path.cwd(), 
                        entity='UCVision')
        

    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        
        self.params = get_sweep_params_dict(config, self.sweep_params)
        redis_db = redis.Redis(host=self.redis_host, port=self.redis_port, db=self.redis_db_num, decode_responses=True)

        result = redis_db.hgetall(f"{self.project}:result")
        result = {k: float(v) for k, v in result.items()}
        
        best_result = redis_db.hgetall(f"{self.project}:best_result") or {}
        best_result = {k: float(v) for k, v in best_result.items()}

        current_step = config.hydra.job.num

        if 'psnr' not in best_result or result['psnr'] > best_result['psnr']:
            best_result = result
            redis_db.hset(f"{self.project}:best_result", mapping=best_result)
            
            data = [current_step] + list(self.params.values()) + list(result.values())
            table = wandb.Table(columns=['step'] + list(self.params.keys()) + list(result.keys()))
            table.add_data(*data)
            self.run.log({"optuna_optimization_best_value/best_value": table}, step=current_step)

        self.run.log({f"optuna_optimization_metric/{metric}": value for metric, value in result.items()}, step=current_step)
        self.run.log({f"optuna_optimization_param/{param_name}": float(param_value) for param_name, param_value in self.params.items()}, step=current_step)
        self.run.log({f"optuna_optimization_best_value/best_psnr": best_result['psnr']}, step=current_step)

    
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None: 
        self.run.finish()