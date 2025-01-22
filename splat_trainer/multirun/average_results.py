import logging
import os
from pathlib import Path
import time
import traceback
from typing import Any, Optional

import numpy as np
import pandas as pd
import redis
import socket
import wandb
from hydra.experimental.callback import Callback
from omegaconf import DictConfig

from splat_trainer.multirun.util import compute_average_across_scenes, save_to_csv



class AverageResults(Callback):
    def __init__(self, output_dir: str, redis_port: int, redis_db: int) -> None:
        self.output_dir = output_dir
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.redis_host = socket.gethostname()
        self.redis_port = redis_port
        self.redis_db = redis_db
        
        
    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None: 
        self.project = config.project
        self.group = config.logger.group
        

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None: 
        
        try:
            results_file = os.path.join(self.output_dir, 'results.json')
            average, df = compute_average_across_scenes(results_file)
            self.log.info(f"Averaged result across the scenes: {average}")
            
        except Exception as e:
            self.log.error(f"Error occurred while averaging the results: {e}")
            self.log.error(f"Stack trace: {traceback.format_exc()}")

        assert average, "Error: average is empty!"
        
        redis_db = redis.Redis(host=self.redis_host, port=self.redis_port, db=self.redis_db, decode_responses=True)
        if 'push_stats' in config.hydra.callbacks:
            redis_db.hset(config.conn.redis.key_for_graphite, mapping=average)
            
        if config.algorithm != 'grid_search':
            redis_db.hset(config.conn.redis.key_for_optuna_result, mapping=average)
        
        # TODO: log time series data 
        # if 'log_best_result' in config.hydra.callbacks:
        #     timestamp = int(time.time() * 1000)
        #     redis_db.set("project", self.project)
        #     for metric, value in average.items():
        #         # redis_db.execute_command("TS.ADD", f"{self.project}:result:{metric}", timestamp, value)
        #     # params = df['param'].iloc[0].to_dict()
        #     # redis_db.hset(f"{self.project}:params:{timestamp}", mapping=params)
        #     # redis_db.zadd
            
            
        self.log_average_to_wandb(df, self.project, self.group)
            
        output_file = Path(self.output_dir) / "averaged_results.csv"
        save_to_csv(df, output_file)
        self.log.info(f"Averaged results saved to {output_file}")
        
        return


    def log_average_to_wandb(self, df: pd.DataFrame,
                project: Optional[str]=None, 
                group: Optional[str]=None,) -> None:

        if project:
            try:
                run = wandb.init(project=project, 
                                group="average_result",
                                name=f"averaged_result__{group}", 
                                dir=Path.cwd(), 
                                entity='UCVision',
                                settings=wandb.Settings(silent=True))
                
                df.columns = [":".join([name for name in col if name]) if isinstance(col, tuple) else str(col) for col in df.columns]
                df = df.drop(columns=["index"], errors="ignore")
                df = df.dropna(axis=1, how='all')
                
                for col in df.columns[df.columns.str.startswith("average_result")]:
                    df.loc[df[col].duplicated(), col] = np.nan
                
                run.log({f"average_result": wandb.Table(dataframe=df)})
                
                run.finish()
                self.log.info(f"Averaged results uploaded to wandb.")
                
            except wandb.errors.UsageError as e:
                self.log.error(f"Failed to upload results to wandb: {e}")
            except Exception as e:
                self.log.error(f"Unexpected error occurred while logging to wandb: {e}")




    
