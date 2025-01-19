import argparse
import logging
import json
import os
import time
from datetime import datetime
from pathlib import Path
import traceback
from typing import Any, Optional, Union

import pandas as pd
import redis
import rq
import socket
import wandb
from filelock import FileLock
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf

from splat_trainer.multirun.deploy import kill_rq_worker_by_name


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)



class AverageResult(Callback):
    def __init__(self, output_dir: str, sweep_params: DictConfig) -> None:
        self.output_dir = output_dir
        self.sweep_params = sweep_params
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.redis_host = socket.gethostname()
        
        
    def _get_redis_client(self) -> redis.Redis:
        return redis.Redis(host=self.redis_host, port=6379, decode_responses=True)
        
        
    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        redis_client = self._get_redis_client()
        project = OmegaConf.select(config, 'project')
        group = OmegaConf.select(config, 'logger.group')
        
        redis_client.set('project', str(project))
        redis_client.set('group', str(group))
        
        self.params = {".".join(k.lstrip('+').split(".")[-2:]): v for k, v in (override.split('=') 
                    for override in OmegaConf.select(config, "hydra.overrides.task")) if k in self.sweep_params}
                
        self.start_time = time.time()
        
    
    
    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
     
        job_num = config.hydra.job.num
        hostname = socket.gethostname()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S" )

        if job_return.status == JobStatus.COMPLETED:
            self.log.info(f"Job {job_num} succeeded with return value: {job_return.return_value}. Saving job result...")

            assert isinstance(job_return.return_value, dict), f"job return value must be a dictionary, but got type {type(job_return.return_value)}"
            
            job_return.return_value["runtime"] = time.time() - self.start_time if self.start_time else None
            
            result_data = {
                "group": OmegaConf.select(config, 'logger.group'),
                "job_num": job_num,
                "test_scene": self.params.pop('test_scene', OmegaConf.select(config, 'test_scene')),
                "params": self.params,
                "result": job_return.return_value,
            }

            results_file = os.path.join(self.output_dir, "results.json")
            self.save_to_json(results_file, result_data)
            self.log.info(f"Job {job_num} result has been successfully saved to {results_file}.\n")


        else:
            self.log.error(f"Job {job_num} failed with error: {job_return._return_value}")
            job = rq.get_current_job()
            job_data = {
                "job_num": job_num,
                "job_id": job.id,
                "params": self.params,
                "error_info": {
                    "error_type": type(job_return._return_value).__name__,
                    "error_message": str(job_return._return_value),
                },
                "hostname": hostname,
                "timestamp": timestamp,
                "status": 'failed'
            }
            
            if "Permission" in str(job_return._return_value):
                kill_rq_worker_by_name()

            failed_jobs_file = os.path.join(self.output_dir, "failed_jobs.json")
            self.save_to_json(failed_jobs_file, job_data)
            self.log.info(f"Job {job_num} has failed and the details have been saved to {failed_jobs_file}.\n")



    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None: 
        redis_client = self._get_redis_client() 
        project = redis_client.get('project')
        group = redis_client.get('group')

        try:
            result, df = average_results(self.output_dir)
            self.log.info(f"Best result: {result}")

        except Exception as e:
            self.log.error(f"Error occurred while averaging the results: {e}")
            self.log.error(f"Stack trace: {traceback.format_exc()}")
        
        redis_client.hset('multirun_result:1', mapping=result)
        
        self.log_wandb(df, project, group)
        self.save_to_csv(df, self.output_dir)
        
        return


    @staticmethod
    def load_data(file: Union[Path, str]) -> dict:
        data = []
        if os.path.exists(file):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
            except Exception as e:
                raise RuntimeError(f"An unexpected error occurred while reading file '{file}': {e}")
            
        return data


    @staticmethod
    def dump_data(file: Union[Path, str], data: dict) -> None:
        try:
            with open(file, "w") as f:
                json.dump(data, f, indent=4)

        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while writing to file '{file}': {e}")


    def save_to_json(self, file: str, data: dict) -> None:  
        lockfile = file + ".lock"
        lock = FileLock(lockfile)

        with lock: 
            all_data = self.load_data(file)
            all_data.append(data)
            self.dump_data(file, all_data)
            
            
    @staticmethod
    def save_to_csv(df: pd.DataFrame, file_path: str):
        output_file = Path(file_path) / "averaged_results.csv"
        df.to_csv(output_file, index=False)
        log.info(f"Averaged results saved to {output_file}")
        
    
    @staticmethod
    def log_wandb(df: pd.DataFrame,
                project: Optional[str]=None, 
                group: Optional[str]=None,):

        if project:
            try:
                run = wandb.init(project=project, 
                                group=group,
                                name=f"averaged_result__{group}", 
                                dir=Path.cwd(), 
                                entity='UCVision',
                                settings=wandb.Settings(silent=True))
                
                df.columns = [".".join(map(str, filter(pd.notna, col))) if isinstance(col, tuple) else str(col) for col in df.columns]
                df = df.drop(columns=["index"], errors="ignore")
                run.log({f"average_result": wandb.Table(dataframe=df)})
                
                run.finish()
                log.info(f"Averaged results uploaded to wandb.")
                
            except wandb.errors.UsageError as e:
                log.error(f"Failed to upload results to wandb: {e}")
            except Exception as e:
                log.error(f"Unexpected error occurred while logging to wandb: {e}")
            
    
def average_results(file_path: Path | str | None):
    
    if not file_path:
        file_path = get_args().path
        
        
    result = AverageResult.load_data(Path(file_path) / "results.json")
    
    df = pd.json_normalize(result, sep=':')
    df.columns = pd.MultiIndex.from_tuples([col.split(":") for col in df.columns])
    
    param_columns = [("params", col) for col in df.get("params", pd.DataFrame()).columns.tolist()]
    result_columns = [("result", col) for col in df.get("result", pd.DataFrame()).columns.tolist()]
    
    if param_columns:
        df = df.set_index(param_columns)
        result_avg = df.groupby(level=[0, 1])[result_columns].mean()
    else:
        result_avg = df[result_columns].mean().to_frame().T

    result_avg.columns = pd.MultiIndex.from_product([["average_result"], result_avg['result'].columns])
    
    df = df.join(result_avg).reset_index()
    df.sort_values(by=('average_result', 'train_psnr'), ascending=False)

    result = result_avg.iloc[0]['average_result'].to_dict()
    
    return result, df

    
def get_args():
  args = argparse.ArgumentParser(description='Average training results across the scenes.')
  args.add_argument('path', type=str, help='Path to the result file')

  return args.parse_args()




if __name__ == "__main__":
    args = get_args()
    result, df = average_results(args.path)
    AverageResult.save_to_csv(df, args.path)