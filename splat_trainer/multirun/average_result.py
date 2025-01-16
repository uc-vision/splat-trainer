import argparse
import logging
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import psutil
import redis
import rq
import socket
import traceback
import wandb
from filelock import FileLock
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf

from splat_trainer.multirun.deploy import kill_rq_worker_by_name



class AverageResult(Callback):
    def __init__(self, output_dir: str, sweep_params: DictConfig) -> None:
        self.output_dir = output_dir
        self.results_file = os.path.join(output_dir, "results.json")
        self.failed_jobs_file = os.path.join(output_dir, "failed_jobs.json")
        self.params = sweep_params
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.redis_host = socket.gethostname()
        self.start_time = None
        
    def _get_redis_client(self) -> redis.Redis:
        return redis.Redis(host=self.redis_host, port=6379, decode_responses=True)
        
    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        redis_client = self._get_redis_client()
        project = OmegaConf.select(config, 'project')
        redis_client.set('project', project)
        self.start_time = time.time()
        
        # job = rq.get_current_job()
        # hostname = socket.gethostname()
        
        # job.descripton = hostname + '__' + job.description
        # job.save()
    
    
    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        end_time = time.time()
        if self.start_time is not None:
            runtime = end_time - self.start_time
        else:
            runtime = None
        job_num = config.hydra.job.num
        params = {k: v for k, v in (override.split('=') 
                    for override in OmegaConf.select(config, "hydra.overrides.task"))}
        hostname = socket.gethostname()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S" )
        job = rq.get_current_job()

        if job_return.status == JobStatus.COMPLETED:
            self.log.info(f"Job {job_num} succeeded with return value: {job_return.return_value}. Saving job result...")

            assert isinstance(job_return.return_value, dict), f"job return value must be a dictionary, but got type {type(job_return.return_value)}"
            result_data = {
                "job_num": job_num,
                "params": params,
                "result": job_return.return_value,
                "hostname": hostname,
                "timestamp": timestamp,
                "status":"succeed",
            }
            
            result_data["result"]["runtime"] = runtime

            self.save_to_json(self.results_file, result_data)
            self.log.info(f"Job {job_num} result has been successfully saved to {self.results_file}.\n")


        else:
            self.log.error(f"Job {job_num} failed with error: {job_return._return_value}")
            job = rq.get_current_job()
            job_data = {
                "job_num": job_num,
                "job_id": job.id,
                "params": params,
                "status": 'failed',
                "error_info": {
                    "error_type": type(job_return._return_value).__name__,
                    "error_message": str(job_return._return_value),
                },
                "hostname": hostname,
                "timestamp": timestamp
            }

            self.save_to_json(self.failed_jobs_file, job_data)
            self.log.info(f"Job {job_num} has failed and the details have been saved to {self.failed_jobs_file}.\n")
            
            if "Permission" in str(job_return._return_value):
                kill_rq_worker_by_name()



    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None: 
        redis_client = self._get_redis_client() 
        project = redis_client.get('project')
        base_path = Path(__file__).parents[2]

        try:
            average_results(project, self.log, base_path / self.output_dir, redis_client)

        except Exception as e:
            self.log.error(f"Error occurred while averaging the results: {e}")
            self.log.error(f"Stack trace: {traceback.format_exc()}")
            return


    @staticmethod
    def load_data(file: Union[Path, str]) -> dict:
        if os.path.exists(file):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
            except Exception as e:
                raise RuntimeError(f"An unexpected error occurred while reading file '{file}': {e}")
        else:
            data = {} 
            
        return data


    @staticmethod
    def dump_data(file: Union[Path, str], data: dict) -> None:
        try:
            with open(file, "w") as f:
                json.dump(data, f, indent=4)

        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while writing to file '{file}': {e}")


    def save_to_json(self, file: str, data: dict) -> None:
        job_num = data['job_num']
        
        lockfile = file + ".lock"
        lock = FileLock(lockfile)

        with lock: 
            all_data = self.load_data(file)
            all_data[job_num] = data
            self.dump_data(file, all_data)


def average_results(project: Optional[str]=None, 
                    log: Optional[logging.Logger]=None, 
                    file_path: Union[Path, str] = None,
                    redis_client: Optional[redis.Redis]=None) -> None:
    if not log:
        log = logging.getLogger(__name__)

    if not file_path:
        file_path = get_args().path
        
    result_dict = AverageResult.load_data(Path(file_path) / "results.json")
    
    if not result_dict:
        print(f"Cannot find 'results.json' file under {file_path}.")
        return
    
    metric_results = defaultdict(list)
    test_scene_set = set()

    for job_num, result_data in result_dict.items():
        params = result_data["params"]
        result = result_data['result']
        param_values = tuple(value for param, value in params.items() if param != 'test_scene')
        param_names = tuple(param for param in params.keys() if param != 'test_scene')
        test_scene = params.get('test_scene')
        test_scene_set.add(test_scene)
        for metric, value in result.items():
            metric_results[metric].append((param_values, job_num, test_scene, value))

    log.info(f"Averaging {len(result_dict)} training results across {len(test_scene_set)} scenes...")

    for metric, metric_list in metric_results.items():
        updated_metric_list = []
        results = defaultdict(list)
        for param_values, job_num, test_scene, value in metric_list:
            results[param_values].append((job_num, test_scene, value))

        for k in results:
            results[k].sort(key=lambda x: x[1])

        for param_values, job_list in results.items():
            
            assert len(job_list) == len(test_scene_set), "Missing jobs."
            total_value = sum(value for _, _, value in job_list)
            avg_value = total_value / len(job_list)

            updated_job_list = [(param_values, job_num, test_scene, value, avg_value) for job_num, test_scene, value in job_list]
            updated_metric_list += updated_job_list
        updated_metric_list.sort(key=lambda x: x[-1], reverse=True if 'psnr' or 'ssim' in metric else False)
        metric_results[metric] = updated_metric_list


    log_average_results = {}
    for metric, values in metric_results.items():
        log_average_results[metric] = values[0][-1]
    redis_client.hset('multirun_result:1', mapping=log_average_results)

  
    averaged_results = []
    for metric, job_list in metric_results.items():
        for param_values, job_num, test_scene, value, avg_value in job_list:
            averaged_results.append((job_num, *param_values, test_scene, value, avg_value, metric))
              
    df = pd.DataFrame(averaged_results, columns=['job_num', *param_names, 'test_scene', 'value', 'avg_value', 'metric'])

    for col in [*param_names]:
        df.loc[df.index % len(test_scene_set) != 0, col] = ''
        
    if project:
        try:
            name = f"averaged_result_{int(time.time())}"
            run = wandb.init(project=project, 
                             name=name, 
                             dir=Path.cwd(), 
                             group=f'metrics_average', 
                             entity='UCVision',
                             settings=wandb.Settings(silent=True))
            
            for metric_name, metric_df in df.groupby("metric"):
                metric_df = metric_df.iloc[:, :-1]
                result_table = wandb.Table(dataframe=metric_df)
                run.log({f"{metric_name}_average": result_table})
            
            run.finish()
            log.info(f"Averaged results uploaded to wandb.")
            
        except wandb.errors.UsageError as e:
            log.error(f"Failed to upload results to wandb: {e}")
        except Exception as e:
            log.error(f"Unexpected error occurred while logging to wandb: {e}")
        
    for col in ['metric']:
        df.loc[df[col].duplicated(), col] = ''
    
    for col in ['avg_value']:
        df.loc[df[col].duplicated(), col] = np.nan
        
    output_file = Path(file_path) / "averaged_results.csv"
    df.to_csv(output_file, index=False)
    log.info(f"Averaged results saved to {output_file}")
    
    return



def get_args():
  args = argparse.ArgumentParser(description='Average training results across the scenes.')
  args.add_argument('path', type=str, help='Path to the result file')

  return args.parse_args()



if __name__ == "__main__":
    average_results()