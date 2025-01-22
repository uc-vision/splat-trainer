import logging
import os
import time
from datetime import datetime
from typing import Any

import socket
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf

from splat_trainer.multirun.deploy import shutdown_workers_on_host
from splat_trainer.multirun.util import save_to_json



class LogJobResult(Callback):
    def __init__(self, output_dir: str, sweep_params: DictConfig, redis_port: int, redis_db: int) -> None:
        self.output_dir = output_dir
        self.sweep_params = sweep_params
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_host = socket.gethostname()
        
        
    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        
        self.params = {".".join(k.lstrip('+').split(".")[-2:]): v for k, v in (override.split('=') 
                    for override in OmegaConf.select(config, "hydra.overrides.task")) if k in self.sweep_params}
                
        self.start_time = time.time()
    
    
    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
     
        job_num = config.hydra.job.num
        hostname = socket.gethostname()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S" )

        if job_return.status == JobStatus.COMPLETED:
            self.log.info(f"Job {job_num} succeeded with return value: {job_return.return_value}. Saving job result...")

            assert  isinstance(job_return.return_value, (dict, float, int)), f"job return value must be a dictionary or a float/int, but got type {type(job_return.return_value)}"
            
            result_data = {
                "group": OmegaConf.select(config, 'logger.group'),
                "job_num": job_num,
                "params": self.params,
                "result": job_return.return_value if isinstance(job_return.return_value, dict) else {"metric": job_return.return_value},
            }
            
            result_data["result"]["runtime"] = time.time() - self.start_time if self.start_time else None
            
            results_file = os.path.join(self.output_dir, "results.json")
            save_to_json(results_file, result_data)
            self.log.info(f"Job {job_num} result has been successfully saved to {results_file}.\n")


        else:
            self.log.error(f"Job {job_num} failed with error: {job_return._return_value}")
            job_data = {
                "job_num": job_num,
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
                redis_url = f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
                shutdown_workers_on_host(redis_url, hostname)

            failed_jobs_file = os.path.join(self.output_dir, "failed_jobs.json")
            save_to_json(failed_jobs_file, job_data)
            self.log.info(f"Job {job_num} has failed and the details have been saved to {failed_jobs_file}.\n")