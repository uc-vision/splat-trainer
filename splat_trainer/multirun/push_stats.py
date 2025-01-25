import time
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
import traceback
from typing import Any, Optional

from fabric import Connection
from grafana_analytics.plugins.util import GraphitePusher
from hydra.experimental.callback import Callback
from omegaconf import DictConfig
import redis
import socket



log = logging.getLogger(__name__)


class PushStats(Callback):
    def __init__(self,
                 user: str,
                 docker_host: str,
                 remote_host: str,
                 local_port: int, 
                 remote_port: int,
                 redis_port: int,
                 redis_db_num: int,
                 result_key: str) -> None:
        self.user = user
        self.docker_host = docker_host
        self.remote_host = remote_host
        self.local_port = local_port
        self.remote_port = remote_port
        
        self.redis_host = socket.gethostname()
        self.redis_port = redis_port
        self.redis_db_num = redis_db_num
        
        self.result_key = result_key
        
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        

        
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None: 
        redis_client = redis.Redis(host=self.redis_host, port=self.redis_port, db=self.redis_db_num, decode_responses=True)
        averaged_results = redis_client.hgetall(self.result_key)
        assert averaged_results, "Error: result data not found or empty in Redis."

        self.push_stats(averaged_results)

        return
    
    @retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
    def push_stats(self, averaged_results: Optional[dict]=None):
        
        self.log.info(f"Start pushing training result {averaged_results} to Graphite database...")
        
        try:
            graphite_pusher = GraphitePusher(hostname='127.0.0.1', port=self.local_port, prefix='training_stats.test.')

            with Connection(self.remote_host, user=self.user).forward_local(self.local_port, self.remote_port, self.docker_host):
                timestamp = int(time.time())
                for metric, value in averaged_results.items():
                    graphite_pusher.push_stat(metric, float(value), timestamp)

                time.sleep(10)
                
            self.log.info(f"Training metrics pushed to Graphite.")
                
        except Exception as e:
            log.error(f"Failed pushing data to Graphite database with error: {e}")
            self.log.error(f"Stack trace: {traceback.format_exc()}")
            raise

    
    