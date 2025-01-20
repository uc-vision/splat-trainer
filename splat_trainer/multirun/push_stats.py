import time
import logging
from typing import Any, Optional

from fabric import Connection
from omegaconf import DictConfig
import redis
import socket
import traceback

from hydra.experimental.callback import Callback
from grafana_analytics.plugins.util import GraphitePusher



class PushStats(Callback):
    def __init__(self,
                 user: str,
                 docker_host: str,
                 remote_host: str,
                 local_port: int, 
                 remote_port: int,
                 redis_port: int,
                 result_key: str) -> None:
        self.user = user
        self.docker_host = docker_host
        self.remote_host = remote_host
        self.local_port = local_port
        self.remote_port = remote_port
        
        self.redis_host = socket.gethostname()
        self.redis_port = redis_port
        
        self.result_key = result_key
        
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None: 
        self.log.error(f"Start pushing training metrics to Graphite database...")
        try: 
            self.push_stats()
            self.log.info(f"Training metrics pushed to Graphite.")
            self.log.setLevel(logging.INFO)
            
        except Exception as e:
            self.log.error(f"Error occurred while pushing data to Graphite: {e}")
            self.log.error(f"Stack trace: {traceback.format_exc()}")

        return
    

    def push_stats(self, averaged_results: Optional[dict]=None):
        
        if not averaged_results:
            redis_client = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=True)
            averaged_results = redis_client.hgetall(self.result_key)

        graphite_pusher = GraphitePusher(hostname='127.0.0.1', port=self.local_port, prefix='training_stats.test.')

        with Connection(self.remote_host, user=self.user).forward_local(self.local_port, self.remote_port, self.docker_host):
            timestamp = int(time.time())
            for metric, value in averaged_results.items():
                graphite_pusher.push_stat(metric, float(value), timestamp)

            time.sleep(10)
        