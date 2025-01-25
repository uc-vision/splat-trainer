import logging
from typing import Any

import socket
from hydra.experimental.callback import Callback
from omegaconf import DictConfig

from splat_trainer.multirun.deploy import flush_db



class FlushDB(Callback):
    def __init__(self, redis_port: int, redis_db_num: int) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        redis_host = socket.gethostname()
        self.redis_url = f"redis://{redis_host}:{redis_port}/{redis_db_num}"
        
        
    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None: 
        flush_db(self.redis_url)
        

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None: 
        flush_db(self.redis_url)
        