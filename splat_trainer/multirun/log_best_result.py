import logging
from pathlib import Path
from typing import Any

import pandas as pd
import redis
import socket
import wandb
from omegaconf import DictConfig, OmegaConf
from hydra.experimental.callback import Callback



log = logging.getLogger(__name__)


class LogBestResult(Callback):
    def __init__(self, redis_port: int) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.redis_host = socket.gethostname()
        self.redis_port = redis_port


    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None: 
        self.project = config.project

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None: 
        pass
                
        # TODO: track best result and log to wandb