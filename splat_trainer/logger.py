from abc import ABCMeta, abstractmethod
from functools import partial
from beartype.typing import Dict, List
from beartype import beartype
import torch
import wandb  
import logging
from pathlib import Path
from threading import Thread
from queue import Queue

import os

# os.environ["WANDB_SILENT"] = "true"

class Logger(metaclass=ABCMeta):

  @abstractmethod
  def log_table(self, name, rows:List[Dict], step):
    raise NotImplementedError
  
  @abstractmethod
  def log(self, data:dict, step:int):
    raise NotImplementedError
  
  @abstractmethod
  def close(self):
    raise NotImplementedError
  
  




class WandbLogger:
  def __init__(self, project:str | None, name:str | None, log_config:dict, dir:str | None = None):
    if dir is not None:
      dir = Path(dir).mkdir(parents=True, exist_ok=True)

    self.run = wandb.init(project=project, name=name, config=log_config, dir=dir)
    # logger = logging.getLogger("wandb")
    # logger.setLevel(logging.WARNING)

    self.queue = Queue()
    self.log_thread = Thread(target=self.log_loop)
    self.log_thread.start()


  def log_loop(self):
    item = self.queue.get()
    while item is not None:
      item()

      item = self.queue.get()


  def close(self):
    self.queue.put(None)
    self.log_thread.join()

    self.run.finish(quiet=True)
    

    
  @beartype
  def log_table(self, name, rows:List[Dict], step):
    table = wandb.Table(columns=list(rows[0].keys()))
    for row in rows:
      table.add_data(*row.values())
                        
    self.log({name:table}, step=step)


  @beartype
  def log(self, data:dict, step:int):
    self.queue.put(partial(self.run.log, data, step=step) )

  @beartype
  def log_image(self, name:str, image:torch.Tensor, caption:str | None = None, step:int = 0):
    
    def log():
      nonlocal image, caption, step
      
      image = (image * 255).to(torch.uint8).cpu().numpy()
      image = wandb.Image(image, mode="RGB", caption=caption)

      self.run.log({name : image}, step=step)

    self.queue.put(log)
