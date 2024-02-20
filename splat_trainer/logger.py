from abc import ABCMeta, abstractmethod
from beartype.typing import Dict, List
from beartype import beartype
import torch
import wandb  
import logging
from pathlib import Path
from threading import Thread
from queue import Queue
import cv2

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
  
  


def numpy_image(tensor:torch.Tensor, caption:str | None = None):
  image = (tensor * 255).to(torch.uint8).cpu().numpy()
  cv2.imshow("image", image)
  cv2.waitKey(0)


  return wandb.Image(image, mode="RGB", caption=caption)

class WandbLogger:
  def __init__(self, project:str | None, name:str | None, log_config:dict, dir:str | None = None):
    if dir is not None:
      dir = Path(dir).mkdir(parents=True, exist_ok=True)

    self.run = wandb.init(project=project, name=name, config=log_config, dir=dir)
    logger = logging.getLogger("wandb")
    logger.setLevel(logging.WARNING)

    self.queue = Queue()
    self.log_thread = Thread(target=self.log_loop)
    self.log_thread.start()


  def log_loop(self):
    item = self.queue.get()
    while item is not None:
      data, step = item

      self.run.log(data, step=step, commit=True)
      item = self.queue.get()


  def close(self):
    self.queue.put(None)
    self.log_thread.join()
    self.run.finish()
    

    
  @beartype
  def log_table(self, name, rows:List[Dict], step):
    table = wandb.Table(columns=list(rows[0].keys()))
    for row in rows:
      table.add_data(*row.values())
                        
    self.log({name:table}, step=step)


  @beartype
  def log(self, data:dict, step:int):
    self.queue.put((data, step))

