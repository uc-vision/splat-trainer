from abc import ABCMeta, abstractmethod
from beartype import beartype
import torch
import wandb  
import logging
from pathlib import Path
from threading import Thread
from queue import Queue

class Logger(metaclass=ABCMeta):

  @abstractmethod
  def log_eval(self, name, ref_image:torch.Tensor, image:torch.Tensor, psnr:float):
    raise NotImplementedError


def numpy_image(tensor:torch.Tensor):
  return (tensor * 255).to(torch.uint8).cpu().numpy()

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
      self.run.log(data, step=step)
      item = self.queue.get()

    
  @beartype
  def log_eval(self, name, filename:str, psnr:float, step:int):
    table = wandb.Table(columns=["filename", "psnr"])
    table.add_data(filename, psnr)
    self.log({name:table}, step=step)


  @beartype
  def log(self, data:dict, step:int):
    self.queue.put((data, step))