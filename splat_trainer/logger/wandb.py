from functools import partial
from numbers import Number
from beartype.typing import Dict, List
from beartype import beartype
import torch
import wandb  
from pathlib import Path
from threading import Thread
from queue import Queue

from splat_trainer.logger.histogram import Histogram

from .logger import Logger

class WandbLogger(Logger):
  def __init__(self, project:str | None, name:str | None, log_config:dict, dir:str | None = None):
    if dir is not None:
      dir = Path(dir).mkdir(parents=True, exist_ok=True)
    self.run = wandb.init(project=project, name=name, config=log_config, dir=dir)

    self.queue = Queue()
    self.log_thread = Thread(target=self.worker)
    self.log_thread.start()


  def worker(self):
    item = self.queue.get()
    while item is not None:
      item()
      item = self.queue.get()


  def close(self):
    self.queue.put(None)
    self.log_thread.join()

    self.run.finish(quiet=True)
    
    
  @beartype
  def log_evaluations(self, name, rows:List[Dict], step):
    table = wandb.Table(columns=list(rows[0].keys()))
    for row in rows:
      table.add_data(*row.values())
                        
    self.log_data({name:table}, step=step)


  @beartype
  def log_data(self, data:dict, step:int):
    self.queue.put(partial(self.run.log, data, step=step) )

  @beartype
  def log_value(self, name:str, value:Number, step:int):
    self.log_data({name:value}, step=step)

  @beartype
  def log_values(self, name:str, data:dict, step:int):
    self.log_data({f"{name}/{k}":value for k, value in data.items()}, step=step)


  @beartype
  def log_image(self, name:str, image:torch.Tensor, step:int = 0):
    
    def log():
      nonlocal image, step
      
      image = (image * 255).to(torch.uint8).cpu().numpy()
      image = wandb.Image(image, mode="RGB")
      self.run.log({name : image}, step=step)

    self.queue.put(log)


  @beartype
  def log_histogram(self, name:str, values:torch.Tensor | Histogram, step:int):
    if isinstance(values, Histogram):
      hist = wandb.Histogram(np_histogram=
        (values.counts.cpu().numpy(), values.bins.cpu().numpy()))
      
      self.log_data({name:hist}, step=step)

    elif isinstance(values, torch.Tensor):
      self.log_data({name:wandb.Histogram(values)}, step=step)

  