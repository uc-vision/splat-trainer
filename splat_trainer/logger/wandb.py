from functools import partial
import json
from numbers import Number
from beartype.typing import Dict, List
from beartype import beartype
import torch
import wandb  
from pathlib import Path
from threading import Thread
from queue import Queue

from splat_trainer.config import Progress
from splat_trainer.logger.histogram import Histogram
from splat_trainer.util.pointcloud import PointCloud

from .logger import Logger

class WandbLogger(Logger):

  @beartype
  def __init__(self, project:str | None, 
               entity:str | None, 
               name:str | None=None):
    
    dir = Path.cwd()
    settings = wandb.Settings(start_method='thread', quiet=True)

    self.run = wandb.init(project=project, name=name, 
                          dir=dir, entity=entity, settings=settings)
    
  

    self.queue = Queue()
    self.log_thread = Thread(target=self.worker)
    self.log_thread.start()
    self.current_step = 0

  
  def step(self, progress:Progress):
    self.current_step = progress.step

  def worker(self):
    item = self.queue.get()
    while item is not None:
      item()
      item = self.queue.get()


  def close(self):
    self.queue.put(None)
    self.log_thread.join()

    self.run.finish()
    

  @beartype
  def log_config(self, config:Dict):
    self.run.config.update(config)

  
  @beartype
  def log_evaluations(self, name, rows:Dict[str, Dict]):
    first_row = next(iter(rows.values()))
    columns = list(first_row.keys())

    table = wandb.Table(columns=["filename"] + columns)
    for k, row in rows.items():
      table.add_data(k, *row.values())
                        
    self.log_data({name:table})


  @beartype
  def log_data(self, data:dict):
    self.queue.put(partial(self.run.log, data, step=self.current_step) )

  @beartype
  def log_value(self, name:str, value:Number):
    self.log_data({name:value})

  @beartype
  def log_values(self, name:str, data:dict):
    self.log_data({f"{name}/{k}":value for k, value in data.items()})


  @beartype
  def log_image(self, name:str, image:torch.Tensor, compressed:bool = True, caption:str | None = None):
    
    def log(image:torch.Tensor, step:int):
      
      image = (image * 255).to(torch.uint8).cpu().numpy()
      image = wandb.Image(image, mode="RGB", caption=caption, file_type="jpg" if compressed else "png")
      self.run.log({name : image}, step=step)

    self.queue.put(partial(log, image, self.current_step))


  def log_cloud(self, name:str, points:PointCloud):

    def log(points:PointCloud, step:int):
      data = torch.cat([points.points, points.colors * 255], dim=1).cpu().numpy()

      image = wandb.Object3D(data)
      self.run.log({name : image}, step=step)

    self.queue.put(partial(log, points, self.current_step))


  @beartype
  def log_histogram(self, name:str, values:torch.Tensor | Histogram):
    def log(values:torch.Tensor | Histogram, step:int):
      try:
        if isinstance(values, Histogram):
          counts_norm = values.counts / values.counts.sum()

          hist = wandb.Histogram(np_histogram=
            (counts_norm.cpu().numpy(), values.bins.cpu().numpy()))
          
          self.run.log({name:hist}, step=step)

        elif isinstance(values, torch.Tensor):
          self.run.log({name:wandb.Histogram(values.cpu().numpy())}, step=step)
      except Exception as e:
        print(f"Error logging histogram {name}: {e}")

    self.queue.put(partial(log, values, self.current_step))


  @beartype
  def log_json(self, name:str, data:dict):
    self.log_data({name:json.dumps(data, indent=2)})
