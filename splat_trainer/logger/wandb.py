from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property, partial
import json
from numbers import Number
from queue import PriorityQueue, Queue
from threading import Thread
from typing import Callable, Optional
from beartype.typing import Dict, List
from beartype import beartype
import torch
import wandb  
from pathlib import Path

from splat_trainer.config import Progress
from splat_trainer.logger.histogram import Histogram
from splat_trainer.util.pointcloud import PointCloud

from .logger import Logger



@dataclass(order=True)
class LogItem:
    step: int
    item: Callable = field(compare=False)


  
class WandbLogger(Logger):

  @beartype
  def __init__(self, project:str | None, 
               entity:str | None, 
               name:str | None=None,
               workers:int=16):
    
    dir = Path.cwd()
    settings = wandb.Settings(start_method='thread', quiet=True)

    self.run = wandb.init(project=project, name=name, 
                          dir=dir, entity=entity, settings=settings)
    
    self.queue = PriorityQueue()
    self.thread = Thread(target=self.worker)

    self.executor = ThreadPoolExecutor(max_workers=workers)
    self.thread.start()
    self.current_step = 0

  
  def step(self, progress:Progress):
    self.current_step = progress.step

  def worker(self):
    while True:
      item:LogItem = self.queue.get()
      if self.queue.qsize() > 20:
        print(f"queue size: {self.queue.qsize()}")

      data = item.item()
      if data is None:
        break
      self.run.log(data, step=item.step)


  def enqueue_future(self, f:Callable):
    future = self.executor.submit(f)
    self.queue.put(LogItem(self.current_step, lambda: future.result()))

  def close(self):
    self.queue.put(LogItem(self.current_step, lambda: None))
    self.thread.join()
    self.run.finish()
    

  @beartype
  def log_config(self, config:Dict):
    self.run.config.update(config)

  
  @beartype
  def log_evaluations(self, name, rows:Dict[str, Dict]):
    def f():
      first_row = next(iter(rows.values()))
      columns = list(first_row.keys())

      table = wandb.Table(columns=["filename"] + columns)
      for k, row in rows.items():
        table.add_data(k, *row.values())

      return {name:table}
                          
    self.enqueue_future(f)


  @beartype
  def log_data(self, data:dict):
    self.queue.put(LogItem(self.current_step, lambda: data))

  @beartype
  def log_value(self, name:str, value:Number):
    self.log_data({name:value})

  @beartype
  def log_values(self, name:str, data:dict):
    self.log_data({f"{name}/{k}":value for k, value in data.items()})


  @beartype
  def log_image(self, name:str, image:torch.Tensor, compressed:bool = True, caption:str | None = None):
    
    def f(image:torch.Tensor):
      
      image = (image * 255).to(torch.uint8).cpu().numpy()
      image = wandb.Image(image, mode="RGB", caption=caption, file_type="jpg" if compressed else "png")
      
      return {name : image}

    self.enqueue_future(partial(f, image.detach()))


  def log_cloud(self, name:str, points:PointCloud):

    def f(points:PointCloud):
      data = torch.cat([points.points, points.colors * 255], dim=1).cpu().numpy()

      obj = wandb.Object3D(data)
      return {name : obj}

    self.enqueue_future(partial(f, points.detach()))


  @beartype
  def log_histogram(self, name:str, values:torch.Tensor | Histogram, num_bins:Optional[int] = None):

    def f(values:torch.Tensor | Histogram):
      try:
        if isinstance(values, Histogram):
          counts_norm = values.counts / values.counts.sum()

          hist = wandb.Histogram(np_histogram=
            (counts_norm.cpu().numpy(), values.bins.cpu().numpy()))
          
          return {name:hist}

        elif isinstance(values, torch.Tensor):
          return {name:wandb.Histogram(values.cpu().numpy(), num_bins=num_bins or 64)}
      except Exception as e:
        print(f"Error logging histogram {name}: {e}")

    if isinstance(values, torch.Tensor):
      values = values.detach()

    self.enqueue_future(partial(f, values))


  @beartype
  def log_json(self, name:str, data:dict):
    self.enqueue_future(lambda: {name:json.dumps(data, indent=2)})
