from functools import partial
import json
from pprint import pformat
import subprocess
from typing import Optional
from beartype.typing import Dict, List
from beartype import beartype
import torch

from pathlib import Path
from threading import Thread
from queue import Queue

from torch.utils.tensorboard import SummaryWriter
import tabulate

from splat_trainer.config import Progress
from splat_trainer.logger.histogram import Histogram
from splat_trainer.util.pointcloud import PointCloud

from .logger import Logger


class TensorboardLogger(Logger):
  def __init__(self, 
               flush_secs:int = 30,
               start_server:bool = False):
    
    dir = Path.cwd() / "log"
    dir.mkdir(parents=True, exist_ok=True)  

    self.writer = SummaryWriter(dir, flush_secs=flush_secs) 

    self.queue = Queue()
    self.log_thread = Thread(target=self.worker)
    self.log_thread.start()

    if start_server:
      self.process = subprocess.Popen(["tensorboard", "--logdir", self.writer.log_dir]) 
    else:
      self.process = None

    self.current_step = 0

  def step(self, progress:Progress):
    self.current_step = progress.step


  def log_config(self, config: Dict):
    config_str = pformat(config)
    self.enqueue(self.writer.add_text, "config", config_str, global_step=self.current_step)


  def worker(self):
    item = self.queue.get()
    while item is not None:
      item()
      item = self.queue.get()


  def close(self):
    print("Ending logging thread...")
    self.queue.put(None)
    self.log_thread.join()

    print("Done, closing writer...")

    self.writer.close()

    if self.process is not None:
      self.process.terminate()
    
  @beartype
  def log_evaluations(self, name, rows:Dict[str, Dict]):
    first = next(iter(rows.values()))
    headers = ["filename"] + list(first.keys())
    rows = [[k] + list(row.values()) for k, row in rows.items()]
    table = tabulate.tabulate(rows, headers, tablefmt="pipe", floatfmt=".3f")

    self.enqueue(self.writer.add_text, name, table, global_step=self.current_step)

  def enqueue(self, func, *args, **kwargs):
    self.queue.put(partial(func, *args, **kwargs))

  @beartype
  def log_values(self, name:str, data:dict):
    self.enqueue(self.writer.add_scalars, name, data, global_step=self.current_step) 

  @beartype
  def log_value(self, name:str, value:float):
    self.enqueue(self.writer.add_scalar, name, value, global_step=self.current_step) 


  @beartype
  def log_image(self, name:str, image:torch.Tensor, compressed:bool = True, caption: str | None = None):
    self.enqueue(self.writer.add_image, 
                 tag=name, img_tensor=image, 
                 dataformats="HWC",
                 global_step=self.current_step)
    
  
  def log_cloud(self, name:str, points:PointCloud):
    pass # Not supported by tensorboard

  @beartype
  def log_histogram(self, name:str, values:torch.Tensor | Histogram, num_bins:Optional[int] = None):
    if isinstance(values, torch.Tensor):
      if num_bins is not None:
        self.enqueue(self.writer.add_histogram, name, values, global_step=self.current_step, bins=num_bins)
      else:
        self.enqueue(self.writer.add_histogram, name, values, global_step=self.current_step)
    elif isinstance(values, Histogram):
      self.enqueue(write_histogram, self.writer, name, values, self.current_step)  


  @beartype
  def log_json(self, name:str, data:dict):
    self.enqueue(self.writer.add_text, name, json.dumps(data, indent=2), global_step=self.current_step)

def write_histogram(writer:SummaryWriter, name:str, hist:Histogram, step:int):
  writer.add_histogram_raw(
    name, 
    min=hist.range[0], max=hist.range[1], 
    num=hist.counts.sum().item(),
    sum=hist.sum,
    sum_squares=hist.sum_squares,
    bucket_limits=hist.bins[1:],
    bucket_counts=hist.counts,  
    global_step=step)


    