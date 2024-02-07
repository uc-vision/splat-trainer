from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch

from torch.utils.tensorboard import SummaryWriter


@dataclass 
class TrainConfig:
  model_path: str
  device: str
  load_model: Optional[str] = None



class Trainer:
  def __init__(self, dataset, config):

    self.device = torch.device(config.device)
    self.model_path = Path(config.model_path)

    self.dataset = dataset
    self.logger = SummaryWriter(config.model_path)


  def train(self):

    print("Using model path", self.model_path)
    self.model_path.mkdir(parents=True, exist_ok=True)

    pass