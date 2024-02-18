from abc import ABCMeta, abstractmethod
import torch
import wandb  

class Logger(metaclass=ABCMeta):

  @abstractmethod
  def log_eval(self, name, ref_image:torch.Tensor, image:torch.Tensor, psnr:float):
    raise NotImplementedError


class WandbLogger:
  def __init__(self, project:str | None, name:str | None, log_config:dict):
    self.run = wandb.init(project=project, name=name, config=log_config)


  def log_eval(self, name, filename, ref_image:torch.Tensor, image:torch.Tensor, psnr:float):
    table = wandb.Table(columns=["filename", "image", "rendering", "psnr"])
    table.add_data(filename, ref_image, image, psnr)

    wandb.log({name:table})