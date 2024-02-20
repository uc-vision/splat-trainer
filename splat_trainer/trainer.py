from dataclasses import dataclass
import os
from pathlib import Path
from beartype.typing import Optional
import torch


from taichi_splatting import perspective
from tqdm import tqdm

from splat_trainer.dataset import Dataset
from splat_trainer.logger import Logger

from splat_trainer.scene.gaussians import GaussianScene, LearningRates



@dataclass 
class TrainConfig:
  output_path: str
  device: str
  load_model: Optional[str] = None
  iterations: int = 30000
  learning_rates: LearningRates = LearningRates()

  eval_iterations: int = 1000

  num_neighbors: int = 3
  initial_alpha: float = 0.5
  sh_degree: int = 2

  num_logged_images: int = 5


class Trainer:
  def __init__(self, dataset:Dataset, config:TrainConfig, logger:Logger):

    self.device = torch.device(config.device)

    self.dataset = dataset
    self.config = config
    self.step = 0

    self.logger = logger

    self.camera_poses = dataset.camera_poses().to(self.device)
    self.camera_projection = dataset.camera_projection().to(self.device)

    if config.load_model:
      print("Loading model from", config.load_model)
      self.scene = GaussianScene.load_model(
        config.load_model, lr=config.learning_rates)
    else:
      pcd = dataset.pointcloud()
      print(f"Initializing model from {dataset}")

      self.scene = GaussianScene.from_pointcloud(pcd, lr=config.learning_rates,
                                        num_neighbors=config.num_neighbors,
                                        initial_alpha=config.initial_alpha,
                                        sh_degree=config.sh_degree)
      
      print(self.scene)
      
    self.scene.to(self.device)
    

  def camera_params(self, cam_idx:torch.Tensor, image:torch.Tensor):
      near, far = self.dataset.depth_range

      return perspective.CameraParams(
          T_camera_world=self.camera_poses(cam_idx.unsqueeze(0)),
          T_image_camera=self.camera_projection[cam_idx[1]],
          image_size=(image.shape[1], image.shape[0]),
          near_plane=near,
          far_plane=far
      ).to(self.device)


  def evaluate(self, name, data, limit_log_images:Optional[int]=None):
    def compute_psnr(a, b):
      return -10 * torch.log10(1 / torch.nn.functional.mse_loss(a, b))  
    
    total_psnr = 0.
    n = 0

    with torch.no_grad():
      pbar = tqdm(total=len(data), desc=f"rendering {name}", leave=False)
      for filename, image, camera_params in self.iter_data(data):
        rendering = self.scene.render(camera_params)
        psnr = compute_psnr(rendering.image, image)

        if limit_log_images and n < limit_log_images or limit_log_images is None:
          self.logger.log_eval(name, filename, image, rendering.image, psnr.item())

        total_psnr += psnr
        n += 1

        pbar.update(1)
        pbar.set_postfix(psnr=total_psnr / n)



    return total_psnr / n


  def iter_train(self):
    while True:
      train = self.iter_data(self.dataset.train())
      yield from train

  


  def iter_data(self, iter):
    for filename, image, cam_idx in iter:
      image, cam_idx = [x.to(self.device, non_blocking=True) 
                    for x in (image, cam_idx)]

      camera_params = self.camera_params(cam_idx, image)
      yield filename, image, camera_params


  def train(self):

    print(f"Writing to model path {os.getcwd()}")


    pbar = tqdm(total=self.config.iterations, desc="training")
    self.step = 0

    iter_train = self.iter_train()

    while True:
      if self.step % self.config.eval_iterations == 0:
        n_logged = self.config.num_logged_images

        self.evaluate("train", self.dataset.train(shuffle=False), limit_log_images=n_logged)
        self.evaluate("val", self.dataset.val(), limit_log_images=n_logged)

      filename, image, camera_params = next(iter_train)

      
      self.step += 1
      if self.step % 10 == 0:
        pbar.update(10)
      