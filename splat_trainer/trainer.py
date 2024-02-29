from dataclasses import dataclass, replace
import math
import os

from beartype.typing import Optional
import numpy as np
import torch

from taichi_splatting import perspective
from tqdm import tqdm

from splat_trainer.dataset import Dataset
from splat_trainer.logger import Logger
from splat_trainer.logger.histogram import Histogram

from splat_trainer.scene.gaussians import GaussianScene, SceneConfig
from splat_trainer.util.containers import transpose_rows

@dataclass(kw_only=True)
class TrainConfig:
  output_path: str
  device: str
  iterations: int 
  scene: SceneConfig

  load_model: Optional[str] = None

  eval_iterations: int = 1000
  num_logged_images: int = 5
  log_interval: int = 20


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
      self.scene = config.scene.load_model(config.load_model)
    else:
      print(f"Initializing model from {dataset}")
      self.scene = config.scene.from_pointcloud(dataset.pointcloud())
      
      print(self.scene)
      
    self.scene.to(self.device)
    self.pbar = None
    

  def camera_params(self, cam_idx:torch.Tensor, image:torch.Tensor):
      near, far = self.dataset.depth_range

      return perspective.CameraParams(
          T_camera_world=self.camera_poses(cam_idx.unsqueeze(0)),
          T_image_camera=self.camera_projection[cam_idx[1]],
          image_size=(image.shape[1], image.shape[0]),
          near_plane=near,
          far_plane=far
      ).to(self.device, dtype=torch.float32)


  def log_image(self, name, image, caption=None):
    return self.logger.log_image(name, image, caption=caption, step=self.step)

  def log_value(self, name, value):
    return self.logger.log_value(name, value, step=self.step) 

  def log_values(self, name, values):
    return self.logger.log_values(name, values, step=self.step)  

  def log_histogram(self, name, values):
    return self.logger.log_histogram(name, values, step=self.step)


  def evaluate_dataset(self, name, data, limit_log_images:Optional[int]=None):
    if len(data) == 0:
      return

    rows = []
    radius_hist = Histogram.empty(range=(-1, 3), num_bins=20, device=self.device) 

    with torch.no_grad():
      pbar = tqdm(total=len(data), desc=f"rendering {name}", leave=False)
      for filename, image, camera_params in self.iter_data(data):
        rendering = self.scene.render(camera_params, compute_radii=True)
        psnr = compute_psnr(rendering.image, image)
        l1 = torch.nn.functional.l1_loss(rendering.image, image)

        radius_hist = radius_hist.append(rendering.radii.log() / math.log(10.0), trim=False)

        if limit_log_images and len(rows) < limit_log_images or limit_log_images is None:
          self.log_image(f"{name}/{filename}/render", rendering.image, caption=f"{filename} PSNR={psnr:.2f} L1={l1:.2f}")
          if self.step == 0:
            self.log_image(f"{name}/{filename}/image", image, caption=filename)
        
        eval = dict(filename=filename, psnr = psnr.item(), l1 = l1.item())
        rows.append(eval)
        
        pbar.update(1)
        pbar.set_postfix(psnr=f"{psnr.item():.2f}", l1=f"{l1.item():.4f}")

    self.logger.log_evaluations(f"{name}/evals", rows, step=self.step)
    totals = transpose_rows(rows)

    self.log_value(f"{name}/psnr", np.mean(totals['psnr']) )
    self.log_value(f"{name}/l1", np.mean(totals['l1']) )

    self.log_histogram(f"{name}/psnr_hist", torch.tensor(totals['psnr']))
    self.log_histogram(f"{name}/radius_hist", radius_hist)

    



  def evaluate(self):
    n_logged = self.config.num_logged_images

    self.evaluate_dataset("train", self.dataset.train(shuffle=False), limit_log_images=n_logged)
    self.evaluate_dataset("val", self.dataset.val(), limit_log_images=n_logged)

  def iter_train(self):
    while True:
      train = self.iter_data(self.dataset.train())
      yield from train

  

  def iter_data(self, iter):
    for filename, image, cam_idx in iter:
      with torch.no_grad():
        image, cam_idx = [x.to(self.device, non_blocking=True) 
                      for x in (image, cam_idx)]
        
        image = image.to(dtype=torch.float) / 255.0

        camera_params = self.camera_params(cam_idx, image)
      
      yield filename, image, camera_params


  def training_step(self, filename, image, camera_params):
    self.scene.zero_grad()

    rendering = self.scene.render(camera_params, compute_split_heuristics=True)
    loss = torch.nn.functional.l1_loss(rendering.image, image)
    loss.backward()
    
    psnr = compute_psnr(rendering.image, image)

    self.scene.step()
    (visible, in_view) =  self.scene.add_training_statistics(rendering)

    self.step += 1
    return dict(l1 = loss.item(), psnr = psnr.item(), visible=visible, in_view=in_view)



  def train(self):
    print(f"Writing to model path {os.getcwd()}")


    self.pbar = tqdm(total=self.config.iterations, desc="training")
    self.step = 0
    since_densify = 0

    iter_train = self.iter_train()

    while self.step < self.config.iterations:
      if self.step % self.config.eval_iterations == 0:
        self.evaluate()
        self.scene.update_learning_rate(self.dataset.scene_scale(), self.step)

      if since_densify >= 100:
        self.scene.log_point_statistics(self.logger, self.step)
        since_densify = 0

      steps = [self.training_step(*next(iter_train)) for _ in range(10)]
      since_densify += 1

      if self.step % self.config.log_interval  == 0:
        steps = transpose_rows(steps)

        self.pbar.update(self.config.log_interval)

        means = {k:np.mean(v) for k, v in steps.items()}
        self.pbar.set_postfix(
          loss=f"{means['l1']:.4f}",
          psnr = f"{means['psnr']:.2f}")
        
        self.log_values("train", means)


      
    self.pbar.close()
    self.evaluate()



  def close(self):
    if self.pbar is not None:
      self.pbar.close()

    print("Closing trainer")
    self.logger.close()

def compute_psnr(a, b):
  return 10 * torch.log10(1 / torch.nn.functional.mse_loss(a, b))  