from dataclasses import dataclass
import math
import os

from beartype.typing import Optional
import numpy as np
import torch

import torch.nn.functional as F
from torchmetrics.image  import MultiScaleStructuralSimilarityIndexMeasure


from taichi_splatting import perspective
from tqdm import tqdm

from splat_trainer.dataset import Dataset
from splat_trainer.logger import Logger
from splat_trainer.logger.histogram import Histogram

from splat_trainer.scene.gaussian_scene import  SceneConfig
from splat_trainer.util.containers import transpose_rows
from splat_trainer.util.misc import strided_indexes

from splat_trainer.scene.controller import ControllerConfig

@dataclass(kw_only=True)
class TrainConfig:
  output_path: str
  device: str
  steps: int 
  scene: SceneConfig
  controller: ControllerConfig

  load_model: Optional[str] = None

  densify_interval: int = 50
  update_steps: int = 10

  eval_steps: int = 1000
  num_logged_images: int = 5
  log_interval: int = 20

  ssim_weight: float = 0.0
  ssim_scale: float = 0.5



class Trainer:
  def __init__(self, dataset:Dataset, config:TrainConfig, logger:Logger):

    self.device = torch.device(config.device)

    self.dataset = dataset
    self.config = config
    self.step = 0

    self.logger = logger

    self.ssim =  MultiScaleStructuralSimilarityIndexMeasure(
        data_range=1.0, kernel_size=11).to(self.device)


    self.camera_poses = dataset.camera_poses().to(self.device)
    self.camera_projection = dataset.camera_projection().to(self.device)

    if config.load_model:
      print("Loading model from", config.load_model)
      self.scene = config.scene.load_model(config.load_model)
    else:
      print(f"Initializing model from {dataset}")
      self.scene = config.scene.from_pointcloud(dataset.pointcloud())
      
      print(self.scene)

    self.render_start, self.render_end = [torch.cuda.Event(enable_timing = True) for _ in range(2)]
      
    self.scene.to(self.device)
    self.controller = config.controller.make_controller(
      self.scene,  config.densify_interval, config.steps)

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



  def evaluate_dataset(self, name, data, log_count:int=0):
    if len(data) == 0:
      return {}

    rows = []
    radius_hist = Histogram.empty(range=(-1, 3), num_bins=20, device=self.device) 
    log_indexes = strided_indexes(log_count, len(data)) 

    with torch.no_grad():
      pbar = tqdm(total=len(data), desc=f"rendering {name}", leave=False)
      for i, (filename, image, camera_params) in enumerate(self.iter_data(data)):

        rendering = self.scene.render(camera_params, compute_radii=True)
        psnr = compute_psnr(rendering.image, image)
        l1 = torch.nn.functional.l1_loss(rendering.image, image)

        radius_hist = radius_hist.append(rendering.radii.log() / math.log(10.0), trim=False)


        if i in log_indexes:
          image_id = filename.replace("/", "_")
          self.log_image(f"{name}_images/{image_id}/render", rendering.image, caption=f"{filename} PSNR={psnr:.2f} L1={l1:.2f}")
          if self.step == 0:
            self.log_image(f"{name}_images/{image_id}/image", image, caption=filename)
        
        eval = dict(filename=filename, psnr = psnr.item(), l1 = l1.item())
        rows.append(eval)
        
        pbar.update(1)
        pbar.set_postfix(psnr=f"{psnr.item():.2f}", l1=f"{l1.item():.4f}")

    self.logger.log_evaluations(f"{name}/evals", rows, step=self.step)
    totals = transpose_rows(rows)
    mean_l1, mean_psnr = np.mean(totals['l1']), np.mean(totals['psnr'])

    self.log_value(f"{name}/psnr", mean_psnr) 
    self.log_value(f"{name}/l1", mean_l1) 

    self.log_histogram(f"{name}/psnr_hist", torch.tensor(totals['psnr']))
    self.log_histogram(f"{name}/radius_hist", radius_hist)

    return {f"{name}_psnr":mean_psnr}


  def evaluate(self):
    n_logged = self.config.num_logged_images

    train = self.evaluate_dataset("train", self.dataset.train(shuffle=False), log_count=n_logged)
    val = self.evaluate_dataset("val", self.dataset.val(), log_count=n_logged)

    return {**train, **val}

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

  @torch.compile
  def compute_ssim(self, image, ref, scale=1.0):
      image1 = ref.unsqueeze(0).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
      image2 = image.unsqueeze(0).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

      return 1. - self.ssim(F.interpolate(image1, scale_factor=scale), F.interpolate(image2, scale_factor=scale))
      

  def losses(self, rendering, image):
    l1 = torch.nn.functional.l1_loss(rendering.image, image)
    
    if self.config.ssim_weight > 0:  
      ssim = self.compute_ssim(rendering.image, image, scale=self.config.ssim_scale)
      loss = l1 + ssim * self.config.ssim_weight

      return loss, dict(l1=l1.item(), ssim=ssim.item())
    else:
      return l1, dict(l1=l1.item())


  def training_step(self, filename, image, camera_params):
    self.scene.zero_grad()

    self.render_start.record()
    rendering = self.scene.render(camera_params, compute_split_heuristics=True)

    loss, losses = self.losses(rendering, image)
    loss.backward()
    self.render_end.record()

    self.scene.step()
    (visible, in_view) =  self.controller.add_rendering(rendering)

    self.step += 1
    return dict(**losses, visible=visible, in_view=in_view, 
                render_ms=self.render_start.elapsed_time(self.render_end))


  def train(self):
    print(f"Writing to model path {os.getcwd()}")


    self.pbar = tqdm(total=self.config.steps, desc="training")
    self.step = 0
    since_densify = 0
    densify_metrics = dict(n = self.scene.num_points)

    iter_train = self.iter_train()
    start_event, end_event = [torch.cuda.Event(enable_timing = True) for _ in range(2)]

    while self.step < self.config.steps:
      if self.step % self.config.eval_steps == 0:
        eval_metrics = self.evaluate()
        self.scene.update_learning_rate(self.dataset.scene_scale(), self.step, self.config.steps)

      if since_densify >= self.config.densify_interval:
        self.controller.log_histograms(self.logger, self.step)
        densify_metrics = self.controller.densify_and_prune(self.step)

        self.log_values("densify", densify_metrics)

        since_densify = 0

      start_event.record()
      steps = [self.training_step(*next(iter_train)) for _ in range(self.config.update_steps)]
      end_event.record()

      since_densify += len(steps)

      if self.step % self.config.log_interval  == 0:
        steps = transpose_rows(steps)

        self.pbar.update(self.config.log_interval)

        means = {k:np.mean(v) for k, v in steps.items()}
        metrics = {k:f"{means[k]:.4f}" for k in ['l1', 'ssim'] if k in means}

        self.pbar.set_postfix(**metrics, **eval_metrics, **densify_metrics)        
        self.log_values("train", means)

        self.log_value("train/step_ms", 
                start_event.elapsed_time(end_event) / self.config.update_steps) 

    eval_metrics = self.evaluate()
    self.pbar.set_postfix(**metrics, **eval_metrics)        
    self.pbar.close()


    return (eval_metrics.get('eval_train_psnr', 0.0) 
            + eval_metrics.get('val_psnr', 0.0)) / 2.0
    

  def close(self):
    if self.pbar is not None:
      self.pbar.close()

    print("Closing trainer")
    self.logger.close()

def compute_psnr(a, b):
  return 10 * torch.log10(1 / torch.nn.functional.mse_loss(a, b))  