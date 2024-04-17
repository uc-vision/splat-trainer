from dataclasses import dataclass
import math
import os
from pathlib import Path

from beartype.typing import Optional
import numpy as np
import torch

import torch.nn.functional as F
from torchmetrics.image  import MultiScaleStructuralSimilarityIndexMeasure

 
from taichi_splatting import Rendering, perspective
from tqdm import tqdm

from splat_trainer.dataset import Dataset
from splat_trainer.image_scaler import ImageScaler, NullScaler
from splat_trainer.gaussians.loading import from_pointcloud

from splat_trainer.logger import Logger
from splat_trainer.logger.histogram import Histogram

from splat_trainer.scene.sh_scene import  GaussianSceneConfig
from splat_trainer.scheduler import Scheduler, Uniform
from splat_trainer.util.colorize import colorize_depth, get_cv_colormap
from splat_trainer.util.containers import transpose_rows
from splat_trainer.util.misc import CudaTimer, strided_indexes

from splat_trainer.controller import ControllerConfig

@dataclass(kw_only=True)
class TrainConfig:
  output_path: str
  device: str
  steps: int 
  scene: GaussianSceneConfig
  controller: ControllerConfig

  load_model: Optional[str] = None

  num_neighbors:int   = 3
  initial_point_scale:float = 0.5
  initial_alpha:float = 0.5 

  densify_interval: int = 50
  update_steps: int = 10

  eval_steps: int = 1000
  num_logged_images: int = 5
  log_interval: int = 20

  ssim_weight: float = 0.0
  ssim_scale: float = 0.5

  blur_cov: float = 0.3

  lr_scheduler: Scheduler = Uniform()
  image_scaler: ImageScaler = NullScaler()
  

class Trainer:
  def __init__(self, dataset:Dataset, config:TrainConfig, logger:Logger):

    self.device = torch.device(config.device)

    self.dataset = dataset
    self.config = config
    self.step = 0
    self.logger = logger
    self.image_scale = self.config.image_scaler.update(0)

    self.blur_cov = config.blur_cov
    self.output_path = Path(config.output_path or os.getcwd())

    self.ssim = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=1.0, kernel_size=11).to(self.device)

    self.camera_table = dataset.camera_table()
    self.camera_table.to(self.device)
    self.camera_table.requires_grad_(False)
    

    print(f"Initializing model from {dataset}")
    initial_gaussians = from_pointcloud(dataset.pointcloud(), 
                                        initial_scale=config.initial_point_scale,
                                        initial_alpha=config.initial_alpha,
                                        num_neighbors=config.num_neighbors)
    
    self.scene = config.scene.from_color_gaussians(initial_gaussians, self.camera_table, self.device)
    print(self.scene)

    self.render_timers = [CudaTimer() for _ in range(self.config.log_interval)]

    self.controller = config.controller.make_controller(
      self.scene,  config.densify_interval, config.steps)
    
    self.color_map = get_cv_colormap().to(self.device)

    self.pbar = None
    

  def camera_params(self, cam_idx:torch.Tensor, image:torch.Tensor):
        near, far = self.dataset.depth_range
        camera_t_world, image_t_camera = self.camera_table(cam_idx)

        return perspective.CameraParams(
            T_camera_world=camera_t_world.squeeze(0),
            T_image_camera=image_t_camera.squeeze(0),
            image_size=(image.shape[1], image.shape[0]),
            near_plane=near,
            far_plane=far,
            blur_cov=self.blur_cov
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

    pbar = tqdm(total=len(data), desc=f"rendering {name}", leave=False)
    for i, (filename, camera_params, cam_idx, image) in enumerate(self.iter_data(data)):

      rendering = self.scene.render(camera_params, cam_idx, compute_radii=True, render_depth=True)
      psnr = compute_psnr(rendering.image, image)
      l1 = torch.nn.functional.l1_loss(rendering.image, image)

      radius_hist = radius_hist.append(rendering.radii.log() / math.log(10.0), trim=False)

      if i in log_indexes:
        image_id = filename.replace("/", "_")
        self.log_image(f"{name}_images/{image_id}/render", rendering.image, caption=f"{filename} PSNR={psnr:.2f} L1={l1:.2f}")
        self.log_image(f"{name}_images/{image_id}/depth", 
            colorize_depth(self.color_map, rendering.depth, 0.1), caption=filename)

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

    self.scene.write_to(self.output_path / "point_cloud" , f"model_{self.step}")
    # self.scene.log(self.logger, self.step)

    return {**train, **val}
  


  def iter_train(self):
    while True:
      train = self.iter_data(self.dataset.train())
      yield from train

  

  def iter_data(self, iter):
    for filename, image, image_idx in iter:
      image, image_idx = [x.to(self.device, non_blocking=True) 
                    for x in (image, image_idx)]
      
      image = image.to(dtype=torch.float) / 255.0
      camera_params = self.camera_params(image_idx, image)

      yield filename, camera_params, image_idx.squeeze(0), image

  @torch.compile()
  def compute_ssim(self, image:torch.Tensor, ref:torch.Tensor, scale:float=1.0):
      image1 = ref.unsqueeze(0).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
      image2 = image.unsqueeze(0).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

      return 1. - self.ssim(F.interpolate(image1, scale_factor=scale), F.interpolate(image2, scale_factor=scale))
      

  def losses(self, rendering:Rendering, image):
    l1 = torch.nn.functional.l1_loss(rendering.image, image)
    
    if self.config.ssim_weight > 0:  
      ssim = self.compute_ssim(rendering.image, image, scale=self.config.ssim_scale)
      loss = l1 + ssim * self.config.ssim_weight 
      return loss, dict(l1=l1.item(), ssim=ssim.item())

    else:
      return l1, dict(l1=l1.item())


  def training_step(self, filename, camera_params, image_idx, image, timer):
    image, camera_params = self.config.image_scaler(image, camera_params, self.image_scale)

    with timer:
      rendering = self.scene.render(camera_params, image_idx, compute_split_heuristics=True)

      loss, losses = self.losses(rendering, image)
      loss.backward()

    (visible, in_view) =  self.controller.add_rendering(rendering)
    self.scene.step(visible)

    self.scene.zero_grad()

    self.step += 1
    return dict(**losses, visible=visible.shape[0], in_view=in_view.shape[0])


  def train(self):
    print(f"Writing to model path {os.getcwd()}")


    self.pbar = tqdm(total=self.config.steps, desc="training")
    self.step = 0
    since_densify = 0
    densify_metrics = dict(n = self.scene.num_points)

    iter_train = self.iter_train()
    step_timer = CudaTimer()

    while self.step < self.config.steps:


      if self.step % self.config.eval_steps == 0:
          eval_metrics = self.evaluate()
          self.image_scale = self.config.image_scaler.update(self.step)

          lr_scale = self.config.lr_scheduler(self.step, self.config.steps)
          self.scene.update_learning_rate(lr_scale)

          self.log_values("train", dict(lr_scale=lr_scale, blur_cov=self.blur_cov, image_scale=self.image_scale))

      if since_densify >= self.config.densify_interval:
        self.controller.log_histograms(self.logger, self.step)
        densify_metrics = self.controller.densify_and_prune(self.step)

        self.log_values("densify", densify_metrics)
        since_densify = 0
      
      torch.cuda.empty_cache()

      with torch.enable_grad():
        with step_timer:
          steps = [self.training_step(*next(iter_train), timer=timer) 
                  for timer in self.render_timers]

      since_densify += len(steps)

      if self.step % self.config.log_interval  == 0:
        steps = transpose_rows(steps)

        self.pbar.update(self.config.log_interval)

        means = {k:np.mean(v) for k, v in steps.items()}
        metrics = {k:f"{means[k]:.4f}" for k in ['l1', 'ssim', 'reg'] if k in means}

        self.pbar.set_postfix(**metrics, **eval_metrics, **densify_metrics)        
        self.log_values("train", means)

        # skip first step as it will include compiling kernels
        if self.step > self.config.log_interval:
          torch.cuda.synchronize()

          self.log_values("timer", 
                  dict(step_ms=step_timer.ellapsed() / self.config.update_steps,
                  render=sum([timer.ellapsed() for timer in self.render_timers]) / self.config.update_steps
                ))

    eval_metrics = self.evaluate()
    self.pbar.set_postfix(**metrics, **eval_metrics)        
    self.pbar.close()


    return (eval_metrics.get('eval_train_psnr', 0.0) 
            + eval_metrics.get('val_psnr', 0.0)) / 2.0
    

  def close(self):
    if self.pbar is not None:
      self.pbar.close()


def compute_psnr(a, b):
  return 10 * torch.log10(1 / torch.nn.functional.mse_loss(a, b))  