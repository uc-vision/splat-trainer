from dataclasses import dataclass, replace
from functools import partial
import gc
import heapq
import json
import math
import os
from pathlib import Path

from beartype.typing import Optional
import numpy as np
from omegaconf import OmegaConf
import torch

import torch.nn.functional as F
from torchmetrics.image  import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image import StructuralSimilarityIndexMeasure


from termcolor import colored

 
from splat_trainer.camera_table.camera_table import CameraInfo
from splat_trainer.controller.controller import Controller
from splat_trainer.scene.scene import GaussianScene
from splat_trainer.util.visibility import crop_cloud, random_cloud, random_points
from taichi_splatting import Gaussians3D, RasterConfig, Rendering, perspective
from tqdm import tqdm 

from splat_trainer.dataset import Dataset
from splat_trainer.gaussians.loading import from_pointcloud

from splat_trainer.logger import Logger
from splat_trainer.logger.histogram import Histogram

from splat_trainer.scene.sh_scene import  GaussianSceneConfig
from splat_trainer.scheduler import Scheduler, Uniform
from splat_trainer.util.colorize import colorize, get_cv_colormap
from splat_trainer.util.containers import transpose_rows
from splat_trainer.util.misc import CudaTimer, strided_indexes

from splat_trainer.controller import ControllerConfig

@dataclass(kw_only=True)
class TrainConfig:
  device: str

  steps: int 
  scene: GaussianSceneConfig
  controller: ControllerConfig

  load_model: Optional[str] = None

  num_neighbors:int   = 3
  initial_point_scale:float = 0.5
  initial_alpha:float = 0.5 

  max_initial_points: Optional[int] = None

  background_points : int = 0
  densify_interval: int = 50

  eval_steps: int = 1000
  num_logged_images: int = 5
  log_worst_images: int = 2

  log_interval: int = 10

  ssim_weight: float = 0.2
  ssim_scale: float = 0.5

  scale_reg: float = 100.0
  opacity_reg: float = 0.1

  blur_cov: float = 0.3
  antialias: bool = True

  save_checkpoints: bool = False
  save_output: bool = True

  lr_scheduler: Scheduler = Uniform()
  raster_config: RasterConfig = RasterConfig()
  


class Trainer:
  def __init__(self, config:TrainConfig,
                scene:GaussianScene, 
                controller:Controller,
                dataset:Dataset,  
              
              logger:Logger,
              step = 0,
      ):

    self.device = torch.device(config.device)
    self.controller = controller
    self.scene = scene
    self.dataset = dataset

    self.camera_info = dataset.camera_info().to(self.device)

    self.config = config
    self.logger = logger
    self.step = step

    self.last_checkpoint = None
    

    self.ssim = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=1.0, kernel_size=11, betas=(0.3, 0.3, 0.3)).to(self.device)
    
    self.render_timers = [CudaTimer() for _ in range(self.config.log_interval)]

    self.color_map = get_cv_colormap().to(self.device)
    self.pbar = None


  @staticmethod
  def initialize(config:TrainConfig, dataset:Dataset, logger:Logger):

    device = torch.device(config.device)
    camera_info = dataset.camera_info().to(device)

    print(f"Initializing model from {dataset}")
    points = dataset.pointcloud().to(device)
    cropped = crop_cloud(camera_info, points)

    # random subset of cropped
    if config.max_initial_points is not None:
      perm = torch.randperm(cropped.batch_size[0])
      cropped = cropped[perm[:config.max_initial_points]]

    if cropped.batch_size[0] == 0:
      raise ValueError("No points visible in dataset images, check input data!")

    print(colored(f"Using {cropped.batch_size[0]} points from original {points.batch_size[0]}", 'yellow'))
  
    initial_gaussians:Gaussians3D = from_pointcloud(cropped, 
                                        initial_scale=config.initial_point_scale,
                                        initial_alpha=config.initial_alpha,
                                        num_neighbors=config.num_neighbors)
    
    if config.background_points > 0:
      near, _ = camera_info.depth_range
      bg_points = random_cloud(camera_info, config.background_points, min_depth=near * 2)
    
      bg_gaussians = from_pointcloud(bg_points, 
                                        initial_scale=config.initial_point_scale,
                                        initial_alpha=config.initial_alpha,
                                        num_neighbors=config.num_neighbors)
      
      initial_gaussians = initial_gaussians.concat(bg_gaussians)
      

    scene = config.scene.from_color_gaussians(initial_gaussians, camera_info.camera_table, device)
    controller = config.controller.make_controller(scene)

    if config.save_output:
      output_path = Path.cwd()

      cropped.save_ply(output_path / "input.ply")
      with open(output_path / "cameras.json", "w") as f:
        json.dump(dataset.camera_json(camera_info.camera_table), f)

    return Trainer(config, scene, controller, dataset, logger)
      

  def state_dict(self):
    return dict(step=self.step, 
                scene=self.scene.state_dict(), 
                controller=self.controller.state_dict())
  

  @property
  def output_path(self):
    return Path.cwd() 

  def write_checkpoint(self):
    path = self.output_path / "checkpoint" / f"checkpoint_{self.step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = self.state_dict()
    torch.save(checkpoint, path)


  @staticmethod
  def from_state_dict(config:TrainConfig, dataset:Dataset, logger:Logger, state_dict:dict):

    scene = config.scene.from_state_dict(state_dict['scene'], dataset.camera_info().camera_table)
    controller = config.controller.from_state_dict(state_dict['controller'], scene)

    return Trainer(config, scene, controller, dataset, logger, step=state_dict['step'])
    
    
  @property
  def camera_table(self):
    return self.camera_info.camera_table
  
  @property
  def blur_cov(self):
    return  self.config.blur_cov if not self.config.antialias else 0.0


  def camera_params(self, cam_idx:torch.Tensor, image:torch.Tensor):
        near, far = self.dataset.depth_range()
        camera_t_world, projection = self.camera_table.lookup(cam_idx)

        return perspective.CameraParams(
            T_camera_world=camera_t_world,
            projection=projection,
            image_size=(image.shape[1], image.shape[0]),
            near_plane=near,
            far_plane=far,
        ).to(self.device, dtype=torch.float32)


  def log_image(self, name, image, caption=None):
    return self.logger.log_image(name, image, caption=caption, step=self.step)
  

  def log_value(self, name, value):
    return self.logger.log_value(name, value, step=self.step) 

  def log_values(self, name, values):
    return self.logger.log_values(name, values, step=self.step)  

  def log_histogram(self, name, values):
    return self.logger.log_histogram(name, values, step=self.step)


  def log_rendering(self, name:str, filename:str, rendering:Rendering, image:torch.Tensor,
                     psnr:float, l1:float, log_image:bool=True):
    self.log_image(f"{name}/render", rendering.image, 
                    caption=f"{filename} PSNR={psnr:.2f} L1={l1:.2f} step={self.step}")
    self.log_image(f"{name}/depth", 
        colorize(self.color_map, rendering.ndc_depth), caption=filename)
    
    if log_image:
      self.log_image(f"{name}/image", image, caption=filename)

  def evaluate_dataset(self, name, data, log_count:int=0, worst_count:int=0):
    if len(data) == 0:
      return {}

    rows = []
    radius_hist = Histogram.empty(range=(-1, 3), num_bins=20, device=self.device) 

    worst = []

    log_indexes = strided_indexes(log_count, len(data)) 

    pbar = tqdm(total=len(data), desc=f"rendering {name}", leave=False)
    for i, (filename, camera_params, cam_idx, image) in enumerate(self.iter_data(data)):

      config = replace(self.config.raster_config, compute_split_heuristics=True, 
                        antialias=self.config.antialias,
                       blur_cov=self.blur_cov)
      rendering = self.scene.render(camera_params, config, cam_idx, render_depth=True)
      
      psnr = compute_psnr(rendering.image, image)
      l1 = torch.nn.functional.l1_loss(rendering.image, image)

      radius_hist = radius_hist.append(rendering.radii.log() / math.log(10.0), trim=False)
      image_id = filename.replace("/", "_")

      if i in log_indexes:
        self.log_rendering(f"{name}_images/{image_id}", filename, rendering, image, 
                           psnr.item(), l1.item(), log_image=self.step == 0)

      add_worst = heapq.heappush if len(worst) < worst_count else heapq.heappushpop
      add_worst(worst, (-psnr.item(), l1.item(), rendering.detach(), image, image_id))
      
      eval = dict(filename=filename, psnr = psnr.item(), l1 = l1.item())
      rows.append(eval)
      
      pbar.update(1)
      pbar.set_postfix(psnr=f"{psnr.item():.2f}", l1=f"{l1.item():.4f}")

    for i, (neg_psnr, l1, rendering, image, filename) in enumerate(worst):
      self.log_rendering(f"worst_{name}/{i}", filename, rendering, image,
                         -neg_psnr, l1, log_image=True)

    self.logger.log_evaluations(f"eval_{name}/evals", rows, step=self.step)
    totals = transpose_rows(rows)
    mean_l1, mean_psnr = np.mean(totals['l1']), np.mean(totals['psnr'])

    self.log_value(f"eval_{name}/psnr", mean_psnr) 
    self.log_value(f"eval_{name}/l1", mean_l1) 

    self.log_histogram(f"eval_{name}/psnr_hist", torch.tensor(totals['psnr']))
    self.log_histogram(f"eval_{name}/radius_hist", radius_hist)

    return {f"{name}_psnr": float(mean_psnr)}


  def evaluate(self, write_outputs=False):

    train = self.evaluate_dataset("train", self.dataset.train(shuffle=False), 
      log_count=self.config.num_logged_images, worst_count=self.config.log_worst_images)
    val = self.evaluate_dataset("val", self.dataset.val(), 
      log_count=self.config.num_logged_images, worst_count=self.config.log_worst_images)

    if write_outputs:
      iteration_path = self.output_path / f"point_cloud/iteration_{self.step}"
      iteration_path.mkdir(parents=True, exist_ok=True)

      self.scene.write_to(iteration_path)
      self.write_checkpoint()
      
      camera_json = self.dataset.camera_json(self.camera_table)
      with open(iteration_path / "cameras.json", "w") as f:
        json.dump(camera_json, f)

    self.scene.log(self.logger, self.step)

    return {**train, **val}
  

  def iter_train(self):
    while True:
      train = self.iter_data(self.dataset.train())
      yield from train

  

  def iter_data(self, iter):
    for filename, image, image_idx in iter:
      image = image.to(self.device, non_blocking=True) 
      
      image = image.to(dtype=torch.float) / 255.0
      camera_params = self.camera_params(image_idx, image)

      yield filename, camera_params, image_idx, image



  @torch.compile
  def compute_ssim(self, image:torch.Tensor, ref:torch.Tensor, scale:float=1.0):
      image1 = ref.unsqueeze(0).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
      image2 = image.unsqueeze(0).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

      if scale == 1.0:
        return 1.0 - self.ssim(image1, image2)
      else:
        return 1.0 - self.ssim(F.interpolate(image1, scale_factor=scale), F.interpolate(image2, scale_factor=scale))
      

  def losses(self, rendering:Rendering, image):
    l1 = torch.nn.functional.l1_loss(rendering.image, image)
    
    losses = dict(l1=l1.item())
    loss = l1

    if self.config.ssim_weight > 0:  
      ssim = self.compute_ssim(rendering.image, image, scale=self.config.ssim_scale)
      loss += ssim * self.config.ssim_weight 
      losses["ssim"] = ssim.item()


    area = rendering.area / (rendering.camera.focal_length[0]**2)
    reg_loss = (  self.scene.opacity.mean() * self.config.opacity_reg
                + (area / rendering.point_depth.squeeze(1)).mean() * self.config.scale_reg)
    
    losses["reg"] = reg_loss.item()
    loss += reg_loss 


    return loss, losses



  def training_step(self, filename, camera_params, image_idx, image, timer):

    with timer:
      config = replace(self.config.raster_config, compute_split_heuristics=True, 
                       antialias=self.config.antialias,
                       blur_cov=self.blur_cov)  
      
      rendering = self.scene.render(camera_params, config, image_idx)

      loss, losses = self.losses(rendering, image)
      loss.backward()

    with torch.no_grad():
      metrics =  self.controller.step(rendering, self.step)
      self.scene.step(rendering, self.step)
      
    del loss

    self.step += 1
    return dict(**losses, **metrics)


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
          eval_metrics = self.evaluate(self.config.save_checkpoints)

          lr_scale = self.config.lr_scheduler(self.step, self.config.steps)
          self.scene.update_learning_rate(lr_scale)

          self.log_values("train", dict(lr_scale=lr_scale, blur_cov=self.blur_cov))
          torch.cuda.empty_cache()


      if since_densify >= self.config.densify_interval:
        self.controller.log_histograms(self.logger, self.step)
        densify_metrics = self.controller.densify_and_prune(self.step, self.config.steps)

        self.log_values("densify", densify_metrics)
        since_densify = 0
      
        

      with torch.enable_grad():
        with step_timer:
          steps = [self.training_step(*next(iter_train), timer=timer) 
                  for timer in self.render_timers]

      torch.cuda.empty_cache()
      since_densify += len(steps)

      if self.step % self.config.log_interval  == 0:
        steps = transpose_rows(steps)

        self.pbar.update(self.config.log_interval)

        means = {k:np.mean(v) for k, v in steps.items()}
        metrics = {k:f"{means[k]:.4f}" for k in ['l1', 'ssim', 'reg'] if k in means}
        densify_pbar = {k:f"{densify_metrics[k]}" for k in ['split', 'prune', 'n'] if k in densify_metrics}

        self.pbar.set_postfix(**metrics, **eval_metrics, **densify_pbar)        
        self.log_values("train", means)

        # skip first step as it will include compiling kernels
        if self.step > self.config.log_interval:
          torch.cuda.synchronize()

          self.log_values("timer", 
                  dict(step_ms=step_timer.ellapsed() / self.config.log_interval,
                  render=sum([timer.ellapsed() for timer in self.render_timers]) / self.config.log_interval
                ))

    eval_metrics = self.evaluate(self.config.save_output)

    self.pbar.set_postfix(**metrics, **eval_metrics)        
    self.pbar.close()

    return eval_metrics
    

  def close(self):
    if self.pbar is not None:
      self.pbar.close()


def compute_psnr(a, b):
  return 10 * torch.log10(1 / torch.nn.functional.mse_loss(a, b))  