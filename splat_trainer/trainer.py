from dataclasses import dataclass, replace
from functools import partial
import heapq
import json
import math
from numbers import Number
from pathlib import Path
from typing import Tuple, TypeVar

from tqdm import tqdm 
from termcolor import colored

from beartype import beartype
from beartype.typing import Optional
import numpy as np
import torch

from fused_ssim import fused_ssim
import torch.nn.functional as F
# import torchmetrics.image as torchmetrics
 
from splat_trainer.controller.controller import Controller
from splat_trainer.scene.scene import GaussianScene
from splat_trainer.config import Varying
from splat_trainer.util.pointcloud import PointCloud
from splat_trainer.util.visibility import crop_cloud, random_cloud
from taichi_splatting import Gaussians3D, RasterConfig, Rendering
from taichi_splatting.perspective import CameraParams

from splat_trainer.dataset import Dataset
from splat_trainer.gaussians.loading import from_pointcloud

from splat_trainer.logger import Logger
from splat_trainer.logger.histogram import Histogram

from splat_trainer.scene.sh_scene import  GaussianSceneConfig
from splat_trainer.util.colorize import colorize, get_cv_colormap
from splat_trainer.util.containers import transpose_rows
from splat_trainer.util.misc import CudaTimer, next_multiple, strided_indexes

from splat_trainer.controller import ControllerConfig


from splat_trainer.util.lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)

T = TypeVar("T")


@beartype
@dataclass(kw_only=True)
class TrainConfig:
  device: str

  steps: int 
  scene: GaussianSceneConfig
  controller: ControllerConfig

  load_model: Optional[str] = None

  num_neighbors:int  
  initial_point_scale:float 
  initial_alpha:float 

  limit_points: Optional[int] = None

  initial_points : int 
  add_initial_points: bool = False
  load_dataset_cloud: bool = True

  eval_steps: int    
  log_interval: int  = 20

  num_logged_images: int = 8
  log_worst_images: int  = 2

  densify_interval: Varying[int]

  ssim_weight: float
  l1_weight: float
  ssim_levels: int = 3

  scale_reg: Varying[float]
  opacity_reg: Varying[float]
  aspect_reg: Varying[float]

  blur_cov: float
  antialias: bool = True

  save_checkpoints: bool = False
  save_output: bool = True

  use_bilateral_grid: bool = False
  bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

  # lr_scheduler: Scheduler = Uniform()
  lr: Varying[float]

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
    self.render_timers = [CudaTimer() for _ in range(self.config.log_interval)]

    self.color_map = get_cv_colormap().to(self.device)
    self.pbar = None

    self.ssim = partial(fused_ssim, padding="valid")


    # if config.use_bilateral_grid:
    #   self.bil_grids = BilateralGrid(
    #     len(self.dataset.train_cameras),
    #     grid_X=config.bilateral_grid_shape[0],
    #     grid_Y=config.bilateral_grid_shape[1],
    #     grid_W=config.bilateral_grid_shape[2],
    #   ).to(self.device)
    #   self.bil_grid_optimizer = torch.optim.Adam(
    #     self.bil_grids.parameters(),
    #     lr=2e-3,
    #     eps=1e-15,
    #   )
    #   self.bil_grid_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    #     [
    #       torch.optim.lr_scheduler.LinearLR(
    #           self.bil_grid_optimizer,
    #           start_factor=0.01,
    #           total_iters=1000,
    #       ),
    #       torch.optim.lr_scheduler.ExponentialLR(
    #           self.bil_grid_optimizer, gamma=0.01 ** (1.0 / config.steps)
    #       ),
    #     ]
    #   )

    # self.ssim = torch.compile(torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0, sigma=1.5).to(self.device))
    # self.ssim = torchmetrics.MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=11, sigma=1.5).to(self.device)



  @staticmethod
  def get_initial_points(config:TrainConfig, dataset:Dataset) -> PointCloud:
    device = torch.device(config.device)
    camera_info = dataset.camera_info().to(device)

    dataset_cloud = dataset.pointcloud() if config.load_dataset_cloud else None
    points = None

    if dataset_cloud is not None:
      points = dataset_cloud.to(device)
      points = crop_cloud(camera_info, points)

      if points.batch_size[0] == 0:
        raise ValueError("No points visible in dataset images, check input data!")

      print(colored(f"Using {points.batch_size[0]} points from original {dataset_cloud.batch_size[0]}", 'yellow'))
    
      if config.limit_points is not None:
        print(f"Limiting {points.batch_size[0]} points to {config.limit_points}")
        random_indices = torch.randperm(points.batch_size[0])[:config.limit_points]
        points = points[random_indices]
      
    if config.add_initial_points or dataset_cloud is None:
      near, _ = camera_info.depth_range
      random_points = random_cloud(camera_info, config.initial_points)
    
      if points is not None:
        print(f"Adding {random_points.batch_size[0]} random points")
        points = points.concat(random_points)
      else:
        print(f"Using {random_points.batch_size[0]} random points")
        points = random_points

    return points


  @staticmethod
  def initialize(config:TrainConfig, dataset:Dataset, logger:Logger):

    device = torch.device(config.device)
    camera_info = dataset.camera_info().to(device)

    print(f"Initializing points from {dataset}")

    initial_points = Trainer.get_initial_points(config, dataset)
    initial_gaussians:Gaussians3D = from_pointcloud(initial_points, 
                                          initial_scale=config.initial_point_scale,
                                          initial_alpha=config.initial_alpha,
                                          num_neighbors=config.num_neighbors)

    scene = config.scene.from_color_gaussians(initial_gaussians, camera_info.camera_table, device)
    controller = config.controller.make_controller(scene)

    if config.save_output:
      output_path = Path.cwd()

      initial_points.save_ply(output_path / "input.ply")
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
  
  @property
  def t(self):
    return self.step / self.config.steps
  

  def __repr__(self):
    return f"Trainer(step={self.step}, scene={self.scene} controller={self.controller})"


  def camera_params(self, cam_idx:torch.Tensor, image:torch.Tensor):
        near, far = self.dataset.depth_range()
        camera_t_world, projection = self.camera_table.lookup(cam_idx)

        return CameraParams(
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
        if self.config.use_bilateral_grid:
          cc_image = color_correct(rendering.image.unsqueeze(0), image.unsqueeze(0)).squeeze(0)
          rendering = replace(rendering, image=cc_image)
          cc_psnr = compute_psnr(cc_image, image)
          self.log_rendering(f"{name}_images/{image_id}", filename, rendering, image, 
                            cc_psnr.item(), l1.item(), log_image=self.step == 0)
          
        else:
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




  def compute_ssim(self, pred:torch.Tensor, ref:torch.Tensor, levels:int=4):
      ref = ref.unsqueeze(0).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
      pred = pred.unsqueeze(0).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

      loss = 1.0 - self.ssim(pred, ref)

      for i in range(1, levels):
        pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
        ref = F.avg_pool2d(ref, kernel_size=2, stride=2)

        loss += (1.0 - self.ssim(pred, ref)) 

      return loss / levels
  

  def losses(self, rendering:Rendering, image):
    losses = {}
    loss = 0.0

    if self.config.l1_weight > 0:
      l1 = torch.nn.functional.l1_loss(rendering.image, image)
      losses["l1"] = l1.item()
      loss = l1 * self.config.l1_weight


    if self.config.ssim_weight > 0:  
      ssim = self.compute_ssim(rendering.image, image, self.config.ssim_levels)
      loss += ssim * self.config.ssim_weight 
      losses["ssim"] = ssim.item()




    aspect = rendering.scale.max(-1).values / rendering.scale.min(-1).values
    
    scale_term = rendering.scale / rendering.camera.focal_length[0]
    reg_loss = (  self.scene.opacity.mean() * self.config.opacity_reg(self.t)
                 + scale_term.mean() * self.config.scale_reg(self.t)
                 + self.config.aspect_reg(self.t) * aspect.mean())
                  
    
    losses["reg"] = reg_loss.item()
    loss += reg_loss 


    return loss, losses


  def training_step(self, filename, camera_params, image_idx, image, timer):

    with timer:
      config = replace(self.config.raster_config, compute_split_heuristics=True, 
                       antialias=self.config.antialias,
                       blur_cov=self.blur_cov)  
      
      rendering = self.scene.render(camera_params, config, image_idx)

      if self.config.use_bilateral_grid:
        grid_y, grid_x = torch.meshgrid(
            (torch.arange(image.shape[0], device=self.device) + 0.5) / image.shape[0],
            (torch.arange(image.shape[1], device=self.device) + 0.5) / image.shape[1],
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        corrected_image = slice(self.bil_grids, grid_xy, rendering.image.unsqueeze(0), torch.tensor([image_idx]))["rgb"].squeeze(0)
        rendering = replace(rendering, image=corrected_image)

      loss, losses = self.losses(rendering, image)
      
      if self.config.use_bilateral_grid:
        tvloss = 10 * total_variation_loss(self.bil_grids.grids)
        loss += tvloss
        self.log_value("train/tvloss", tvloss.item())

      loss.backward()

    with torch.no_grad():
      metrics =  self.controller.step(rendering, self.step)
      self.scene.step(rendering, self.step)

      if self.config.use_bilateral_grid:
        self.bil_grid_optimizer.step()
        self.bil_grid_optimizer.zero_grad(set_to_none=True)
        self.bil_grid_scheduler.step()

      
    del loss

    self.step += 1
    return dict(**losses, **metrics)

  


  def train(self):

    self.pbar = tqdm(total=self.config.steps - self.step, desc="training")
    densify_metrics = dict(n = self.scene.num_points)
    next_densify = next_multiple(self.step, self.config.densify_interval(self.t))

    metrics = {}
    eval_metrics = {}
    
    iter_train = self.iter_train()
    step_timer = CudaTimer()

    while self.step < self.config.steps:

      if self.step % self.config.eval_steps == 0:
          eval_metrics = self.evaluate(self.config.save_checkpoints)

          lr_scale = self.config.lr(self.t)
          self.scene.update_learning_rate(lr_scale)

          self.log_values("train", dict(lr_scale=lr_scale, blur_cov=self.blur_cov))
          torch.cuda.empty_cache()

      if self.step - next_densify > 0:
        self.controller.log_histograms(self.logger, self.step)
        densify_metrics = self.controller.densify_and_prune(self.step, self.config.steps)

        self.log_values("densify", densify_metrics)
        next_densify += self.config.densify_interval(self.t)
      

      with torch.enable_grad():
        with step_timer:
          steps = [self.training_step(*next(iter_train), timer=timer) 
                  for timer in self.render_timers]

      torch.cuda.empty_cache()

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
