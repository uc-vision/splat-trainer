# Python standard library
from dataclasses import replace
from enum import Enum
from functools import partial
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Callable, Iterable, Iterator, Optional, Tuple

# Third party packages
from beartype import beartype
from beartype.typing import Optional
from fused_ssim import fused_ssim
import numpy as np
import pandas as pd
from pydispatch import Dispatcher

from tensordict import TensorDict
from termcolor import colored
import torch
import torch.nn.functional as F
from tqdm import tqdm

from taichi_splatting import Gaussians3D, Rendering
from taichi_splatting.perspective import CameraParams

# Local imports
from splat_trainer.camera_table.camera_table import Camera, Label, camera_json
from splat_trainer.dataset import Dataset
from splat_trainer.dataset.dataset import CameraView


from splat_trainer.config import Progress, eval_varying
from splat_trainer.controller.controller import Controller

from splat_trainer.debug.optim import print_stats, print_table
from splat_trainer.logger import Logger
from splat_trainer.logger.logger import CompositeLogger, StateLogger

from splat_trainer.gaussians.loading import from_pointcloud
from splat_trainer.trainer.config import TrainConfig
from splat_trainer.trainer.evaluation import Evaluation
from splat_trainer.util.pointcloud import PointCloud

from splat_trainer.scene.io import read_gaussians, write_gaussians
from splat_trainer.scene.scene import GaussianScene

from splat_trainer.visibility import cluster
from splat_trainer.visibility.query_points import crop_cloud, random_cloud

from splat_trainer.util.colorize import colorize, get_cv_colormap
from splat_trainer.util.containers import mean_rows, transpose_rows

from splat_trainer.util.misc import Heap, next_multiple



  

class TrainerState(Enum):
  Stopped = 0
  Training = 1
  Paused = 2

class Trainer(Dispatcher):
  _events_ = ["on_update"]

  def __init__(self, config:TrainConfig,
                scene:GaussianScene, 
                controller:Controller,
                dataset:Dataset,              
                logger:Logger,
                step = 0,
                view_clustering:Optional[cluster.ViewClustering] = None,
                view_counts:Optional[torch.Tensor] = None
      ):

    self.device = torch.device(config.device)
    self.controller = controller
    self.scene = scene
    self.dataset = dataset

    self.camera_table = dataset.camera_table().to(self.device)

    self.config = config
    self.logger = logger

    self.step = step
    self.state = TrainerState.Stopped


    self.state_logger = StateLogger()
    self.logger = CompositeLogger(self.state_logger, logger)


    self.color_map = get_cv_colormap().to(self.device)
    self.ssim = partial(fused_ssim, padding="valid")
    self.pbar = None

    n = self.camera_table.count_label(label=Label.Training)
    if view_counts is None:
      self.view_counts: torch.Tensor = torch.zeros((n,), device=self.device)
    else:
      self.view_counts = view_counts.to(self.device)

    self.view_clustering = view_clustering

  @staticmethod
  def get_initial_points(config:TrainConfig, dataset:Dataset) -> PointCloud:
    device = torch.device(config.device)
    camera_table = dataset.camera_table().to(device)
    cameras = camera_table.cameras

    dataset_cloud:Optional[PointCloud] = dataset.pointcloud() if config.load_dataset_cloud else None
    points = None

    if dataset_cloud is not None:
      points = dataset_cloud.to(device)
      points = crop_cloud(cameras, points)

      if points.batch_size[0] == 0:
        raise ValueError("No points visible in dataset images, check input data!")

      print(colored(f"Using {points.batch_size[0]} points from original {dataset_cloud.batch_size[0]}", 'yellow'))
    
      if config.limit_points is not None:
        print(f"Limiting {points.batch_size[0]} points to {config.limit_points}")
        random_indices = torch.randperm(points.batch_size[0])[:config.limit_points]
        points = points[random_indices]
      
    if config.add_initial_points or dataset_cloud is None:
      random_points = random_cloud(cameras, config.initial_points)
      if points is not None:
        print(f"Adding {random_points.batch_size[0]} random points")
        points = torch.cat([points, random_points], dim=0)
      else:
        print(f"Using {random_points.batch_size[0]} random points")
        points = random_points

    return points


  @staticmethod
  def initialize(config:TrainConfig, dataset:Dataset, logger:Logger):

    device = torch.device(config.device)
    camera_table = dataset.camera_table().to(device)


    print(f"Initializing points from {dataset}")

    initial_points = Trainer.get_initial_points(config, dataset)
    initial_gaussians:Gaussians3D = from_pointcloud(initial_points, 
                                          initial_scale=config.initial_point_scale,
                                          initial_alpha=config.initial_alpha,
                                          num_neighbors=config.num_neighbors)
    


    scene = config.scene.from_color_gaussians(initial_gaussians, camera_table, device, logger)
    controller = config.controller.make_controller(scene, logger)

    if config.save_output:
      output_path = Path.cwd()

      initial_points.save_ply(output_path / "input.ply")
      with open(output_path / "cameras.json", "w") as f:
        json.dump(camera_json(camera_table), f)

    return Trainer(config, scene, controller, dataset, logger)
      

    
  @staticmethod
  def from_state_dict(config:TrainConfig, dataset:Dataset, logger:Logger, state_dict:dict):
    device = torch.device(config.device)
    camera_table = dataset.camera_table().to(device)

    scene = config.scene.from_state_dict(state_dict['scene'], camera_table, logger)
    controller = config.controller.from_state_dict(state_dict['controller'], scene, logger) 

    if state_dict['view_clustering'] is not None:
      view_clustering = cluster.ViewClustering.from_state_dict(state_dict['view_clustering'])
    else:
      view_clustering = None

    return Trainer(config, scene, controller, dataset, logger, 
                   step=state_dict['step'],
                   view_clustering=view_clustering,
                   view_counts=state_dict.get('view_counts', None))

  def state_dict(self):

    return dict(step=self.step, 
                scene=self.scene.state_dict(), 
                controller=self.controller.state_dict(),
                
                view_clustering=self.view_clustering.state_dict(),
                view_counts=self.view_counts)
  


  def clone(self) -> 'Trainer':
    state = self.state_dict()
    return self.from_state_dict(self.config, self.dataset, self.logger, state)

  @property
  def output_path(self):
    return Path.cwd() 

  def paths(self, step:Optional[int]=None):

    if step is None:
      step = self.step

    paths = dict(
      checkpoint = self.output_path / "checkpoint" / f"checkpoint_{step}.pt",
      point_cloud = self.output_path / "point_cloud" / f"iteration_{step}" / "point_cloud.ply",
      cameras = self.output_path / "checkpoint" / f"cameras.json",
      workspace = self.output_path 
    )

    for path in paths.values():
      path.parent.mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(**paths)
  
  def print(self, str:str):
    if self.pbar is not None:
      self.pbar.write(str)
    else:
      print(str)

  def write_checkpoint(self):
    paths = self.paths()
    
    with open(paths.cameras, "w") as f:
      json.dump(camera_json(self.camera_table), f)

    checkpoint = self.state_dict()
    torch.save(checkpoint, paths.checkpoint)

    write_gaussians(paths.point_cloud, self.scene.to_sh_gaussians(), with_sh=True)  
    self.print(f"Checkpoint saved to {colored(paths.checkpoint, 'light_green')}")




  def load_cloud(self) -> Gaussians3D:
    paths = self.paths()
    if paths.point_cloud.exists():  
      gaussians = read_gaussians(paths.point_cloud, with_sh=True)
    else:
      gaussians = self.scene.to_sh_gaussians()

    return gaussians.to(self.device)

  @property
  def progress(self) -> Progress:
    return Progress(step=self.step, total_steps=self.total_steps)
  
  @property
  def total_steps(self) -> int:
    return self.config.steps
  
  def __repr__(self):
    return f"Trainer(step={self.step}, scene={self.scene} controller={self.controller})"


  def camera_params(self, cam_idx:torch.Tensor):
    camera:Camera = self.camera_table[cam_idx].item()
    near, far = camera.depth_range

    return CameraParams(
            T_camera_world=camera.camera_t_world,
            projection=camera.intrinsics,
            image_size=camera.image_size,
            near_plane=near,
            far_plane=far,
        ).to(self.device, dtype=torch.float32)


  def update_config(self, **kwargs):
    self.config = replace(self.config, **kwargs)

  @beartype
  def render(self, camera_params:CameraParams, image_idx:Optional[int]=None, **options):

    return self.scene.render(camera_params, image_idx,  **options, 
      antialias=self.config.antialias, compute_visibility=True,
      blur_cov=0.0 if self.config.antialias is True else self.config.blur_cov)



  @beartype
  def evaluate_image(self, filename:str, camera_params:CameraParams, image_idx:int, source_image:torch.Tensor):
    rendering = self.render(camera_params, image_idx, render_median_depth=True).detach()
    return Evaluation(filename, rendering, source_image)



  def evaluate_training(self, name: str, data: Iterable[CameraView]):
    """Evaluate training set, log a selection of images, and worst performing images by psnr
       Compute view clustering
    """
    # Track the n worst images by psnr
    worst = Heap(self.config.log_worst_images) 
    view_features = []

    # Track metrics for all images
    metrics = {}

    point_clusters = cluster.PointClusters.cluster(self.scene.points['position'], self.config.vis_clusters)
    log_interval = len(data) // (self.config.num_logged_images + 1)

    pbar = tqdm(total=len(data), desc=f"rendering {name}", leave=False)
    for i, image_data in enumerate(self.iter_data(data)):
        eval = self.evaluate_image(*image_data)

        if (i + 1) % log_interval == 0:
            self.log_evaluation_images(f"{name}_images/{eval.image_id}", eval, log_source=self.step == 0)
            
        worst.push(-eval.metrics['psnr'], eval)

        metrics[eval.filename] = eval.metrics
        view_features.append(point_clusters.view_features(*eval.rendering.visible))
        
        pbar.update(1)
        pbar.set_postfix(**{k:f"{v:.3f}" for k, v in eval.metrics.items()})

    for i, (_, eval) in enumerate(worst):
        self.log_evaluation_images(f"worst_{name}/{i}", eval, log_source=True)

    self.log_evaluation_table(name, metrics)
    self.view_clustering = cluster.ViewClustering(point_clusters, torch.stack(view_features))

  
  def evaluate_dataset(self, name:str, data:Iterable[CameraView]):
    # Track metrics for all images
    metrics = {}
    metrics_cc = {}

    pbar = tqdm(total=len(data), desc=f"rendering {name}", leave=False)
    for i, image_data in enumerate(self.iter_data(data)):
        eval = self.evaluate_image(*image_data)
        eval_cc = eval.color_corrected()

        metrics[eval.filename] = eval.metrics
        metrics_cc[eval.filename] = eval_cc.metrics

        self.log_evaluation_images(f"{name}_images/{eval_cc.image_id}", eval_cc, log_source=self.step == 0)

        pbar.update(1)
        pbar.set_postfix(**{k:f"{v:.3f}" for k, v in eval_cc.metrics.items()})

    self.log_evaluation_table(name, metrics)
    self.log_evaluation_table(f"{name}_cc", metrics_cc)

  def log_colormapped(self, name, values):
    colorized = colorize(self.color_map, values)
    self.logger.log_image(name, colorized, compressed=False)



  def log_evaluation_images(self, name:str, eval:Evaluation, log_source:bool=True):
    self.logger.log_image(f"{name}/render", eval.rendering.image, 
                    caption=f"{eval.filename} PSNR={eval.psnr:.2f} L1={eval.l1:.2f} ssim={eval.ssim:.2f}")
    self.logger.log_image(f"{name}/depth", 
        colorize(self.color_map, eval.rendering.ndc_median_depth), caption=eval.filename)
    
    if log_source:
      self.logger.log_image(f"{name}/image", eval.source_image, caption=eval.filename)


  def log_evaluation_table(self, name:str, metrics:dict):
    self.logger.log_evaluations(f"eval_{name}/evals", metrics)
    totals = transpose_rows(list(metrics.values()))
    means = {k:float(np.mean(v)) for k, v in totals.items()}

    for k, v in totals.items():
        self.logger.log_value(f"eval_{name}/{k}", means[k])
        self.logger.log_histogram(f"eval_{name}/{k}_hist", torch.tensor(v))


  def evaluate(self):
    self.evaluate_training("train", self.dataset.train(shuffle=False))
    if self.camera_table.count_label(Label.Validation) > 0:
      self.evaluate_dataset("val", self.dataset.val())

  

  @property
  def view_weighting(self):
    return 1 / (self.view_counts + 1)

  def select_cluster(self) -> torch.Tensor:
    """ Select a cluster of cameras to train on for one densify/prune cycle.
    """
    assert self.view_clustering is not None, "View clustering not initialized, call evaluate() first"

    cluster_idx = self.view_clustering.select_batch(1 / (1 + self.view_weighting), 
              self.config.min_group_size, self.config.overlap_threshold)
    

    # Lookup camera table to get camera indexes 
    all_train_idx = self.camera_table.has_label(Label.Training)
    return all_train_idx[cluster_idx]
  
  
  @beartype
  def sample_batch(self, batch_size:int) -> torch.Tensor:

    weighting = F.normalize(1 / (self.view_counts + 1), p=1, dim=0)
    batch_idx = self.view_clustering.sample_batch(weighting, 
              batch_size, self.config.overlap_temperature)
    
    self.view_counts[batch_idx] += 1

    return batch_idx





  def load_data(self, camera_view:CameraView) -> Tuple[str, CameraParams, int, torch.Tensor]:
    filename, image, image_idx = camera_view
    
    image = image.to(self.device, non_blocking=True) 
    image = image.to(dtype=torch.float) / 255.0
    camera_params = self.camera_params(image_idx)

    return filename, camera_params, image_idx, image
  
  def iter_data(self, iter:Iterator[CameraView]) -> Iterator[Tuple[str, CameraParams, int, torch.Tensor]]:
    for camera_view in iter:
      yield self.load_data(camera_view)



  def compute_ssim_loss(self, pred:torch.Tensor, ref:torch.Tensor, levels:int=4):
      ref = ref.unsqueeze(0).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
      pred = pred.unsqueeze(0).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

      ssim = self.ssim(pred, ref)
      loss = 1.0 - ssim

      for i in range(1, levels):
        pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
        ref = F.avg_pool2d(ref, kernel_size=2, stride=2)

        loss += (1.0 - self.ssim(pred, ref)) 

      return loss / levels, ssim.item()
  


  def reg_loss(self, rendering:Rendering) -> Tuple[torch.Tensor, dict]:
    vis = rendering.point_visibility 

    scale_term =  (rendering.point_scale / rendering.camera.focal_length[0]).pow(2)
    aspect_term = vis * (rendering.point_scale.max(-1).values / (rendering.point_scale.min(-1).values + 1e-6))
    opacity_term = vis * (rendering.point_opacity)

    scale, opacity, aspect = [eval_varying(x, self.progress) 
          for x in [self.config.scale_reg, self.config.opacity_reg, self.config.aspect_reg]]

    
    regs = dict(
      scale_reg     =  (vis.unsqueeze(-1) * scale_term).mean() * scale,
      opacity_reg   =  (vis * opacity_term).mean() * opacity,  
      aspect_reg    =  (vis * aspect_term).mean() * aspect
    )

    # include total as "reg"
    metrics = {k:v.item() for k, v in regs.items()} 
    total = sum(regs.values())

    metrics["reg"] = total.item()
    return total, metrics

  def losses(self, rendering:Rendering, image:torch.Tensor):
    metrics = {}
    loss = 0.0

    if self.config.l1_weight > 0:
      l1 = torch.nn.functional.l1_loss(rendering.image, image)
      metrics["l1"] = l1.item()
      loss = l1 * self.config.l1_weight


    if self.config.ssim_weight > 0:  
      ssim_loss, ssim_metric = self.compute_ssim_loss(rendering.image, image, self.config.ssim_levels)
      loss += ssim_loss * self.config.ssim_weight 
      metrics["ssim"] = ssim_metric


    reg_loss, reg_losses = self.reg_loss(rendering)
    metrics.update(reg_losses)
    loss += reg_loss 

    return loss, metrics

  def evaluate_batch_with(self, batch_idx:torch.Tensor, f:Callable[[int, Rendering], None]) -> dict:
    iter_batch = self.iter_data(self.dataset.loader(batch_idx.cpu().numpy()))

    loss_metrics = []
    for filename, camera_params, image_idx, image in iter_batch:

      with torch.enable_grad():
        rendering = self.render(camera_params, image_idx, compute_point_heuristic=True)

        loss, metrics = self.losses(rendering, image)
        loss.backward()

      f(image_idx, rendering)
      loss_metrics.append(metrics)

    return loss_metrics
  
  def training_step(self, batch_idx:torch.Tensor) -> dict:
    def f(image_idx:int, rendering:Rendering):
      self.scene.add_rendering(image_idx, rendering)
      self.controller.add_rendering(image_idx, rendering)

    loss_metrics = self.evaluate_batch_with(batch_idx, f)

    self.scene.step(self.progress)
    self.controller.step(self.progress)
    self.logger.step(self.progress)

    self.step += batch_idx.shape[0]
    return dict(**mean_rows(loss_metrics),  t = self.progress.t)




  @property
  def all_parameters(self) -> TensorDict:
    return self.scene.all_parameters.to_dict()
  
  def zero_grad(self):
    self.scene.zero_grad()

  def batch_summary(self):
    heuristics, metrics = self.evaluate_batch(torch.arange(len(self.camera_table), device=self.device))
    print_stats(self.all_parameters)
    print_stats(heuristics.to_tensordict())
    print_table(pd.DataFrame([mean_rows(metrics)]))



  def is_training(self):
    return self.state in (TrainerState.Training, TrainerState.Paused)
  
  def set_paused(self, paused:bool):
    assert self.is_training()
    self.state = TrainerState.Paused if paused else TrainerState.Training
    self.pbar.set_description_str(self.state.name)    

  
  def checkpoint(self, save:bool=True):
    self.evaluate()

    self.scene.log_checkpoint()
    self.controller.log_checkpoint()

    if save and self.config.save_output:
      self.write_checkpoint()

    torch.cuda.empty_cache()

  def update_progress(self):
    self.pbar.update(self.progress.step - self.pbar.n)
    desc = []

    if "densify" in self.state_logger:
      pruned, split, n = [self.state_logger[k].value for k in ["densify/n_pruned", "densify/n_split", "densify/n_total"]]
      desc.append(f"points(-{pruned:d} +{split*2:d} = {n:d})")

    if "train" in self.state_logger:
      l1, ssim, reg = [self.state_logger[k].value for k in ["train/l1", "train/ssim", "train/reg"]]
      desc.append(f"train(l1:{l1:.2f} ssim:{ssim:.2f} reg:{reg:.4f})")

    if "eval_train" in self.state_logger:
      psnr, ssim = [self.state_logger[k].value for k in ["eval_train/psnr", "eval_train/ssim"]]
      desc.append(f"eval(psnr:{psnr:.2f} ssim:{ssim:.2f})")

    self.pbar.set_postfix_str(" ".join(desc))


  def train(self, state:TrainerState = TrainerState.Training):
    self.state = state
    self.pbar = tqdm(initial=self.step, total=self.config.steps, desc=self.state.name)

    next_densify = next_multiple(self.step, eval_varying(self.config.densify_interval, self.progress))
    while self.step < self.config.steps:

      self.emit("on_update")
      if self.state == TrainerState.Paused:
        time.sleep(0.1)
        continue
      
      if self.step - next_densify > 0:

        torch.cuda.empty_cache()
        self.controller.densify_and_prune(self.progress)
        next_densify += eval_varying(self.config.densify_interval, self.progress)

      if self.step % self.config.eval_steps == 0:
        self.checkpoint(self.config.save_checkpoints)

      batch_size = eval_varying(self.config.batch_size, self.progress)
      steps = [self.training_step(self.sample_batch(batch_size)) 
              for _ in range(self.config.log_interval)]

      self.logger.log_values("train", mean_rows(steps))
      self.logger.log_values("train", dict(step=self.step, batch_size=batch_size))
      self.update_progress()


    self.checkpoint(True)

    self.state = TrainerState.Stopped
    self.pbar.close()

    

  def close(self):
    self.logger.close()

