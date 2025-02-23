# Python standard library
from collections import deque
from dataclasses import replace
from enum import Enum
from functools import partial
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
from beartype.typing import Callable, Iterator, List, Optional, Sequence


# Third party packages
from beartype import beartype
from fused_ssim import fused_ssim
import numpy as np
from pydispatch import Dispatcher
import colored_traceback

from splat_trainer.dataset.normalization import Normalization
from tensordict import TensorDict
from termcolor import colored
import torch
import torch.nn.functional as F
from tqdm import tqdm

from taichi_splatting import Gaussians3D, Rendering
from taichi_splatting.perspective import CameraParams

# Local imports
from splat_trainer.camera_table.camera_table import Camera, camera_json
from splat_trainer.dataset import Dataset
from splat_trainer.dataset.dataset import ImageView

from splat_trainer.config import Progress
from splat_trainer.controller.controller import Controller

from splat_trainer.gaussians.loading import to_pointcloud
from splat_trainer.logger import Logger
from splat_trainer.logger.logger import LoggerWithState

from splat_trainer.scene.io import read_gaussians, write_gaussians
from splat_trainer.scene.scene import GaussianScene

from splat_trainer.visibility import cluster

from splat_trainer.util.colorize import colorize, get_cv_colormap
from splat_trainer.util.containers import transpose_rows

from splat_trainer.util.misc import Heap, format_dict

from taichi_splatting.torch_lib.util import count_nonfinite
from . import exception

from .config import TrainConfig
from .evaluation import Evaluation
from .init import get_initial_gaussians
from .loading import ThreadedLoader
from .view_selection import ViewSelection

colored_traceback.add_hook()

class TrainerState(Enum):
  Stopped = 0
  Training = 1
  Paused = 2


class Trainer(Dispatcher):
  _events_ = ["on_update"]

  @beartype
  def __init__(self, config:TrainConfig,
                scene:GaussianScene, 
                controller:Controller,
                dataset:Dataset,              
                logger:LoggerWithState,
                
                view_selection:ViewSelection,
                step = 0,
                evaluation_log:Optional[List[dict]] = None,

                view_clustering:Optional[cluster.ViewClustering] = None,
      ):
    
    assert torch.get_float32_matmul_precision() == 'highest', \
      f"lower precision causes inaccuracy in 4x4 transforms {torch.get_float32_matmul_precision()}"
        

    self.device = torch.device(config.device)
    self.controller = controller
    self.scene = scene
    self.dataset = dataset

    self.camera_table = dataset.camera_table.to(self.device)

    self.config = config
    self.logger = logger

    self.step = step

    self.last_checkpoint = step
    self.state = TrainerState.Stopped

    self.logger = logger
    self.loader = None

    self.view_selection = view_selection
    self.color_map = get_cv_colormap().to(self.device)
    self.ssim = partial(fused_ssim, padding="valid")
    self.pbar = None

    self.view_clustering = view_clustering
    self.evaluation_log = evaluation_log or []
    
    self.running_time = deque(maxlen=10)
    self.last_time = None

  @staticmethod
  def initialize(config:TrainConfig, dataset:Dataset, logger:Logger):

    device = torch.device(config.device)
    camera_table = dataset.camera_table.to(device)

    print(f"Loading from {colored(dataset, 'light_green')}")
    initial_gaussians = get_initial_gaussians(config.cloud_init, dataset, device)

    if not isinstance(logger, LoggerWithState):
      logger = LoggerWithState(logger)

    scene = config.scene.from_color_gaussians(initial_gaussians, camera_table, device, logger)
    progress = Progress(step=0, total_steps=config.total_steps, logging_step=False)
    
    controller = config.controller.make_controller(scene, config.target_points, progress, logger)
    view_selector = config.view_selection.create(camera_table)

    trainer = Trainer(config, scene, controller, dataset, logger, view_selector)

    # Write initial cloud and cameras in original coordinates (undo scene normalisation)
    paths = trainer.paths()
    if config.save_output:
      trainer.write_cameras(paths.cameras)

      initial = to_pointcloud(dataset.to_original.transform_gaussians(initial_gaussians))
      initial.save_ply(paths.point_cloud)

    return trainer

  @property
  def unnormalize(self) -> Normalization:
    return self.dataset.to_original


  @staticmethod
  def from_state_dict(config:TrainConfig, dataset:Dataset, logger:Logger, state_dict:dict):
    device = torch.device(config.device)
    camera_table = dataset.camera_table.to(device)
  
    if not isinstance(logger, LoggerWithState):
      logger = LoggerWithState(logger)

    step = state_dict['step']

    scene = config.scene.from_state_dict(state_dict['scene'], camera_table, logger)

    progress = Progress(step=step, total_steps=config.total_steps)
    controller = config.controller.from_state_dict(
      state_dict['controller'], scene, 
      target_points=config.target_points,
      progress=progress, logger=logger) 

    view_selection = config.view_selection.from_state_dict(state_dict['view_selection'], camera_table)

    if state_dict['view_clustering'] is not None:
      view_clustering = cluster.ViewClustering.from_state_dict(state_dict['view_clustering'])
    else:
      view_clustering = None

    evaluation_log = state_dict.get('evaluation_log', None)

    return Trainer(config, scene, controller, dataset, logger, 
                   view_selection=view_selection,
                   step=step, evaluation_log=evaluation_log,
                   view_clustering=view_clustering)

  def state_dict(self):

    return dict(step=self.step,
                scene=self.scene.state_dict(), 
                controller=self.controller.state_dict(),
                view_selection=self.view_selection.state_dict(),
                view_clustering=self.view_clustering.state_dict())
  


  def clone(self) -> 'Trainer':
    state = self.state_dict()
    return self.from_state_dict(self.config, self.dataset, self.logger, state)

  def replace(self, logger:Optional[Logger]=None, **kwargs) -> 'Trainer':
    """ Basically, clone with changed config."""
    trainer = self.state_dict()
    
    config = replace(self.config, **kwargs)
    return self.from_state_dict(config, self.dataset, logger or self.logger, trainer)

  def update_config(self, **kwargs):
    self.config = replace(self.config, **kwargs)
    


  @property
  def output_path(self):
    return Path.cwd() 

  def paths(self, step:Optional[int]=None):

    if step is None:
      step = self.step

    paths = dict(
      checkpoint = self.output_path / "checkpoint" / f"checkpoint_{step}.pt",
      point_cloud = self.output_path / "point_cloud" / f"iteration_{step}" / "point_cloud.ply",
      cameras = self.output_path / "cameras.json",
      workspace = self.output_path 
    )

    for path in paths.values():
      path.parent.mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(**paths)
  
  def print(self, str:str):

    interactive = os.environ.get("TQDM_DISABLE", False)
    if self.pbar is not None and interactive:
      self.pbar.write(str)
    else:
      print(str)

  def write_cameras(self, filename:str):
    with open(filename, "w") as f:
      cameras = self.unnormalize.transform_cameras(self.camera_table.cameras)
      json.dump(camera_json(cameras), f, indent=2, sort_keys=True)


  def write_checkpoint(self):
    paths = self.paths()

    checkpoint = self.state_dict()
    torch.save(checkpoint, paths.checkpoint)

    write_gaussians(paths.point_cloud, self.sh_gaussians(), with_sh=True)  
    self.print(f"Checkpoint saved to {colored(paths.checkpoint, 'light_green')}")


  @property
  def scene_transform(self):
    translation, scale = self.dataset.scene_transform
    return translation.to(self.device), scale

  def sh_gaussians(self) -> Gaussians3D:
    return self.unnormalize.transform_gaussians(self.scene.to_sh_gaussians())

  def load_cloud(self) -> Gaussians3D:
    paths = self.paths()
    if paths.point_cloud.exists():  
      gaussians = read_gaussians(paths.point_cloud, with_sh=True)
    else:
      gaussians = self.sh_gaussians()

    return gaussians.to(self.device)


  @property
  def progress(self) -> Progress:
    return Progress(step=self.step, 
                    total_steps=self.config.total_steps,
                    logging_step=self.is_logging_step)
  
  @property
  def is_logging_step(self):
    return self.step % self.config.log_interval == 0

  def __repr__(self):
    return f"Trainer(step={self.step}, scene={self.scene} controller={self.controller})"


  def camera_params(self, cam_idx:int) -> CameraParams:
    camera:Camera = self.camera_table[cam_idx].item()
    near, far = camera.depth_range

    return CameraParams(
            T_camera_world=camera.camera_t_world,
            projection=camera.intrinsics,
            image_size=camera.image_size,
            near_plane=near,
            far_plane=far,
        ).to(self.device, dtype=torch.float32)



  @beartype
  def render(self, camera_params:CameraParams, image_idx:Optional[int]=None, **options) -> Rendering:

    return self.scene.render(camera_params, image_idx,  **options, 
      antialias=self.config.antialias, compute_visibility=True, compute_point_heuristic=True,
      blur_cov=0.0 if self.config.antialias is True else self.config.blur_cov)



  @beartype
  def evaluate_image(self, image_view:ImageView):
    image_view = self.load_data(image_view)
    camera_params = self.camera_params(image_view.image_idx)

    rendering = self.render(camera_params, image_view.image_idx, render_median_depth=True)
    return Evaluation(image_view.filename, rendering.detach(), image_view.image)

  def iter_evaluations(self, image_views:Sequence[ImageView]) -> Iterator[Evaluation]:
    for image_view in image_views:
      yield self.evaluate_image(image_view)


  def evaluate_training(self, name: str, image_views: Sequence[ImageView], random_seed:int=0):
    """Evaluate training set, log a selection of images, and worst performing images by psnr.
       Compute view clustering.
    """
    # Track the n worst images by psnr
    worst = Heap(self.config.log_worst_images) 
    view_features = []

    # Track metrics for all images
    metrics = {}
    point_clusters = cluster.PointClusters.cluster(self.scene.gaussians.position, self.config.vis_clusters)

    # record how many times each point is visible from each camera
    point_visible = torch.zeros(self.scene.num_points, dtype=torch.int, device=self.device)

    rng = np.random.RandomState(random_seed)
    log_indices = set(rng.choice(len(image_views), min(self.config.num_logged_images, len(image_views)), replace=False))

    pbar = tqdm(total=len(image_views), desc=f"Evaluating {name}", leave=False, 
                disable=os.environ.get("TQDM_DISABLE", False))
    for i, image_view in enumerate(image_views):
        eval = self.evaluate_image(image_view)

        if i in log_indices:
            self.log_evaluation_images(f"{name}_images/{eval.image_id}", eval, log_source=self.step == 0)
            
        worst.push(-eval.metrics['psnr'], eval)

        metrics[eval.filename] = eval.metrics
        points = eval.rendering.points.visible
        view_features.append(point_clusters.view_features(points.idx, points.visibility))

        point_visible[points.idx] += 1
        
        pbar.update(1)
        pbar.set_postfix_str(", ".join([f"{k}:{v:.4f}" for k, v in eval.metrics.items()]))

    for i, (_, eval) in enumerate(worst):
        self.log_evaluation_images(f"{name}_images/worst_{i}", eval, log_source=True)

    self.logger.log_histogram(f"eval_{name}/points_visible", point_visible)

    self.log_evaluation_table(name, metrics)
    self.view_clustering = cluster.ViewClustering(point_clusters, torch.stack(view_features))
    

  
  def evaluate_dataset(self, name:str, image_views:Sequence[ImageView], random_seed:int=0):
    """ Evaluate dataset, log images and color corrected images.
    """
    # Track metrics for all images
    metrics = {}
    metrics_cc = {}

    rng = np.random.RandomState(random_seed)
    log_indices = set(rng.choice(len(image_views), min(self.config.num_logged_images, len(image_views)), replace=False))


    pbar = tqdm(total=len(image_views), desc=f"Evaluating {name}", leave=False, 
                disable=os.environ.get("TQDM_DISABLE", False))
    for i, image_view in enumerate(image_views):
        eval = self.evaluate_image(image_view)
        eval_cc = eval.color_corrected()

        metrics[eval.filename] = eval.metrics
        metrics_cc[eval_cc.filename] = eval_cc.metrics

        if i in log_indices:
          self.log_evaluation_images(f"{name}_images/{eval_cc.image_id}", eval_cc, log_source=self.step == 0)

        pbar.update(1)
        pbar.set_postfix_str(format_dict(eval_cc.metrics, precision=3))


    self.log_evaluation_table(name, metrics)
    self.log_evaluation_table(f"{name}_cc", metrics_cc)

  

  def log_colormapped(self, name, values):
    colorized = colorize(self.color_map, values)
    self.logger.log_image(name, colorized, compressed=False)


  def log_evaluation_images(self, name:str, eval:Evaluation, log_source:bool=True):
    if self.config.log_images:
      self.logger.log_image(f"{name}/render", eval.rendering.image, 
                      caption=f"{eval.filename} PSNR={eval.psnr:.3f} L1={eval.l1:.2f} ssim={eval.ssim:.2f}")
      self.logger.log_image(f"{name}/depth", 
          colorize(self.color_map, eval.rendering.median_ndc_image), caption=eval.filename)
      
      if log_source:
        self.logger.log_image(f"{name}/image", eval.source_image, caption=eval.filename)


  def log_evaluation_table(self, name:str, metrics:dict):
    self.logger.log_evaluations(f"eval_{name}/evals", metrics)

    metrics = transpose_rows(list(metrics.values()))
    means = {k:np.mean(v) for k, v in metrics.items()}

    for k, v in metrics.items():
        self.logger.log_value(f"eval_{name}/{k}", means[k])
        self.logger.log_histogram(f"eval_{name}/{k}_hist", torch.tensor(v))



  def evaluate(self):
    self.print(f"Performing evaluation at step {self.step}")

    self.evaluate_training("train", self.dataset.train(shuffle=False))
    val = self.dataset.val()
    if len(val) > 0:
      self.evaluate_dataset("val", val)

    means = self.eval_metrics()
    self.print(f"step={self.step:<6d} n={self.scene.num_points:<7} {format_dict(means,  precision=4)}")

    self.update_progress()

    return means
  

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


  def compute_losses(self, rendering:Rendering, image:torch.Tensor):
    
    def compute_l1(rendering:Rendering, image:torch.Tensor):
      l1 = torch.nn.functional.l1_loss(rendering.image, image)
      weighted_l1 = l1 * self.config.l1_weight 
      return weighted_l1, l1.item()
    
    def compute_mse(rendering:Rendering, image:torch.Tensor):
      mse = torch.nn.functional.mse_loss(rendering.image, image)
      weighted_mse = mse * self.config.mse_weight 
      return weighted_mse, mse.item()
    
    def compute_ssim(rendering:Rendering, image:torch.Tensor):
      ssim_loss, ssim_metric = self.compute_ssim_loss(rendering.image, image, self.config.ssim_levels)
      weighted_ssim = ssim_loss * self.config.ssim_weight 
      return weighted_ssim, ssim_metric

    loss_funcs = dict(l1=compute_l1, mse=compute_mse, ssim=compute_ssim)
    loss_metrics = [func(rendering, image) for func in loss_funcs.values()]

    losses, metrics = zip(*loss_metrics)

    reg_loss = self.scene.reg_loss(rendering, self.progress)
    total = sum(losses) + reg_loss 
    
    if self.is_logging_step:
      loss_dict = {k:v.item() for k, v in zip(loss_funcs.keys(), losses)}
      metrics_dict = dict(zip(loss_funcs.keys(), metrics))
      psnr = 10 * math.log10(1 / metrics_dict['mse'])  

      self.logger.log_values("train/loss", dict(**loss_dict, reg=reg_loss.item(), total=total.item()))
      self.logger.log_values("train/metrics", dict(**metrics_dict, psnr = psnr))
    
    return total

  def evaluate_backward_with(self, batch:List[ImageView], f:Callable[[int, Rendering], None]):
    for image_view in batch:

      camera_params = self.camera_params(image_view.image_idx)
      with torch.enable_grad():
        rendering = self.render(camera_params, image_view.image_idx)

        if rendering.points.num_visible == 0:
          filename = image_view.filename
          raise exception.TrainingException(f"No visible points: {filename} - check training parameters or dataset camera poses")

        loss = self.compute_losses(rendering, image_view.image) 
        loss.backward()

      f(image_view.image_idx, rendering)

  

  def log_rendering_histograms(self, rendering:Rendering):
    points = rendering.points.visible
    
    def log_scale_histogram(name:str, values:torch.Tensor, min_val:float = 1e-12):
      valid = values[values > min_val]
      self.logger.log_histogram(name, torch.log10(valid))

    log_scale_histogram("rendering/log10_prune_cost", points.prune_cost, min_val=1e-20)
    log_scale_histogram("rendering/log10_split_score", points.split_score, min_val=1e-10)
    log_scale_histogram("rendering/log10_max_scale_px", points.screen_scale, min_val=1e-6)
    log_scale_histogram("rendering/log10_visibility", points.visibility, min_val=1e-10)


  def training_step(self, batch:List[ImageView]):
    self.step += len(batch)

    @torch.no_grad()
    def f(image_idx:int, rendering:Rendering):
      self.scene.add_rendering(image_idx, rendering)
      self.controller.add_rendering(image_idx, rendering, self.progress)

      if self.config.log_details and self.is_logging_step:
        self.log_rendering_histograms(rendering)

    self.evaluate_backward_with(batch, f)

    self.scene.step(self.progress, log_details=self.config.log_details and self.is_logging_step)
    self.logger.step(self.progress)


  def load_data(self, image_view:ImageView) -> ImageView:

    image = image_view.image.to(self.device, non_blocking=True) 
    image = image.to(dtype=torch.float) / 255.0

    return replace(image_view, image=image)

  def load_batch(self, batch_idx:torch.Tensor) -> List[ImageView]:
    return [self.load_data(view) for view in self.dataset.loader(batch_idx)]

  def iter_batches(self) -> Iterator[List[ImageView]]:
    while True:
      batch_idx = self.view_selection.select_images(self.view_clustering, self.progress)
      yield [self.load_data(view) for view in self.dataset.loader(batch_idx)]
        
  @property
  def all_parameters(self) -> TensorDict:
    return self.scene.all_parameters.to_dict()
  
  def zero_grad(self):
    self.scene.zero_grad()


  def is_training(self):
    return self.state in (TrainerState.Training, TrainerState.Paused)
  
  def set_paused(self, paused:bool):
    assert self.is_training()
    self.state = TrainerState.Paused if paused else TrainerState.Training
    self.pbar.set_description_str(self.state.name)    

  
  def checkpoint(self, save:bool=True):
    def check_nan(t, name):
      d = count_nonfinite(t, name)
      if len(d) > 0:
        raise exception.NaNParameterException(f'Non-finite entries detected: {d}')
      
    check_nan(self.scene.state_dict(), "scene")
    

    self.scene.log_checkpoint(self.progress)
    metrics = self.evaluate()

    if len(self.evaluation_log) > 0:
      ssim = metrics['train_ssim']
      prev_ssim = self.evaluation_log[-1]['train_ssim']
      initial_ssim = self.evaluation_log[0]['train_ssim']

      if ssim < initial_ssim:
        raise exception.NoProgressException("Training aborted - ssim is worse than untrained ssim")
      
      if prev_ssim > ssim + self.config.max_ssim_regression:
        raise exception.NoProgressException(f"Training aborted - ssim regression exceeded {prev_ssim:4f} - {ssim:4f} > {self.config.max_ssim_regression:4f}")

    self.evaluation_log.append(dict(step = self.step, **metrics))
    
    if save and self.config.save_output:
      self.write_checkpoint()

    self.last_checkpoint = self.step
    torch.cuda.empty_cache()

  def pbar_metrics(self):
    """ Fetch metrics from logger and format them for the progress bar """
    desc = []

    if "densify" in self.logger:
      pruned, split, n = [self.logger[k].value 
          for k in ["densify/prune", "densify/split", "densify/n"]]
      desc.append(f"points(+{split:d} -{pruned:d} = {n:d})")

    if "train/metrics" in self.logger:
      ssim, psnr = [self.logger[k].value 
          for k in ["train/metrics/ssim", "train/metrics/psnr"]]
      
      desc.append(f"ssim:{ssim:.3f} psnr:{psnr:.3f}")

    if "train/loss" in self.logger:
      values = self.logger["train/loss"]
      losses = [f"{k}:{v.value:.3f}" for k, v in values.items()]
      desc.append(f"losses({', '.join(losses)})")

    return desc

  def update_progress(self):
    if self.pbar is not None:
      
      self.pbar.update(self.progress.step - self.pbar.n)
      self.pbar.set_postfix_str(" ".join(self.pbar_metrics()))

      current_time = time.time()

      # Start timing after the first logging interval (or evaluation) to avoid startup time
      if self.last_time is not None:  
        self.running_time.append(current_time - self.last_time)

        step_rate = self.config.log_interval / np.mean(self.running_time)
        self.logger.log_value("train/step_rate", step_rate)

        # Check for excessively slow training
        if self.config.min_step_rate is not None and len(self.running_time)  == self.running_time.maxlen:

          # give some extra leeway near the start
          if step_rate < self.config.min_step_rate:
            raise exception.TrainingTimeoutException(f"Step rate lower than config.min_step_rate: {step_rate:.3f} < {self.config.min_step_rate:.3f}")

      self.last_time = current_time

  def train(self, state:TrainerState = TrainerState.Training):
    self.state = state
    # Explicitly load images to avoid progress bars being messed up
    self.dataset.load_images()

    self.checkpoint(self.config.save_checkpoints)

    self.print(f"Beginning training for {self.config.total_steps - self.step} steps, eval_steps={self.config.eval_steps}")


    # Create batch loader that loads log_interval batches at a time
    self.loader = ThreadedLoader(self.iter_batches())
    self.pbar = tqdm(initial=self.step, total=self.config.total_steps, desc=self.state.name,
                     disable=os.environ.get("TQDM_DISABLE", False))

    while self.step < self.config.total_steps:

      images = self.loader.next()
      self.training_step(images)

      if self.last_checkpoint + self.config.eval_steps <= self.step:
          self.checkpoint(self.config.save_checkpoints or self.step == self.config.total_steps)     
          self.last_time = None 

      self.controller.step(self.progress)

      if self.is_logging_step:
        self.emit("on_update")
        if self.state == TrainerState.Paused:
            time.sleep(0.1)
            continue

        self.update_progress()

    self.state = TrainerState.Stopped
    self.pbar.close()
  
    return self.eval_metrics()
  
  def eval_metrics(self, metrics:tuple[str, ...]=("ssim", "psnr")):
    result = {}
    for category in ["train", "val", "val_cc"]:
      name = f"eval_{category}"
      if name in self.logger:
        values = self.logger[name]
        result.update({f"{category}_{k}": v.value for k, v in values.items() if k in metrics})

    return result

  def close(self):
    self.logger.close()

    if self.loader is not None:
      self.loader.stop()  # Ensure we clean up the thread



