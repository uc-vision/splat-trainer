# Python standard library
from dataclasses import replace
from enum import Enum
from functools import partial
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Callable, Iterator, List, Optional, Sequence, Tuple


# Third party packages
from beartype import beartype
from beartype.typing import Optional
from fused_ssim import fused_ssim
import numpy as np
from pydispatch import Dispatcher
import colored_traceback
colored_traceback.add_hook()


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
from splat_trainer.dataset.dataset import ImageView

from splat_trainer.config import Progress, eval_varying
from splat_trainer.controller.controller import Controller

from splat_trainer.gaussians.loading import to_pointcloud
from splat_trainer.logger import Logger
from splat_trainer.logger.logger import CompositeLogger, StateLogger

from splat_trainer.scene.io import read_gaussians, write_gaussians
from splat_trainer.scene.scene import GaussianScene

from splat_trainer.visibility import cluster

from splat_trainer.util.colorize import colorize, get_cv_colormap
from splat_trainer.util.containers import mean_rows, transpose_rows

from splat_trainer.util.misc import Heap, next_multiple


from .config import TrainConfig
from .evaluation import Evaluation
from .init import get_initial_gaussians
from .loading import ThreadedLoader
from .view_selection import ViewSelection

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
                logger:CompositeLogger,
                
                view_selection:ViewSelection,

                step = 0,
                view_clustering:Optional[cluster.ViewClustering] = None,
      ):

    self.device = torch.device(config.device)
    self.controller = controller
    self.scene = scene
    self.dataset = dataset

    self.camera_table = dataset.camera_table().to(self.device)

    self.config = config
    self.logger = logger

    self.step = step
    self.last_checkpoint = step
    self.state = TrainerState.Stopped


    self.state_logger = StateLogger()
    self.logger = logger.add_logger(self.state_logger)
    self.loader = None

    self.view_selection = view_selection
    self.color_map = get_cv_colormap().to(self.device)
    self.ssim = partial(fused_ssim, padding="valid")
    self.pbar = None

    self.view_clustering = view_clustering




  @staticmethod
  def initialize(config:TrainConfig, dataset:Dataset, logger:Logger):

    device = torch.device(config.device)
    camera_table = dataset.camera_table().to(device)

    print(f"Initializing points from {dataset}")

    initial_gaussians = get_initial_gaussians(config.cloud_init, dataset, device)
    logger = CompositeLogger(logger)

    scene = config.scene.from_color_gaussians(initial_gaussians, camera_table, device, logger)
    controller = config.controller.make_controller(scene, logger)
    view_selector = config.view_selection.create(camera_table)

    if config.save_output:
      output_path = Path.cwd()

      to_pointcloud(initial_gaussians).save_ply(output_path / "input.ply")
      with open(output_path / "cameras.json", "w") as f:
        json.dump(camera_json(camera_table), f)

    return Trainer(config, scene, controller, dataset, logger, view_selector)
      

    
  @staticmethod
  def from_state_dict(config:TrainConfig, dataset:Dataset, logger:Logger, state_dict:dict):
    device = torch.device(config.device)
    camera_table = dataset.camera_table().to(device)

    logger = CompositeLogger(logger)

    scene = config.scene.from_state_dict(state_dict['scene'], camera_table, logger)
    controller = config.controller.from_state_dict(state_dict['controller'], scene, logger) 

    view_selection = config.view_selection.from_state_dict(state_dict['view_selection'], camera_table)

    if state_dict['view_clustering'] is not None:
      view_clustering = cluster.ViewClustering.from_state_dict(state_dict['view_clustering'])
    else:
      view_clustering = None

    return Trainer(config, scene, controller, dataset, logger, 
                   view_selection=view_selection,
                   step=state_dict['step'],
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
    return Progress(step=self.step, total_steps=self.config.total_steps)
  

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


  def update_config(self, **kwargs):
    self.config = replace(self.config, **kwargs)

  @beartype
  def render(self, camera_params:CameraParams, image_idx:Optional[int]=None, **options):

    return self.scene.render(camera_params, image_idx,  **options, 
      antialias=self.config.antialias, compute_visibility=True,
      blur_cov=0.0 if self.config.antialias is True else self.config.blur_cov)



  @beartype
  def evaluate_image(self, image_view:ImageView):
    image_view = self.load_data(image_view)

    camera_params = self.camera_params(image_view.image_idx)
    rendering = self.render(camera_params, image_view.image_idx, render_median_depth=True).detach()
    return Evaluation(image_view.filename, rendering, image_view.image)

  def evaluations(self, image_views:Sequence[ImageView]) -> Iterator[Evaluation]:
    for image_view in image_views:
      yield self.evaluate_image(image_view)


  def evaluate_training(self, name: str, image_views: Sequence[ImageView]):
    """Evaluate training set, log a selection of images, and worst performing images by psnr.
       Compute view clustering.
    """
    # Track the n worst images by psnr
    worst = Heap(self.config.log_worst_images) 
    view_features = []

    # Track metrics for all images
    metrics = {}

    point_clusters = cluster.PointClusters.cluster(self.scene.points['position'], self.config.vis_clusters)
    log_interval = len(image_views) // (self.config.num_logged_images + 1)

    pbar = tqdm(total=len(image_views), desc=f"Evaluating {name}", leave=False)
    for i, image_view in enumerate(image_views):
        eval = self.evaluate_image(image_view)

        if (i + 1) % log_interval == 0:
            self.log_evaluation_images(f"{name}_images/{eval.image_id}", eval, log_source=self.step == 0)
            
        worst.push(-eval.metrics['psnr'], eval)

        metrics[eval.filename] = eval.metrics
        view_features.append(point_clusters.view_features(*eval.rendering.visible))
        
        pbar.update(1)
        pbar.set_postfix_str(", ".join([f"{k}:{v:.3f}" for k, v in eval.metrics.items()]))

    for i, (_, eval) in enumerate(worst):
        self.log_evaluation_images(f"worst_{name}/{i}", eval, log_source=True)

    self.log_evaluation_table(name, metrics)
    self.view_clustering = cluster.ViewClustering(point_clusters, torch.stack(view_features))
    

  
  def evaluate_dataset(self, name:str, image_views:Sequence[ImageView]):
    """ Evaluate dataset, log images and color corrected images.
    """
    # Track metrics for all images
    metrics = {}
    metrics_cc = {}

    pbar = tqdm(total=len(image_views), desc=f"Evaluating {name}", leave=False)
    for i, image_view in enumerate(image_views):
        eval = self.evaluate_image(image_view)
        eval_cc = eval.color_corrected()

        metrics[eval.filename] = eval.metrics
        metrics_cc[eval_cc.filename] = eval_cc.metrics

        self.log_evaluation_images(f"{name}_images/{eval_cc.image_id}", eval_cc, log_source=self.step == 0)

        pbar.update(1)
        pbar.set_postfix_str(", ".join([f"{k}:{v:.3f}" for k, v in eval_cc.metrics.items()]))


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
    means = mean_rows(list(metrics.values()))

    for k, v in means.items():
        self.logger.log_value(f"eval_{name}/{k}", v)
        self.logger.log_histogram(f"eval_{name}/{k}_hist", torch.tensor(v))


  def evaluate(self):
    self.evaluate_training("train", self.dataset.train(shuffle=False))
    val = self.dataset.val()
    if len(val) > 0:
      self.evaluate_dataset("val", val)

    self.update_progress()
  

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
    aspect_term = (rendering.point_scale.max(-1).values / (rendering.point_scale.min(-1).values + 1e-6))
    opacity_term = rendering.point_opacity

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
    loss = torch.tensor(0.0, device=self.device)

    if self.config.l1_weight > 0:
      l1 = torch.nn.functional.l1_loss(rendering.image, image)
      metrics["l1"] = l1.item()
      loss += l1 * self.config.l1_weight 

    if self.config.mse_weight > 0:
      mse = torch.nn.functional.mse_loss(rendering.image, image)
      metrics["mse"] = mse.item()
      loss += mse * self.config.mse_weight 

    if self.config.ssim_weight > 0:  
      ssim_loss, ssim_metric = self.compute_ssim_loss(rendering.image, image, self.config.ssim_levels)
      loss += ssim_loss * self.config.ssim_weight 
      metrics["ssim"] = ssim_metric

    reg_loss, reg_losses = self.reg_loss(rendering)
    metrics.update(reg_losses)
    loss += reg_loss 

    return loss, metrics

  def evaluate_backward_with(self, batch:List[ImageView], f:Callable[[int, Rendering], None]) -> dict:
    metrics = []
    for image_view in batch:

      camera_params = self.camera_params(image_view.image_idx)
      with torch.enable_grad():
        rendering = self.render(camera_params, image_view.image_idx, compute_point_heuristic=True)

        loss, loss_metrics = self.losses(rendering, image_view.image)
        loss.backward()

      f(image_view.image_idx, rendering)
      metrics.append(loss_metrics)

    return metrics
  
  def training_step(self, batch:List[ImageView]) -> dict:
    def f(image_idx:int, rendering:Rendering):
      self.scene.add_rendering(image_idx, rendering)
      self.controller.add_rendering(image_idx, rendering)

    metrics = self.evaluate_backward_with(batch, f)
    self.step += len(batch)

    self.scene.step(self.progress)
    self.controller.step(self.progress)
    self.logger.step(self.progress)

    return dict(**mean_rows(metrics), t = self.progress.t)
  

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
    self.evaluate()
    
    self.scene.log_checkpoint()
    self.controller.log_checkpoint()

    if save and self.config.save_output:
      self.write_checkpoint()

    self.last_checkpoint = self.step
    torch.cuda.empty_cache()

  def update_progress(self):
    if self.pbar is None:
      return

    self.pbar.update(self.progress.step - self.pbar.n)
    desc = []

    if "densify" in self.state_logger:
      pruned, split, n = [self.state_logger[k].value for k in ["densify/prune", "densify/split", "densify/n"]]
      desc.append(f"points(+{split:d} -{pruned:d} = {n:d})")

    if "train" in self.state_logger:
      values = self.state_logger["train"]
      for k in ["l1", "mse", "ssim", "reg"]:
        if k in values:
          desc.append(f"{k}:{values[k].value:.3f}")
      

    if "eval_train" in self.state_logger:
      psnr, ssim = [self.state_logger[k].value for k in ["eval_train/psnr", "eval_train/ssim"]]
      desc.append(f"eval(psnr:{psnr:.3f} ssim:{ssim:.3f})")

    self.pbar.set_postfix_str(" ".join(desc))


  def train(self, state:TrainerState = TrainerState.Training):
    self.state = state
    next_densify = next_multiple(self.step, eval_varying(self.config.densify_interval, self.progress))

    self.checkpoint(self.config.save_checkpoints)

    # Create batch loader that loads log_interval batches at a time
    self.loader = ThreadedLoader(self.iter_batches())
    self.pbar = tqdm(initial=self.step, total=self.config.total_steps, desc=self.state.name)

    while self.step < self.config.total_steps:
        self.emit("on_update")
        if self.state == TrainerState.Paused:
            time.sleep(0.1)
            continue

        if self.last_checkpoint + self.config.eval_steps <= self.step:
            self.checkpoint(self.config.save_checkpoints)

        if self.step - next_densify >= 0:
            next_densify += eval_varying(self.config.densify_interval, self.progress)
            if next_densify < self.config.total_steps:
                self.controller.densify_and_prune(self.progress)

        steps = [self.training_step(self.loader.next()) for _ in range(self.config.log_interval)]

        self.logger.log_values("train", mean_rows(steps))
        self.update_progress()

    self.checkpoint(True)

    self.state = TrainerState.Stopped
    self.pbar.close()

  def close(self):
    self.logger.close()

    if self.loader is not None:
      self.loader.stop()  # Ensure we clean up the thread



