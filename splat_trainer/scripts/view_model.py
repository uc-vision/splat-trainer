from argparse import ArgumentParser
from dataclasses import replace
import hydra
from pathlib import Path
import time
import torch
from typing import Tuple

import nerfview
from omegaconf import OmegaConf
import torch
import viser

from .evaluate import find_checkpoint
from splat_trainer.config import add_resolvers
from taichi_splatting.perspective import CameraParams
from taichi_splatting.taichi_queue import TaichiQueue




def parse_args():
  parser = ArgumentParser()
  parser.add_argument("splat_path", type=Path,  help="Path to output folder from splat-trainer")
  parser.add_argument("--step", type=int, default=None, help="Checkpoint from step to evaluate")
  parser.add_argument("--debug", action="store_true", help="Enable debug in taichi")

  args = parser.parse_args()
  return args



def main():
  args = parse_args()
  
  import taichi as ti
  TaichiQueue.init(arch=ti.cuda, debug=args.debug, device_memory_GB=0.1, threaded=True)
  
  add_resolvers()
  config = OmegaConf.load(args.splat_path / "config.yaml")
  
  train_config = hydra.utils.instantiate(config.trainer)
  dataset = hydra.utils.instantiate(config.dataset)
  
  checkpoint = find_checkpoint(args.splat_path, args.step)
  print(f"Loading checkpoint {checkpoint}")
  
  state_dict = torch.load(checkpoint, weights_only=True)
  scene = train_config.scene.from_state_dict(state_dict['scene'], dataset.view_info().camera_table)
  
  
  def viewer_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
    
    c2w = camera_state.c2w
    K = camera_state.get_K(img_wh)
    c2w = torch.from_numpy(c2w).float().to(train_config.device)
    K = torch.from_numpy(K).float().to(train_config.device)
    projection = torch.tensor([K[0,0], K[1,1], K[0,2], K[1,2]], device=train_config.device)
    near, far = dataset.depth_range()
    camera_params = CameraParams(projection=projection,
                                T_camera_world=c2w,
                                near_plane=near,
                                far_plane=far,
                                image_size=img_wh)

    config = replace(train_config.raster_config, compute_point_heuristics=True,
                    antialias=train_config.antialias,
                    blur_cov=train_config.blur_cov)
    rendering = scene.render(camera_params, config, 0)

    return rendering.image.detach().cpu().numpy()
  
  _ = nerfview.Viewer(
      server=viser.ViserServer(port=8080, verbose=False),
      render_fn=viewer_render_fn,
      mode="rendering",
  )
  print("Viewer running... Ctrl+C to exit.")
  time.sleep(100000)






if __name__ == "__main__":
  main()