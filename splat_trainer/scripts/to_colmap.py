from pathlib import Path
import hydra
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored
from splat_trainer.logger.logger import Logger, NullLogger
from splat_trainer import config

import torch
from splat_trainer.scripts.train_scan import cfg_from_args
from splat_trainer.trainer import Trainer



config.add_resolvers()

def export_colmap(trainer: Trainer):
    import pycolmap

    # Get cameras from trainer
    cameras = trainer.dataset.cameras

    # Create a COLMAP reconstruction object
    reconstruction = pycolmap.Reconstruction()

    # Add cameras to the reconstruction
    for idx, camera in enumerate(cameras):
        # Assuming camera parameters are available in the trainer's dataset
        # Adjust these lines based on your specific camera representation
        width, height = camera.image_size
        fx, fy = camera.focal_length
        cx, cy = camera.principal_point

        colmap_camera = pycolmap.Camera(
            model="PINHOLE",
            width=width,
            height=height,
            params=[fx, fy, cx, cy]
        )
        reconstruction.add_camera(colmap_camera, camera_id=idx)

    # Export the reconstruction to a COLMAP format
    output_path = Path("sparse") / "0"
    output_path.mkdir(parents=True, exist_ok=True)

    cloud = trainer.dataset.points

    reconstruction.write_text(output_path)
    print(f"Cameras exported to COLMAP format in {output_path}")


def export_with_config(cfg) -> dict | str:
  import taichi as ti
  from splat_trainer.trainer import Trainer

  torch.set_grad_enabled(False)
  torch.set_float32_matmul_precision('high')

  torch.set_printoptions(precision=4, sci_mode=False)
  np.set_printoptions(precision=4, suppress=True)

  output_path = Path.cwd()
  print(f"Exporting to {colored(output_path, 'light_green')}")

  logger:Logger = NullLogger()

  try:
    
    train_config = hydra.utils.instantiate(cfg.trainer)
    dataset = hydra.utils.instantiate(cfg.dataset)
    
    trainer = Trainer.initialize(train_config, dataset, logger)
    
    export_colmap(trainer)

  except KeyboardInterrupt:
    pass

  finally:
    if trainer is not None:
      trainer.close()
    
    logger.close()
    
def main():
  cfg = cfg_from_args()
  export_with_config(cfg)


if __name__ == "__main__":
  main()