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
from splat_trainer.util.transforms import split_rt



config.add_resolvers()



def export_colmap(trainer: Trainer):
    import pycolmap

    # Get cameras from trainer
    cameras = trainer.dataset.camera_table()

    # Create a COLMAP reconstruction object
    reconstruction = pycolmap.Reconstruction()

    # Add cameras to the reconstruction
    for idx in range(cameras.num_images):
        camera_t_world, proj = cameras.lookup(idx)
        fx, fy, cx, cy = proj.cpu().tolist()
        
        # Assuming image sizes are available in the trainer's dataset
        width, height = trainer.dataset.image_sizes()[idx].tolist()

        colmap_camera = pycolmap.Camera(
            model="PINHOLE",
            width=width,
            height=height,
            params=[fx, fy, cx, cy]
        )
        reconstruction.add_camera(colmap_camera, camera_id=idx)

        # Add image to the reconstruction
        r, t = split_rt(torch.linalg.inv(camera_t_world))
        qvec = pycolmap.rotmat_to_qvec(r.cpu().numpy())
        tvec = t.cpu().numpy()
        
        image_name = trainer.dataset.all_cameras[idx].filename
        reconstruction.add_image(
            pycolmap.Image(
                name=image_name,
                camera_id=idx,
                qvec=qvec,
                tvec=tvec
            ),
            image_id=idx
        )
    # Export the reconstruction to a COLMAP format
    output_path = Path("sparse") / "0"
    output_path.mkdir(parents=True, exist_ok=True)

    cloud = Trainer.get_initial_points(trainer.config, trainer.dataset)

    positions = cloud.position.cpu().numpy()
    colors = (cloud.color.cpu().numpy() * 255).astype(np.uint8)

    # Add points to the reconstruction
    for i, (position, color) in enumerate(zip(positions, colors)):
        reconstruction.add_point3D(
            xyz=position,
            color=color,
            track=pycolmap.Track()
        )

    print(f"Added {len(cloud.position)} points to the reconstruction")

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