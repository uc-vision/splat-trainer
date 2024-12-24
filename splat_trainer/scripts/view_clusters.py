from multiprocessing import Queue
from taichi_splatting import Gaussians3D
import torch.nn.functional as F

from splat_trainer.camera_table.camera_table import Cameras, Camera
from splat_trainer.scene.io import read_gaussians
from splat_trainer.scripts.checkpoint import arguments, with_trainer
from splat_trainer.trainer import Trainer


from splat_viewer.viewer import Viewer, show_workspace, FOVCamera
from splat_viewer.gaussians import Workspace, Gaussians

from splat_trainer.visibility.cluster import PointClusters


def to_fov_camera(camera:Camera) -> FOVCamera:
  near, far = camera.depth_range
  return FOVCamera(position=camera.position, 
                   rotation=camera.rotation, 
                   image_size=camera.image_size,
                   image_name=camera.image_name,
                   
                   focal_length=camera.focal_length,
                   principal_point=camera.principal_point,

                   near=near,
                   far=far)


def main():
  parser = arguments()
  parser.add_argument("--k", type=int, default=1024, help="Number of clusters")
  
  args = parser.parse_args()


  def f(trainer:Trainer):
    paths = trainer.paths()

    if paths.point_cloud.exists():  
      gaussians = read_gaussians(paths.point_cloud, with_sh=True)
    else:
      gaussians = trainer.scene.to_sh_gaussians()
    gaussians = gaussians.to(trainer.device)

    batch_indexes = trainer.select_batch(args.batch_size, temperature=args.temperature)
    cameras = trainer.camera_table.cameras[batch_indexes]

    workspace = Workspace.load(paths.workspace).replace(
      cameras=[to_fov_camera(c.item()) for c in cameras])
    
    clusters = PointClusters(gaussians.position, args.k)
    gaussian_labelled = Gaussians.from_gaussians3d(gaussians).replace(
      instance_label=clusters.point_labels.unsqueeze(1))
  
    show_workspace(workspace, gaussians=gaussian_labelled)

  with_trainer(f, args)
