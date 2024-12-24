from typing import Iterator, Tuple

from taichi_splatting import Rendering
from taichi_splatting.torch_lib.projection import ndc_depth, CameraParams
import torch
import torch.nn.functional as F
from tqdm import tqdm

from splat_trainer.camera_table.camera_table import Cameras
from splat_trainer.scene.io import read_gaussians
from splat_trainer.scene.scene import GaussianScene
from splat_trainer.scripts.checkpoint import arguments, with_trainer
from splat_trainer.scripts.view_batching import sh_gaussians_to_cloud
from splat_trainer.trainer import Trainer

from splat_trainer.util.view_cameras import CameraViewer
from splat_trainer.visibility import cluster
from splat_trainer.visibility.query_points import foreground_points
import fast_pytorch_kmeans
import distinctipy



def evaluate_visibility(scene:GaussianScene, 
                        cameras:Cameras,  image_scale:float=1.0
                        ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
  
  for i in range(cameras.batch_size[0]):
    camera_params:CameraParams = cameras[i].item().resized(image_scale).to_camera_params()
    rendering:Rendering = scene.render(camera_params, compute_visibility=True)
    
    # depth = ndc_depth(rendering.point_depth, near=camera_params.near_plane, far=camera_params.far_plane)
    idx, vis = rendering.visible
    
    yield idx, vis #/ depth.squeeze(1)

def get_visibility_features(scene:GaussianScene, cameras:Cameras, num_clusters:int):
  visibility_clusters = cluster.PointClusters(scene.points.position, num_clusters)
  
  for i, (idx, vis) in tqdm(enumerate(evaluate_visibility(scene, cameras)), 
                            total=cameras.batch_size[0], 
                            desc="Evaluating visibility"):
    visibility_clusters.add_visible(i, idx, vis)

  return visibility_clusters


def cluster_views(features: torch.Tensor, n: int):
  kmeans = fast_pytorch_kmeans.KMeans(n_clusters=n, mode="euclidean")
  return kmeans.fit_predict(features)



def make_colors(k:int):
  """ Make k distinct colors """
  # Generate k visually distinct colors
  colors = distinctipy.get_colors(k, pastel_factor=0.0)
  return torch.tensor(colors, dtype=torch.float32)

def main():
  parser = arguments()
  parser.add_argument("--k", type=int, default=4, help="Number of clusters")
  args = parser.parse_args()


  def f(trainer:Trainer):
    paths = trainer.paths()

    if paths.point_cloud.exists():  
      gaussians = read_gaussians(paths.point_cloud, with_sh=True)
    else:
      gaussians = trainer.scene.to_sh_gaussians()
    gaussians = gaussians.to(trainer.device)

    point_cloud = sh_gaussians_to_cloud(gaussians)
    cameras:Cameras = trainer.camera_table.cameras

    fg_mask = foreground_points(cameras, point_cloud.points)
    point_cloud = point_cloud[fg_mask]
    
    camera_viewer = CameraViewer(cameras, point_cloud)

    vis_clusters = get_visibility_features(trainer.scene, cameras, 8192)
    features = vis_clusters.feature_matrix(normalize=False)
    
    label_colors = make_colors(args.k).to(trainer.device)

    while camera_viewer.is_active():
      labels = cluster_views(features, args.k)
      camera_viewer.colorize_cameras(label_colors[labels])

      camera_viewer.wait_key()

  with_trainer(f, args)
